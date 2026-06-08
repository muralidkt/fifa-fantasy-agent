"""Deterministic, JSON-returning tools the conversational agent can call.

Every tool reuses the existing engine (``engine.py``, ``optimize``, ``scoring``, ``rules``)
and returns a plain dict so it can be tested without an LLM and fed straight back to Claude.
The two *write* tools (``propose_override``, ``save_squad``) never touch disk on their own:
they take a ``confirm`` callback and only write when it returns True — that callback is the
human y/N gate owned by ``loop.py``. The ILP remains the only thing that selects a team;
``lock``/``ban`` are hard constraints handed to the solver, not picks made by the model.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .. import rules, scoring
from ..engine import _load_squad, _pipeline, _planning_rows, _save_squad
from ..ingest.ratings import normalize_name
from ..model.expected_points import load_projection_overrides
from ..optimize import rows_from_projections
from ..optimize.squad import optimize_squad
from ..optimize.transfers import (
    optimize_squad_sequence,
    optimize_transfers,
    optimize_transfers_sequence,
)

# Tools that mutate files. The loop must route these through the human confirm gate.
WRITE_TOOLS = {"propose_override", "save_squad"}

# Override fields the agent may set, with validation. (Mirrors data/overrides.yaml schema.)
_UNIT_FIELDS = {"start_prob", "quality", "goal_share", "assist_share"}  # [0, 1]
_NONNEG_FIELDS = {"penalty_xg", "xpts_multiplier"}                       # >= 0
_FREE_FIELDS = {"xpts_delta"}                                           # any float
_FLAG_FIELDS = {"captain_avoid"}                                        # bool
_TEXT_FIELDS = {"notes"}                                                # str
_OVERRIDE_FIELDS = _UNIT_FIELDS | _NONNEG_FIELDS | _FREE_FIELDS | _FLAG_FIELDS | _TEXT_FIELDS


@dataclass
class _LastResult:
    kind: str            # "build" | "advise"
    round_id: int
    ds: Any
    selected: Any        # SelectedSquad
    banked: int


class AgentTools:
    """Stateful façade: caches projections per round and remembers the last build/advise so a
    follow-up like 'save that' or 'now bench him' works without recomputing context."""

    def __init__(self, cfg: dict, *, default_round: int | None = None,
                 odds: bool = True, ratings: bool = True, mode: str | None = None):
        self.cfg = cfg
        self.default_round = default_round
        self.odds = odds
        self.ratings = ratings
        self.mode = mode or cfg["optimize"].get("mode", "balanced")
        self.cache_dir = cfg["data"]["cache_dir"]
        self._proj_cache: dict[tuple, tuple] = {}
        self.last: _LastResult | None = None

    # --- internal helpers ---------------------------------------------------
    def _round(self, round_id: int | None) -> int:
        rid = round_id if round_id is not None else self.default_round
        if rid is None:
            raise ValueError("no round specified and no default round set")
        return int(rid)

    def _pipeline(self, round_id: int, odds: bool | None, ratings: bool | None):
        odds = self.odds if odds is None else odds
        ratings = self.ratings if ratings is None else ratings
        key = (round_id, odds, ratings)
        if key not in self._proj_cache:
            self._proj_cache[key] = _pipeline(
                self.cfg, round_id, use_odds=odds, use_ratings=ratings)
        return self._proj_cache[key]

    def _resolve_player(self, ds, query) -> Any:
        """Resolve an id/name to a Player, raising with candidates if ambiguous/unknown."""
        by_id = {p.id: p for p in ds.players}
        q = str(query).strip()
        if q.isdigit() and int(q) in by_id:
            return by_id[int(q)]
        for p in ds.players:                      # exact name (case-insensitive)
            if p.name.lower() == q.lower():
                return p
        nq = normalize_name(q)
        exact = [p for p in ds.players if normalize_name(p.name) == nq]
        if len(exact) == 1:
            return exact[0]
        partial = [p for p in ds.players if nq and nq in normalize_name(p.name)]
        if len(partial) == 1:
            return partial[0]
        candidates = exact or partial
        if not candidates:
            raise ValueError(f"no player matches {query!r}")
        names = ", ".join(sorted(f"{p.name} (#{p.id})" for p in candidates[:8]))
        raise ValueError(f"{query!r} is ambiguous; did you mean: {names}")

    def _resolve_ids(self, ds, queries) -> set[int]:
        return {self._resolve_player(ds, q).id for q in (queries or [])}

    def _squad_summary(self, ds, projections, selected, round_id: int) -> dict:
        by_id = {p.id: p for p in ds.players}
        starters = set(selected.starter_ids)
        lines = []
        for pid in selected.squad_ids:
            pr = projections.get(pid)
            p = by_id.get(pid)
            team = ds.squads[p.squad_id].name if p and p.squad_id in ds.squads else "?"
            lines.append({
                "id": pid,
                "name": p.name if p else str(pid),
                "pos": p.position if p else "?",
                "team": team,
                "opp": pr.opponent_name if pr else "?",
                "price": round(p.price, 1) if p else None,
                "xpts": pr.xpts if pr else 0.0,
                "start_prob": pr.start_prob if pr else None,
                "role": "XI" if pid in starters else "bench",
                "captain": pid == selected.captain_id,
                "vice": pid == selected.vice_id,
            })
        cap, vc = by_id.get(selected.captain_id), by_id.get(selected.vice_id)
        return {
            "round": round_id,
            "formation": selected.formation,
            "captain": cap.name if cap else str(selected.captain_id),
            "vice": vc.name if vc else str(selected.vice_id),
            "total_cost": round(selected.total_cost, 1),
            "xi_xpts": round(selected.xi_xpts, 2),
            "total_xpts_with_captain": round(selected.total_xpts, 2),
            "players": lines,
        }

    # --- read / compute tools ----------------------------------------------
    def show_squad(self) -> dict:
        path = Path(self.cache_dir) / "my_squad.yaml"
        if not path.exists():
            return {"saved": False, "message": "No squad saved yet. Run build_squad then save_squad."}
        data = _load_squad(str(path))
        return {"saved": True, **{k: data.get(k) for k in
                ("round", "banked_transfer", "player_ids", "starters", "bench",
                 "captain", "vice", "names")}}

    def build_squad(self, round: int | None = None, horizon: int = 1,
                    odds: bool | None = None, ratings: bool | None = None,
                    mode: str | None = None, lock: list | None = None,
                    ban: list | None = None) -> dict:
        round_id = self._round(round)
        ds, projections, rnd = self._pipeline(round_id, odds, ratings)
        mode = mode or self.mode
        rows = rows_from_projections(projections, ds.squads, mode=mode)
        lock_ids = self._resolve_ids(ds, lock)
        ban_ids = self._resolve_ids(ds, ban)
        note = None
        if (lock_ids or ban_ids) and horizon > 1:
            note = "lock/ban apply to a single round; horizon look-ahead was disabled."
            horizon = 1
        if horizon > 1:
            rows_by_round, budgets, caps, windows = _planning_rows(
                self.cfg, round_id, horizon, rows, use_elo=False,
                use_odds=self.odds if odds is None else odds,
                use_ratings=self.ratings if ratings is None else ratings, mode=mode)
            selected = optimize_squad_sequence(
                rows_by_round, free_transfers_after_round=windows,
                budget_by_round=budgets, nation_cap_by_round=caps,
                bench_weight=self.cfg["optimize"]["bench_weight"],
                hit_threshold=self.cfg["optimize"]["hit_threshold"])
        else:
            selected = optimize_squad(
                rows, budget=rules.budget_for_stage(rnd.stage),
                nation_cap=rules.nation_cap_for_stage(rnd.stage),
                bench_weight=self.cfg["optimize"]["bench_weight"],
                lock_ids=lock_ids, ban_ids=ban_ids)
        self.last = _LastResult("build", round_id, ds, selected, banked=0)
        out = self._squad_summary(ds, projections, selected, round_id)
        if note:
            out["note"] = note
        return out

    def advise_transfers(self, round: int | None = None, free: float | None = None,
                         horizon: int = 1, odds: bool | None = None,
                         ratings: bool | None = None, mode: str | None = None,
                         lock: list | None = None, ban: list | None = None) -> dict:
        round_id = self._round(round)
        path = Path(self.cache_dir) / "my_squad.yaml"
        if not path.exists():
            return {"error": "No saved squad. Build and save a squad before asking for transfers."}
        squad_state = _load_squad(str(path))
        current_ids = squad_state["player_ids"]
        banked = int(squad_state.get("banked_transfer", 0))

        ds, projections, rnd = self._pipeline(round_id, odds, ratings)
        mode = mode or self.mode
        rows = rows_from_projections(projections, ds.squads, mode=mode)
        lock_ids = self._resolve_ids(ds, lock)
        ban_ids = self._resolve_ids(ds, ban)
        free_n = free if free is not None else rules.free_transfers_for_round(round_id, banked)
        note = None
        if (lock_ids or ban_ids) and horizon > 1:
            note = "lock/ban apply to a single round; horizon look-ahead was disabled."
            horizon = 1
        if horizon > 1:
            rows_by_round, budgets, caps, windows = _planning_rows(
                self.cfg, round_id, horizon, rows, use_elo=False,
                use_odds=self.odds if odds is None else odds,
                use_ratings=self.ratings if ratings is None else ratings, mode=mode)
            plan = optimize_transfers_sequence(
                rows_by_round, current_ids, free_transfers_by_round=[free_n] + windows,
                budget_by_round=budgets, nation_cap_by_round=caps,
                bench_weight=self.cfg["optimize"]["bench_weight"],
                hit_threshold=self.cfg["optimize"]["hit_threshold"])
        else:
            plan = optimize_transfers(
                rows, current_ids, free_transfers=free_n,
                budget=rules.budget_for_stage(rnd.stage),
                nation_cap=rules.nation_cap_for_stage(rnd.stage),
                bench_weight=self.cfg["optimize"]["bench_weight"],
                hit_threshold=self.cfg["optimize"]["hit_threshold"],
                lock_ids=lock_ids, ban_ids=ban_ids)
        banked_after = rules.banked_transfer_after_round(round_id, free_n, plan.transfers)
        self.last = _LastResult("advise", round_id, ds, plan.squad, banked=banked_after)
        names = {p.id: p.name for p in ds.players}
        out = {
            "round": round_id,
            "free_transfers": "unlimited" if free_n == float("inf") else free_n,
            "transfers_in": [names.get(i, i) for i in plan.in_ids],
            "transfers_out": [names.get(i, i) for i in plan.out_ids],
            "paid_transfers": plan.paid_transfers,
            "hit_points": plan.hit_points,
            "squad": self._squad_summary(ds, projections, plan.squad, round_id),
        }
        if note:
            out["note"] = note
        return out

    def score_round(self, round: int | None = None, hits: int = 0) -> dict:
        round_id = self._round(round)
        path = Path(self.cache_dir) / "my_squad.yaml"
        if not path.exists():
            return {"error": "No saved squad to score."}
        squad = _load_squad(str(path))
        from ..ingest import fifa
        ds = fifa.load(self.cache_dir, ttl_hours=self.cfg["data"]["cache_ttl_hours"])
        card = scoring.score_matchday(ds, squad, round_id, hits=hits)
        return {
            "round": card.round_id,
            "total": card.total,
            "xi_points": card.xi_points,
            "captain": card.captain_name,
            "captain_bonus": card.captain_bonus,
            "hits": card.hits,
            "played_any": card.played_any,
            "autosubs": card.autosubs,
            "lines": [{"name": l.name, "pos": l.position, "points": l.points, "role": l.role}
                      for l in card.lines],
        }

    def explain_player(self, player: str, round: int | None = None,
                       odds: bool | None = None, ratings: bool | None = None) -> dict:
        round_id = self._round(round)
        ds, projections, rnd = self._pipeline(round_id, odds, ratings)
        p = self._resolve_player(ds, player)
        pr = projections.get(p.id)
        overrides = load_projection_overrides(self.cfg["expected_points"].get("overrides_path"))
        has_override = (str(p.id) in overrides or p.name in overrides
                        or normalize_name(p.name) in overrides)
        if pr is None:
            return {"name": p.name, "round": round_id, "message": "No projection (no fixture this round)."}
        return {
            "name": p.name,
            "id": p.id,
            "position": p.position,
            "team": ds.squads[p.squad_id].name if p.squad_id in ds.squads else "?",
            "price": p.price,
            "round": round_id,
            "opponent": pr.opponent_name,
            "xpts": pr.xpts,
            "start_prob": pr.start_prob,
            "quality": pr.quality,
            "components": pr.components,
            "ownership": getattr(p, "ownership", None),
            "one_to_watch": getattr(p, "one_to_watch", False),
            "override_active": bool(has_override),
        }

    def explain_optimizer(self) -> dict:
        return {
            "objective": ("Maximise sum(xPts over the 11 starters) + bench_weight * sum(xPts "
                          "over the 4 bench). Solving the 15 and the XI together so the budget "
                          "is spent on players who actually start."),
            "solver": "Integer Linear Program (PuLP / CBC). The LLM never selects — it can only "
                      "add lock/ban constraints or edit projection overrides and re-solve.",
            "hard_constraints": {
                "squad_size": rules.SQUAD_SIZE,
                "composition": rules.SQUAD_COMPOSITION,
                "xi_size": rules.XI_SIZE,
                "formation_bounds": {k: list(v) for k, v in rules.FORMATION_BOUNDS.items()},
                "budget_$m": {"group": rules.budget_for_stage("GROUP"),
                              "knockout": rules.budget_for_stage("R32")},
                "nation_cap_by_stage": {s: rules.nation_cap_for_stage(s)
                                        for s in ("GROUP", "R32", "R16", "QF", "SF", "F")},
            },
            "levers_available_to_you": {
                "lock": "force a player into the squad (hard x==1 constraint)",
                "ban": "force a player out of the squad (hard x==0 constraint)",
                "overrides": "soft signals via data/overrides.yaml (start_prob, captain_avoid, "
                             "xpts_multiplier, ...) that nudge projections before the solve",
                "mode": "balanced | safe | upside | differential | template",
                "horizon": "value this round plus N-1 future rounds (group stage only)",
            },
        }

    def get_rules(self, topic: str | None = None) -> dict:
        verify = _verify_in_app_fields()
        scoring_tbl = asdict(rules.SCORING)
        out = {
            "squad": {"size": rules.SQUAD_SIZE, "composition": rules.SQUAD_COMPOSITION,
                      "xi_size": rules.XI_SIZE,
                      "formation_bounds": {k: list(v) for k, v in rules.FORMATION_BOUNDS.items()}},
            "budget_$m": {"group": rules.budget_for_stage("GROUP"),
                          "knockout": rules.budget_for_stage("R32")},
            "nation_cap_by_stage": {s: rules.nation_cap_for_stage(s)
                                    for s in ("GROUP", "R32", "R16", "QF", "SF", "F")},
            "free_transfers_by_round": {k: ("unlimited" if v == float("inf") else v)
                                        for k, v in rules.FREE_TRANSFERS_BY_ROUND.items()},
            "transfer_hit": rules.TRANSFER_HIT,
            "scoring": scoring_tbl,
            "captain_multiplier": rules.SCORING.captain_multiplier,
            "verify_in_app": verify,
            "_note": ("Fields under 'verify_in_app' are disputed in 2026 sources — tell the user "
                      "to confirm them on play.fifa.com rather than asserting them as fact."),
        }
        if topic:
            t = topic.lower()
            filtered = {k: v for k, v in out.items()
                        if t in k.lower() or (isinstance(v, dict) and any(t in str(kk).lower() for kk in v))}
            out = {"topic": topic, **(filtered or out)}
        return out

    def list_overrides(self) -> dict:
        path = self.cfg["expected_points"].get("overrides_path")
        data = load_projection_overrides(path)
        return {"path": path, "count": len(data), "overrides": data}

    # --- write tools (gated by confirm) ------------------------------------
    def propose_override(self, player: str, fields: dict, round: int | None = None,
                         confirm: Callable[[str], bool] | None = None) -> dict:
        clean, errors = _validate_override_fields(fields)
        if errors:
            return {"error": "invalid override fields", "details": errors,
                    "allowed_fields": sorted(_OVERRIDE_FIELDS)}
        if not clean:
            return {"error": "no valid fields to set"}
        # Resolve the player against whatever round context we have (default round).
        ds, _, _ = self._pipeline(self._round(round), None, None)
        p = self._resolve_player(ds, player)

        path = Path(self.cfg["expected_points"].get("overrides_path", "data/overrides.yaml"))
        container, players = _read_overrides_file(path)
        before = dict(players.get(p.name, {}))
        after = dict(before)
        if round is not None:
            rounds = dict(after.get("rounds", {}))
            scoped = dict(rounds.get(int(round), {}))
            scoped.update(clean)
            rounds[int(round)] = scoped
            after["rounds"] = rounds
        else:
            after.update(clean)

        diff = _diff_text(p.name, before, after, round)
        if confirm is None or not confirm(diff):
            return {"applied": False, "reason": "declined by user", "diff": diff}
        players[p.name] = after
        container["players"] = players
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(container, allow_unicode=True, sort_keys=False))
        self._proj_cache.clear()  # projections must be recomputed with the new override
        return {"applied": True, "player": p.name, "fields": clean,
                "round": round, "path": str(path)}

    def save_squad(self, source: str = "build",
                   confirm: Callable[[str], bool] | None = None) -> dict:
        if self.last is None:
            return {"error": "Nothing to save — run build_squad or advise_transfers first."}
        if source not in ("build", "advise"):
            return {"error": "source must be 'build' or 'advise'"}
        last = self.last
        names = {p.id: p.name for p in last.ds.players}
        new_ids = set(last.selected.squad_ids)
        path = Path(self.cache_dir) / "my_squad.yaml"
        old_ids = set(_load_squad(str(path))["player_ids"]) if path.exists() else set()
        added = sorted(names.get(i, str(i)) for i in new_ids - old_ids)
        removed = sorted(names.get(i, str(i)) for i in old_ids - new_ids)
        diff = (f"Save {last.kind} result for round {last.round_id} to {path}\n"
                f"  + in : {', '.join(added) or '—'}\n"
                f"  - out: {', '.join(removed) or '—'}\n"
                f"  captain: {names.get(last.selected.captain_id)}  "
                f"banked: {last.banked}")
        if confirm is None or not confirm(diff):
            return {"applied": False, "reason": "declined by user", "diff": diff}
        _save_squad(self.cache_dir, last.ds, last.selected, last.round_id,
                    banked_transfer=last.banked)
        return {"applied": True, "round": last.round_id, "path": str(path),
                "added": added, "removed": removed}


# --- module-level helpers ---------------------------------------------------
def _validate_override_fields(fields: dict) -> tuple[dict, list[str]]:
    clean: dict[str, Any] = {}
    errors: list[str] = []
    for k, v in (fields or {}).items():
        if k not in _OVERRIDE_FIELDS:
            errors.append(f"unknown field {k!r}")
            continue
        try:
            if k in _FLAG_FIELDS:
                clean[k] = bool(v)
            elif k in _TEXT_FIELDS:
                clean[k] = str(v)
            elif k in _UNIT_FIELDS:
                fv = float(v)
                if not 0.0 <= fv <= 1.0:
                    errors.append(f"{k}={v} out of range [0, 1]")
                else:
                    clean[k] = fv
            elif k in _NONNEG_FIELDS:
                fv = float(v)
                if fv < 0:
                    errors.append(f"{k}={v} must be >= 0")
                else:
                    clean[k] = fv
            else:  # free float
                clean[k] = float(v)
        except (TypeError, ValueError):
            errors.append(f"{k}={v!r} is not the right type")
    return clean, errors


def _read_overrides_file(path: Path) -> tuple[dict, dict]:
    """Return (container, players) where container is the full doc to write back."""
    if not path.exists():
        return {}, {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        return {}, {}
    if isinstance(raw.get("players"), dict):
        return raw, dict(raw["players"])
    # File is a bare name->override map; normalise to a players-wrapped doc on write.
    return {}, {k: v for k, v in raw.items() if isinstance(v, dict)}


def _diff_text(name: str, before: dict, after: dict, round: int | None) -> str:
    scope = f" (round {round})" if round is not None else ""
    keys = sorted(set(before) | set(after))
    lines = [f"Override for {name}{scope}:"]
    for k in keys:
        b, a = before.get(k), after.get(k)
        if b == a:
            continue
        lines.append(f"  {k}: {b!r} -> {a!r}")
    if len(lines) == 1:
        lines.append("  (no change)")
    return "\n".join(lines)


def _verify_in_app_fields() -> dict[str, str]:
    """Scan rules.py for `# VERIFY in-app` markers so the agent flags disputed values."""
    src = Path(rules.__file__).read_text().splitlines()
    out: dict[str, str] = {}
    for line in src:
        if "VERIFY in-app" not in line:
            continue
        m = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]", line)
        if m:
            note = line.split("# VERIFY in-app", 1)[1].strip() or "confirm on play.fifa.com"
            out[m.group(1)] = note
    return out
