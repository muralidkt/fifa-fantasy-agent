"""Compute your team's matchday score from FIFA's official per-player points.

FIFA publishes each player's actual points in the same public `players.json`
(`stats.roundPoints` -> per round). This module reads those for your saved squad and applies
the game's scoring mechanics: starting XI only, captain doubled (vice if the captain didn't
play), and auto-substitutions (a non-playing starter is replaced, in bench order, by the next
bench player who played and keeps the formation legal). Pre-tournament every points value is
0, so a scorecard before any games shows 0 and flags "not played yet".

Also provides `points_from_stats()` — the rules-based scoring of raw match events — for
what-if checks and to keep the encoded scoring (rules.py) honest against reality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ingest.fifa import Dataset
from .optimize import Row
from .optimize.squad import choose_xi
from .rules import FORMATION_BOUNDS, SCORING, XI_SIZE


@dataclass
class PlayerScore:
    pid: int
    name: str
    position: str
    points: float
    role: str          # "starter" | "bench" | "subbed-in" | "subbed-out" | "(C)" | "(V→C)"
    played: bool


@dataclass
class ScoreCard:
    round_id: int
    total: float
    xi_points: float
    captain_id: int | None
    captain_name: str
    captain_bonus: float
    hits: int
    autosubs: list[tuple[int, int]] = field(default_factory=list)  # (out_pid, in_pid)
    lines: list[PlayerScore] = field(default_factory=list)
    played_any: bool = False


def player_points(ds: Dataset, round_id: int) -> dict[int, float]:
    """player_id -> official FIFA points for the round."""
    return {p.id: p.points_for_round(round_id) for p in ds.players}


def _played(points: float) -> bool:
    # No minutes field is exposed, so we treat any non-zero score as "featured".
    return points != 0


def score_matchday(ds: Dataset, squad: dict, round_id: int, *, hits: int = 0) -> ScoreCard:
    """Score a saved squad dict (keys: player_ids, [starters], [bench], captain, [vice])."""
    pts = player_points(ds, round_id)
    pos = {p.id: p.position for p in ds.players}
    names = {p.id: p.name for p in ds.players}
    squad_ids = squad["player_ids"]
    played_any = any(pts.get(pid, 0) for pid in squad_ids)

    starters = squad.get("starters")
    bench = squad.get("bench")
    if not starters:  # older save / no XI stored: pick best legal XI by actual points
        starters, bench = _fallback_xi(squad_ids, pos, pts)

    final_xi, subs = _apply_autosubs(starters, bench or [], pos, pts)
    final_set = set(final_xi)

    captain, vice = squad.get("captain"), squad.get("vice")
    eff_captain = captain if (captain in final_set and _played(pts.get(captain, 0))) else vice
    captain_bonus = (
        pts.get(eff_captain, 0) * (SCORING.captain_multiplier - 1)
        if eff_captain in final_set else 0.0
    )

    xi_points = sum(pts.get(pid, 0) for pid in final_xi)
    total = xi_points + captain_bonus - hits

    subbed_out = {o for o, _ in subs}
    subbed_in = {i for _, i in subs}
    lines: list[PlayerScore] = []
    for pid in squad_ids:
        if pid == eff_captain and pid in final_set:
            role = "(C)" if pid == captain else "(V→C)"
        elif pid in subbed_in:
            role = "subbed-in"
        elif pid in subbed_out:
            role = "subbed-out"
        elif pid in final_set:
            role = "starter"
        else:
            role = "bench"
        lines.append(PlayerScore(pid, names.get(pid, str(pid)), pos.get(pid, "?"),
                                 pts.get(pid, 0), role, _played(pts.get(pid, 0))))
    lines.sort(key=lambda s: (s.role == "bench", {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}.get(s.position, 9)))

    return ScoreCard(
        round_id=round_id, total=round(total, 1), xi_points=round(xi_points, 1),
        captain_id=eff_captain, captain_name=names.get(eff_captain, "—"),
        captain_bonus=round(captain_bonus, 1), hits=hits, autosubs=subs,
        lines=lines, played_any=played_any,
    )


def _apply_autosubs(starters, bench_order, pos, pts):
    """FIFA-style auto-subs: replace non-playing starters, in bench order, keeping formation legal."""
    final = list(starters)
    counts = _counts(final, pos)
    used: set[int] = set()
    subs: list[tuple[int, int]] = []
    for s in starters:
        if _played(pts.get(s, 0)):
            continue
        for b in bench_order:
            if b in used or not _played(pts.get(b, 0)):
                continue
            trial = dict(counts)
            trial[pos[s]] -= 1
            trial[pos.get(b, "?")] = trial.get(pos.get(b, "?"), 0) + 1
            if _formation_ok(trial):
                final[final.index(s)] = b
                counts = trial
                used.add(b)
                subs.append((s, b))
                break
    return final, subs


def _counts(ids, pos) -> dict[str, int]:
    c: dict[str, int] = {}
    for pid in ids:
        c[pos.get(pid, "?")] = c.get(pos.get(pid, "?"), 0) + 1
    return c


def _formation_ok(counts: dict[str, int]) -> bool:
    if sum(counts.values()) != XI_SIZE:
        return False
    return all(lo <= counts.get(p, 0) <= hi for p, (lo, hi) in FORMATION_BOUNDS.items())


def _fallback_xi(squad_ids, pos, pts):
    rows = [Row(pid=pid, name="", position=pos.get(pid, "?"), price=0.0,
                squad_id=0, squad_name="", xpts=pts.get(pid, 0)) for pid in squad_ids]
    sel = choose_xi(rows)
    return sel.starter_ids, sel.bench_ids


# --- rules-based scoring of raw events (what-if / verification) ---------------
def points_from_stats(position: str, *, minutes: int = 0, goals: int = 0, assists: int = 0,
                      clean_sheet: bool = False, conceded: int = 0, saves: int = 0,
                      pens_saved: int = 0, pens_missed: int = 0, own_goals: int = 0,
                      yellow: int = 0, red: int = 0) -> float:
    """Points a player would score for a set of match events, per rules.py SCORING."""
    s = SCORING
    p = 0.0
    if minutes >= 60:
        p += s.appearance_long
    elif minutes >= 1:
        p += s.appearance_short
    p += goals * s.goal.get(position, 0)
    p += assists * s.assist
    if clean_sheet and minutes >= 60:
        p += s.clean_sheet.get(position, 0)
    if position in ("GK", "DEF"):
        p += (s.goals_conceded_per) * (conceded // s.goals_conceded_unit)
    if position == "GK":
        p += s.saves_per * (saves // s.saves_unit)
        p += pens_saved * s.penalty_saved
    p += pens_missed * s.penalty_missed
    p += own_goals * s.own_goal
    p += yellow * s.yellow_card
    p += red * s.red_card
    return float(p)
