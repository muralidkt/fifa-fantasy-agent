"""Shared engine helpers: config, the data→projection pipeline, planning rows, persistence.

These were lifted out of ``cli.py`` so both the Click commands and the conversational agent
(``agent/``) drive the exact same deterministic pipeline. Nothing here is LLM-aware beyond the
optional ``llm_adjust`` layer the CLI already exposed; the ILP remains the only thing that
selects a team.
"""
from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console

from .ingest import fifa, fixture_odds, ratings, strength
from .model import llm_adjust
from .model.expected_points import load_projection_overrides, project_round
from .model.opponent import OpponentModel
from .optimize import rows_from_projections
from . import rules

# record=True lets the CLI export everything rendered this command to a styled HTML file.
console = Console(record=True)


DEFAULTS = {
    "data": {"cache_dir": "data", "cache_ttl_hours": 6},
    "strength": {"source": "price", "odds_stage": "Quarterfinals",
                 "beta": 1.5, "base_goals": 1.35, "home_advantage": 0.20},
    "expected_points": {
        "form_weight": 0.55, "price_weight": 0.45, "ratings_weight": 0.35,
        "one_to_watch_bonus": 0.10, "start_prob_floor": 0.15,
        "overrides_path": "data/overrides.yaml", "captain_start_weight": 0.75,
        "fixture_odds_path": "data/fixture_odds.yaml",
    },
    "optimize": {"bench_weight": 0.12, "hit_threshold": 1.0,
                 "mode": "balanced", "horizon_decay": 0.70},
    "llm": {"enabled": False, "model": "claude-sonnet-4-6", "top_n": 40},
}


def _load_config(path: str = "config.yaml") -> dict:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    p = Path(path)
    if p.exists():
        loaded = yaml.safe_load(p.read_text()) or {}
        for section, values in loaded.items():
            cfg.setdefault(section, {}).update(values or {})
    return cfg


def _deadline(rnd: fifa.Round) -> str:
    return rnd.start_date.replace("T", " ")


def _pipeline(cfg: dict, round_id: int, *, use_elo: bool = False, use_odds: bool = False,
              use_ratings: bool = False, use_llm: bool = False, news: str | None = None):
    """Load data and compute per-player projections for a round. Returns (ds, projections, round)."""
    cache = cfg["data"]["cache_dir"]
    ds = fifa.load(cache, ttl_hours=cfg["data"]["cache_ttl_hours"])
    rnd = ds.round(round_id)

    src = "odds" if use_odds else "elo" if use_elo else cfg["strength"]["source"]
    strengths = strength.team_strength(
        ds, source=src, cache_dir=cache, stage=cfg["strength"]["odds_stage"],
    )
    if use_odds:
        console.print(f"[dim]strength: Polymarket implied 'reach {cfg['strength']['odds_stage']}' odds[/dim]")
    model = OpponentModel(
        strengths,
        beta=cfg["strength"]["beta"],
        base_goals=cfg["strength"]["base_goals"],
        home_advantage=cfg["strength"]["home_advantage"],
    )

    rating_pct, ratings_weight = None, 0.0
    if use_ratings:
        raw = ratings.load_overall_ratings(cache)
        if raw:
            rating_pct = ratings.rating_percentiles(raw)
            ratings_weight = cfg["expected_points"]["ratings_weight"]
            matched = sum(1 for p in ds.players if ratings.normalize_name(p.name) in rating_pct)
            console.print(f"[dim]SoFIFA ratings: {len(raw)} loaded, matched {matched} players[/dim]")
        else:
            console.print("[yellow]--ratings set but data/sofifa.csv is empty/missing "
                          "(run `fantasy fetch-ratings`).[/yellow]")

    ep = cfg["expected_points"]
    overrides = load_projection_overrides(ep.get("overrides_path"))
    fixture_expectations = fixture_odds.load_fixture_expectations(ep.get("fixture_odds_path"), ds)
    projections = project_round(
        ds, round_id, model,
        form_weight=ep["form_weight"], price_weight=ep["price_weight"],
        one_to_watch_bonus=ep["one_to_watch_bonus"], start_prob_floor=ep["start_prob_floor"],
        rating_pct=rating_pct, ratings_weight=ratings_weight,
        overrides=overrides, captain_start_weight=ep.get("captain_start_weight", 0.75),
        fixture_expectations=fixture_expectations,
    )
    if overrides:
        console.print(f"[dim]projection overrides: {len(overrides)} loaded from {ep.get('overrides_path')}[/dim]")
    if fixture_expectations:
        console.print(f"[dim]fixture odds: {len(fixture_expectations) // 2} fixture(s) loaded from "
                      f"{ep.get('fixture_odds_path')}[/dim]")

    if use_llm or cfg["llm"]["enabled"]:
        adjustments = llm_adjust.adjust(
            ds, projections, model=cfg["llm"]["model"], top_n=cfg["llm"]["top_n"], news=news,
        )
        if adjustments:
            console.print(f"[dim]LLM adjusted {len(adjustments)} players[/dim]")
            projections = llm_adjust.apply(projections, adjustments)
        else:
            console.print("[yellow]LLM layer requested but inactive "
                          "(install `anthropic` and set ANTHROPIC_API_KEY).[/yellow]")
    return ds, projections, rnd


def _planning_rows(
    cfg: dict,
    round_id: int,
    horizon: int,
    current_rows,
    *,
    use_elo: bool,
    use_odds: bool,
    use_ratings: bool,
    mode: str,
):
    rows_by_round = [(current_rows, 1.0)]
    budgets = []
    caps = []
    transfer_windows = []
    decay = float(cfg["optimize"].get("horizon_decay", 0.70))
    stage = rules.stage_for_round(round_id)
    budgets.append(rules.budget_for_stage(stage))
    caps.append(rules.nation_cap_for_stage(stage))
    for offset in range(1, horizon):
        future_round = round_id + offset
        if future_round > 8:
            break
        try:
            f_ds, f_proj, f_rnd = _pipeline(
                cfg, future_round, use_elo=use_elo, use_odds=use_odds,
                use_ratings=use_ratings, use_llm=False, news=None,
            )
        except Exception:
            break
        if not f_rnd.fixtures:
            break
        rows_by_round.append((rows_from_projections(f_proj, f_ds.squads, mode=mode), decay ** offset))
        budgets.append(rules.budget_for_stage(f_rnd.stage))
        caps.append(rules.nation_cap_for_stage(f_rnd.stage))
        transfer_windows.append(rules.free_transfers_for_round(future_round))
    return rows_by_round, budgets, caps, transfer_windows


def _save_squad(cache_dir: str, ds: fifa.Dataset, selected, round_id: int, *, banked_transfer: int = 0) -> None:
    by_id = {p.id: p for p in ds.players}
    data = {
        "round": round_id,
        "banked_transfer": int(banked_transfer),
        "player_ids": selected.squad_ids,
        "starters": selected.starter_ids,
        "bench": selected.bench_ids,
        "captain": selected.captain_id,
        "vice": selected.vice_id,
        "names": {pid: by_id[pid].name for pid in selected.squad_ids if pid in by_id},
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "my_squad.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def _load_squad(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "player_ids" not in data:
        raise click.ClickException(f"Saved squad at {path} does not contain player_ids.")
    return data
