"""`fantasy` command-line interface: refresh data, build a squad, advise transfers."""
from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console

from .ingest import fifa, ratings, strength
from .model import llm_adjust
from .model.expected_points import load_projection_overrides, project_round
from .model.opponent import OpponentModel
from .optimize import rows_from_projections
from .optimize.squad import optimize_squad
from .optimize.transfers import optimize_transfers
from . import report
from . import rules

console = Console(record=True)  # record=True lets us export the rendered report to HTML


def _export_html(path: str) -> None:
    """Save everything rendered so far this command to a styled HTML file."""
    console.save_html(path, clear=False)
    console.print(f"[green]Wrote[/green] HTML report to {path}")

DEFAULTS = {
    "data": {"cache_dir": "data", "cache_ttl_hours": 6},
    "strength": {"source": "price", "odds_stage": "Quarterfinals",
                 "beta": 1.5, "base_goals": 1.35, "home_advantage": 0.20},
    "expected_points": {
        "form_weight": 0.55, "price_weight": 0.45, "ratings_weight": 0.35,
        "one_to_watch_bonus": 0.10, "start_prob_floor": 0.15,
        "overrides_path": "data/overrides.yaml", "captain_start_weight": 0.75,
    },
    "optimize": {"bench_weight": 0.12, "hit_threshold": 1.0},
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
    projections = project_round(
        ds, round_id, model,
        form_weight=ep["form_weight"], price_weight=ep["price_weight"],
        one_to_watch_bonus=ep["one_to_watch_bonus"], start_prob_floor=ep["start_prob_floor"],
        rating_pct=rating_pct, ratings_weight=ratings_weight,
        overrides=overrides, captain_start_weight=ep.get("captain_start_weight", 0.75),
    )
    if overrides:
        console.print(f"[dim]projection overrides: {len(overrides)} loaded from {ep.get('overrides_path')}[/dim]")

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


# --- commands ---------------------------------------------------------------
@click.group()
def main() -> None:
    """Autonomous FIFA World Cup 2026 Fantasy assistant (recommend-only)."""


@main.command()
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def refresh(config_path: str) -> None:
    """Download the latest FIFA public JSON into the data cache."""
    cfg = _load_config(config_path)
    cache = cfg["data"]["cache_dir"]
    console.print("Fetching FIFA fantasy data…")
    fifa.fetch(cache, force=True)
    ds = fifa.load(cache, auto_fetch=False)
    console.print(
        f"[green]OK[/green] {len(ds.squads)} teams · {len(ds.rounds)} rounds · "
        f"{sum(1 for p in ds.players if p.available)} active players cached in {cache}/"
    )


@main.command(name="fetch-ratings")
@click.option("--fc-version", default="latest", show_default=True, help="EA FC / SoFIFA version to pull.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def fetch_ratings(fc_version: str, config_path: str) -> None:
    """Populate data/sofifa.csv with SoFIFA overall ratings (needs `pip install -e '.[ratings]'`)."""
    cfg = _load_config(config_path)
    try:
        n = ratings.fetch_via_soccerdata(cfg["data"]["cache_dir"], fc_version=fc_version)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    console.print(f"[green]Wrote[/green] {n} ratings to {cfg['data']['cache_dir']}/sofifa.csv")


@main.command()
@click.option("--round", "round_id", default=1, show_default=True, help="Matchday/round id (1-8).")
@click.option("--elo", is_flag=True, help="Use World Football Elo from data/elo.csv for strength.")
@click.option("--odds", is_flag=True, help="Use Polymarket implied odds for team strength (free, live).")
@click.option("--ratings", "use_ratings", is_flag=True, help="Blend SoFIFA overall ratings (data/sofifa.csv) into quality.")
@click.option("--llm", is_flag=True, help="Enable the optional Claude adjustment layer.")
@click.option("--news", type=click.Path(exists=True), default=None, help="Team-news text file for the LLM layer.")
@click.option("--save", is_flag=True, help="Save the squad to data/my_squad.yaml for later `advise`.")
@click.option("--html", "html_path", default=None, help="Also write the rendered report to this HTML file.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def build(round_id: int, elo: bool, odds: bool, use_ratings: bool, llm: bool,
          news: str | None, save: bool, html_path: str | None, config_path: str) -> None:
    """Build the optimal squad + XI + captain for a round (default: Matchday 1)."""
    cfg = _load_config(config_path)
    news_text = Path(news).read_text() if news else None
    ds, projections, rnd = _pipeline(cfg, round_id, use_elo=elo, use_odds=odds,
                                     use_ratings=use_ratings, use_llm=llm, news=news_text)

    rows = rows_from_projections(projections, ds.squads)
    selected = optimize_squad(
        rows,
        budget=rules.budget_for_stage(rnd.stage),
        nation_cap=rules.nation_cap_for_stage(rnd.stage),
        bench_weight=cfg["optimize"]["bench_weight"],
    )
    report.render_squad(console, ds, projections, selected,
                        round_id=round_id, deadline=_deadline(rnd))

    if save:
        _save_squad(cfg["data"]["cache_dir"], ds, selected, round_id)
        console.print(f"[green]Saved[/green] squad to {cfg['data']['cache_dir']}/my_squad.yaml")
    if html_path:
        _export_html(html_path)


@main.command()
@click.option("--round", "round_id", required=True, type=int, help="Upcoming matchday/round id (1-8).")
@click.option("--squad", "squad_path", default=None, help="Path to saved squad (default data/my_squad.yaml).")
@click.option("--free", "free_override", default=None, type=float, help="Override free-transfer count.")
@click.option("--elo", is_flag=True, help="Use World Football Elo from data/elo.csv for strength.")
@click.option("--odds", is_flag=True, help="Use Polymarket implied odds for team strength (free, live).")
@click.option("--ratings", "use_ratings", is_flag=True, help="Blend SoFIFA overall ratings (data/sofifa.csv) into quality.")
@click.option("--llm", is_flag=True, help="Enable the optional Claude adjustment layer.")
@click.option("--news", type=click.Path(exists=True), default=None, help="Team-news text file for the LLM layer.")
@click.option("--html", "html_path", default=None, help="Also write the rendered report to this HTML file.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def advise(round_id: int, squad_path: str | None, free_override: float | None,
           elo: bool, odds: bool, use_ratings: bool, llm: bool, news: str | None,
           html_path: str | None, config_path: str) -> None:
    """Recommend transfers for a round given your current saved squad."""
    cfg = _load_config(config_path)
    cache = cfg["data"]["cache_dir"]
    squad_path = squad_path or str(Path(cache) / "my_squad.yaml")
    if not Path(squad_path).exists():
        raise click.ClickException(
            f"No saved squad at {squad_path}. Run `fantasy build --save` first."
        )
    current_ids = yaml.safe_load(Path(squad_path).read_text())["player_ids"]

    news_text = Path(news).read_text() if news else None
    ds, projections, rnd = _pipeline(cfg, round_id, use_elo=elo, use_odds=odds,
                                     use_ratings=use_ratings, use_llm=llm, news=news_text)

    rows = rows_from_projections(projections, ds.squads)
    free = free_override if free_override is not None else rules.free_transfers_for_round(round_id)
    plan = optimize_transfers(
        rows, current_ids,
        free_transfers=free,
        budget=rules.budget_for_stage(rnd.stage),
        nation_cap=rules.nation_cap_for_stage(rnd.stage),
        bench_weight=cfg["optimize"]["bench_weight"],
        hit_threshold=cfg["optimize"]["hit_threshold"],
    )
    report.render_transfers(console, ds, projections, plan,
                            round_id=round_id, deadline=_deadline(rnd))
    if html_path:
        _export_html(html_path)


@main.command()
@click.option("--round", "round_id", default=1, show_default=True, type=int, help="Matchday/round id (1-8).")
@click.option("--advise", is_flag=True, help="Advise transfers on your saved squad (else build a fresh squad).")
@click.option("--squad", "squad_path", default=None, help="Path to saved squad (default data/my_squad.yaml).")
@click.option("--elo", is_flag=True, help="Use World Football Elo from data/elo.csv for strength.")
@click.option("--odds", is_flag=True, help="Use Polymarket implied odds for team strength (free, live).")
@click.option("--ratings", "use_ratings", is_flag=True, help="Blend SoFIFA overall ratings (data/sofifa.csv) into quality.")
@click.option("--llm", is_flag=True, help="Enable the bounded research subloop (Claude). Off = zero tokens.")
@click.option("--news", type=click.Path(exists=True), default=None, help="Team-news text file for the subloop.")
@click.option("--max-research", default=2, show_default=True, type=int, help="Max research-subloop iterations.")
@click.option("--yes", "auto_approve", is_flag=True, help="Skip the interactive approval gate.")
@click.option("--free", "free_override", default=None, type=float,
              help="Override free-transfer count in --advise mode.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def run(round_id: int, advise: bool, squad_path: str | None, elo: bool, odds: bool, use_ratings: bool,
        llm: bool, news: str | None, max_research: int, auto_approve: bool,
        free_override: float | None, config_path: str) -> None:
    """Run the full LangGraph agent: data -> assess -> analyze -> (research) -> synthesize -> approve."""
    try:
        from .graph import run as run_graph
    except ImportError as exc:
        raise click.ClickException(
            f"LangGraph not installed ({exc}). Install with: pip install -e '.[graph]'"
        )
    cfg = _load_config(config_path)
    mode = "advise" if advise else "build"
    current_ids = None
    if advise:
        path = squad_path or str(Path(cfg["data"]["cache_dir"]) / "my_squad.yaml")
        if not Path(path).exists():
            raise click.ClickException(f"No saved squad at {path}. Run `fantasy run --save`-equivalent build first.")
        current_ids = yaml.safe_load(Path(path).read_text())["player_ids"]

    news_text = Path(news).read_text() if news else None
    state = run_graph(
        round_id, mode=mode, config=cfg, use_elo=elo, use_odds=odds, use_ratings=use_ratings,
        use_llm=llm, news=news_text,
        max_research_iters=max_research, auto_approve=auto_approve, current_ids=current_ids,
        free_transfers=free_override,
    )
    if not state.get("data_ready"):
        console.print("[yellow]Round not drawn yet — opponents unknown. Try again once the bracket is set.[/yellow]")
    console.print("\n[dim]graph trace:[/dim] " + " → ".join(state.get("log", [])))


def _save_squad(cache_dir: str, ds: fifa.Dataset, selected, round_id: int) -> None:
    by_id = {p.id: p for p in ds.players}
    data = {
        "round": round_id,
        "player_ids": selected.squad_ids,
        "starters": selected.starter_ids,
        "bench": selected.bench_ids,
        "captain": selected.captain_id,
        "vice": selected.vice_id,
        "names": {pid: by_id[pid].name for pid in selected.squad_ids if pid in by_id},
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "my_squad.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


@main.command()
@click.option("--round", "round_id", required=True, type=int, help="Matchday/round id to score (1-8).")
@click.option("--squad", "squad_path", default=None, help="Path to saved squad (default data/my_squad.yaml).")
@click.option("--hits", default=0, type=int, help="Points deducted for transfers taken this round.")
@click.option("--no-refresh", is_flag=True, help="Use cached data instead of fetching the latest points.")
@click.option("--html", "html_path", default=None, help="Also write the scorecard to this HTML file.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def score(round_id: int, squad_path: str | None, hits: int, no_refresh: bool,
          html_path: str | None, config_path: str) -> None:
    """Calculate your team's actual points for a matchday from FIFA's official scores."""
    from . import scoring

    cfg = _load_config(config_path)
    cache = cfg["data"]["cache_dir"]
    squad_path = squad_path or str(Path(cache) / "my_squad.yaml")
    if not Path(squad_path).exists():
        raise click.ClickException(f"No saved squad at {squad_path}. Run `fantasy build --save` first.")
    squad = yaml.safe_load(Path(squad_path).read_text())

    if not no_refresh:
        try:
            fifa.fetch(cache, force=True)  # pull the latest official points
        except Exception as exc:
            console.print(f"[yellow]Could not refresh ({exc}); scoring from cached data.[/yellow]")
    ds = fifa.load(cache, auto_fetch=False)

    card = scoring.score_matchday(ds, squad, round_id, hits=hits)
    report.render_scorecard(console, ds, card)
    if html_path:
        _export_html(html_path)


if __name__ == "__main__":
    main()
