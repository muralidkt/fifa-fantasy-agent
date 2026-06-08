"""`fantasy` command-line interface: refresh data, build a squad, advise transfers."""
from __future__ import annotations

from pathlib import Path

import click

from .ingest import fifa, ratings
from .optimize import rows_from_projections
from .optimize.squad import optimize_squad
from .optimize.transfers import optimize_squad_sequence, optimize_transfers, optimize_transfers_sequence
from . import report
from . import rules
from .engine import (
    DEFAULTS, console, _deadline, _load_config, _load_squad, _pipeline,
    _planning_rows, _save_squad,
)

# DEFAULTS / _load_config / _pipeline / _planning_rows / _save_squad / _load_squad / console
# live in engine.py and are imported above so both the CLI and the agent share one pipeline.
__all__ = ["main", "DEFAULTS"]


def _export_html(path: str) -> None:
    """Save everything rendered so far this command to a styled HTML file."""
    console.save_html(path, clear=False)
    console.print(f"[green]Wrote[/green] HTML report to {path}")


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
@click.option("--mode", "opt_mode", default=None,
              type=click.Choice(["balanced", "safe", "upside", "differential", "template"]),
              help="Recommendation style; defaults to optimize.mode in config.")
@click.option("--html", "html_path", default=None, help="Also write the rendered report to this HTML file.")
@click.option("--horizon", default=1, show_default=True, type=click.IntRange(1, 8),
              help="Number of rounds to plan for, respecting future transfer limits.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def build(round_id: int, elo: bool, odds: bool, use_ratings: bool, llm: bool,
          news: str | None, save: bool, opt_mode: str | None,
          html_path: str | None, horizon: int, config_path: str) -> None:
    """Build the optimal squad + XI + captain for a round (default: Matchday 1)."""
    cfg = _load_config(config_path)
    news_text = Path(news).read_text() if news else None
    ds, projections, rnd = _pipeline(cfg, round_id, use_elo=elo, use_odds=odds,
                                     use_ratings=use_ratings, use_llm=llm, news=news_text)

    mode = opt_mode or cfg["optimize"].get("mode", "balanced")
    rows = rows_from_projections(projections, ds.squads, mode=mode)
    rows_by_round, budgets, caps, transfer_windows = _planning_rows(
        cfg, round_id, horizon, rows, use_elo=elo, use_odds=odds,
        use_ratings=use_ratings, mode=mode,
    )
    if len(rows_by_round) > 1:
        selected = optimize_squad_sequence(
            rows_by_round,
            free_transfers_after_round=transfer_windows,
            budget_by_round=budgets,
            nation_cap_by_round=caps,
            bench_weight=cfg["optimize"]["bench_weight"],
            hit_threshold=cfg["optimize"]["hit_threshold"],
        )
        console.print(f"[dim]squad horizon: {len(rows_by_round)} round(s), respecting transfer windows[/dim]")
    else:
        selected = optimize_squad(
            rows,
            budget=rules.budget_for_stage(rnd.stage),
            nation_cap=rules.nation_cap_for_stage(rnd.stage),
            bench_weight=cfg["optimize"]["bench_weight"],
        )
    report.render_squad(console, ds, projections, selected,
                        round_id=round_id, deadline=_deadline(rnd))

    if save:
        _save_squad(cfg["data"]["cache_dir"], ds, selected, round_id, banked_transfer=0)
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
@click.option("--mode", "opt_mode", default=None,
              type=click.Choice(["balanced", "safe", "upside", "differential", "template"]),
              help="Recommendation style; defaults to optimize.mode in config.")
@click.option("--horizon", default=1, show_default=True, type=click.IntRange(1, 8),
              help="Number of rounds to value for transfer advice.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def advise(round_id: int, squad_path: str | None, free_override: float | None,
           elo: bool, odds: bool, use_ratings: bool, llm: bool, news: str | None,
           html_path: str | None, opt_mode: str | None, horizon: int, config_path: str) -> None:
    """Recommend transfers for a round given your current saved squad."""
    cfg = _load_config(config_path)
    cache = cfg["data"]["cache_dir"]
    squad_path = squad_path or str(Path(cache) / "my_squad.yaml")
    if not Path(squad_path).exists():
        raise click.ClickException(
            f"No saved squad at {squad_path}. Run `fantasy build --save` first."
        )
    squad_state = _load_squad(squad_path)
    current_ids = squad_state["player_ids"]

    news_text = Path(news).read_text() if news else None
    ds, projections, rnd = _pipeline(cfg, round_id, use_elo=elo, use_odds=odds,
                                     use_ratings=use_ratings, use_llm=llm, news=news_text)

    mode = opt_mode or cfg["optimize"].get("mode", "balanced")
    rows = rows_from_projections(projections, ds.squads, mode=mode)
    banked = int(squad_state.get("banked_transfer", 0))
    free = free_override if free_override is not None else rules.free_transfers_for_round(round_id, banked)
    if horizon > 1:
        rows_by_round, budgets, caps, future_windows = _planning_rows(
            cfg, round_id, horizon, rows, use_elo=elo, use_odds=odds,
            use_ratings=use_ratings, mode=mode,
        )
        plan = optimize_transfers_sequence(
            rows_by_round,
            current_ids,
            free_transfers_by_round=[free] + future_windows,
            budget_by_round=budgets,
            nation_cap_by_round=caps,
            bench_weight=cfg["optimize"]["bench_weight"],
            hit_threshold=cfg["optimize"]["hit_threshold"],
        )
        console.print(f"[dim]transfer horizon: {len(rows_by_round)} round(s), respecting transfer windows[/dim]")
    else:
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
    _save_squad(
        cfg["data"]["cache_dir"], ds, plan.squad, round_id,
        banked_transfer=rules.banked_transfer_after_round(round_id, free, plan.transfers),
    )
    console.print(f"[green]Saved[/green] squad to {cfg['data']['cache_dir']}/my_squad.yaml")
    if html_path:
        _export_html(html_path)


@main.command()
@click.option("--round", "--matchday", "round_id", default=1, show_default=True,
              type=int, help="Matchday/round id (1-8).")
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
@click.option("--mode", "opt_mode", default=None,
              type=click.Choice(["balanced", "safe", "upside", "differential", "template"]),
              help="Recommendation style; defaults to optimize.mode in config.")
@click.option("--horizon", default=3, show_default=True, type=click.IntRange(1, 8),
              help="Number of rounds to plan for, respecting transfer limits.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def run(round_id: int, advise: bool, squad_path: str | None, elo: bool, odds: bool, use_ratings: bool,
        llm: bool, news: str | None, max_research: int, auto_approve: bool,
        free_override: float | None, opt_mode: str | None, horizon: int, config_path: str) -> None:
    """Run the full agent flow.

    Matchday 1 builds and saves a squad. Later matchdays load the saved squad,
    apply the available transfer window, save the updated squad, and report the
    best XI/captain for that matchday.
    """
    try:
        from .graph import run as run_graph
    except ImportError as exc:
        raise click.ClickException(
            f"LangGraph not installed ({exc}). Install with: pip install -e '.[graph]'"
        )
    cfg = _load_config(config_path)
    if opt_mode:
        cfg.setdefault("optimize", {})["mode"] = opt_mode
    mode = "advise" if advise or round_id > 1 else "build"
    current_ids = None
    banked = 0
    if advise:
        path = squad_path or str(Path(cfg["data"]["cache_dir"]) / "my_squad.yaml")
        if not Path(path).exists():
            raise click.ClickException(f"No saved squad at {path}. Run `fantasy run --save`-equivalent build first.")
        squad_state = _load_squad(path)
        current_ids = squad_state["player_ids"]
        banked = int(squad_state.get("banked_transfer", 0))
    elif mode == "advise":
        path = squad_path or str(Path(cfg["data"]["cache_dir"]) / "my_squad.yaml")
        if not Path(path).exists():
            raise click.ClickException(f"No saved squad at {path}. Run `fantasy run --matchday 1 --yes` first.")
        squad_state = _load_squad(path)
        current_ids = squad_state["player_ids"]
        banked = int(squad_state.get("banked_transfer", 0))
        free_override = free_override if free_override is not None else rules.free_transfers_for_round(round_id, banked)

    news_text = Path(news).read_text() if news else None
    state = run_graph(
        round_id, mode=mode, config=cfg, use_elo=elo, use_odds=odds, use_ratings=use_ratings,
        use_llm=llm, news=news_text,
        max_research_iters=max_research, auto_approve=auto_approve, current_ids=current_ids,
        free_transfers=free_override, horizon=horizon,
    )
    if not state.get("data_ready"):
        console.print("[yellow]Round not drawn yet — opponents unknown. Try again once the bracket is set.[/yellow]")
    console.print("\n[dim]graph trace:[/dim] " + " → ".join(state.get("log", [])))


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
    squad = _load_squad(squad_path)

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


@main.command()
@click.option("--round", "round_id", default=None, type=int,
              help="Default matchday/round id (1-8) for tools that need one.")
@click.option("--model", default=None, help="Anthropic model id (defaults to llm.model in config).")
@click.option("--odds/--no-odds", default=True, show_default=True,
              help="Use Polymarket implied odds for team strength.")
@click.option("--ratings/--no-ratings", "use_ratings", default=True, show_default=True,
              help="Blend SoFIFA overall ratings into player quality.")
@click.option("--mode", "opt_mode", default=None,
              type=click.Choice(["balanced", "safe", "upside", "differential", "template"]),
              help="Recommendation style; defaults to optimize.mode in config.")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def chat(round_id: int | None, model: str | None, odds: bool, use_ratings: bool,
         opt_mode: str | None, config_path: str) -> None:
    """Talk to the agent about your team (interactive). Needs `pip install -e '.[llm]'` + ANTHROPIC_API_KEY.

    A conversational front-end over build/advise/score/explain. The ILP still picks the team;
    the agent can lock/ban players, edit overrides (with a y/N gate), and explain the rules.
    """
    from .agent import loop

    cfg = _load_config(config_path)
    loop.run(cfg, model=model, default_round=round_id,
             odds=odds, ratings=use_ratings, mode=opt_mode)


if __name__ == "__main__":
    main()
