"""Render recommendations to the terminal (rich tables)."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ingest.fifa import Dataset
from .model.expected_points import Projection
from .optimize.squad import SelectedSquad
from .optimize.transfers import TransferPlan
from .scoring import ScoreCard

_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def render_squad(
    console: Console,
    ds: Dataset,
    projections: dict[int, Projection],
    squad: SelectedSquad,
    *,
    round_id: int,
    deadline: str,
    title: str = "Recommended squad",
) -> None:
    console.print(
        Panel.fit(
            f"[bold]{title}[/bold]  ·  Matchday {round_id}  ·  "
            f"formation [cyan]{squad.formation}[/cyan]\n"
            f"cost [yellow]${squad.total_cost:.1f}m[/yellow]  ·  "
            f"projected [green]{squad.total_xpts:.1f} pts[/green] (with captain)\n"
            f"[dim]Deadline: {deadline}  ·  enter this team at play.fifa.com/fantasy[/dim]",
            border_style="green",
        )
    )

    console.print(_player_table(ds, projections, squad.starter_ids, squad, "Starting XI"))
    console.print(_player_table(ds, projections, squad.bench_ids, squad, "Bench"))


def _player_table(
    ds: Dataset,
    projections: dict[int, Projection],
    ids: list[int],
    squad: SelectedSquad,
    heading: str,
) -> Table:
    t = Table(title=heading, title_justify="left", header_style="bold")
    t.add_column("Pos")
    t.add_column("Player")
    t.add_column("Team")
    t.add_column("vs")
    t.add_column("Price", justify="right")
    t.add_column("xPts", justify="right")
    t.add_column("Start", justify="right")
    t.add_column("")

    ordered = sorted(ids, key=lambda p: (_POS_ORDER.get(projections[p].player.position, 9),
                                         -projections[p].xpts))
    for pid in ordered:
        pr = projections[pid]
        tag = ""
        if pid == squad.captain_id:
            tag = "[bold red](C)[/bold red]"
        elif pid == squad.vice_id:
            tag = "[red](V)[/red]"
        team = ds.squads[pr.player.squad_id].name if pr.player.squad_id in ds.squads else "?"
        start = f"{pr.start_prob:.0%}"
        if pr.start_prob < 0.85:
            start = f"[yellow]{start}[/yellow]"
        t.add_row(
            pr.player.position,
            pr.player.name,
            team,
            pr.opponent_name,
            f"${pr.player.price:.1f}m",
            f"{pr.xpts:.2f}",
            start,
            tag,
        )
    return t


def render_transfers(
    console: Console,
    ds: Dataset,
    projections: dict[int, Projection],
    plan: TransferPlan,
    *,
    round_id: int,
    deadline: str,
) -> None:
    if not plan.in_ids:
        console.print(Panel.fit("[bold]No transfers recommended[/bold] — hold your squad.",
                                border_style="cyan"))
    else:
        t = Table(title=f"Transfers for Matchday {round_id}", header_style="bold")
        t.add_column("OUT", style="red")
        t.add_column("→")
        t.add_column("IN", style="green")
        for out_id, in_id in zip(plan.out_ids, plan.in_ids):
            po, pi = projections.get(out_id), projections.get(in_id)
            t.add_row(
                f"{po.player.name} ({po.player.position}) {po.xpts:.2f}" if po else str(out_id),
                "→",
                f"{pi.player.name} ({pi.player.position}) vs {pi.opponent_name} {pi.xpts:.2f}" if pi else str(in_id),
            )
        console.print(t)
        note = f"{plan.transfers} transfer(s)"
        if plan.paid_transfers:
            note += f" · {plan.paid_transfers} paid (−{plan.hit_points} pts hit)"
        console.print(f"[dim]{note}[/dim]")

    render_squad(console, ds, projections, plan.squad, round_id=round_id,
                 deadline=deadline, title="Squad after transfers")


def render_scorecard(console: Console, ds: Dataset, card: ScoreCard) -> None:
    """Show the actual matchday score from FIFA's official per-player points."""
    by_id = {p.id: p for p in ds.players}
    header = (f"[bold]Matchday {card.round_id} score[/bold]   "
              f"[green]{card.total:.0f} pts[/green]\n"
              f"XI {card.xi_points:.0f}  +  captain {card.captain_name} "
              f"(+{card.captain_bonus:.0f})"
              + (f"  −  hits {card.hits}" if card.hits else ""))
    console.print(Panel.fit(header, border_style="green" if card.played_any else "yellow"))

    if not card.played_any:
        console.print("[yellow]No points yet for this round — matches not played "
                      "(or run `fantasy refresh`/wait until after kickoff).[/yellow]")

    t = Table(header_style="bold")
    t.add_column("Pos"); t.add_column("Player"); t.add_column("Team")
    t.add_column("Pts", justify="right"); t.add_column("Role")
    for ln in card.lines:
        team = ds.squads[by_id[ln.pid].squad_id].name if ln.pid in by_id and by_id[ln.pid].squad_id in ds.squads else ""
        style = "" if ln.role != "bench" else "dim"
        tag = {"(C)": "[bold red](C)[/bold red]", "(V→C)": "[red](V→C)[/red]",
               "subbed-in": "[green]▲ in[/green]", "subbed-out": "[red]▼ out[/red]"}.get(ln.role, ln.role)
        t.add_row(ln.position, ln.name, team, f"{ln.points:.0f}", tag, style=style)
    console.print(t)
    for out_id, in_id in card.autosubs:
        console.print(f"[dim]auto-sub: {by_id[out_id].name} → {by_id[in_id].name}[/dim]")
