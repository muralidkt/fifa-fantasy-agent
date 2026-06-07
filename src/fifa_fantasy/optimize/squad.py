"""Pick the optimal 15-man squad AND starting XI in one ILP.

Maximises  sum(xPts over starters) + bench_weight * sum(xPts over bench)
subject to the 2/5/5/3 squad shape, budget, per-nation cap, a valid formation, and
exactly 11 starters. Solving squad + XI together means budget is spent on the players who
actually start, parking cheap fillers on the bench. Uses PuLP's bundled CBC solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import pulp

from ..rules import FORMATION_BOUNDS, SQUAD_COMPOSITION, SQUAD_SIZE, XI_SIZE, SCORING
from . import Row


@dataclass
class SelectedSquad:
    squad_ids: list[int]
    starter_ids: list[int]
    bench_ids: list[int]            # ordered: outfield by xPts desc, then reserve GK
    captain_id: int
    vice_id: int
    formation: str                  # e.g. "4-4-2"
    total_cost: float
    xi_xpts: float                  # expected points of the XI (excl. captain bonus)
    total_xpts: float               # XI + captain doubling


def optimize_squad(rows: list[Row], *, budget: float, nation_cap: int,
                   bench_weight: float = 0.12) -> SelectedSquad:
    rows = [r for r in rows]
    idx = {r.pid: r for r in rows}
    prob = pulp.LpProblem("squad", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("pick", [r.pid for r in rows], cat="Binary")   # in 15
    y = pulp.LpVariable.dicts("start", [r.pid for r in rows], cat="Binary")  # in XI

    prob += pulp.lpSum(
        y[r.pid] * r.objective + bench_weight * (x[r.pid] - y[r.pid]) * r.objective
        for r in rows
    )

    # Squad-level constraints.
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    prob += pulp.lpSum(r.price * x[r.pid] for r in rows) <= budget
    for pos, n in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(x[r.pid] for r in rows if r.position == pos) == n
    for sid in {r.squad_id for r in rows}:
        prob += pulp.lpSum(x[r.pid] for r in rows if r.squad_id == sid) <= nation_cap

    # Starting-XI constraints.
    for r in rows:
        prob += y[r.pid] <= x[r.pid]
    prob += pulp.lpSum(y.values()) == XI_SIZE
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        s = pulp.lpSum(y[r.pid] for r in rows if r.position == pos)
        prob += s >= lo
        prob += s <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"squad optimisation failed: {pulp.LpStatus[prob.status]}")

    squad_ids = [r.pid for r in rows if x[r.pid].value() > 0.5]
    starter_ids = [r.pid for r in rows if y[r.pid].value() > 0.5]
    return _assemble(idx, squad_ids, starter_ids)


def _assemble(idx: dict[int, Row], squad_ids: list[int], starter_ids: list[int]) -> SelectedSquad:
    starters = sorted(starter_ids, key=lambda p: idx[p].armband_score, reverse=True)
    captain_id, vice_id = starters[0], starters[1]

    bench = [p for p in squad_ids if p not in set(starter_ids)]
    bench_out = sorted((p for p in bench if idx[p].position != "GK"),
                       key=lambda p: idx[p].xpts, reverse=True)
    bench_gk = [p for p in bench if idx[p].position == "GK"]
    bench_ids = bench_out + bench_gk

    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for p in starter_ids:
        if idx[p].position in counts:
            counts[idx[p].position] += 1
    formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"

    total_cost = sum(idx[p].price for p in squad_ids)
    xi_xpts = sum(idx[p].xpts for p in starter_ids)
    total_xpts = xi_xpts + idx[captain_id].xpts * (SCORING.captain_multiplier - 1)
    return SelectedSquad(
        squad_ids=squad_ids,
        starter_ids=starter_ids,
        bench_ids=bench_ids,
        captain_id=captain_id,
        vice_id=vice_id,
        formation=formation,
        total_cost=round(total_cost, 1),
        xi_xpts=round(xi_xpts, 2),
        total_xpts=round(total_xpts, 2),
    )


def choose_xi(rows_15: list[Row]) -> SelectedSquad:
    """Given a fixed 15, pick the best legal XI + captain/vice (used after transfers)."""
    idx = {r.pid: r for r in rows_15}
    prob = pulp.LpProblem("xi", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("start", [r.pid for r in rows_15], cat="Binary")
    prob += pulp.lpSum(y[r.pid] * r.objective for r in rows_15)
    prob += pulp.lpSum(y.values()) == XI_SIZE
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        s = pulp.lpSum(y[r.pid] for r in rows_15 if r.position == pos)
        prob += s >= lo
        prob += s <= hi
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"XI optimisation failed: {pulp.LpStatus[prob.status]}")
    starter_ids = [r.pid for r in rows_15 if y[r.pid].value() > 0.5]
    return _assemble(idx, [r.pid for r in rows_15], starter_ids)
