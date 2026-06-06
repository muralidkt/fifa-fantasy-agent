"""Matchday transfer optimisation.

Given your current 15 and the projections for an upcoming round, find the new 15 + XI that
maximises expected points, where each transfer beyond the free allowance costs -3 points.
Prices are fixed all tournament, so selling price == buying price (no sell-value bookkeeping).
"""
from __future__ import annotations

from dataclasses import dataclass

import pulp

from ..rules import FORMATION_BOUNDS, SQUAD_COMPOSITION, SQUAD_SIZE, TRANSFER_HIT, XI_SIZE
from . import Row
from .squad import SelectedSquad, _assemble


@dataclass
class TransferPlan:
    in_ids: list[int]
    out_ids: list[int]
    transfers: int
    paid_transfers: int     # beyond the free allowance
    hit_points: int         # points deducted (paid_transfers * 3)
    squad: SelectedSquad


def optimize_transfers(
    rows: list[Row],
    current_ids: list[int],
    *,
    free_transfers: float,
    budget: float,
    nation_cap: int,
    bench_weight: float = 0.12,
    hit_threshold: float = 1.0,
) -> TransferPlan:
    idx = {r.pid: r for r in rows}
    current = [pid for pid in current_ids if pid in idx]
    current_set = set(current)

    prob = pulp.LpProblem("transfers", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("pick", [r.pid for r in rows], cat="Binary")
    y = pulp.LpVariable.dicts("start", [r.pid for r in rows], cat="Binary")

    objective = pulp.lpSum(
        y[r.pid] * r.xpts + bench_weight * (x[r.pid] - y[r.pid]) * r.xpts for r in rows
    )

    # Transfers = how many of the current squad we drop (= number bought, squad size fixed).
    transfers_expr = SQUAD_SIZE - pulp.lpSum(x[pid] for pid in current)
    if free_transfers != float("inf"):
        extra = pulp.LpVariable("extra_transfers", lowBound=0, cat="Integer")
        prob += extra >= transfers_expr - free_transfers
        # Discourage marginal hits: a paid transfer must clear the 3pt cost + threshold.
        objective -= (TRANSFER_HIT + hit_threshold) * extra

    prob += objective

    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    prob += pulp.lpSum(r.price * x[r.pid] for r in rows) <= budget
    for pos, n in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(x[r.pid] for r in rows if r.position == pos) == n
    for sid in {r.squad_id for r in rows}:
        prob += pulp.lpSum(x[r.pid] for r in rows if r.squad_id == sid) <= nation_cap
    for r in rows:
        prob += y[r.pid] <= x[r.pid]
    prob += pulp.lpSum(y.values()) == XI_SIZE
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        s = pulp.lpSum(y[r.pid] for r in rows if r.position == pos)
        prob += s >= lo
        prob += s <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"transfer optimisation failed: {pulp.LpStatus[prob.status]}")

    new_ids = [r.pid for r in rows if x[r.pid].value() > 0.5]
    starter_ids = [r.pid for r in rows if y[r.pid].value() > 0.5]
    new_set = set(new_ids)

    out_ids = [pid for pid in current if pid not in new_set]
    in_ids = [pid for pid in new_ids if pid not in current_set]
    transfers = len(in_ids)
    paid = 0 if free_transfers == float("inf") else max(0, transfers - int(free_transfers))
    squad = _assemble(idx, new_ids, starter_ids)
    return TransferPlan(
        in_ids=in_ids,
        out_ids=out_ids,
        transfers=transfers,
        paid_transfers=paid,
        hit_points=paid * TRANSFER_HIT,
        squad=squad,
    )
