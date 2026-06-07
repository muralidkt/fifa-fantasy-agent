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
        y[r.pid] * r.objective + bench_weight * (x[r.pid] - y[r.pid]) * r.objective
        for r in rows
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


def optimize_transfers_horizon(
    rows_by_round: list[tuple[list[Row], float]],
    current_ids: list[int],
    *,
    free_transfers: float,
    budget: float,
    nation_cap: int,
    bench_weight: float = 0.12,
    hit_threshold: float = 1.0,
) -> TransferPlan:
    """Optimize transfers over several rounds with weighted future projections.

    The resulting 15-man squad is fixed across the horizon. The solver can pick a different
    legal XI for each round, so short-term benching and fixture swings are valued correctly.
    The first row set is used for prices, metadata, reporting, and the returned XI.
    """
    if not rows_by_round:
        raise ValueError("rows_by_round must contain at least one round")

    first_rows = rows_by_round[0][0]
    idx = {r.pid: r for r in first_rows}
    pids = [r.pid for r in first_rows]
    current = [pid for pid in current_ids if pid in idx]
    current_set = set(current)
    round_maps = [({r.pid: r for r in rows}, weight) for rows, weight in rows_by_round]

    prob = pulp.LpProblem("transfers_horizon", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("pick", pids, cat="Binary")
    y = {
        ridx: pulp.LpVariable.dicts(f"start_r{ridx}", pids, cat="Binary")
        for ridx in range(len(round_maps))
    }

    objective = pulp.lpSum(
        weight * (
            y[ridx][pid] * row_map[pid].objective
            + bench_weight * (x[pid] - y[ridx][pid]) * row_map[pid].objective
        )
        for ridx, (row_map, weight) in enumerate(round_maps)
        for pid in pids
        if pid in row_map
    )

    transfers_expr = SQUAD_SIZE - pulp.lpSum(x[pid] for pid in current)
    if free_transfers != float("inf"):
        extra = pulp.LpVariable("extra_transfers", lowBound=0, cat="Integer")
        prob += extra >= transfers_expr - free_transfers
        objective -= (TRANSFER_HIT + hit_threshold) * extra

    prob += objective
    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    prob += pulp.lpSum(idx[pid].price * x[pid] for pid in pids) <= budget
    for pos, n in SQUAD_COMPOSITION.items():
        prob += pulp.lpSum(x[pid] for pid in pids if idx[pid].position == pos) == n
    for sid in {r.squad_id for r in first_rows}:
        prob += pulp.lpSum(x[pid] for pid in pids if idx[pid].squad_id == sid) <= nation_cap

    for ridx in y:
        for pid in pids:
            prob += y[ridx][pid] <= x[pid]
        prob += pulp.lpSum(y[ridx].values()) == XI_SIZE
        for pos, (lo, hi) in FORMATION_BOUNDS.items():
            s = pulp.lpSum(y[ridx][pid] for pid in pids if idx[pid].position == pos)
            prob += s >= lo
            prob += s <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"horizon transfer optimisation failed: {pulp.LpStatus[prob.status]}")

    new_ids = [pid for pid in pids if x[pid].value() > 0.5]
    starter_ids = [pid for pid in pids if y[0][pid].value() > 0.5]
    new_set = set(new_ids)
    out_ids = [pid for pid in current if pid not in new_set]
    in_ids = [pid for pid in new_ids if pid not in current_set]
    transfers = len(in_ids)
    paid = 0 if free_transfers == float("inf") else max(0, transfers - int(free_transfers))
    return TransferPlan(
        in_ids=in_ids,
        out_ids=out_ids,
        transfers=transfers,
        paid_transfers=paid,
        hit_points=paid * TRANSFER_HIT,
        squad=_assemble(idx, new_ids, starter_ids),
    )


def optimize_transfers_sequence(
    rows_by_round: list[tuple[list[Row], float]],
    current_ids: list[int],
    *,
    free_transfers_by_round: list[float],
    budget_by_round: list[float],
    nation_cap_by_round: list[int],
    bench_weight: float = 0.12,
    hit_threshold: float = 1.0,
) -> TransferPlan:
    """Optimize this round's transfers while planning later transfer windows.

    Unlike ``optimize_transfers_horizon``, this models a different 15-man squad
    in each future round and constrains each transition by that round's transfer
    allocation. The returned transfer plan is only for the immediate upcoming
    round; future transfers are planning scaffolding.
    """
    if not rows_by_round:
        raise ValueError("rows_by_round must contain at least one round")
    if len(budget_by_round) != len(rows_by_round) or len(nation_cap_by_round) != len(rows_by_round):
        raise ValueError("budget_by_round and nation_cap_by_round must match rows_by_round")
    if len(free_transfers_by_round) != len(rows_by_round):
        raise ValueError("free_transfers_by_round must include the current round transition")

    first_rows = rows_by_round[0][0]
    idx = {r.pid: r for r in first_rows}
    pids = [r.pid for r in first_rows]
    current = [pid for pid in current_ids if pid in idx]
    current_set = set(current)
    round_maps = [({r.pid: r for r in rows}, weight) for rows, weight in rows_by_round]

    prob = pulp.LpProblem("transfers_sequence", pulp.LpMaximize)
    x = {
        ridx: pulp.LpVariable.dicts(f"pick_r{ridx}", pids, cat="Binary")
        for ridx in range(len(round_maps))
    }
    y = {
        ridx: pulp.LpVariable.dicts(f"start_r{ridx}", pids, cat="Binary")
        for ridx in range(len(round_maps))
    }

    objective = pulp.lpSum(
        weight * (
            y[ridx][pid] * row_map[pid].objective
            + bench_weight * (x[ridx][pid] - y[ridx][pid]) * row_map[pid].objective
        )
        for ridx, (row_map, weight) in enumerate(round_maps)
        for pid in pids
        if pid in row_map
    )

    extra_first = None
    for ridx in range(len(round_maps)):
        free = free_transfers_by_round[ridx]
        if ridx == 0:
            transfers_expr = SQUAD_SIZE - pulp.lpSum(x[0][pid] for pid in current)
        else:
            keep = pulp.LpVariable.dicts(f"keep_r{ridx}", pids, cat="Binary")
            for pid in pids:
                prob += keep[pid] <= x[ridx - 1][pid]
                prob += keep[pid] <= x[ridx][pid]
                prob += keep[pid] >= x[ridx - 1][pid] + x[ridx][pid] - 1
            transfers_expr = SQUAD_SIZE - pulp.lpSum(keep.values())
        if free != float("inf"):
            extra = pulp.LpVariable(f"extra_transfers_r{ridx}", lowBound=0, cat="Integer")
            prob += extra >= transfers_expr - free
            objective -= (TRANSFER_HIT + hit_threshold) * extra
            if ridx == 0:
                extra_first = extra

    prob += objective

    for ridx in range(len(round_maps)):
        row_map, _ = round_maps[ridx]
        prob += pulp.lpSum(x[ridx].values()) == SQUAD_SIZE
        prob += pulp.lpSum(idx[pid].price * x[ridx][pid] for pid in pids) <= budget_by_round[ridx]
        for pos, n in SQUAD_COMPOSITION.items():
            prob += pulp.lpSum(x[ridx][pid] for pid in pids if idx[pid].position == pos) == n
        for sid in {r.squad_id for r in first_rows}:
            prob += pulp.lpSum(x[ridx][pid] for pid in pids if idx[pid].squad_id == sid) <= nation_cap_by_round[ridx]
        for pid in pids:
            prob += y[ridx][pid] <= x[ridx][pid]
            if pid not in row_map:
                prob += y[ridx][pid] == 0
        prob += pulp.lpSum(y[ridx].values()) == XI_SIZE
        for pos, (lo, hi) in FORMATION_BOUNDS.items():
            s = pulp.lpSum(y[ridx][pid] for pid in pids if idx[pid].position == pos)
            prob += s >= lo
            prob += s <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"sequence transfer optimisation failed: {pulp.LpStatus[prob.status]}")

    new_ids = [pid for pid in pids if x[0][pid].value() > 0.5]
    starter_ids = [pid for pid in pids if y[0][pid].value() > 0.5]
    new_set = set(new_ids)
    out_ids = [pid for pid in current if pid not in new_set]
    in_ids = [pid for pid in new_ids if pid not in current_set]
    transfers = len(in_ids)
    free = free_transfers_by_round[0]
    paid = 0 if free == float("inf") else max(0, transfers - int(free))
    if extra_first is not None:
        paid = int(round(extra_first.value() or 0))
    return TransferPlan(
        in_ids=in_ids,
        out_ids=out_ids,
        transfers=transfers,
        paid_transfers=paid,
        hit_points=paid * TRANSFER_HIT,
        squad=_assemble(idx, new_ids, starter_ids),
    )


def optimize_squad_sequence(
    rows_by_round: list[tuple[list[Row], float]],
    *,
    free_transfers_after_round: list[float],
    budget_by_round: list[float],
    nation_cap_by_round: list[int],
    bench_weight: float = 0.12,
    hit_threshold: float = 1.0,
) -> SelectedSquad:
    """Choose this round's initial squad while planning future transfer windows."""
    if len(free_transfers_after_round) != max(0, len(rows_by_round) - 1):
        raise ValueError("free_transfers_after_round must have one value per future transition")
    if not rows_by_round:
        raise ValueError("rows_by_round must contain at least one round")

    first_rows = rows_by_round[0][0]
    idx = {r.pid: r for r in first_rows}
    pids = [r.pid for r in first_rows]
    round_maps = [({r.pid: r for r in rows}, weight) for rows, weight in rows_by_round]

    prob = pulp.LpProblem("squad_sequence", pulp.LpMaximize)
    x = {
        ridx: pulp.LpVariable.dicts(f"pick_r{ridx}", pids, cat="Binary")
        for ridx in range(len(round_maps))
    }
    y = {
        ridx: pulp.LpVariable.dicts(f"start_r{ridx}", pids, cat="Binary")
        for ridx in range(len(round_maps))
    }

    objective = pulp.lpSum(
        weight * (
            y[ridx][pid] * row_map[pid].objective
            + bench_weight * (x[ridx][pid] - y[ridx][pid]) * row_map[pid].objective
        )
        for ridx, (row_map, weight) in enumerate(round_maps)
        for pid in pids
        if pid in row_map
    )

    for ridx, free in enumerate(free_transfers_after_round, start=1):
        keep = pulp.LpVariable.dicts(f"keep_r{ridx}", pids, cat="Binary")
        for pid in pids:
            prob += keep[pid] <= x[ridx - 1][pid]
            prob += keep[pid] <= x[ridx][pid]
            prob += keep[pid] >= x[ridx - 1][pid] + x[ridx][pid] - 1
        transfers_expr = SQUAD_SIZE - pulp.lpSum(keep.values())
        if free != float("inf"):
            extra = pulp.LpVariable(f"extra_transfers_r{ridx}", lowBound=0, cat="Integer")
            prob += extra >= transfers_expr - free
            objective -= (TRANSFER_HIT + hit_threshold) * extra

    prob += objective

    for ridx in range(len(round_maps)):
        row_map, _ = round_maps[ridx]
        prob += pulp.lpSum(x[ridx].values()) == SQUAD_SIZE
        prob += pulp.lpSum(idx[pid].price * x[ridx][pid] for pid in pids) <= budget_by_round[ridx]
        for pos, n in SQUAD_COMPOSITION.items():
            prob += pulp.lpSum(x[ridx][pid] for pid in pids if idx[pid].position == pos) == n
        for sid in {r.squad_id for r in first_rows}:
            prob += pulp.lpSum(x[ridx][pid] for pid in pids if idx[pid].squad_id == sid) <= nation_cap_by_round[ridx]
        for pid in pids:
            prob += y[ridx][pid] <= x[ridx][pid]
            if pid not in row_map:
                prob += y[ridx][pid] == 0
        prob += pulp.lpSum(y[ridx].values()) == XI_SIZE
        for pos, (lo, hi) in FORMATION_BOUNDS.items():
            s = pulp.lpSum(y[ridx][pid] for pid in pids if idx[pid].position == pos)
            prob += s >= lo
            prob += s <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"sequence squad optimisation failed: {pulp.LpStatus[prob.status]}")

    squad_ids = [pid for pid in pids if x[0][pid].value() > 0.5]
    starter_ids = [pid for pid in pids if y[0][pid].value() > 0.5]
    return _assemble(idx, squad_ids, starter_ids)
