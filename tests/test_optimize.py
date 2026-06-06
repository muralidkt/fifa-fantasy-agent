"""ILP correctness on a synthetic pool: legal squad, valid XI, transfer limits respected."""
from fifa_fantasy.optimize import Row
from fifa_fantasy.optimize.squad import optimize_squad, choose_xi
from fifa_fantasy.optimize.transfers import optimize_transfers
from fifa_fantasy.rules import FORMATION_BOUNDS, SQUAD_COMPOSITION

PER_NATION = {"GK": 2, "DEF": 4, "MID": 4, "FWD": 3}


def _pool(n_nations: int = 10) -> list[Row]:
    rows: list[Row] = []
    pid = 0
    for nation in range(n_nations):
        for pos, count in PER_NATION.items():
            for k in range(count):
                pid += 1
                # cheap-but-decent and pricey-and-good options; xpts loosely tracks price
                price = 4.0 + k * 0.7
                rows.append(Row(pid=pid, name=f"p{pid}", position=pos, price=price,
                                squad_id=nation, squad_name=f"N{nation}", xpts=price * 0.9 + (pid % 5) * 0.1))
    return rows


def _assert_legal(squad, rows):
    idx = {r.pid: r for r in rows}
    assert len(squad.squad_ids) == 15
    by_pos = {}
    for pid in squad.squad_ids:
        by_pos[idx[pid].position] = by_pos.get(idx[pid].position, 0) + 1
    assert by_pos == SQUAD_COMPOSITION
    # nation cap (default 3)
    by_nat = {}
    for pid in squad.squad_ids:
        by_nat[idx[pid].squad_id] = by_nat.get(idx[pid].squad_id, 0) + 1
    assert max(by_nat.values()) <= 3
    # XI valid
    assert len(squad.starter_ids) == 11
    xi_pos = {}
    for pid in squad.starter_ids:
        xi_pos[idx[pid].position] = xi_pos.get(idx[pid].position, 0) + 1
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        assert lo <= xi_pos.get(pos, 0) <= hi
    assert set(squad.starter_ids).issubset(set(squad.squad_ids))


def test_squad_is_legal_and_within_budget():
    rows = _pool()
    squad = optimize_squad(rows, budget=100.0, nation_cap=3)
    _assert_legal(squad, rows)
    assert squad.total_cost <= 100.0


def test_captain_is_highest_xpts_starter():
    rows = _pool()
    squad = optimize_squad(rows, budget=100.0, nation_cap=3)
    idx = {r.pid: r for r in rows}
    best = max(squad.starter_ids, key=lambda p: idx[p].xpts)
    assert squad.captain_id == best
    assert squad.captain_id != squad.vice_id


def test_choose_xi_on_fixed_15():
    rows = _pool()
    squad = optimize_squad(rows, budget=100.0, nation_cap=3)
    fifteen = [r for r in rows if r.pid in set(squad.squad_ids)]
    xi = choose_xi(fifteen)
    _assert_legal(xi, rows)


def test_transfers_respect_free_allowance():
    rows = _pool()
    squad = optimize_squad(rows, budget=100.0, nation_cap=3)
    # Big threshold => never take a paid hit; with free=2 at most 2 changes.
    plan = optimize_transfers(rows, squad.squad_ids, free_transfers=2,
                              budget=100.0, nation_cap=3, hit_threshold=1000.0)
    assert plan.transfers <= 2
    assert plan.paid_transfers == 0
    _assert_legal(plan.squad, rows)
