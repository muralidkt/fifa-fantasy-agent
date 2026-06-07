from fifa_fantasy import rules


def test_squad_composition_sums_to_15():
    assert sum(rules.SQUAD_COMPOSITION.values()) == rules.SQUAD_SIZE == 15


def test_budget_rises_in_knockouts():
    assert rules.budget_for_stage("GROUP") == 100.0
    assert rules.budget_for_stage("R32") == 105.0
    assert rules.budget_for_stage("F") == 105.0


def test_nation_cap_scales_by_stage():
    assert rules.nation_cap_for_stage("GROUP") == 3
    assert rules.nation_cap_for_stage("R32") == 3
    assert rules.nation_cap_for_stage("R16") == 4
    assert rules.nation_cap_for_stage("F") == 8


def test_free_transfers():
    assert rules.free_transfers_for_round(1) == float("inf")  # initial build
    assert rules.free_transfers_for_round(2) == 2
    assert rules.free_transfers_for_round(3, banked=1) == 3
    assert rules.free_transfers_for_round(4) == float("inf")  # group->R32 reset
    assert rules.banked_transfer_after_round(2, 2, 1) == 1
    assert rules.banked_transfer_after_round(2, 2, 2) == 0


def test_scoring_values():
    s = rules.SCORING
    assert s.goal == {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5}
    assert s.assist == 3
    assert s.clean_sheet["DEF"] == 5 and s.clean_sheet["MID"] == 1
    assert s.captain_multiplier == 2
