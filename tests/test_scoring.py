"""Matchday scoring: official-points totals (captain doubling + auto-subs) and rules-based events."""
from fifa_fantasy import scoring
from fifa_fantasy.ingest.fifa import Dataset, Player


def _p(pid, pos, pts):
    return Player(id=pid, name=f"p{pid}", squad_id=1, position=pos, price=5.0, status="playing",
                  ownership=0.0, form=0.0, avg_points=0.0, total_points=0.0, one_to_watch=False,
                  round_points={1: pts})


def _ds(players):
    return Dataset(squads={}, rounds=[], players=players)


def test_score_with_captain_double_and_autosub():
    players = [
        _p(1, "GK", 6),   # starter
        _p(2, "DEF", 2), _p(3, "DEF", 2), _p(4, "DEF", 2), _p(5, "DEF", 2),
        _p(6, "MID", 0),  # starter who DIDN'T play -> should be auto-subbed
        _p(7, "MID", 4), _p(8, "MID", 4),
        _p(9, "FWD", 10), # captain
        _p(10, "FWD", 5), _p(11, "FWD", 3),
        _p(12, "GK", 3),  # bench
        _p(13, "DEF", 5), # bench
        _p(14, "MID", 7), # bench, played -> comes in for #6
        _p(15, "MID", 1), # bench
    ]
    squad = {
        "player_ids": [p.id for p in players],
        "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "bench": [14, 13, 15, 12],   # bench order
        "captain": 9, "vice": 10,
    }
    card = scoring.score_matchday(_ds(players), squad, 1)
    # XI after sub: 6,8 +... 6(GK)+2*4(DEF)+4+4+7(MID m14 in)+10+5+3(FWD) = 6+8+4+4+7+18 = 47
    assert card.xi_points == 47
    assert card.captain_bonus == 10          # captain #9 played, doubled
    assert card.total == 57
    assert (6, 14) in card.autosubs          # #6 out, #14 in
    assert card.played_any is True


def test_vice_captain_takes_double_when_captain_did_not_play():
    players = [_p(1, "GK", 5)] + [_p(i, "DEF", 2) for i in range(2, 6)] \
        + [_p(i, "MID", 3) for i in range(6, 9)] + [_p(9, "FWD", 0), _p(10, "FWD", 8), _p(11, "FWD", 4)] \
        + [_p(12, "GK", 0), _p(13, "DEF", 0), _p(14, "MID", 0), _p(15, "FWD", 0)]
    squad = {"player_ids": [p.id for p in players], "starters": list(range(1, 12)),
             "bench": [13, 14, 15, 12], "captain": 9, "vice": 10}
    card = scoring.score_matchday(_ds(players), squad, 1)
    assert card.captain_id == 10             # captain #9 scored 0 (didn't play) -> vice #10
    assert card.captain_bonus == 8


def test_points_from_stats_matches_rules():
    # Forward, 75 min, 1 goal + 1 assist
    assert scoring.points_from_stats("FWD", minutes=75, goals=1, assists=1) == 2 + 5 + 3
    # Defender, 90 min, clean sheet
    assert scoring.points_from_stats("DEF", minutes=90, clean_sheet=True, conceded=0) == 2 + 5
    # Goalkeeper, 90 min, clean sheet, 6 saves -> 2 + 5 + (6//3)*1
    assert scoring.points_from_stats("GK", minutes=90, clean_sheet=True, saves=6) == 2 + 5 + 2


def test_no_points_yet_flagged():
    players = [_p(i, "MID", 0) for i in range(1, 16)]
    squad = {"player_ids": [p.id for p in players], "starters": list(range(1, 12)),
             "bench": [12, 13, 14, 15], "captain": 1, "vice": 2}
    card = scoring.score_matchday(_ds(players), squad, 1)
    assert card.played_any is False
    assert card.total == 0
