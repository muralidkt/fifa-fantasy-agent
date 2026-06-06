"""Parse the cached real 2026 data (offline). Requires data/*.json (already seeded)."""
from pathlib import Path

import pytest

from fifa_fantasy.ingest import fifa

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "players.json").exists(), reason="run `fantasy refresh` to populate data cache"
)


def _ds():
    return fifa.load(DATA, auto_fetch=False)


def test_loads_48_teams_and_8_rounds():
    ds = _ds()
    assert len(ds.squads) == 48
    assert [r.id for r in ds.rounds] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds.rounds[0].stage == "GROUP" and ds.rounds[3].stage == "R32"


def test_active_squads_are_26_man():
    ds = _ds()
    active = [p for p in ds.players if p.available]
    assert len(active) == 48 * 26  # 1248


def test_opponent_map_is_symmetric_for_round1():
    ds = _ds()
    opp = ds.opponent_map(1)
    # Round 1 fixture 1 is Mexico (28) vs South Africa (40).
    assert opp[28].squad_id == 40 and opp[28].is_home is True
    assert opp[40].squad_id == 28 and opp[40].is_home is False


def test_every_team_plays_once_in_a_group_round():
    ds = _ds()
    opp = ds.opponent_map(1)
    assert len(opp) == 48  # all 48 teams have exactly one opponent in MD1
