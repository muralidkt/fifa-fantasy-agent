"""Tests for the new data sources: Polymarket odds strength + SoFIFA ratings blend."""
import csv
from pathlib import Path

import pytest

from fifa_fantasy.ingest import fifa, odds, ratings, strength

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "players.json").exists(), reason="run `fantasy refresh` to populate data cache"
)


def _ds():
    return fifa.load(DATA, auto_fetch=False)


# --- Polymarket name matcher (no network) -----------------------------------
def test_name_matcher_handles_aliases():
    ds = _ds()
    lookup = odds._fifa_lookup(ds)
    expect = {
        "Cape Verde": "Cabo Verde", "DR Congo": "Congo DR", "Ivory Coast": "Côte d'Ivoire",
        "South Korea": "Korea Republic", "Iran": "IR Iran", "Turkiye": "Türkiye",
        "Curacao": "Curaçao", "Brazil": "Brazil", "USA": "USA",
    }
    for pm_name, fifa_name in expect.items():
        sid = odds._match(pm_name, lookup)
        assert sid is not None, f"no match for {pm_name}"
        assert ds.squads[sid].name == fifa_name


def test_odds_strength_falls_back_to_price_without_cache(tmp_path):
    """With no polymarket cache and no network, odds source must not crash — falls back to price."""
    ds = _ds()
    s = strength.team_strength(ds, source="odds", cache_dir=str(tmp_path), stage="Quarterfinals")
    assert len(s) == 48
    assert all(0.0 <= v <= 1.0 for v in s.values())


# --- SoFIFA ratings blend ----------------------------------------------------
def test_ratings_load_and_percentiles(tmp_path):
    csv_path = tmp_path / "sofifa.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "overall"])
        w.writerow(["Kylian Mbappé", 91])
        w.writerow(["Some Journeyman", 68])
    r = ratings.load_overall_ratings(tmp_path)
    assert r[ratings.normalize_name("Kylian Mbappe")] == 91  # accent-insensitive
    pct = ratings.rating_percentiles(r)
    assert pct[ratings.normalize_name("Kylian Mbappé")] == 1.0
    assert pct[ratings.normalize_name("Some Journeyman")] == 0.0


def test_ratings_blend_changes_quality(tmp_path):
    """A high rating for an otherwise cheap player should lift his projected points."""
    from fifa_fantasy.model.expected_points import project_round
    from fifa_fantasy.model.opponent import OpponentModel

    ds = _ds()
    model = OpponentModel(strength.team_strength(ds, cache_dir=str(DATA)))
    # pick a genuinely cheap player to amplify the rating effect
    cheap = min((p for p in ds.players if p.available and p.position == "MID"), key=lambda p: p.price)

    base = project_round(ds, 1, model)[cheap.id].xpts
    rp = {ratings.normalize_name(cheap.name): 1.0}  # treat him as top-rated
    boosted = project_round(ds, 1, model, rating_pct=rp, ratings_weight=0.5)[cheap.id].xpts
    assert boosted > base


def test_round_specific_overrides_adjust_projection():
    from fifa_fantasy.model.expected_points import project_round
    from fifa_fantasy.model.opponent import OpponentModel

    ds = _ds()
    model = OpponentModel(strength.team_strength(ds, cache_dir=str(DATA)))
    player = next(p for p in ds.players if p.available and p.position == "FWD")
    base = project_round(ds, 1, model)[player.id]
    overrides = {
        str(player.id): {
            "rounds": {
                "1": {
                    "start_prob": 0.25,
                    "goal_share": 0.60,
                    "assist_share": 0.10,
                    "penalty_xg": 0.15,
                }
            }
        }
    }
    adjusted = project_round(ds, 1, model, overrides=overrides)[player.id]
    assert adjusted.start_prob == 0.25
    assert adjusted.components["player_xg"] > base.components["player_xg"]


def test_fixture_expectations_override_generic_opponent_model(tmp_path):
    from fifa_fantasy.ingest import fixture_odds
    from fifa_fantasy.model.expected_points import project_round
    from fifa_fantasy.model.opponent import OpponentModel

    ds = _ds()
    fx = ds.round(1).fixtures[0]
    path = tmp_path / "fixture_odds.yaml"
    path.write_text(
        "fixtures:\n"
        f"  {fx.id}:\n"
        "    home_xg: 3.0\n"
        "    away_xg: 0.2\n"
        "    home_clean_sheet: 0.85\n"
    )
    expectations = fixture_odds.load_fixture_expectations(path, ds)
    model = OpponentModel(strength.team_strength(ds, cache_dir=str(DATA)))
    home_player = next(p for p in ds.players if p.available and p.squad_id == fx.home_id and p.position == "FWD")
    base = project_round(ds, 1, model)[home_player.id]
    adjusted = project_round(ds, 1, model, fixture_expectations=expectations)[home_player.id]
    assert adjusted.components["player_xg"] > base.components["player_xg"]
    assert expectations[(fx.id, fx.home_id)].clean_sheet == 0.85
