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
