"""Manual fixture-specific expectations from betting odds or trusted projections.

The live sources for match odds vary by country and availability, so this module provides a
stable local interface instead of baking in a fragile scraper. Populate `data/fixture_odds.yaml`
with per-fixture xG values and the projection model will use them over generic team strength.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from ..model.opponent import MatchExpectation, _poisson_win_prob
from .fifa import Dataset


def load_fixture_expectations(
    path: str | Path | None,
    ds: Dataset,
) -> dict[tuple[int, int], MatchExpectation]:
    """Return {(fixture_id, squad_id): MatchExpectation}.

    YAML shape:

    fixtures:
      19:
        home_xg: 1.6
        away_xg: 0.8
        home_clean_sheet: 0.46  # optional; defaults to exp(-away_xg)
        away_clean_sheet: 0.20  # optional; defaults to exp(-home_xg)
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    fixtures = raw.get("fixtures", raw) if isinstance(raw, dict) else {}
    by_fixture = {fx.id: fx for rnd in ds.rounds for fx in rnd.fixtures}
    out: dict[tuple[int, int], MatchExpectation] = {}
    for key, value in fixtures.items():
        if not isinstance(value, dict):
            continue
        try:
            fixture_id = int(key)
            home_xg = float(value["home_xg"])
            away_xg = float(value["away_xg"])
        except (TypeError, ValueError, KeyError):
            continue
        fx = by_fixture.get(fixture_id)
        if fx is None:
            continue
        home_cs = _float_or(value.get("home_clean_sheet"), math.exp(-away_xg))
        away_cs = _float_or(value.get("away_clean_sheet"), math.exp(-home_xg))
        out[(fixture_id, fx.home_id)] = MatchExpectation(
            xgf=home_xg,
            xga=away_xg,
            clean_sheet=home_cs,
            win=_poisson_win_prob(home_xg, away_xg),
        )
        out[(fixture_id, fx.away_id)] = MatchExpectation(
            xgf=away_xg,
            xga=home_xg,
            clean_sheet=away_cs,
            win=_poisson_win_prob(away_xg, home_xg),
        )
    return out


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
