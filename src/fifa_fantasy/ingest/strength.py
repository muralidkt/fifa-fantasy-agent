"""National-team strength = how tough an opponent each team is.

Default ("price"): a team's strength is the sum of its most expensive players' FIFA
prices, normalised across all 48 teams to [0, 1]. FIFA prices its players by expected
output, so richer squads are stronger — a self-contained metric needing no external data.

Optional ("elo"): if `data/elo.csv` exists (columns `team,elo`), use World Football Elo
instead (better predictive signal), matching on team name; teams not found fall back to
their price-based strength.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .fifa import Dataset

_TOP_N_FOR_STRENGTH = 15  # value a team by its likely-contributing core, not its 26th player


def team_strength(ds: Dataset, *, source: str = "price", cache_dir: str | Path = "data",
                  stage: str = "Quarterfinals") -> dict[int, float]:
    """Return squad_id -> strength in [0, 1] (1 = strongest of the 48 teams).

    Each overlay signal (elo / odds) is normalised on its OWN scale, then merged over the
    price-based baseline, so teams missing from the overlay keep a comparable price strength
    instead of being crushed by a different unit's range.
    """
    base = _normalise(_price_raw(ds))
    if source == "elo":
        elo = _load_elo(Path(cache_dir) / "elo.csv", ds)
        if elo:
            base = {**base, **_normalise(elo)}
    elif source == "odds":
        from . import odds
        market = odds.market_strength_raw(ds, cache_dir, stage=stage)
        if market:
            base = {**base, **_normalise(market)}
    return base


def _price_raw(ds: Dataset) -> dict[int, float]:
    raw: dict[int, float] = {}
    for sid, players in ds.players_by_squad().items():
        top = sorted((p.price for p in players if p.available), reverse=True)[:_TOP_N_FOR_STRENGTH]
        raw[sid] = sum(top)
    return raw


def _load_elo(path: Path, ds: Dataset) -> dict[int, float]:
    if not path.exists():
        return {}
    by_name = {s.name.lower(): sid for s in ds.squads.values()}
    by_abbr = {s.abbr.lower(): sid for s in ds.squads.values()}
    out: dict[int, float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = (row.get("team") or "").strip().lower()
            sid = by_name.get(key) or by_abbr.get(key)
            if sid is not None:
                try:
                    out[sid] = float(row["elo"])
                except (KeyError, ValueError):
                    pass
    return out


def _normalise(raw: dict[int, float]) -> dict[int, float]:
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    if hi <= lo:
        return {sid: 0.5 for sid in raw}
    return {sid: (v - lo) / (hi - lo) for sid, v in raw.items()}
