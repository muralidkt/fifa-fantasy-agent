"""Forward-looking team strength from Polymarket prediction markets (free, no auth).

Polymarket's "World Cup: Nation To Reach <stage>" events each contain 48 binary markets
(one per nation); the "Yes" price is the market-implied probability that team reaches that
stage — an excellent, live, forward-looking strength signal. We pull one stage market,
map team names to FIFA squad ids, and return log-probability strengths for `strength.py`
to normalise. Unmatched teams fall back to the price-based strength.

Endpoint (public, no key): https://gamma-api.polymarket.com/events?tag_slug=world-cup
"""
from __future__ import annotations

import json
import math
import time
import unicodedata
from pathlib import Path

import requests

from .fifa import Dataset

GAMMA_URL = "https://gamma-api.polymarket.com/events"
CACHE_FILE = "polymarket.json"
_PROB_FLOOR = 0.01  # avoid log(0) for no-hope nations

# Stage event titles to try, best spread first. ("Nation To Reach <X>")
STAGE_TITLES = {
    "Quarterfinals": "Nation To Reach Quarterfinals",
    "Round of 16": "Nation To Reach Round of 16",
    "Semifinals": "Nation To Reach Semifinals",
}

# Polymarket name (normalised) -> FIFA name (normalised). Only the genuine mismatches.
_ALIASES = {
    "drcongo": "congodr",
    "capeverde": "caboverde",
    "ivorycoast": "cotedivoire",
    "southkorea": "korearepublic",
    "iran": "iriran",
}


def market_strength_raw(ds: Dataset, cache_dir: str | Path = "data", *,
                        stage: str = "Quarterfinals", ttl_hours: float = 12.0) -> dict[int, float]:
    """squad_id -> log(implied probability) for teams found in the market (others omitted)."""
    probs = _team_probabilities(cache_dir, stage=stage, ttl_hours=ttl_hours)
    if not probs:
        return {}
    lookup = _fifa_lookup(ds)
    out: dict[int, float] = {}
    for name, p in probs.items():
        sid = _match(name, lookup)
        if sid is not None:
            out[sid] = math.log(max(p, _PROB_FLOOR))
    return out


def _team_probabilities(cache_dir: str | Path, *, stage: str, ttl_hours: float) -> dict[str, float]:
    events = _load_events(cache_dir, ttl_hours=ttl_hours)
    target = STAGE_TITLES.get(stage, STAGE_TITLES["Quarterfinals"]).lower()
    # find the requested stage event, else the first available stage event
    event = _find_event(events, target) or next(
        (e for t in STAGE_TITLES.values() for e in [_find_event(events, t.lower())] if e), None
    )
    if event is None:
        return {}
    probs: dict[str, float] = {}
    for m in event.get("markets", []):
        team = m.get("groupItemTitle")
        prices = m.get("outcomePrices")
        if not team or not prices:
            continue
        try:
            yes = float(json.loads(prices)[0])  # ["Yes","No"] -> first price is P(Yes)
        except (ValueError, IndexError, TypeError):
            continue
        probs[team] = yes
    return probs


def _find_event(events: list[dict], title_sub: str) -> dict | None:
    for e in events:
        if title_sub in (e.get("title") or "").lower():
            return e
    return None


def _load_events(cache_dir: str | Path, *, ttl_hours: float) -> list[dict]:
    path = Path(cache_dir) / CACHE_FILE
    if not path.exists() or _age_hours(path) >= ttl_hours:
        try:
            resp = requests.get(
                GAMMA_URL,
                params={"tag_slug": "world-cup", "closed": "false", "limit": 300},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            json.loads(json.dumps(data))  # validate serialisable
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        except Exception:
            if not path.exists():
                return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("data", [])


def _age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600.0


# --- name matching -----------------------------------------------------------
def _normalise(name: str) -> str:
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return "".join(c for c in n.lower() if c.isalnum())


def _fifa_lookup(ds: Dataset) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for s in ds.squads.values():
        lookup[_normalise(s.name)] = s.id
        if s.abbr:
            lookup[_normalise(s.abbr)] = s.id
    return lookup


def _match(name: str, lookup: dict[str, int]) -> int | None:
    norm = _normalise(name)
    norm = _ALIASES.get(norm, norm)
    return lookup.get(norm)
