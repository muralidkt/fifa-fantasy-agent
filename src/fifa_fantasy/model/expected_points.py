"""Per-player expected fantasy points for a given matchday.

xPts(player) = appearance + attacking + defensive contributions, where:
  * quality   = price percentile within position, blended with FIFA in-game form once games start
  * start_prob= how likely the player starts (price rank within his squad & position)
  * the opponent for THIS round scales everything (weak opponent -> more goals & clean sheets)

Because the opponent changes every matchday, so does xPts — which is what drives the
matchday transfer suggestions. All coefficients are transparent heuristics, tunable below
and via config.yaml. They are priors, not a trained model.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..ingest.fifa import Dataset, Player
from ..ingest.ratings import normalize_name
from ..rules import SCORING
from .opponent import OpponentModel

# Share of a team's expected goals that flows to a top-quality player of each position,
# split into goal-scoring and assisting propensity. Heuristic.
GOAL_COEFF = {"GK": 0.0, "DEF": 0.06, "MID": 0.14, "FWD": 0.30}
ASSIST_COEFF = {"GK": 0.0, "DEF": 0.05, "MID": 0.12, "FWD": 0.10}
GK_SAVES_PER_OPP_GOAL = 2.5  # expected saves scale with opponent's expected goals


@dataclass(frozen=True)
class Projection:
    player: Player
    xpts: float
    opponent_id: int | None
    opponent_name: str
    start_prob: float
    quality: float
    components: dict[str, Any]


def project_round(
    ds: Dataset,
    round_id: int,
    model: OpponentModel,
    *,
    form_weight: float = 0.55,
    price_weight: float = 0.45,
    one_to_watch_bonus: float = 0.10,
    start_prob_floor: float = 0.15,
    rating_pct: dict[str, float] | None = None,
    ratings_weight: float = 0.0,
    overrides: dict[str, dict[str, Any]] | None = None,
    captain_start_weight: float = 0.75,
) -> dict[int, Projection]:
    """Return player_id -> Projection for the given round.

    `rating_pct` (normalised name -> percentile) is the optional SoFIFA quality signal; it is
    blended in only when present and `ratings_weight` > 0.
    """
    opponents = ds.opponent_map(round_id)
    by_squad = ds.players_by_squad()
    price_pct = _position_price_percentiles(ds.players)
    start_probs = _start_probabilities(by_squad, start_prob_floor)
    max_form = max((p.form for p in ds.players), default=0.0)
    weights = {"price": price_weight, "form": form_weight, "ratings": ratings_weight}
    overrides = overrides or {}

    out: dict[int, Projection] = {}
    for p in ds.players:
        if not p.available:
            continue
        opp = opponents.get(p.squad_id)
        if opp is None:  # team has no fixture this round (eliminated / bye)
            out[p.id] = Projection(p, 0.0, None, "—", 0.0, 0.0, {})
            continue

        quality = _quality(p, price_pct, max_form, weights, rating_pct)
        override = _override_for(p, overrides)
        start = _override_float(override, "start_prob", start_probs[p.id])
        start = max(0.0, min(1.0, start))
        if "quality" in override:
            quality = max(0.0, min(1.0, _override_float(override, "quality", quality)))
        exp = model.expectation(p.squad_id, opp.squad_id, team_home=opp.is_home)

        comp = _score_components(p, quality, start, exp)
        xpts = sum(comp.values())
        if p.one_to_watch:
            xpts *= 1.0 + one_to_watch_bonus
        xpts *= _override_float(override, "xpts_multiplier", 1.0)
        xpts += _override_float(override, "xpts_delta", 0.0)
        xpts = max(0.0, xpts)
        captain_score = xpts * (captain_start_weight + (1 - captain_start_weight) * start)
        if override.get("captain_avoid"):
            captain_score = 0.0
        comp["captain_score"] = round(captain_score, 3)
        if override.get("notes"):
            comp["risk_note"] = str(override["notes"])

        out[p.id] = Projection(
            player=p,
            xpts=round(xpts, 3),
            opponent_id=opp.squad_id,
            opponent_name=ds.squads[opp.squad_id].name if opp.squad_id in ds.squads else "?",
            start_prob=round(start, 3),
            quality=round(quality, 3),
            components={k: round(v, 3) if isinstance(v, int | float) else v for k, v in comp.items()},
        )
    return out


def load_projection_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional per-player projection overrides from YAML.

    Accepts either:
      players:
        Lionel Messi:
          start_prob: 0.95
    or a top-level mapping of player names/ids to override dictionaries.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    players = raw.get("players", raw) if isinstance(raw, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in players.items():
        if isinstance(value, dict):
            out[str(key)] = value
    return out


def _score_components(p: Player, quality: float, start: float, exp) -> dict[str, float]:
    pos = p.position
    s = SCORING

    appearance = start * s.appearance_long + (1 - start) * 0.15 * s.appearance_short

    player_xg = exp.xgf * GOAL_COEFF.get(pos, 0.0) * quality
    player_xa = exp.xgf * ASSIST_COEFF.get(pos, 0.0) * quality
    attacking = start * (player_xg * s.goal[pos] + player_xa * s.assist)

    defensive = 0.0
    cs_value = s.clean_sheet.get(pos, 0)
    if cs_value:
        defensive += start * exp.clean_sheet * cs_value
    if pos in ("GK", "DEF"):
        # expected concede penalty (-1 per 2, first not penalised)
        defensive += start * (s.goals_conceded_per / s.goals_conceded_unit) * max(0.0, exp.xga - 1.0)
    if pos == "GK":
        exp_saves = exp.xga * GK_SAVES_PER_OPP_GOAL
        defensive += start * (s.saves_per / s.saves_unit) * exp_saves

    return {"appearance": appearance, "attacking": attacking, "defensive": defensive}


def _override_for(p: Player, overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return (
        overrides.get(str(p.id))
        or overrides.get(p.name)
        or overrides.get(normalize_name(p.name))
        or {}
    )


def _override_float(override: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(override.get(key, default))
    except (TypeError, ValueError):
        return default


def _quality(p: Player, price_pct: dict[int, float], max_form: float,
             weights: dict[str, float], rating_pct: dict[str, float] | None) -> float:
    """Weighted blend of whatever quality signals are available for this player.

    Price percentile is always present; FIFA form is added once games are played; the SoFIFA
    rating percentile is added when supplied. Weights are renormalised over present signals,
    so a missing signal never penalises a player (pre-tournament = pure price).
    """
    signals = [(price_pct[p.id], weights["price"])]
    if max_form > 0 and p.form > 0:
        signals.append((p.form / max_form, weights["form"]))
    if rating_pct and weights.get("ratings", 0) > 0:
        rp = rating_pct.get(normalize_name(p.name))
        if rp is not None:
            signals.append((rp, weights["ratings"]))
    total = sum(w for _, w in signals)
    return sum(v * w for v, w in signals) / total if total else price_pct[p.id]


def _position_price_percentiles(players: list[Player]) -> dict[int, float]:
    """player_id -> price percentile within its position (0=cheapest, 1=most expensive)."""
    sorted_prices: dict[str, list[float]] = {}
    for p in players:
        sorted_prices.setdefault(p.position, []).append(p.price)
    for v in sorted_prices.values():
        v.sort()
    out: dict[int, float] = {}
    for p in players:
        arr = sorted_prices[p.position]
        n = len(arr)
        out[p.id] = 0.5 if n <= 1 else bisect_left(arr, p.price) / (n - 1)
    return out


def _start_probabilities(by_squad: dict[int, list[Player]], floor: float) -> dict[int, float]:
    """Within each squad+position, rank by price; expensive players are likelier to start."""
    out: dict[int, float] = {}
    for players in by_squad.values():
        by_pos: dict[str, list[Player]] = {}
        for p in players:
            if p.available:
                by_pos.setdefault(p.position, []).append(p)
        for group in by_pos.values():
            group.sort(key=lambda p: p.price, reverse=True)
            n = len(group)
            for rank, p in enumerate(group):
                pct = 1.0 if n <= 1 else 1.0 - rank / (n - 1)
                out[p.id] = floor + (1 - floor) * pct
    return out
