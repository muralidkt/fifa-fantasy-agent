"""Optional Claude layer: nudge expected points for soft signals the model can't see
(injury doubts, rotation risk, suspensions, hot/cold news).

OFF by default (config llm.enabled / --llm). Only the top-N candidates are sent, so token
use is bounded and one-shot. Degrades to a no-op if `anthropic` isn't installed or no API
key is set. Returns {player_id: (multiplier, reason)} for the caller to apply.
"""
from __future__ import annotations

import json
import os

from ..ingest.fifa import Dataset
from .expected_points import Projection


def adjust(
    ds: Dataset,
    projections: dict[int, Projection],
    *,
    model: str = "claude-sonnet-4-6",
    top_n: int = 40,
    news: str | None = None,
    api_key: str | None = None,
    focus_ids: list[int] | None = None,
) -> dict[int, tuple[float, str]]:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {}
    try:
        import anthropic
    except ImportError:
        return {}

    if focus_ids:  # subloop: vet a specific set of flagged players
        top = [projections[pid] for pid in focus_ids if pid in projections]
    else:
        top = sorted(projections.values(), key=lambda p: p.xpts, reverse=True)[:top_n]
    table = [
        {
            "id": pr.player.id,
            "name": pr.player.name,
            "team": ds.squads[pr.player.squad_id].name if pr.player.squad_id in ds.squads else "?",
            "pos": pr.player.position,
            "price": pr.player.price,
            "vs": pr.opponent_name,
            "xpts": pr.xpts,
        }
        for pr in top
    ]

    prompt = (
        "You are a World Cup fantasy-football analyst. For each player below, return a "
        "multiplier in [0.0, 1.3] reflecting CURRENT team news only: ~1.0 = no change, "
        "<1.0 for injury/rotation/suspension doubt (0.0 = ruled out), >1.0 for confirmed "
        "nailed starter in great form. Be conservative; default to 1.0 when unsure.\n"
        "Respond with ONLY a JSON array of {\"id\": int, \"m\": float, \"why\": str (<=8 words)}.\n"
    )
    if news:
        prompt += f"\nTeam news to consider:\n{news}\n"
    prompt += f"\nPlayers:\n{json.dumps(table, ensure_ascii=False)}"

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return _parse(text)


def _parse(text: str) -> dict[int, tuple[float, str]]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, tuple[float, str]] = {}
    for item in data:
        try:
            mult = max(0.0, min(1.3, float(item["m"])))
            out[int(item["id"])] = (mult, str(item.get("why", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def apply(projections: dict[int, Projection], adjustments: dict[int, tuple[float, str]]) -> dict[int, Projection]:
    """Return a new projections dict with multipliers applied to xpts."""
    if not adjustments:
        return projections
    from dataclasses import replace

    out: dict[int, Projection] = {}
    for pid, pr in projections.items():
        if pid in adjustments and adjustments[pid][0] != 1.0:
            out[pid] = replace(pr, xpts=round(pr.xpts * adjustments[pid][0], 3))
        else:
            out[pid] = pr
    return out
