"""Integer-linear-programming optimizers for squad selection and matchday transfers."""
from __future__ import annotations

from dataclasses import dataclass

from ..model.expected_points import Projection


@dataclass(frozen=True)
class Row:
    """Flat view of a candidate player for the optimizer."""
    pid: int
    name: str
    position: str
    price: float
    squad_id: int
    squad_name: str
    xpts: float
    start_prob: float = 1.0
    objective_xpts: float | None = None
    captain_score: float | None = None
    risk_note: str = ""
    ownership: float = 0.0

    @property
    def armband_score(self) -> float:
        return self.xpts if self.captain_score is None else self.captain_score

    @property
    def objective(self) -> float:
        return self.xpts if self.objective_xpts is None else self.objective_xpts


def rows_from_projections(projections: dict[int, Projection], squads, *, mode: str = "balanced") -> list[Row]:
    rows: list[Row] = []
    for proj in projections.values():
        p = proj.player
        rows.append(
            Row(
                pid=p.id,
                name=p.name,
                position=p.position,
                price=p.price,
                squad_id=p.squad_id,
                squad_name=squads[p.squad_id].name if p.squad_id in squads else "?",
                xpts=proj.xpts,
                start_prob=proj.start_prob,
                objective_xpts=_objective_xpts(proj, mode),
                captain_score=proj.components.get("captain_score"),
                risk_note=str(proj.components.get("risk_note", "")),
                ownership=p.ownership,
            )
        )
    return rows


def _objective_xpts(proj: Projection, mode: str = "balanced") -> float:
    """Recommendation-mode score used by the optimizer.

    Displayed xPts remains the raw projection. This only nudges squad selection.
    """
    xpts = proj.xpts
    start = proj.start_prob
    ownership = proj.player.ownership
    if mode == "safe":
        return xpts * (0.65 + 0.35 * start)
    if mode == "upside":
        return xpts * (1.0 + 0.08 * (1.0 - start)) + 0.02 * max(0.0, 10.0 - ownership)
    if mode == "differential":
        return xpts + 0.05 * max(0.0, 15.0 - ownership)
    if mode == "template":
        return xpts + 0.03 * min(ownership, 50.0)
    return xpts
