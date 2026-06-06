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
    captain_score: float | None = None
    risk_note: str = ""

    @property
    def armband_score(self) -> float:
        return self.xpts if self.captain_score is None else self.captain_score


def rows_from_projections(projections: dict[int, Projection], squads) -> list[Row]:
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
                captain_score=proj.components.get("captain_score"),
                risk_note=str(proj.components.get("risk_note", "")),
            )
        )
    return rows
