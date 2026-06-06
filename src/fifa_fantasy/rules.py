"""FIFA World Cup 2026 Fantasy — game rules, the single source of truth.

Values come from FIFA's 2026 rules explainers (see README "Sources"). A few disputed
values are marked `# VERIFY in-app` — confirm against the live rules page on
play.fifa.com before relying on them competitively. Anything here can be overridden by
editing this file; nothing about the rules is hard-coded elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Squad composition -------------------------------------------------------
SQUAD_SIZE = 15
# Exact number of each position in the 15-man squad.
SQUAD_COMPOSITION: dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Starting XI formation bounds (1 GK always; valid FIFA formations e.g. 3-4-3, 4-4-2, 5-3-2).
XI_SIZE = 11
FORMATION_BOUNDS: dict[str, tuple[int, int]] = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (3, 5),
    "FWD": (1, 3),
}

POSITIONS = ("GK", "DEF", "MID", "FWD")

# --- Stages / matchdays ------------------------------------------------------
# rounds.json `stage` -> our notion. Round ids 1..8.
GROUP_STAGES = {"GROUP"}
KNOCKOUT_STAGES = {"R32", "R16", "QF", "SF", "F"}


def budget_for_stage(stage: str) -> float:
    """Squad budget in $m. $100m group stage, $105m from the Round of 32."""
    return 100.0 if stage in GROUP_STAGES else 105.0


def nation_cap_for_stage(stage: str) -> int:
    """Max players from one nation. Scales as teams are eliminated."""
    return {
        "GROUP": 3,
        "R32": 3,
        "R16": 4,
        "QF": 5,
        "SF": 6,
        "F": 8,
    }.get(stage, 3)


def free_transfers_for_round(round_id: int) -> float:
    """Free transfers available going INTO the given round id.

    inf == unlimited (the pre-MD1 build window and the MD3->R32 reset).
    Group rollover (2 + 1 banked) is approximated as 2; pass --free to override.
    """
    return {
        1: float("inf"),  # initial squad: unlimited
        2: 2,
        3: 2,
        4: float("inf"),  # unlimited reset between group stage and Round of 32
        5: 4,
        6: 4,
        7: 5,
        8: 6,
    }.get(round_id, 2)


TRANSFER_HIT = 3  # points deducted per extra transfer beyond the free allowance


# --- Scoring -----------------------------------------------------------------
@dataclass(frozen=True)
class Scoring:
    """Points awarded per event, by player position where it matters."""

    appearance_short: int = 1          # played 1-59 minutes
    appearance_long: int = 2           # played 60+ minutes (total, not additional)

    # Goal scored, by position.
    goal: dict[str, int] = field(default_factory=lambda: {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5})
    goal_outside_box_bonus: int = 1    # +1 for a goal from outside the box / direct free-kick

    assist: int = 3                    # primary assist only

    # Clean sheet (requires 60+ minutes), by position.
    clean_sheet: dict[str, int] = field(default_factory=lambda: {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0})

    goals_conceded_per: int = -1       # -1 for every 2 conceded (the first is not penalised)
    goals_conceded_unit: int = 2
    saves_per: int = 1                 # +1 per 3 saves (GK)
    saves_unit: int = 3

    penalty_saved: int = 5             # VERIFY in-app (some 2026 sources say +3)
    penalty_won: int = 2
    penalty_missed: int = -2           # VERIFY in-app (not consistently quoted)
    penalty_conceded: int = -1
    own_goal: int = -2
    yellow_card: int = -1
    red_card: int = -3                 # VERIFY in-app (some 2026 sources say -2)

    # Per-position performance bonuses.
    mid_tackles_per: int = 1           # MID +1 per 3 tackles
    mid_tackles_unit: int = 3
    mid_chances_per: int = 1           # MID +1 per 2 chances created
    mid_chances_unit: int = 2
    fwd_shots_on_target_per: int = 1   # FWD +1 per 2 shots on target
    fwd_shots_on_target_unit: int = 2

    scouting_bonus: int = 2            # +2 if a player scores >4 pts AND is owned by <5%
    scouting_ownership_threshold: float = 5.0
    scouting_points_threshold: int = 4

    captain_multiplier: int = 2        # captain scores double


SCORING = Scoring()


def stage_for_round(round_id: int) -> str:
    """Static fallback map round id -> stage (rounds.json is authoritative when loaded)."""
    return {1: "GROUP", 2: "GROUP", 3: "GROUP", 4: "R32", 5: "R16", 6: "QF", 7: "SF", 8: "F"}.get(
        round_id, "GROUP"
    )
