"""Turn a team-strength gap into match expectations (goals for/against, clean-sheet prob).

A transparent, dependency-free Poisson model:
  factor(t)      = exp(beta * (strength[t] - 0.5))        # multiplicative quality factor
  xGF(T vs O)    = base_goals * factor[T] / factor[O]     # expected goals for T
  xGA(T)         = xGF(O vs T)                             # = expected goals against T
  clean_sheet(T) = exp(-xGA)                              # Poisson P(0 conceded)
A small home advantage nudges the home side's factor up. `beta` and `base_goals` are
tunable in config.yaml. (Swap in penaltyblog's Dixon-Coles here for a sharper model.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchExpectation:
    xgf: float          # expected goals scored by the team
    xga: float          # expected goals conceded by the team
    clean_sheet: float  # probability of conceding 0
    win: float          # probability the team wins


class OpponentModel:
    def __init__(self, strength: dict[int, float], *, beta: float = 1.5,
                 base_goals: float = 1.35, home_advantage: float = 0.20):
        self.strength = strength
        self.beta = beta
        self.base_goals = base_goals
        self.home_advantage = home_advantage

    def _factor(self, squad_id: int, *, home: bool) -> float:
        s = self.strength.get(squad_id, 0.5)
        if home:
            s += self.home_advantage * 0.5
        return math.exp(self.beta * (s - 0.5))

    def expectation(self, team_id: int, opp_id: int, *, team_home: bool) -> MatchExpectation:
        ft = self._factor(team_id, home=team_home)
        fo = self._factor(opp_id, home=not team_home)
        xgf = self.base_goals * ft / fo
        xga = self.base_goals * fo / ft
        cs = math.exp(-xga)
        win = _poisson_win_prob(xgf, xga)
        return MatchExpectation(xgf=xgf, xga=xga, clean_sheet=cs, win=win)


def _poisson_win_prob(xgf: float, xga: float, *, max_goals: int = 8) -> float:
    """P(team scores more than it concedes) under independent Poisson goals."""
    pf = [_poisson_pmf(k, xgf) for k in range(max_goals + 1)]
    pa = [_poisson_pmf(k, xga) for k in range(max_goals + 1)]
    win = 0.0
    for i in range(max_goals + 1):
        for j in range(i):
            win += pf[i] * pa[j]
    return win


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)
