# Data And Model

This document explains the main data inputs and modeling assumptions behind the fantasy
recommendations.

## Data inputs

| Source | File/cache | Used for |
| --- | --- | --- |
| FIFA public fantasy JSON | `data/players.json`, `data/rounds.json`, `data/squads.json` | Players, prices, positions, squads, fixtures, availability, form, official points. |
| Polymarket odds | `data/polymarket.json` | Optional team-strength signal with `--odds`. |
| Fixture odds / xG | `data/fixture_odds.yaml` | Optional per-fixture xG and clean-sheet probabilities. |
| World Football Elo | `data/elo.csv` | Optional team-strength signal with `--elo`. |
| SoFIFA / EA FC ratings | `data/sofifa.csv` | Optional player-quality signal with `--ratings`. |
| Manual overrides | `data/overrides.yaml` | Start probability, score adjustments, captain avoidance, risk notes. |
| Team news text | user-supplied file via `--news` | Optional LLM score adjustments when `--llm` is enabled. |

The core pipeline works without optional external files. Missing optional signals are skipped or
fall back to safer defaults.

## Projection model

For each available player in a round:

1. Find the player's actual opponent from `rounds.json`.
2. Estimate team strength from price, odds, or Elo.
3. Convert strength gap into goals for, goals against, and clean-sheet probability, unless
   `data/fixture_odds.yaml` supplies fixture-specific expectations.
4. Estimate player quality from price percentile, in-game form, and optional ratings.
5. Estimate start probability from price rank within squad and position.
6. Apply manual overrides, including round-specific starts and player attacking shares.
7. Convert appearance, attacking, defensive, and goalkeeper save expectations into fantasy xPts.

The output is a `Projection` containing:

| Field | Meaning |
| --- | --- |
| `xpts` | Expected fantasy points for this matchday. |
| `start_prob` | Estimated chance the player starts. |
| `quality` | Blended player-quality percentile. |
| `components` | Breakdown values plus captain score and optional risk notes. |

## Optimizer model

The optimizer is an integer linear program. It chooses:

| Decision | Constraint |
| --- | --- |
| 15-player squad | Must match required squad composition. |
| Starting XI | Must satisfy formation bounds. |
| Budget | Must stay under stage budget. |
| Nation cap | Must respect per-stage max players per nation. |
| Transfers | Must respect current squad and transfer-hit costs in `advise` mode. |

The objective is:

```text
starting_xpts + bench_weight * bench_xpts
```

Captain and vice are selected after the XI is chosen using the armband score:

```text
captain_score = xPts * (captain_start_weight + (1 - captain_start_weight) * start_prob)
```

This keeps squad selection point-maximizing while making captaincy more robust to rotation risk.

`--mode` changes the optimizer objective only; reports still show raw xPts. `--horizon` on direct
`advise` optimizes one fixed 15 across multiple future rounds while allowing a different legal XI in
each round, with future rounds discounted by `optimize.horizon_decay`.

## Known limitations

| Limitation | Impact | Mitigation |
| --- | --- | --- |
| Start probability is heuristic by default | Rotation-risk players can be overvalued | Use `data/overrides.yaml`. |
| SoFIFA coverage may be sparse | Ratings help only matched players | Improve `data/sofifa.csv`. |
| Tournament odds are not fixture odds | Team strength may miss match-specific context | Add manual multipliers or future fixture-odds integration. |
| Public FIFA data is read-only | Agent cannot submit teams | Manually enter recommendations in the FIFA app/site. |
| Rules may change or be disputed | Scoring constants can drift | Verify `rules.py` against in-app rules before deadline. |

## Recommended future improvements

1. Fixture-specific betting odds: win probability, clean-sheet odds, totals, anytime scorer.
2. Predicted-lineup ingestion from reliable public sources.
3. Broader player ratings and aliases for better name matching.
4. Backtesting against completed matchdays.
5. Separate output modes for max projection, safe, and differential squads.
