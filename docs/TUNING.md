# Tuning Guide

This project is only as good as the projection inputs. The optimizer will find the best legal team
for the scores it receives, so most performance work should improve player xPts, start probability,
and captain risk.

## Recommended matchday workflow

1. Refresh the public FIFA data.

   ```bash
   fantasy refresh
   ```

2. Update team-news overrides in `data/overrides.yaml`.

3. Run the agent with the strongest available deterministic signals.

   ```bash
   fantasy run --odds --ratings --yes
   ```

4. Check the `Start` column and rationale watchlist before entering the team manually.

5. Re-run close to deadline after lineups or reliable team news changes.

## Manual projection overrides

Overrides are loaded from `expected_points.overrides_path` in `config.yaml`, defaulting to
`data/overrides.yaml`.

Example:

```yaml
players:
  Mikel Oyarzabal:
    start_prob: 0.72
    captain_avoid: true
    notes: "verify Spain starting XI; rotation risk"
  Gonçalo Ramos:
    start_prob: 0.74
    notes: "verify Portugal striker role"
  Lionel Messi:
    start_prob: 0.95
```

Keys can be exact player names, normalized names, or FIFA player ids.

Supported fields:

| Field | Meaning |
| --- | --- |
| `start_prob` | Replaces the model's start probability. Use `0.0` to `1.0`. |
| `quality` | Replaces blended player-quality percentile. Use `0.0` to `1.0`. |
| `xpts_multiplier` | Multiplies final projected points. Use for broad boosts/penalties. |
| `xpts_delta` | Adds or subtracts final projected points. Use for specific judgement calls. |
| `captain_avoid` | Prevents captain/vice selection without removing the player from the squad. |
| `notes` | Appears in rationale/watchlists so judgement remains auditable. |

## Captain logic

Raw xPts is not enough for captaincy because a benched captain is far more damaging than a benched
ordinary starter. The agent computes:

```text
captain_score = xPts * (captain_start_weight + (1 - captain_start_weight) * start_prob)
```

The default `captain_start_weight` is `0.75`. Lower values punish start risk more aggressively.

Practical settings:

| Scenario | Suggested value |
| --- | --- |
| Conservative captaincy | `0.55` to `0.70` |
| Balanced default | `0.75` |
| Chase upside | `0.85` to `0.95` |

Use `captain_avoid: true` for players you still like as starters but do not trust enough for the
armband.

## SoFIFA ratings coverage

`--ratings` helps only when `data/sofifa.csv` contains matching player names. If the agent reports
low match coverage, improve the CSV before tuning model weights.

CSV format:

```csv
name,overall
Lionel Messi,88
Pedri,86
```

Run:

```bash
fantasy build --odds --ratings
```

The command prints how many ratings loaded and how many players matched.

## Risk interpretation

Treat the `Start` column as a prior, not a confirmed lineup. Values come from price rank unless
overridden.

Suggested interpretation:

| Start probability | Meaning |
| --- | --- |
| `95-100%` | Model thinks the player is highly likely to start. |
| `85-94%` | Good pick, but worth checking team news. |
| `70-84%` | Rotation risk; strong upside must justify selection. |
| `<70%` | Avoid unless you have specific team-news confidence. |

## What to improve first

1. Add reliable `start_prob` overrides for uncertain forwards, attacking midfielders, and captain
   candidates.
2. Improve `data/sofifa.csv` coverage and name matching.
3. Add fixture-specific odds or manual `xpts_multiplier` adjustments for strong/weak matchups.
4. Use `--llm --news team_news.txt` only when you have useful curated news text.

