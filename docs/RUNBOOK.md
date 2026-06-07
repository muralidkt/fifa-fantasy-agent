# Runbook

Operational commands for using the agent during a tournament.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,graph]"
```

Optional extras:

```bash
pip install -e ".[ratings]"
pip install -e ".[llm]"
```

The LLM layer also requires:

```bash
export ANTHROPIC_API_KEY=...
```

## Daily commands

Refresh FIFA public JSON:

```bash
fantasy refresh
```

Build a squad directly:

```bash
fantasy build --odds --ratings --save
```

Run the full graph agent:

```bash
fantasy run --odds --ratings --yes
```

Advise transfers for a later matchday:

```bash
fantasy run --round 2 --advise --odds --ratings --yes
fantasy advise --round 2 --horizon 3 --odds --ratings
```

Score a completed matchday:

```bash
fantasy score --round 1
```

## Important files

| File | Purpose |
| --- | --- |
| `config.yaml` | Tunable weights, cache TTL, strength source, optimization settings. |
| `data/players.json` | Cached FIFA player data. |
| `data/rounds.json` | Cached rounds and fixtures. |
| `data/squads.json` | Cached national-team metadata. |
| `data/polymarket.json` | Cached Polymarket odds source when `--odds` is used. |
| `data/sofifa.csv` | Optional player ratings input for `--ratings`. |
| `data/overrides.yaml` | Manual team-news, start-risk, and captain-risk overrides. |
| `data/my_squad.yaml` | Last approved/saved squad, used by `advise` and `score`. |
| `data/report.html` | Optional HTML report output when requested. |

## HTML reports

Direct commands can export rich output to HTML:

```bash
fantasy build --odds --ratings --html data/report.html
fantasy advise --round 2 --odds --ratings --html data/report.html
fantasy score --round 1 --html data/report.html
```

## Pre-deadline checklist

1. Run `fantasy refresh`.
2. Update `data/overrides.yaml` for injuries, expected benchings, and captain risk.
3. Run `fantasy run --odds --ratings --yes`.
4. Review the `Start` column, captain, vice, and rationale watchlist.
5. Manually enter the squad at `play.fifa.com/fantasy`.
6. Save the terminal result or generate an HTML report if you want an audit trail.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No saved squad at data/my_squad.yaml` | Run `fantasy build --save` or `fantasy run --yes` in build mode first. |
| `LangGraph not installed` | Run `pip install -e ".[graph]"`. |
| Low SoFIFA match count | Add more rows to `data/sofifa.csv` using `name,overall`. |
| `--odds` unavailable/offline | The agent falls back to price-based strength where possible. Re-run later if you want fresh odds. |
| Tests fail with missing packages | Use `.venv/bin/python -m pytest`, not a system Python without dependencies. |
