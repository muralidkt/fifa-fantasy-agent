# CLAUDE.md — guidance for agents working in this repo

FIFA World Cup 2026 Fantasy agent: it learns the rules, pulls live data, projects each
player's expected points against their actual matchday opponent, and recommends the optimal
squad / XI / captain / transfers — **recommend-only** (you enter the team yourself).

## Run & test (always use the project venv)
```bash
.venv/bin/python -m pytest -q          # full suite (28 tests)
.venv/bin/python -m pip install -e ".[dev]"   # if deps missing
fantasy refresh                        # pull latest FIFA public JSON into data/
fantasy build --save                   # MD1 squad   (see docs/USAGE.md for all commands)
```
Note: `pytest` on the bare interpreter fails to collect (missing `requests`); the deps live
in `.venv`, so invoke tests via `.venv/bin/python -m pytest`.

## Design invariants (do not break these)
- **The optimizer and model are PURE functions.** The ILP (PuLP/CBC) enforces every hard rule
  (budget, 2/5/5/3, per-nation cap, formation). Never let the LLM pick the team — it only
  returns bounded `0.0–1.3` score multipliers inside the `research` subloop.
- **`rules.py` is the single source of truth** for scoring, budget, caps, and transfers.
  Don't hard-code rule values elsewhere; a few are marked `# VERIFY in-app` (see docs/RULES.md).
- **Recommend-only.** No write API; automating team submission violates FIFA ToS (ban risk).
  Browser automation is intentionally out of scope unless the user explicitly opts in.
- **Free data only, cached + throttled.** All sources (FIFA JSON, Polymarket, Elo, SoFIFA) are
  free; cache to `data/` (gitignored) and degrade gracefully when offline.
- Match the surrounding code style; keep new code as readable as what's there.

## Layout
```
src/fifa_fantasy/
  rules.py            # game rules + scoring (single source of truth)
  ingest/   fifa.py (public JSON), strength.py, odds.py (Polymarket), ratings.py (SoFIFA)
  model/    opponent.py (Poisson), expected_points.py (xPts), llm_adjust.py (optional Claude)
  optimize/ squad.py (ILP 15+XI), transfers.py (matchday), __init__.py (Row)
  scoring.py          # actual matchday points from FIFA's official roundPoints
  report.py           # rich tables + HTML export
  graph.py            # LangGraph state machine (`fantasy run`)
  cli.py              # `fantasy` commands
```

## Two run paths
- Lean deterministic commands: `build` / `advise` / `score` (zero LLM tokens by default).
- LangGraph orchestration: `fantasy run` — `ensure_data → assess → analyze → risk_check ⇄
  research → synthesize → human_approval → persist`. See docs/ARCHITECTURE.md.

## Docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline, graph, design rationale
- [docs/RULES.md](docs/RULES.md) — encoded 2026 rules & scoring (+ verify-in-app items)
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — every data source, access, caveats
- [docs/USAGE.md](docs/USAGE.md) — command cookbook & weekly routine

## Known follow-ups (living list)
- `graph._expected_fixture_count("F")` assumes 2 fixtures (3rd-place + final, per 2022). Verify
  against live data once the bracket populates; if the Final round has only 1 fixture, it will
  wrongly report "not drawn".
- The readiness guard exists only on the graph path; direct `build`/`advise` don't yet refuse an
  undrawn knockout round (they'd emit a meaningless all-zero-projection squad).
- Highest-ROI accuracy work next: a predicted-lineups/minutes source and a backtest harness
  (replay 2022 to score price vs odds vs odds+ratings on rank-correlation / captain hit-rate).
