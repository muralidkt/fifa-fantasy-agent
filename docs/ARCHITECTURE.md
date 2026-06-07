# Architecture

A deterministic **pipeline + optimizer**, optionally orchestrated by a **LangGraph** state
machine. It is not a ReAct/RAG agent: the intelligence is operations research (ILP) plus a
transparent expected-points model. An LLM can adjust projections from team news, but it never
directly picks an illegal team or bypasses the optimizer.

## Data flow

```
FIFA public JSON ──┐
(prices, fixtures, ├─► team strength ─► opponent model ─► expected points ─► ILP optimizer ─► report
 ownership, points)│   (per nation)     (goals / clean    (per player,        (legal 15 + XI
Polymarket / Elo  ─┘                     sheet, Poisson)    per matchday)        + captain)
SoFIFA ratings ───────────────────────────────┘ (quality prior)        │
manual overrides ─────────────────────────────┘ (start risk/news)      │
                                                              optional Claude subloop
                                                              (bounded score multipliers)
```

1. **Ingest** (`ingest/fifa.py`) — three live, no-auth endpoints under `play.fifa.com/json/fantasy/`:
   `squads.json` (48 teams), `rounds.json` (8 rounds + every fixture → each team's opponent per
   matchday), `players.json` (price, position, ownership, form, **official points**). Cached to `data/`.
2. **Team strength** (`ingest/strength.py`) — price-sum by default; `--odds` overlays Polymarket
   implied "reach QF" probabilities; `--elo` overlays `data/elo.csv`. Each overlay is normalised on
   its own scale, then merged over the price baseline.
3. **Opponent model** (`model/opponent.py`) — a transparent Poisson model turns the strength gap for
   *that round's fixture* into expected goals for/against and a clean-sheet probability.
4. **Expected points** (`model/expected_points.py`) — blends player quality (price percentile, FIFA
   form once games start, optional SoFIFA rating), a start-probability heuristic, manual overrides
   including round-specific starts and attacking shares, and the opponent expectation into a
   projected score, priced through `rules.py`.
5. **Optimize** (`optimize/`) — PuLP/CBC integer linear program picks the best legal 15 + XI. It
   maximizes starting XI points plus a small bench value, optionally nudged by recommendation mode,
   then chooses captain and vice using a reliability-aware armband score instead of raw xPts alone.
   For `advise`, it finds the best transfers within the free allowance, taking a -3 hit only when
   it pays off; direct `advise` can value a weighted multi-round horizon.
6. **Report / score** (`report.py`, `scoring.py`) — rich tables (+ HTML export); `score` computes the
   actual matchday total from FIFA's official `roundPoints`.

Because the opponent changes every matchday, so do the projections — which is what drives the
matchday transfer suggestions.

## LangGraph state machine (`fantasy run`, in `graph.py`)

```
START
  → ensure_data        refresh-if-stale, build dataset + opponent model (+ optional ratings)
  → assess_situation   is the round fully drawn? (expected fixture count per stage)
      ├─ not ready → END
      └─ ready → analyze
  → analyze            deterministic xPts + ILP  (pure functions)
  → risk_check ←──────────────────────┐
      ├─ flagged & --llm & !converged & iters left → research ┘   (bounded subloop)
      └─ else → synthesize
  → synthesize         captain + chip advice + rationale
  → human_approval     show team, accept (recommend-only gate; --yes to skip)
  → persist            save squad (build mode)
  → END
```

### Invariants
- **Pure core.** `analyze`/`_optimize` call the same pure functions the CLI uses; the graph only
  orchestrates (refresh, branch, loop, approve, persist).
- **Optimizer owns legality.** Projections, overrides, and LLM adjustments change scores only. Squad
  size, budget, positions, nation caps, formations, and transfer-hit math remain hard constraints in
  the ILP.
- **Manual overrides are first-class inputs.** `data/overrides.yaml` can set start probability,
  quality, score deltas/multipliers, notes, and captain avoidance. This keeps late team-news judgment
  auditable and outside code.
- **Bounded subloop.** `research` runs only with `--llm`, vets `flagged` players, and applies the
  *merged* multipliers once onto a stable `base_projections` (no compounding across passes). It
  stops on convergence or `--max-research`.
- **Fail-closed readiness.** A round is "drawn" only when it has the full expected fixture count and
  every fixture has both teams — partially drawn knockout rounds short-circuit.

## Why no framework for the core / why LangGraph for `run`
The deterministic path is a straight line — plain function calls, fully testable, zero tokens.
LangGraph earns its place only on `run`, where there are real branches (readiness), a real cycle
(the research subloop), and a human-in-the-loop approval gate. See the chat history / README for the
trade-off discussion.

## Key modules

| Module | Responsibility |
| --- | --- |
| `src/fifa_fantasy/cli.py` | Click commands and shared direct-command pipeline. |
| `src/fifa_fantasy/graph.py` | LangGraph orchestration for `fantasy run`. |
| `src/fifa_fantasy/ingest/fifa.py` | FIFA public JSON loading and cache management. |
| `src/fifa_fantasy/ingest/strength.py` | Team-strength source selection and normalization. |
| `src/fifa_fantasy/model/opponent.py` | Matchup expectations from strength gaps. |
| `src/fifa_fantasy/model/expected_points.py` | Player xPts, start probability, overrides, captain score. |
| `src/fifa_fantasy/optimize/squad.py` | Squad/XI ILP and captain/vice assembly. |
| `src/fifa_fantasy/optimize/transfers.py` | Transfer ILP with free-transfer and hit constraints. |
| `src/fifa_fantasy/report.py` | Terminal and HTML rendering. |
| `src/fifa_fantasy/scoring.py` | Actual matchday scoring from FIFA official points. |
| `src/fifa_fantasy/rules.py` | Game rules, scoring constants, budgets, caps, transfer counts. |

## Testing
`tests/` covers rules, ingestion (against cached real data), ILP legality, the data sources
(odds matcher, ratings blend), scoring (captain doubling, auto-subs, rules-based events), and the
graph (build/advise/short-circuit/no-compounding/free-transfer override). Run with
`.venv/bin/python -m pytest -q`.
