# FIFA World Cup 2026 Fantasy Agent

An agent that picks and manages your [FIFA World Cup 2026 Fantasy](https://play.fifa.com/fantasy)
team for you: it learns the rules, pulls live player data, models each player's expected points
against their actual matchday opponent, and recommends the optimal squad, starting XI, captain,
and transfers — within the rules of the game.

**Recommend-only by design.** FIFA exposes a public *read* API but no *write* API, and automating
team submission violates FIFA's Terms of Service (account-ban risk). So the agent computes the
exact team and you enter it in the app (~60 seconds). This captures ~95% of the value — the
analysis is the hard part — with zero risk to your account.

## How it works

```
FIFA public JSON ──┐
(prices, fixtures, ├─► team strength ─► opponent model ─► expected points ─► ILP optimizer ─► report
 ownership, form)  │   (per nation)     (goals/clean      (per player,        (legal 15 + XI
external form/Elo ─┘                     sheet odds)        per matchday)        + captain)
                                                              │
                                                  optional Claude layer
                                                  (injuries / rotation / news)
```

- **Data** (`ingest/fifa.py`): three live, no-auth endpoints under `play.fifa.com/json/fantasy/`
  — `squads.json` (48 teams), `rounds.json` (8 matchdays + every fixture), `players.json`
  (prices, positions, ownership, form). Cached locally; refresh with `fantasy refresh`.
- **Opponent toughness** (`ingest/strength.py` + `model/opponent.py`): each team is rated by the
  value of its squad (or World Football Elo if you drop a `data/elo.csv`); a Poisson model turns
  the strength gap for *that round's fixture* into goal and clean-sheet expectations. Because the
  opponent changes every matchday, so do the recommendations.
- **Expected points** (`model/expected_points.py`): blends player quality (price percentile, plus
  FIFA in-game form once games start), a start-probability heuristic, and the opponent expectation
  into a projected score, priced through the real 2026 scoring rules.
- **Optimizer** (`optimize/`): integer linear programming (PuLP/CBC) picks the best legal 15 and XI
  under budget, the 2/5/5/3 shape, the per-nation cap, and valid formations — and, for transfers,
  respects your free-transfer allowance (taking a −3 hit only when it clearly pays off).
- **Rules** (`rules.py`): the single source of truth for scoring, budget, caps, and transfers.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

fantasy refresh                  # download the latest FIFA data
fantasy build --save             # optimal Matchday 1 squad (saves it for later advice)
fantasy advise --round 2         # transfer suggestions for Matchday 2 vs the new opponents
fantasy score  --round 1         # your actual points once the matchday is played
```

Two ways to run:
- **Direct commands** (`build` / `advise`) — the lean, deterministic path above.
- **`fantasy run`** — the same pipeline orchestrated as a [LangGraph](https://langchain-ai.github.io/langgraph/)
  state machine (`ensure_data → assess → analyze → [research subloop] → synthesize → approve → persist`),
  with a human-approval gate and an optional bounded LLM research subloop:
  ```bash
  pip install -e ".[graph]"
  fantasy run --round 1                 # build, with an interactive "accept this team?" gate
  fantasy run --round 2 --advise        # transfers via the graph
  fantasy run --round 1 --llm --news team_news.txt   # enable the research subloop
  ```

Useful flags: `--odds` (Polymarket implied strength), `--ratings` (blend SoFIFA overalls),
`--elo` (use `data/elo.csv` for strength), `--llm` (enable the Claude layer),
`--news team_news.txt` (feed the LLM team news), `--free N` (override free transfers),
`--yes` (skip the approval gate), `--max-research N` (cap subloop iterations).

## Docs

- [Architecture](docs/ARCHITECTURE.md) — pipeline, graph, core modules, and invariants.
- [Data and model](docs/DATA_AND_MODEL.md) — data sources, projection assumptions, optimizer model.
- [Tuning guide](docs/TUNING.md) — overrides, captain risk, ratings coverage, pre-deadline tuning.
- [Runbook](docs/RUNBOOK.md) — setup, daily commands, important files, troubleshooting.

### Scoring a matchday
FIFA publishes each player's **official points** in the same public `players.json`
(`stats.roundPoints`), so no extra API is needed. `fantasy score --round N` reads them for
your saved squad and computes your matchday total — starting XI only, **captain doubled**
(vice if the captain didn't play), **auto-substitutions** (a non-playing starter is replaced,
in bench order, by the next bench player who played and keeps the formation legal), minus any
`--hits`. Points are 0 until matches are played. `points_from_stats()` in `scoring.py` also
scores raw events against `rules.py` for what-if checks.

### Team-strength & quality signals
- **`--odds`** — team toughness from **Polymarket** implied "reach the Quarterfinals"
  probabilities (free, no key, live, forward-looking). Better than the price-sum default,
  which underrates South American sides and cohesive lower-budget teams. Auto-cached to
  `data/polymarket.json`; falls back to price if offline.
- **`--ratings`** — blend **SoFIFA / EA FC overall ratings** into the player-quality prior
  (helps lower-profile / thin-sample players). Populate `data/sofifa.csv` (`name,overall`) via
  `fantasy fetch-ratings` (`pip install -e '.[ratings]'`) or a Kaggle/GitHub EA FC export;
  SoFIFA blocks direct scraping (HTTP 403), so it's a local-file interface.

### Optional extras
- `pip install -e ".[form]"` — FBref recent-form ingestion via `soccerdata` (sharper pre-tournament
  signal than price alone).
- `pip install -e ".[llm]"` + `export ANTHROPIC_API_KEY=…` — enable the Claude adjustment layer for
  injuries / rotation / team news. Off by default to keep token use at zero.

## Status & roadmap

Working today: data ingestion, opponent-aware expected points, squad/XI/captain optimization,
matchday transfer optimization, the (optional) LLM research subloop, and the LangGraph
state-machine runner (`fantasy run`). Planned: FBref form wiring (`ingest/form.py`), a
Dixon-Coles opponent model via `penaltyblog`, predicted-lineup signals, and a backtest harness.
Browser-automated submission is intentionally **out of scope** (ToS/ban risk).

## ⚠️ Verify before the deadline

Scoring values come from FIFA's 2026 rules explainers; a few are disputed across sources and
marked `# VERIFY in-app` in `rules.py` (penalty save, red card, penalty miss). Confirm them
against the live in-app rules page before relying on them competitively. **Matchday 1 deadline:
20:00 UK, Thursday 11 June 2026.**

## Sources

FIFA public JSON (`play.fifa.com/json/fantasy/`), FIFA 2026 rules explainers (Fantasy Football
Scout et al.), World Football Elo (eloratings.net), FBref/StatsBomb via `soccerdata`.
This project is unofficial and not affiliated with FIFA.
