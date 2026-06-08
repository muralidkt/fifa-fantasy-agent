# Matchday routine — quick reference

A per-matchday cheat sheet for running the agent through the tournament.
This is **recommend-only**: the tool tells you what to do, you enter the team at
`play.fifa.com/fantasy` yourself.

## Key idea: a "matchday" is not a calendar day

The `--round N` id (1–8) is a **fantasy scoring period** that covers several real
fixtures over a few days — not a single day. You run **once per matchday**, not every day:

- **Advise** before the matchday's deadline.
- **Score** after its games finish.

`--round N` selects the matchday; the date does not.

## One-time setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,graph]"
# optional extras:
pip install -e ".[ratings]"     # SoFIFA ratings for --ratings
pip install -e ".[llm]"         # Claude layer for --llm (also needs ANTHROPIC_API_KEY)
```

## Matchday 1 — build & save your squad

```bash
fantasy refresh
fantasy build --round 1 --odds --ratings --horizon 3 --save
```

- `--save` writes the chosen 15 to `data/my_squad.yaml` (used by `advise` and `score`).
- `--horizon 3` values MD1 **plus** the next two rounds when picking the squad
  (decay-weighted by `optimize.horizon_decay = 0.70` in `config.yaml`), so you don't
  pick a squad that's great for MD1 but stranded afterwards.
- Then enter the squad in the FIFA app.

## Each later matchday (MD2, MD3, …)

**1. Before the deadline — get transfer advice (re-saves your squad):**

```bash
fantasy refresh
# (optional) edit data/overrides.yaml for injuries, benchings, captain risk
fantasy advise --round 2 --odds --ratings --horizon 3
```

- Reads `data/my_squad.yaml`, applies that round's free-transfer allowance, and
  recommends transfers in/out + the best XI & captain.
- `--horizon 3` plans this window while looking ahead, so it won't burn a transfer
  that future fixtures make pointless.
- Use `--free N` if the app shows a different free-transfer count than the rules assume.

**2. After the games finish — score what you actually got:**

```bash
fantasy score --round 2 --hits 6      # --hits = points docked for paid transfers (paid × 3)
```

## One-command alternative: `fantasy run`

Wraps the whole flow with the LangGraph state machine (readiness guard + approval gate):

```bash
fantasy run --round 1 --odds --ratings --yes              # MD1: builds + saves
fantasy run --round 2 --advise --odds --ratings --yes     # MD2+: loads, transfers, re-saves
```

- MD1 builds & saves; later matchdays need `--advise` to load the saved squad and apply
  the transfer window.
- `run` defaults to `--horizon 3`. Add `--yes` to skip the interactive approval prompt,
  `--llm` to enable the bounded research subloop (costs tokens; off by default).

## Talk to the agent: `fantasy chat`

A conversational front-end over `build` / `advise` / `score`. Needs the LLM extra and a key:

```bash
pip install -e ".[llm]"          # installs `anthropic`
export ANTHROPIC_API_KEY=sk-...
fantasy refresh
fantasy chat --round 3 --odds --ratings
```

What it's for — ask in plain language:
- *"Why is `<player>` on the bench for MD3?"* → explains the projection (opponent, xPts,
  start prob, component breakdown).
- *"What are the captain / clean-sheet scoring rules?"* → reads `rules.py`, and flags the
  `# VERIFY in-app` items as "confirm on play.fifa.com" rather than asserting them.
- *"Force `<player>` into the squad"* / *"don't pick anyone from `<nation>`"* → re-runs the
  **ILP** with lock/ban constraints. The solver still enforces every rule; the agent never
  hand-picks the team.
- *"`<player>` is a rotation risk, set start prob 0.5 for MD3"* → proposes an
  `overrides.yaml` edit and asks **`Apply? [y/N]`** before writing.
- *"Save that as my squad"* → re-saves `my_squad.yaml` from the last build/advise result,
  again behind a `y/N` gate.

Guardrails: the LLM only routes, explains, and adds bounded constraints/overrides — the
PuLP/CBC optimiser makes every actual selection, and all file writes require your `y/N`.
Without `anthropic`/`ANTHROPIC_API_KEY`, `fantasy chat` prints how to enable it and exits.

> Chips (Wildcard / Bench Boost / Triple Captain / Free Hit) aren't modelled yet, so the
> agent can't advise on chip timing — that's a follow-up.

## Cadence at a glance

| When | Command |
| --- | --- |
| Before MD1 deadline | `fantasy refresh` → `fantasy build --round 1 --odds --ratings --horizon 3 --save` |
| Before each later MD deadline | `fantasy refresh` → `fantasy advise --round N --odds --ratings --horizon 3` |
| After each MD's games finish | `fantasy score --round N --hits <paid×3>` |

## `--horizon` defaults differ

| Command | Default `--horizon` |
| --- | --- |
| `build` | 1 |
| `advise` | 1 |
| `run` | 3 |

Pass `--horizon 3` explicitly on `build`/`advise` if you want the multi-round look-ahead.

## Caveats

- **Knockout rounds:** if a round isn't drawn yet, projections are meaningless. `fantasy run`
  has a readiness guard that refuses; direct `build`/`advise` currently don't, so they'd emit
  an all-zero-projection squad. `--horizon` is most reliable during the group stage, when
  future opponents are known.
- `--odds` offline: the agent falls back to price-based strength. Re-run later for fresh odds.
- Use `.venv/bin/python -m pytest -q` to run tests (system Python lacks the deps).

## HTML reports (optional audit trail)

```bash
fantasy build  --odds --ratings --html data/report.html
fantasy advise --round 2 --odds --ratings --html data/report.html
fantasy score  --round 2 --html data/report.html
```

See also: [docs/RUNBOOK.md](RUNBOOK.md) (full operational reference) and
[docs/TUNING.md](TUNING.md) (`--horizon` / `horizon_decay` tuning).
