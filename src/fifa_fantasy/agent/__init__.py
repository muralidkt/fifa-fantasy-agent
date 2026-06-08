"""Conversational agent over the fantasy engine (Phase 1).

`tools.py` exposes the deterministic engine (build / advise / score / explain / rules /
overrides) as tool functions; `loop.py` drives an Anthropic tool-calling REPL on top of them.
The ILP still selects every team — the agent only routes, explains, and (behind a human y/N
gate) edits overrides or re-saves the squad. See docs and CLAUDE.md design invariants.
"""
