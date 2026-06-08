"""Interactive REPL that lets you talk to the fantasy engine through Claude.

A bounded tool-calling loop: your message → Claude (with the tools in ``tools.py``) → any
tool calls run against the deterministic engine → results back → Claude replies. The ILP
still picks every team; Claude can only re-run it with lock/ban constraints or edit the
soft overrides. File writes go through ``_confirm`` — a y/N prompt owned here in Python, so
the model can never bypass your approval. Degrades to a clear message if ``anthropic`` is
missing or ``ANTHROPIC_API_KEY`` is unset (mirrors the ``--llm`` path).
"""
from __future__ import annotations

import json
import os
from typing import Callable

from rich.panel import Panel

from ..engine import console
from .tools import WRITE_TOOLS, AgentTools

MAX_TOOL_ITERS = 8          # cap tool round-trips per user turn (token guard)
MAX_TOKENS = 2048

_SYSTEM = """You are a World Cup Fantasy assistant operating in a terminal, on top of a \
deterministic optimisation engine. Hard rules you must respect:

- You NEVER pick or invent a team yourself. An Integer Linear Program selects the squad and \
enforces every rule (budget, 2/5/5/3 shape, per-nation cap, formation). To change a result, \
call build_squad / advise_transfers with `lock` (force players in) or `ban` (force out), or \
edit soft projection signals via propose_override — then let the solver re-solve.
- This is recommend-only: there is no team-submission API. You tell the user what to do; they \
enter it in the FIFA app.
- For rules questions, call get_rules and faithfully relay the `verify_in_app` items as \
"confirm on play.fifa.com" rather than stating disputed values as fact.
- Writes (propose_override, save_squad) trigger a y/N confirmation the user must approve; \
explain the change clearly before proposing it.

Be concise. Use the tools rather than guessing numbers. When you show a squad or transfer, \
summarise the key picks, captain, cost, and expected points."""


def _tool_schemas() -> list[dict]:
    pos_mode = ["balanced", "safe", "upside", "differential", "template"]
    players = {"type": "array", "items": {"type": "string"},
               "description": "player names or numeric ids"}
    return [
        {"name": "show_squad", "description": "Show the currently saved squad (data/my_squad.yaml).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "build_squad",
         "description": "Run the squad ILP for a round and return the optimal 15 + XI + captain. Does NOT save.",
         "input_schema": {"type": "object", "properties": {
             "round": {"type": "integer", "description": "matchday 1-8 (defaults to session round)"},
             "horizon": {"type": "integer", "description": "value this + N-1 future rounds (group stage)"},
             "odds": {"type": "boolean"}, "ratings": {"type": "boolean"},
             "mode": {"type": "string", "enum": pos_mode},
             "lock": players, "ban": players}}},
        {"name": "advise_transfers",
         "description": "Recommend transfers for a round against the saved squad. Does NOT save.",
         "input_schema": {"type": "object", "properties": {
             "round": {"type": "integer"}, "free": {"type": "number", "description": "override free-transfer count"},
             "horizon": {"type": "integer"}, "odds": {"type": "boolean"}, "ratings": {"type": "boolean"},
             "mode": {"type": "string", "enum": pos_mode}, "lock": players, "ban": players}}},
        {"name": "score_round",
         "description": "Score the saved squad for a finished round using FIFA's official points.",
         "input_schema": {"type": "object", "properties": {
             "round": {"type": "integer"}, "hits": {"type": "integer", "description": "points docked for paid transfers"}}}},
        {"name": "explain_player",
         "description": "Explain a player's projection for a round: opponent, xPts, start prob, component breakdown.",
         "input_schema": {"type": "object", "properties": {
             "player": {"type": "string"}, "round": {"type": "integer"},
             "odds": {"type": "boolean"}, "ratings": {"type": "boolean"}},
             "required": ["player"]}},
        {"name": "explain_optimizer",
         "description": "Describe how the ILP decides: objective, hard constraints, and the levers you can use.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_rules",
         "description": "Return the encoded 2026 rules & scoring, flagging disputed (verify-in-app) values.",
         "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}}},
        {"name": "list_overrides",
         "description": "Show the current manual projection overrides (data/overrides.yaml).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "propose_override",
         "description": ("Propose a manual projection override for a player (e.g. rotation/injury "
                         "risk). Requires user y/N confirmation before writing. Allowed fields: "
                         "start_prob, quality, goal_share, assist_share (0-1); penalty_xg, "
                         "xpts_multiplier (>=0); xpts_delta (any); captain_avoid (bool); notes (str)."),
         "input_schema": {"type": "object", "properties": {
             "player": {"type": "string"},
             "fields": {"type": "object", "description": "override fields to set"},
             "round": {"type": "integer", "description": "scope to one round only (optional)"}},
             "required": ["player", "fields"]}},
        {"name": "save_squad",
         "description": "Save the last build_squad/advise_transfers result to data/my_squad.yaml. Requires y/N confirmation.",
         "input_schema": {"type": "object", "properties": {
             "source": {"type": "string", "enum": ["build", "advise"]}}}},
    ]


def _dispatch(tools: AgentTools, name: str, args: dict, confirm: Callable[[str], bool]) -> dict:
    try:
        method = getattr(tools, name, None)
        if method is None:
            return {"error": f"unknown tool: {name}"}
        if name in WRITE_TOOLS:
            return method(confirm=confirm, **args)
        return method(**args)
    except Exception as exc:  # surface the error to the model so it can self-correct
        return {"error": f"{type(exc).__name__}: {exc}"}


def _make_confirm() -> Callable[[str], bool]:
    def confirm(diff: str) -> bool:
        console.print(Panel(diff, title="Apply this change?", border_style="yellow"))
        try:
            ans = input("Apply? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")
    return confirm


def run(cfg: dict, *, model: str | None = None, default_round: int | None = None,
        odds: bool = True, ratings: bool = True, mode: str | None = None,
        api_key: str | None = None) -> None:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[yellow]`fantasy chat` needs an Anthropic API key.[/yellow] "
                      r"Set ANTHROPIC_API_KEY and install extras: pip install -e '.\[llm]'.")
        return
    try:
        import anthropic
    except ImportError:
        console.print("[yellow]`anthropic` is not installed.[/yellow] "
                      r"Install it: pip install -e '.\[llm]'.")
        return

    model = model or cfg["llm"]["model"]
    tools = AgentTools(cfg, default_round=default_round, odds=odds, ratings=ratings, mode=mode)
    schemas = _tool_schemas()
    confirm = _make_confirm()
    client = anthropic.Anthropic(api_key=api_key)

    rnd_txt = f" · default round {default_round}" if default_round else ""
    console.print(Panel(
        "[bold]FIFA Fantasy chat[/bold] — ask about your team, players, rules, or transfers.\n"
        "The ILP still picks the team; I can lock/ban players, edit overrides (with your y/N), "
        "and explain everything.\n"
        f"[dim]model: {model}{rnd_txt} · type 'exit' or Ctrl-D to quit[/dim]",
        border_style="cyan"))

    messages: list[dict] = []
    while True:
        try:
            user = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            console.print("[dim]bye[/dim]")
            return

        messages.append({"role": "user", "content": user})
        for _ in range(MAX_TOOL_ITERS):
            resp = client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=_SYSTEM,
                tools=schemas, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})

            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            if text.strip():
                console.print(f"\n[bold cyan]agent ›[/bold cyan] {text.strip()}")

            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                break

            results = []
            for tu in tool_uses:
                console.print(f"[dim]· {tu.name}({json.dumps(tu.input, ensure_ascii=False)})[/dim]")
                result = _dispatch(tools, tu.name, dict(tu.input), confirm)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(result, ensure_ascii=False, default=str)})
            messages.append({"role": "user", "content": results})
        else:
            console.print("[yellow](reached the tool-call limit for this turn)[/yellow]")
