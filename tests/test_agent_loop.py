"""Agent REPL tests with a stubbed Anthropic client — no network, no real model.

Covers the parts that are ours: the y/N write gate, tool dispatch/error handling, and that the
conversation loop actually routes a tool_use block to the engine tools and stops cleanly.
"""
import sys
import types

import pytest

from fifa_fantasy.agent import loop
from fifa_fantasy.agent.loop import _dispatch, _make_confirm


# --- write gate -------------------------------------------------------------
@pytest.mark.parametrize("answer,expected", [("y", True), ("yes", True), ("n", False), ("", False)])
def test_confirm_gate(monkeypatch, answer, expected):
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    assert _make_confirm()("some diff") is expected


# --- dispatch ---------------------------------------------------------------
class _FakeTools:
    def __init__(self):
        self.confirm_seen = None

    def show_squad(self):
        return {"ok": True}

    def save_squad(self, source="build", confirm=None):
        self.confirm_seen = confirm           # write tools must receive the gate
        return {"applied": bool(confirm("diff"))}

    def boom(self):
        raise ValueError("kaboom")


def test_dispatch_routes_confirm_to_write_tools():
    tools = _FakeTools()
    out = _dispatch(tools, "save_squad", {}, confirm=lambda d: True)
    assert out == {"applied": True}
    assert tools.confirm_seen is not None


def test_dispatch_read_tool_needs_no_confirm():
    assert _dispatch(_FakeTools(), "show_squad", {}, confirm=lambda d: False) == {"ok": True}


def test_dispatch_unknown_and_errors_are_returned_not_raised():
    assert "error" in _dispatch(_FakeTools(), "nope", {}, confirm=lambda d: True)
    assert "error" in _dispatch(_FakeTools(), "boom", {}, confirm=lambda d: True)


# --- full loop with a scripted fake client ----------------------------------
def _text(t):
    return types.SimpleNamespace(type="text", text=t)


def _tool_use(name, inp, id="t1"):
    return types.SimpleNamespace(type="tool_use", name=name, input=inp, id=id)


class _FakeMessages:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(content=self._scripted.pop(0))


class _FakeClient:
    def __init__(self, scripted, **_):
        self.messages = _FakeMessages(scripted)


def test_loop_dispatches_tool_then_replies(monkeypatch):
    # Stub the `anthropic` import inside loop.run().
    calls = {"get_rules": 0}

    class StubTools:
        def __init__(self, *a, **k):
            pass

        def get_rules(self, topic=None):
            calls["get_rules"] += 1
            return {"captain_multiplier": 2}

    # Two model responses: first calls get_rules, second is a plain-text reply.
    scripted = [
        [_tool_use("get_rules", {})],
        [_text("Captain scores double.")],
    ]
    fake_anthropic = types.SimpleNamespace(
        Anthropic=lambda **kw: _FakeClient(scripted, **kw))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setattr(loop, "AgentTools", StubTools)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    prompts = iter(["what are the captain rules?", "exit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(prompts))

    cfg = {"llm": {"model": "claude-test"}, "data": {"cache_dir": "data"},
           "optimize": {"mode": "balanced"}}
    loop.run(cfg)

    assert calls["get_rules"] == 1  # the tool was actually dispatched


def test_loop_exits_clean_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"llm": {"model": "x"}, "data": {"cache_dir": "data"}, "optimize": {"mode": "balanced"}}
    loop.run(cfg)  # should print guidance and return, not raise
