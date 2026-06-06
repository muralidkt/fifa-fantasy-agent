"""LangGraph state-machine tests (offline, deterministic). Requires the data cache."""
import copy
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from fifa_fantasy.cli import DEFAULTS
from fifa_fantasy.graph import _optimize, ensure_data, research, run
from fifa_fantasy.rules import SQUAD_COMPOSITION

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "players.json").exists(), reason="run `fantasy refresh` to populate data cache"
)


@pytest.fixture
def cfg(tmp_path):
    """Config pointing at a throwaway cache copy (so persist won't touch real my_squad.yaml)."""
    for f in ("players.json", "rounds.json", "squads.json"):
        shutil.copy(DATA / f, tmp_path / f)
    c = copy.deepcopy(DEFAULTS)
    c["data"]["cache_dir"] = str(tmp_path)
    c["data"]["cache_ttl_hours"] = 10_000  # never auto-refetch during the test
    return c


def test_build_path_produces_legal_squad(cfg):
    state = run(1, mode="build", config=cfg, use_llm=False, auto_approve=True)
    assert state["data_ready"] is True
    squad = state["squad"]
    assert len(squad.squad_ids) == 15
    by_pos = {}
    for pid in squad.squad_ids:
        pos = state["projections"][pid].player.position
        by_pos[pos] = by_pos.get(pos, 0) + 1
    assert by_pos == SQUAD_COMPOSITION
    assert state["approved"] is True
    assert (Path(cfg["data"]["cache_dir"]) / "my_squad.yaml").exists()  # persisted


def test_research_subloop_is_bounded_without_api_key(cfg, monkeypatch):
    """With --llm on but no API key, the subloop must run once and converge (no infinite loop)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = run(1, mode="build", config=cfg, use_llm=True, max_research_iters=3, auto_approve=True)
    assert state["research_iter"] <= 3
    assert state["confidence"] >= 1.0  # converged (LLM inactive -> no change)
    assert len(state["squad"].squad_ids) == 15


def test_undrawn_knockout_round_short_circuits(cfg):
    """R32 has no fixtures yet -> graph reports not-ready and makes no pick."""
    state = run(4, mode="build", config=cfg, use_llm=False, auto_approve=True)
    assert state["data_ready"] is False
    assert "squad" not in state  # never reached analyze


def test_partially_drawn_knockout_round_short_circuits(cfg):
    """A partially populated knockout round is still not enough context to optimize."""
    rounds_path = Path(cfg["data"]["cache_dir"]) / "rounds.json"
    rounds = json.loads(rounds_path.read_text())
    group_fixture = rounds[0]["tournaments"][0]
    for rnd in rounds:
        if rnd["id"] == 4:
            rnd["tournaments"] = [{**group_fixture, "roundId": 4}]
    rounds_path.write_text(json.dumps(rounds))

    state = run(4, mode="build", config=cfg, use_llm=False, auto_approve=True)
    assert state["data_ready"] is False
    assert "squad" not in state


def test_research_adjustments_are_applied_from_base_projection(cfg, monkeypatch):
    """Repeated LLM passes should not compound the same multiplier onto itself."""
    state = {
        "round_id": 1,
        "mode": "build",
        "config": cfg,
        "use_elo": False,
        "use_odds": False,
        "use_ratings": False,
        "use_llm": True,
        "news": None,
        "max_research_iters": 2,
        "auto_approve": True,
        "current_ids": [],
        "research_iter": 0,
        "confidence": 0.0,
        "adjustments": {},
        "log": [],
    }
    state.update(ensure_data(state))
    state.update(_optimize(state, None))
    flagged_id = state["squad"].captain_id
    state["flagged"] = [flagged_id]

    def fake_adjust(*args, **kwargs):
        return {flagged_id: (0.5, "rotation risk")}

    monkeypatch.setattr("fifa_fantasy.graph.llm_adjust.adjust", fake_adjust)
    first = research(state)
    second_state = {**state, **first, "flagged": [flagged_id]}
    second = research(second_state)

    base_xpts = state["base_projections"][flagged_id].xpts
    assert second["projections"][flagged_id].xpts == pytest.approx(round(base_xpts * 0.5, 3))


def test_advise_graph_honors_free_transfer_override(cfg, monkeypatch):
    build_state = run(1, mode="build", config=cfg, use_llm=False, auto_approve=True)
    captured = {}

    from fifa_fantasy.optimize.transfers import optimize_transfers as real_optimize_transfers

    def wrapped_optimize_transfers(*args, **kwargs):
        captured["free_transfers"] = kwargs["free_transfers"]
        return real_optimize_transfers(*args, **kwargs)

    monkeypatch.setattr("fifa_fantasy.graph.optimize_transfers", wrapped_optimize_transfers)
    run(
        2,
        mode="advise",
        config=cfg,
        use_llm=False,
        auto_approve=True,
        current_ids=build_state["squad"].squad_ids,
        free_transfers=1,
    )
    assert captured["free_transfers"] == 1


def test_synthesize_rationale_mentions_captain(cfg):
    state = run(1, mode="build", config=cfg, use_llm=False, auto_approve=True)
    assert "Captain" in state["rationale"]
