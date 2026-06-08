"""Agent tool tests: pure validation/rules helpers, plus the engine-backed tools on the
cached dataset. No LLM is involved — the tools are deterministic dicts."""
import copy
import shutil
from pathlib import Path

import pytest

from fifa_fantasy.engine import DEFAULTS
from fifa_fantasy.agent.tools import (
    AgentTools,
    _validate_override_fields,
    _verify_in_app_fields,
)

DATA = Path(__file__).resolve().parents[1] / "data"
needs_cache = pytest.mark.skipif(
    not (DATA / "players.json").exists(), reason="run `fantasy refresh` to populate data cache")


@pytest.fixture
def cfg(tmp_path):
    for f in ("players.json", "rounds.json", "squads.json"):
        shutil.copy(DATA / f, tmp_path / f)
    c = copy.deepcopy(DEFAULTS)
    c["data"]["cache_dir"] = str(tmp_path)
    c["data"]["cache_ttl_hours"] = 10_000
    c["expected_points"]["overrides_path"] = str(tmp_path / "overrides.yaml")
    c["expected_points"]["fixture_odds_path"] = str(tmp_path / "fixture_odds.yaml")
    return c


# --- pure helpers -----------------------------------------------------------
def test_validate_override_fields_accepts_and_rejects():
    clean, errors = _validate_override_fields(
        {"start_prob": 0.5, "captain_avoid": True, "notes": "rest risk"})
    assert clean == {"start_prob": 0.5, "captain_avoid": True, "notes": "rest risk"}
    assert errors == []

    clean, errors = _validate_override_fields({"start_prob": 1.7, "bogus": 1})
    assert "start_prob" not in clean
    assert any("out of range" in e for e in errors)
    assert any("unknown field" in e for e in errors)


def test_get_rules_flags_verify_in_app(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    rules_out = tools.get_rules()
    verify = rules_out["verify_in_app"]
    assert "penalty_saved" in verify and "red_card" in verify
    # the same markers come straight from rules.py
    assert set(verify) == set(_verify_in_app_fields())


def test_explain_optimizer_lists_hard_constraints(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    out = tools.explain_optimizer()
    assert out["hard_constraints"]["squad_size"] == 15
    assert out["hard_constraints"]["composition"] == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


# --- engine-backed tools ----------------------------------------------------
@needs_cache
def test_build_squad_returns_legal_15(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    out = tools.build_squad(round=1)
    assert len(out["players"]) == 15
    assert sum(1 for p in out["players"] if p["role"] == "XI") == 11
    assert out["captain"] and out["vice"] and out["captain"] != out["vice"]


@needs_cache
def test_explain_player_resolves_name(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    squad = tools.build_squad(round=1)
    name = squad["players"][0]["name"]
    out = tools.explain_player(player=name, round=1)
    assert out["name"] == name
    assert "components" in out and "xpts" in out


@needs_cache
def test_propose_override_gated_by_confirm(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    name = tools.build_squad(round=1)["players"][0]["name"]
    path = Path(cfg["expected_points"]["overrides_path"])

    declined = tools.propose_override(player=name, fields={"start_prob": 0.4},
                                      confirm=lambda diff: False)
    assert declined["applied"] is False
    assert not path.exists()  # nothing written when declined

    applied = tools.propose_override(player=name, fields={"start_prob": 0.4},
                                     confirm=lambda diff: True)
    assert applied["applied"] is True
    assert path.exists()
    assert name in path.read_text()


@needs_cache
def test_save_squad_gated_and_persists(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    tools.build_squad(round=1)
    saved_path = Path(cfg["data"]["cache_dir"]) / "my_squad.yaml"

    assert tools.save_squad(source="build", confirm=lambda d: False)["applied"] is False
    assert not saved_path.exists()

    out = tools.save_squad(source="build", confirm=lambda d: True)
    assert out["applied"] is True
    assert saved_path.exists()


@needs_cache
def test_build_squad_lock_forces_player(cfg):
    tools = AgentTools(cfg, default_round=1, odds=False, ratings=False)
    base = tools.build_squad(round=1)
    in_ids = {p["id"] for p in base["players"]}
    # find a player not selected to lock in
    ds, _, _ = tools._pipeline(1, False, False)
    left_out = next(p for p in ds.players if p.available and p.id not in in_ids)
    locked = tools.build_squad(round=1, lock=[left_out.name])
    assert left_out.id in {p["id"] for p in locked["players"]}
