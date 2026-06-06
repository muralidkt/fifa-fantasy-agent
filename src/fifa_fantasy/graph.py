"""LangGraph state machine that orchestrates the fantasy pipeline.

Flow (a real graph — branches, a cycle, and an approval gate):

  START
    -> ensure_data        refresh cache if stale, build dataset + opponent model
    -> assess_situation   is this round drawn yet? (knockouts fill in later)
        |-- not ready --> END (nothing to decide)
        '-- ready ------> analyze
    -> analyze            deterministic xPts + ILP (pure functions — the math stays pure)
    -> risk_check  <-------------------.
        |-- flagged & llm & !converged + iters left --> research --'   (bounded subloop)
        '-- else --------------------------------------> synthesize
    -> synthesize         merge ILP + LLM adjustments, captain/chip advice, rationale
    -> human_approval     show the team; accept (recommend-only gate)
    -> persist            save the squad (build mode)
    -> END

Design rule: the optimizer/model are PURE functions invoked by nodes. The LLM only enters
via the bounded `research` subloop and only adjusts scores; the solver still enforces all
hard rules. The subloop is skipped entirely unless `--llm` is on (so token cost stays opt-in).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

from .ingest import fifa, ratings, strength
from .model import llm_adjust
from .model.expected_points import load_projection_overrides, project_round
from .model.opponent import OpponentModel
from .optimize import rows_from_projections
from .optimize.squad import optimize_squad
from .optimize.transfers import optimize_transfers
from . import report, rules

console = Console()

START_PROB_FLAG = 0.85   # starters below this start probability are "uncertain" -> vetted by the subloop


class State(TypedDict, total=False):
    # inputs
    round_id: int
    mode: str                 # "build" or "advise"
    config: dict
    use_elo: bool
    use_odds: bool
    use_ratings: bool
    use_llm: bool
    news: str | None
    max_research_iters: int
    auto_approve: bool
    current_ids: list[int]    # advise mode: your current 15
    free_transfers: float | None

    # working state (filled by nodes)
    dataset: Any
    model: Any
    rating_pct: dict
    ratings_weight: float
    overrides: dict
    base_projections: dict
    projections: dict
    squad: Any                # SelectedSquad
    transfer_plan: Any        # TransferPlan | None
    flagged: list[int]
    adjustments: dict
    research_iter: int
    confidence: float
    rationale: str
    approved: bool
    data_ready: bool
    log: Annotated[list[str], operator.add]


# --- nodes -------------------------------------------------------------------
def ensure_data(state: State) -> dict:
    cfg = state["config"]
    cache = cfg["data"]["cache_dir"]
    logs = []
    try:
        fifa.fetch(cache, ttl_hours=cfg["data"]["cache_ttl_hours"])  # refreshes only if stale
    except Exception as exc:  # offline / network — fall back to cache
        logs.append(f"refresh skipped ({exc}); using cached data")
    ds = fifa.load(cache, auto_fetch=False)

    src = "odds" if state.get("use_odds") else "elo" if state.get("use_elo") else cfg["strength"]["source"]
    strengths = strength.team_strength(ds, source=src, cache_dir=cache, stage=cfg["strength"]["odds_stage"])
    model = OpponentModel(
        strengths,
        beta=cfg["strength"]["beta"],
        base_goals=cfg["strength"]["base_goals"],
        home_advantage=cfg["strength"]["home_advantage"],
    )

    rating_pct, ratings_weight = {}, 0.0
    if state.get("use_ratings"):
        raw = ratings.load_overall_ratings(cache)
        if raw:
            rating_pct = ratings.rating_percentiles(raw)
            ratings_weight = cfg["expected_points"]["ratings_weight"]

    ep = cfg["expected_points"]
    overrides = load_projection_overrides(ep.get("overrides_path"))

    logs.append(f"data ready: {len(ds.squads)} teams, strength source={src}"
                + (f", SoFIFA ratings on ({len(rating_pct)})" if rating_pct else ""))
    if overrides:
        logs.append(f"projection overrides loaded ({len(overrides)})")
    return {"dataset": ds, "model": model, "rating_pct": rating_pct,
            "ratings_weight": ratings_weight, "overrides": overrides, "log": logs}


def assess_situation(state: State) -> dict:
    ds, rnd = state["dataset"], state["dataset"].round(state["round_id"])
    expected = _expected_fixture_count(rnd.stage)
    drawn = (
        bool(rnd.fixtures)
        and (expected is None or len(rnd.fixtures) >= expected)
        and all(fx.home_id and fx.away_id for fx in rnd.fixtures)
    )
    msg = (f"Matchday {rnd.id} ({rnd.stage}) is drawn — {len(rnd.fixtures)} fixtures"
           if drawn else
           f"Matchday {rnd.id} ({rnd.stage}) not drawn yet — opponents unknown, nothing to decide")
    return {"data_ready": drawn, "log": [msg]}


def analyze(state: State) -> dict:
    """Deterministic projection + optimisation (re-run after each adjustment)."""
    return _optimize(state, state.get("projections"))


def risk_check(state: State) -> dict:
    """Flag uncertain starters (rotation/availability) for the subloop to vet."""
    squad, proj = state["squad"], state["projections"]
    flagged = {squad.captain_id, squad.vice_id}
    for pid in squad.starter_ids:
        if proj[pid].start_prob < START_PROB_FLAG:
            flagged.add(pid)
    flagged.discard(None)
    return {"flagged": sorted(flagged)}


def research(state: State) -> dict:
    """Bounded agent subloop body: vet flagged players, adjust scores, re-optimise."""
    cfg = state["config"]
    ds, proj = state["dataset"], state["projections"]
    base_proj = state.get("base_projections") or proj
    before = (frozenset(state["squad"].squad_ids), state["squad"].captain_id)

    adjustments = llm_adjust.adjust(
        ds, proj,
        model=cfg["llm"]["model"], top_n=cfg["llm"]["top_n"],
        news=state.get("news"), focus_ids=state.get("flagged"),
    )
    all_adjustments = {**state.get("adjustments", {}), **adjustments}
    new_proj = llm_adjust.apply(base_proj, all_adjustments) if all_adjustments else base_proj
    result = _optimize(state, new_proj)

    after = (frozenset(result["squad"].squad_ids), result["squad"].captain_id)
    converged = before == after
    it = state.get("research_iter", 0) + 1
    note = (f"research pass {it}: {len(adjustments)} adjustments, "
            + ("squad stable" if converged else "squad changed"))
    if not adjustments:
        note = f"research pass {it}: LLM inactive (no key/anthropic) — skipping subloop"
        converged = True
    return {**result, "base_projections": base_proj, "adjustments": all_adjustments,
            "research_iter": it, "confidence": 1.0 if converged else 0.3, "log": [note]}


def synthesize(state: State) -> dict:
    squad, proj, ds = state["squad"], state["projections"], state["dataset"]
    rnd = ds.round(state["round_id"])
    cap = proj[squad.captain_id]
    parts = [
        f"Captain {cap.player.name} ({ds.squads[cap.player.squad_id].name} vs {cap.opponent_name}) "
        f"— best armband score at {cap.components.get('captain_score', cap.xpts):.1f} "
        f"from {cap.xpts:.1f} xPts and {cap.start_prob:.0%} start probability.",
    ]
    uncertain = [
        proj[pid] for pid in squad.starter_ids
        if proj[pid].start_prob < START_PROB_FLAG or proj[pid].components.get("risk_note")
    ]
    if uncertain:
        detail = "; ".join(
            f"{p.player.name} {p.start_prob:.0%}"
            + (f" ({p.components['risk_note']})" if p.components.get("risk_note") else "")
            for p in sorted(uncertain, key=lambda pr: pr.start_prob)[:4]
        )
        parts.append(f"Start-risk watchlist: {detail}.")
    if state.get("transfer_plan") and state["transfer_plan"].in_ids:
        tp = state["transfer_plan"]
        ins = ", ".join(proj[i].player.name for i in tp.in_ids if i in proj)
        parts.append(f"{tp.transfers} transfer(s) for fresh fixtures: in {ins}"
                     + (f" (−{tp.hit_points} pt hit)" if tp.hit_points else ""))
    adj = {k: v for k, v in state.get("adjustments", {}).items() if v[0] != 1.0}
    if adj:
        notable = "; ".join(f"{proj[i].player.name}: {v[1]}" for i, v in list(adj.items())[:4] if i in proj)
        parts.append(f"News adjustments applied — {notable}")
    parts.append(_chip_advice(rnd.stage, rnd.id))
    return {"rationale": " ".join(parts), "log": ["synthesised rationale"]}


def human_approval(state: State) -> dict:
    ds, proj = state["dataset"], state["projections"]
    rnd = ds.round(state["round_id"])
    deadline = rnd.start_date.replace("T", " ")
    if state.get("mode") == "advise" and state.get("transfer_plan"):
        report.render_transfers(console, ds, proj, state["transfer_plan"],
                                round_id=rnd.id, deadline=deadline)
    else:
        report.render_squad(console, ds, proj, state["squad"], round_id=rnd.id, deadline=deadline)
    console.print(Panel.fit(state.get("rationale", ""), title="Rationale", border_style="blue"))

    if state.get("auto_approve"):
        return {"approved": True, "log": ["auto-approved"]}
    import click
    approved = click.confirm("Accept this team?", default=True)
    return {"approved": approved, "log": [f"user {'accepted' if approved else 'rejected'}"]}


def persist(state: State) -> dict:
    if not state.get("approved"):
        return {"log": ["not saved (rejected)"]}
    if state.get("mode") != "build":
        return {"log": ["advise mode — no squad file written"]}
    cfg, ds, squad = state["config"], state["dataset"], state["squad"]
    _save_squad(cfg["data"]["cache_dir"], ds, squad, state["round_id"])
    return {"log": [f"saved squad to {cfg['data']['cache_dir']}/my_squad.yaml"]}


# --- routers -----------------------------------------------------------------
def route_after_assess(state: State) -> str:
    return "analyze" if state.get("data_ready") else "end"


def route_after_risk(state: State) -> str:
    if not state.get("use_llm") or not state.get("flagged"):
        return "synthesize"
    if state.get("research_iter", 0) >= state.get("max_research_iters", 2):
        return "synthesize"
    if state.get("confidence", 0.0) >= 1.0:
        return "synthesize"
    return "research"


# --- helpers -----------------------------------------------------------------
def _optimize(state: State, projections: dict | None) -> dict:
    cfg, ds = state["config"], state["dataset"]
    rnd = ds.round(state["round_id"])
    if projections is None:
        ep = cfg["expected_points"]
        projections = project_round(
            ds, state["round_id"], state["model"],
            form_weight=ep["form_weight"], price_weight=ep["price_weight"],
            one_to_watch_bonus=ep["one_to_watch_bonus"], start_prob_floor=ep["start_prob_floor"],
            rating_pct=state.get("rating_pct") or None, ratings_weight=state.get("ratings_weight", 0.0),
            overrides=state.get("overrides") or None,
            captain_start_weight=ep.get("captain_start_weight", 0.75),
        )
    base_projections = state.get("base_projections") or projections
    rows = rows_from_projections(projections, ds.squads)
    budget = rules.budget_for_stage(rnd.stage)
    cap = rules.nation_cap_for_stage(rnd.stage)
    bench_w = cfg["optimize"]["bench_weight"]

    if state.get("mode") == "advise":
        free_transfers = state.get("free_transfers")
        if free_transfers is None:
            free_transfers = rules.free_transfers_for_round(state["round_id"])
        plan = optimize_transfers(
            rows, state["current_ids"],
            free_transfers=free_transfers,
            budget=budget, nation_cap=cap, bench_weight=bench_w,
            hit_threshold=cfg["optimize"]["hit_threshold"],
        )
        return {"base_projections": base_projections, "projections": projections,
                "squad": plan.squad, "transfer_plan": plan}
    squad = optimize_squad(rows, budget=budget, nation_cap=cap, bench_weight=bench_w)
    return {"base_projections": base_projections, "projections": projections,
            "squad": squad, "transfer_plan": None}


def _chip_advice(stage: str, round_id: int) -> str:
    if stage == "GROUP" and round_id == 1:
        return "Chips: hold Wildcard & boosters — MD1 already has unlimited free transfers."
    if stage == "GROUP":
        return "Chips: consider 12th Man / Maximum Captaincy on a strong double-up fixture round."
    return "Chips: knockouts — Qualification & Mystery boosters unlock at R32; deploy on safe favourites."


def _expected_fixture_count(stage: str) -> int | None:
    return {
        "GROUP": 24,
        "R32": 16,
        "R16": 8,
        "QF": 4,
        "SF": 2,
        "F": 2,
    }.get(stage)


def _save_squad(cache_dir: str, ds, squad, round_id: int) -> None:
    import yaml
    from pathlib import Path

    by_id = {p.id: p for p in ds.players}
    data = {
        "round": round_id,
        "player_ids": squad.squad_ids,
        "starters": squad.starter_ids,
        "bench": squad.bench_ids,
        "captain": squad.captain_id,
        "vice": squad.vice_id,
        "names": {pid: by_id[pid].name for pid in squad.squad_ids if pid in by_id},
    }
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "my_squad.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


# --- graph assembly ----------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("ensure_data", ensure_data)
    g.add_node("assess_situation", assess_situation)
    g.add_node("analyze", analyze)
    g.add_node("risk_check", risk_check)
    g.add_node("research", research)
    g.add_node("synthesize", synthesize)
    g.add_node("human_approval", human_approval)
    g.add_node("persist", persist)

    g.add_edge(START, "ensure_data")
    g.add_edge("ensure_data", "assess_situation")
    g.add_conditional_edges("assess_situation", route_after_assess,
                            {"analyze": "analyze", "end": END})
    g.add_edge("analyze", "risk_check")
    g.add_conditional_edges("risk_check", route_after_risk,
                            {"research": "research", "synthesize": "synthesize"})
    g.add_edge("research", "risk_check")          # the bounded subloop
    g.add_edge("synthesize", "human_approval")
    g.add_edge("human_approval", "persist")
    g.add_edge("persist", END)
    return g.compile()


def run(round_id: int, *, mode: str, config: dict, use_elo: bool = False, use_odds: bool = False,
        use_ratings: bool = False, use_llm: bool = False, news: str | None = None,
        max_research_iters: int = 2, auto_approve: bool = False,
        current_ids: list[int] | None = None, free_transfers: float | None = None) -> State:
    app = build_graph()
    initial: State = {
        "round_id": round_id, "mode": mode, "config": config,
        "use_elo": use_elo, "use_odds": use_odds, "use_ratings": use_ratings,
        "use_llm": use_llm, "news": news,
        "max_research_iters": max_research_iters, "auto_approve": auto_approve,
        "current_ids": current_ids or [], "free_transfers": free_transfers,
        "research_iter": 0, "confidence": 0.0,
        "adjustments": {}, "log": [],
    }
    return app.invoke(initial, config={"recursion_limit": 50})
