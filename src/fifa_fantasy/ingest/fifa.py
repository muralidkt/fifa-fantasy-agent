"""Ingest the FIFA World Cup 2026 Fantasy public JSON (read-only, no auth).

Three live endpoints under https://play.fifa.com/json/fantasy/ :
  - squads.json   -> 48 national teams {id, name, group, abbr, isEliminated}
  - rounds.json   -> 8 rounds; each round's `tournaments[]` are that round's fixtures
  - players.json  -> the full player pool with price, position, ownership, form

Note: there is NO public WRITE API, so this module only READS. The agent recommends a
team; you enter it in the app. (`squads_fifa.json` exists but is stale 2022 data — ignore it.)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE_URL = "https://play.fifa.com/json/fantasy"
FILES = ("squads.json", "rounds.json", "players.json")
_HEADERS = {"User-Agent": "Mozilla/5.0 (fifa-fantasy-agent; personal use)"}


# --- domain types ------------------------------------------------------------
@dataclass(frozen=True)
class Squad:
    id: int
    name: str
    abbr: str
    group: str
    eliminated: bool


@dataclass(frozen=True)
class Fixture:
    id: int
    round_id: int
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    date: str
    status: str


@dataclass(frozen=True)
class Round:
    id: int
    stage: str
    start_date: str       # ISO; this is the matchday DEADLINE (first kickoff of the round)
    status: str
    fixtures: list[Fixture] = field(default_factory=list)


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    squad_id: int
    position: str          # GK / DEF / MID / FWD
    price: float
    status: str            # "playing" (in a 26-man squad) or "transferred" (dropped)
    ownership: float       # percentSelected
    form: float            # FIFA in-game form (0 pre-tournament)
    avg_points: float
    total_points: float    # FIFA official points across all played rounds
    one_to_watch: bool
    last_round_points: float = 0.0
    round_points: dict[int, float] = field(default_factory=dict)  # round_id -> official points

    @property
    def available(self) -> bool:
        return self.status == "playing"

    def points_for_round(self, round_id: int) -> float:
        """FIFA's official fantasy points this player scored in the given round (0 if none/unplayed)."""
        return self.round_points.get(round_id, 0.0)


@dataclass
class Dataset:
    squads: dict[int, Squad]
    rounds: list[Round]
    players: list[Player]

    def round(self, round_id: int) -> Round:
        for r in self.rounds:
            if r.id == round_id:
                return r
        raise KeyError(f"round {round_id} not found")

    def opponent_map(self, round_id: int) -> dict[int, "Opponent"]:
        """squad_id -> who they face in this round (and whether at home)."""
        out: dict[int, Opponent] = {}
        for fx in self.round(round_id).fixtures:
            out[fx.home_id] = Opponent(fx.away_id, is_home=True, fixture=fx)
            out[fx.away_id] = Opponent(fx.home_id, is_home=False, fixture=fx)
        return out

    def players_by_squad(self) -> dict[int, list[Player]]:
        out: dict[int, list[Player]] = {}
        for p in self.players:
            out.setdefault(p.squad_id, []).append(p)
        return out


@dataclass(frozen=True)
class Opponent:
    squad_id: int
    is_home: bool
    fixture: Fixture


# --- fetch / cache -----------------------------------------------------------
def fetch(cache_dir: str | Path, *, force: bool = False, ttl_hours: float = 6.0) -> None:
    """Download the three JSON files into the cache dir (skip if fresh unless `force`)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = cache / fname
        if not force and dest.exists() and _age_hours(dest) < ttl_hours:
            continue
        resp = requests.get(f"{BASE_URL}/{fname}", headers=_HEADERS, timeout=45)
        resp.raise_for_status()
        # Validate it parses before overwriting the cache.
        json.loads(resp.content)
        dest.write_bytes(resp.content)


def _age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600.0


def _read(cache: Path, fname: str):
    path = cache / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `fantasy refresh` first to download FIFA data."
        )
    return json.loads(path.read_text())


# --- parse -------------------------------------------------------------------
def load(cache_dir: str | Path, *, auto_fetch: bool = True, ttl_hours: float = 6.0) -> Dataset:
    """Load (and if needed download) the dataset from the cache dir."""
    cache = Path(cache_dir)
    if auto_fetch and not all((cache / f).exists() for f in FILES):
        fetch(cache, ttl_hours=ttl_hours)

    squads = {
        s["id"]: Squad(
            id=s["id"],
            name=s["name"],
            abbr=s.get("abbr", ""),
            group=s.get("group", ""),
            eliminated=bool(s.get("isEliminated", False)),
        )
        for s in _read(cache, "squads.json")
    }

    rounds: list[Round] = []
    for r in _read(cache, "rounds.json"):
        fixtures = [
            Fixture(
                id=t["id"],
                round_id=r["id"],
                home_id=t["homeSquadId"],
                away_id=t["awaySquadId"],
                home_name=t.get("homeSquadName", ""),
                away_name=t.get("awaySquadName", ""),
                date=t.get("date", ""),
                status=t.get("status", ""),
            )
            for t in r.get("tournaments", [])
        ]
        rounds.append(
            Round(
                id=r["id"],
                stage=r["stage"],
                start_date=r.get("startDate", ""),
                status=r.get("status", ""),
                fixtures=fixtures,
            )
        )
    rounds.sort(key=lambda r: r.id)

    players = [_parse_player(p) for p in _read(cache, "players.json")]
    return Dataset(squads=squads, rounds=rounds, players=players)


def _parse_player(p: dict) -> Player:
    name = p.get("knownName") or " ".join(
        x for x in (p.get("firstName"), p.get("lastName")) if x
    ).strip()
    stats = p.get("stats") or {}
    round_points = _parse_round_points(stats.get("roundPoints"))
    return Player(
        id=p["id"],
        name=name or f"player#{p['id']}",
        squad_id=p["squadId"],
        position=p["position"],
        price=float(p["price"]),
        status=p.get("status", "playing"),
        ownership=float(p.get("percentSelected") or 0.0),
        form=float(stats.get("form") or 0.0),
        avg_points=float(stats.get("avgPoints") or 0.0),
        total_points=float(stats.get("totalPoints") or 0.0),
        one_to_watch=bool(p.get("oneToWatch", False)),
        last_round_points=float(stats.get("lastRoundPoints") or 0.0),
        round_points=round_points,
    )


def _parse_round_points(raw) -> dict[int, float]:
    """Defensively parse FIFA's `roundPoints` into {round_id: points}.

    Empty pre-tournament; once live it is expected to be a list of per-round entries (objects
    or numbers) or a {round: points} mapping — all handled here so a shape change won't break.
    """
    out: dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = float(v)
            except (ValueError, TypeError):
                continue
    elif isinstance(raw, list):
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                rid = item.get("roundId") or item.get("round") or item.get("id") or i
                pts = item.get("points", item.get("score", item.get("value")))
            else:
                rid, pts = i, item
            try:
                out[int(rid)] = float(pts)
            except (ValueError, TypeError):
                continue
    return out
