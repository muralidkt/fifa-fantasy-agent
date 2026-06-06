"""Player quality prior from SoFIFA / EA FC overall ratings (CSV-driven).

Reads `data/sofifa.csv` (columns: `name`, `overall`, optional `team`) and exposes overall
ratings keyed by a normalised name. These are blended into the expected-points quality prior
(see `model/expected_points.py`), so lower-profile / thin-sample players are rated on scouted
ability rather than only their FIFA price — improving the long tail.

Populate the CSV with `fantasy fetch-ratings` (uses the optional `soccerdata` SoFIFA reader,
`pip install -e '.[ratings]'`) or a Kaggle/GitHub EA FC export. SoFIFA's site blocks direct
scraping (HTTP 403), which is why this is a local-file interface.
"""
from __future__ import annotations

import csv
import unicodedata
from bisect import bisect_left
from pathlib import Path

RATINGS_FILE = "sofifa.csv"


def normalize_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "")
    n = "".join(c for c in n if not unicodedata.combining(c))
    return "".join(c for c in n.lower() if c.isalnum())


def load_overall_ratings(cache_dir: str | Path) -> dict[str, float]:
    """normalised name -> overall rating (e.g. 88). Empty dict if no CSV present."""
    path = Path(cache_dir) / RATINGS_FILE
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            name, overall = row.get("name"), row.get("overall")
            if not name or overall in (None, ""):
                continue
            try:
                out[normalize_name(name)] = float(overall)
            except ValueError:
                continue
    return out


def rating_percentiles(ratings: dict[str, float]) -> dict[str, float]:
    """normalised name -> percentile in [0, 1] across the provided ratings."""
    if not ratings:
        return {}
    values = sorted(ratings.values())
    n = len(values)
    return {
        name: (0.5 if n <= 1 else bisect_left(values, v) / (n - 1))
        for name, v in ratings.items()
    }


def fetch_via_soccerdata(cache_dir: str | Path, *, fc_version: str = "latest") -> int:
    """Best-effort populate of sofifa.csv via the optional soccerdata SoFIFA reader.

    Returns the number of player ratings written. Raises a clear error if soccerdata is not
    installed. The exact upstream API varies by soccerdata version, so this is wrapped
    defensively; if it fails, populate sofifa.csv manually (name,overall[,team]).
    """
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise RuntimeError(
            "soccerdata not installed — run `pip install -e '.[ratings]'` "
            "or populate data/sofifa.csv manually (columns: name,overall)."
        ) from exc

    reader = sd.SoFIFA(versions=fc_version)
    df = reader.read_player_ratings()  # MultiIndex/columns vary by version
    rows = _dataframe_to_rows(df)
    path = Path(cache_dir) / RATINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "overall"])
        writer.writerows(rows)
    return len(rows)


def _dataframe_to_rows(df) -> list[tuple[str, float]]:
    """Coerce a soccerdata ratings DataFrame into (name, overall) rows, defensively."""
    df = df.reset_index()
    cols = {str(c).lower(): c for c in df.columns}
    name_col = next((cols[c] for c in ("player", "name") if c in cols), None)
    ovr_col = next((cols[c] for c in ("overall", "ovr", "overall rating") if c in cols), None)
    if name_col is None or ovr_col is None:
        raise RuntimeError(f"unexpected soccerdata columns: {list(df.columns)}")
    rows: list[tuple[str, float]] = []
    for _, r in df.iterrows():
        try:
            rows.append((str(r[name_col]), float(r[ovr_col])))
        except (ValueError, TypeError):
            continue
    return rows
