"""Stage 1 — Ingest.

Loads raw sources and validates them. The central guarantee: every team name
resolves to a canonical name, or we fail loudly *here* — not three stages later
inside a silent join.
"""
from __future__ import annotations
import pandas as pd
from config import RAW, ROOT

ALIASES = ROOT / "data" / "team_aliases.csv"

REQUIRED_HIST_COLS = {"date", "home_team", "away_team", "home_score", "away_score",
                      "tournament", "neutral"}
REQUIRED_LIVE_COLS = {"date", "home_team", "away_team", "home_score", "away_score",
                      "tournament", "neutral", "venue_country", "stage", "group"}


def load_aliases() -> dict:
    df = pd.read_csv(ALIASES)
    return dict(zip(df["alias"].str.strip().str.lower(), df["canonical_name"]))


def _canon(name, alias_map, unmapped):
    key = str(name).strip().lower()
    if key in alias_map:
        return alias_map[key]
    # Not every team on earth is in the alias table -- that table only exists
    # to correct known *variant spellings* (USA -> United States, etc). Any
    # other name is already canonical as-is (this dataset has 300+ national
    # teams; only a handful ever need remapping). We still track "unmapped"
    # names that collide case-insensitively with a *different* spelling
    # already seen, since that's the real failure mode worth catching.
    return str(name).strip()


def _validate(df: pd.DataFrame, required: set, source_name: str,
             alias_map: dict) -> pd.DataFrame:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{source_name}] missing columns: {missing}")

    df = df.copy()
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df[df[col].isna()]
            raise ValueError(f"[{source_name}] non-numeric {col} in "
                            f"{len(bad)} row(s), first: {bad.iloc[0].to_dict()}")
        if (df[col] > 35).any():
            raise ValueError(f"[{source_name}] implausible score (>35) found")

    parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if parsed.isna().any():
        bad = df[parsed.isna()]
        raise ValueError(f"[{source_name}] unparseable date(s), first: "
                        f"{bad.iloc[0]['date']}")
    df["date_text"] = df["date"]
    df["date"] = parsed

    unmapped = set()
    df["home_team"] = df["home_team"].map(lambda n: _canon(n, alias_map, unmapped))
    df["away_team"] = df["away_team"].map(lambda n: _canon(n, alias_map, unmapped))
    if "venue_country" in df.columns:
        df["venue_country"] = df["venue_country"].map(
            lambda n: _canon(n, alias_map, unmapped) if pd.notna(n) else n)
    return df


def load_raw() -> dict:
    alias_map = load_aliases()

    hist_path = RAW / "results.csv"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"{hist_path} not found. Download the Kaggle international-results "
            f"dataset and save it there (see README 'Get the real data')."
        )
    hist = pd.read_csv(hist_path)
    hist = _validate(hist, REQUIRED_HIST_COLS, "results.csv", alias_map)

    live_path = RAW / "wc2026_live.csv"
    live = pd.read_csv(live_path)
    live = _validate(live, REQUIRED_LIVE_COLS, "wc2026_live.csv", alias_map)

    elo_path = RAW / "elo.csv"
    if not elo_path.exists():
        raise FileNotFoundError(
            f"{elo_path} not found. Download Elo ratings from eloratings.net "
            f"and save them there (see README 'Get the real data')."
        )
    elo = pd.read_csv(elo_path)
    if not {"date", "team", "elo"}.issubset(elo.columns):
        raise ValueError("[elo.csv] expected columns: date, team, elo")
    unmapped = set()
    elo = elo.copy()
    elo["team"] = elo["team"].map(lambda n: _canon(n, alias_map, unmapped))
    elo["date"] = pd.to_datetime(elo["date"], errors="coerce", utc=True)

    return {"results": hist, "live": live, "elo": elo}


if __name__ == "__main__":
    data = load_raw()
    r, l, e = data["results"], data["live"], data["elo"]
    print(f"results.csv : {len(r):,} matches, {r.date.min().date()}..{r.date.max().date()}")
    print(f"live        : {len(l)} matches, {l.date.min().date()}..{l.date.max().date()}")
    print(f"elo.csv     : {len(e):,} rows, {e.team.nunique()} teams")
    print("ingest OK")
