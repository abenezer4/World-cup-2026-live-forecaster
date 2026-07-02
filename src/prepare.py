"""Stage 2 — Prepare.

Merge historical + live results into one clean, chronological match table.
Resolve host-vs-neutral by venue country (not a trusted flag), guard against
future-dated rows leaking in as results, and deduplicate deterministically.

Output: data/processed/matches.csv
"""
from __future__ import annotations
import pandas as pd
from config import PROCESSED, TODAY, HOST_COUNTRIES
from ingest import load_raw


def _is_host_match(row) -> bool:
    """True only when a host nation plays a WC match in its own country."""
    if row.get("tournament") != "World Cup":
        return not row.get("neutral", True)
    return (row.get("home_team") in HOST_COUNTRIES and
            row.get("venue_country") == row.get("home_team"))


def build_matches(today: pd.Timestamp = TODAY) -> pd.DataFrame:
    data = load_raw()

    hist = data["results"].copy()
    hist["source"] = "history"
    hist["stage"] = "n/a"
    hist["group"] = "n/a"
    if "venue_country" not in hist.columns:
        hist["venue_country"] = "Various"

    live = data["live"].copy()
    live["source"] = "wc2026"

    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral", "venue_country", "stage", "group", "source"]
    matches = pd.concat([hist[cols], live[cols]], ignore_index=True)

    # --- future-date guard: never let an unplayed fixture count as a result ---
    future = matches["date"] > today
    if future.any():
        n = int(future.sum())
        latest = matches.loc[future, "date"].max().date()
        print(f"  WARNING: dropped {n} future-dated match(es) (up to {latest}) "
              f"— not yet played as of {today.date()}.")
        matches = matches[~future].reset_index(drop=True)

    matches["is_host_match"] = matches.apply(_is_host_match, axis=1)
    matches["result"] = pd.NA
    matches.loc[matches.home_score > matches.away_score, "result"] = "H"
    matches.loc[matches.home_score < matches.away_score, "result"] = "A"
    matches.loc[matches.home_score == matches.away_score, "result"] = "D"

    matches["match_key"] = (
        matches["date"].dt.strftime("%Y-%m-%d") + "|" +
        matches["home_team"] + "|" + matches["away_team"] + "|" +
        matches["tournament"]
    )
    before = len(matches)
    matches = matches.drop_duplicates("match_key").reset_index(drop=True)
    dropped = before - len(matches)

    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED / "matches.csv", index=False)
    print(f"Built {len(matches):,} matches ({dropped} duplicate(s) removed).")
    print(f"  host-advantage matches: {int(matches['is_host_match'].sum())}")
    print(f"  saved -> {PROCESSED / 'matches.csv'}")
    return matches


if __name__ == "__main__":
    build_matches()
