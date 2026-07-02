"""Stage 3 — Features (leakage-guarded).

For every match, compute features using only information available strictly
before kickoff: Elo as-of the match date, and a weight = time-decay ×
competition-importance.

Elo lookup uses a vectorised as-of join (merge_asof) rather than a per-row
table scan -- the naive version is O(n * m) and becomes impractically slow
once results.csv holds real full history (tens of thousands of matches).
merge_asof's own backward-direction semantics are what prevent leakage here
(a match can only join to an Elo snapshot dated on or before it), and that's
checked with one vectorised assertion instead of an O(n^2) per-row scan.
"""
from __future__ import annotations
import pandas as pd
from config import PROCESSED, HALF_LIFE_DAYS, IMPORTANCE, DEFAULT_IMPORTANCE, TODAY
from prepare import build_matches

# How much history to actually fit on. Real historical data can span 150
# years; teams and football itself have changed enough that ancient matches
# add little signal while making every stage slower. 15 years is generous
# for capturing genuine long-run strength while keeping this tractable.
TRAINING_WINDOW_YEARS = 15


def assert_no_leakage(features: pd.DataFrame):
    """Vectorised check: no feature row may be built from an Elo snapshot
    dated after that match. merge_asof(direction='backward') guarantees this
    by construction, but we verify it explicitly rather than trust silently.
    """
    bad_home = features["elo_home_date"] > features["date"]
    bad_away = features["elo_away_date"] > features["date"]
    if bad_home.any() or bad_away.any():
        raise AssertionError(
            f"LEAKAGE: {int((bad_home | bad_away).sum())} row(s) used an "
            f"Elo snapshot dated after the match itself"
        )


def _decay_weight(match_date, ref_date, half_life=HALF_LIFE_DAYS):
    age_days = (ref_date - match_date).dt.days
    return 0.5 ** (age_days / half_life)


def build_features(matches: pd.DataFrame | None = None,
                   elo: pd.DataFrame | None = None,
                   ref_date: pd.Timestamp = TODAY,
                   window_years: int = TRAINING_WINDOW_YEARS) -> pd.DataFrame:
    if matches is None:
        matches = build_matches()
    if elo is None:
        from ingest import load_raw
        elo = load_raw()["elo"]

    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    if window_years:
        cutoff = ref_date - pd.Timedelta(days=365 * window_years)
        matches = matches[matches["date"] >= cutoff].reset_index(drop=True)

    # rename the Elo table's own date column before merging, so merge_asof's
    # output keeps both the match date (join key) and the *snapshot's* date
    # as two distinct columns -- needed to actually verify no leakage.
    elo_sorted = (elo.sort_values("date").reset_index(drop=True)
                 .rename(columns={"date": "snapshot_date"}))
    # normalise timezone-awareness on both sides -- merge_asof requires an
    # exact dtype match, and a CSV round-trip can silently drop tz info.
    if matches["date"].dt.tz is None:
        matches["date"] = matches["date"].dt.tz_localize("UTC")
    if elo_sorted["snapshot_date"].dt.tz is None:
        elo_sorted["snapshot_date"] = elo_sorted["snapshot_date"].dt.tz_localize("UTC")

    def asof_join(side: str) -> pd.DataFrame:
        left = (matches[["date", f"{side}_team"]]
               .rename(columns={f"{side}_team": "team"})
               .reset_index().rename(columns={"index": "_orig_idx"}))
        joined = pd.merge_asof(left.sort_values("date"), elo_sorted,
                               left_on="date", right_on="snapshot_date",
                               by="team", direction="backward")
        return joined.sort_values("_orig_idx").reset_index(drop=True)

    eh = asof_join("home")
    ea = asof_join("away")

    feats = matches.copy()
    feats["elo_home"] = eh["elo"].fillna(1500.0).values
    feats["elo_home_date"] = eh["snapshot_date"].reset_index(drop=True)
    feats["elo_away"] = ea["elo"].fillna(1500.0).values
    feats["elo_away_date"] = ea["snapshot_date"].reset_index(drop=True)

    feats["elo_diff"] = feats["elo_home"] - feats["elo_away"]
    feats["importance"] = feats["tournament"].map(IMPORTANCE).fillna(DEFAULT_IMPORTANCE)
    feats["weight"] = _decay_weight(feats["date"], ref_date) * feats["importance"]

    assert_no_leakage(feats)
    feats = feats.drop(columns=["elo_home_date", "elo_away_date"])
    feats = feats[["date", "home_team", "away_team", "home_score", "away_score",
                   "is_host_match", "elo_home", "elo_away", "elo_diff",
                   "importance", "weight"]]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_csv(PROCESSED / "features.csv", index=False)
    return feats


if __name__ == "__main__":
    f = build_features()
    print(f"features: {len(f):,} rows (last {TRAINING_WINDOW_YEARS} years), "
          f"weight range [{f.weight.min():.4f}, {f.weight.max():.3f}]")
    print("leakage guard: OK (no assertion raised)")
