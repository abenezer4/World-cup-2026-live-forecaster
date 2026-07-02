"""compute_elo.py — builds data/raw/elo.csv from data/raw/results.csv using
the standard World Football Elo methodology (goal-difference-weighted
K-factor, home-advantage adjustment). No external Elo download needed.

Run whenever you refresh results.csv:  python3 src/compute_elo.py
"""
import pandas as pd
from config import RAW


def k_factor(tournament: str) -> int:
    t = str(tournament)
    if t == "FIFA World Cup":
        return 60
    if "Cup" in t or "Championship" in t or "Qualif" in t:
        return 50
    return 30


def goal_diff_multiplier(gd: float) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def compute_elo(matches: pd.DataFrame, base: float = 1500.0,
                home_adv: float = 100.0) -> pd.DataFrame:
    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    rating = {}
    rows = []
    for _, m in matches.iterrows():
        h, a = m.home_team, m.away_team
        rh, ra = rating.get(h, base), rating.get(a, base)
        adv = 0.0 if m.neutral else home_adv
        we = 1 / (10 ** (-(rh + adv - ra) / 400) + 1)
        outcome = (1.0 if m.home_score > m.away_score else
                  (0.0 if m.home_score < m.away_score else 0.5))
        k = k_factor(m.tournament) * goal_diff_multiplier(m.home_score - m.away_score)
        delta = k * (outcome - we)
        rating[h] = rh + delta
        rating[a] = ra - delta
        rows.append((m.date, h, rating[h]))
        rows.append((m.date, a, rating[a]))
    return pd.DataFrame(rows, columns=["date", "team", "elo"])


if __name__ == "__main__":
    df = pd.read_csv(RAW / "results.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    elo = compute_elo(df)
    elo.to_csv(RAW / "elo.csv", index=False)
    print(f"computed Elo for {elo.team.nunique()} teams "
          f"({len(elo):,} rating snapshots) -> {RAW / 'elo.csv'}")
