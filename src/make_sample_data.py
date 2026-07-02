"""Optional helper — generates plausible sample history + Elo so you can test
the full pipeline immediately, before downloading the real Kaggle/eloratings
data. Uses the real 48 team names so it's a drop-in replacement.

Run once: python src/make_sample_data.py
Then swap data/raw/results.csv and data/raw/elo.csv for the real files later.
"""
import numpy as np
import pandas as pd
from config import RAW, ALL_WC_TEAMS

EXTRA_TEAMS = ["Italy", "Denmark", "Poland", "Serbia", "Chile", "Peru",
              "Costa Rica", "Nigeria", "Cameroon", "Wales", "Iceland",
              "Ukraine", "Greece", "Hungary"]
POOL = sorted(set(ALL_WC_TEAMS) | set(EXTRA_TEAMS))


def make_strengths(seed=42):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.normal(0, 0.45, len(POOL)))[::-1] * 0.9
    attack = dict(zip(POOL, base))
    defense = {t: rng.normal(0, 0.35) for t in POOL}
    return attack, defense


def _goals(home, away, attack, defense, r, neutral=True):
    adv = 0.0 if neutral else 0.35
    lam = np.exp(0.15 + attack[home] - defense[away] + adv)
    mu  = np.exp(0.15 + attack[away] - defense[home])
    return r.poisson(lam), r.poisson(mu)


def generate_history(n_years=12, seed=7):
    attack, defense = make_strengths()
    r = np.random.default_rng(seed)
    rows, start = [], pd.Timestamp("2014-01-01")
    comps = ["Friendly"] * 5 + ["Qualifier"] * 3 + ["Continental"] * 2
    for d in range(n_years * 365):
        if r.random() > 0.12:
            continue
        date = start + pd.Timedelta(days=int(d))
        for _ in range(int(r.integers(1, 4))):
            h, a = r.choice(POOL, size=2, replace=False)
            neutral = bool(r.random() < 0.4)
            gh, ga = _goals(h, a, attack, defense, r, neutral=neutral)
            rows.append(dict(date=date.strftime("%Y-%m-%d"), home_team=h,
                            away_team=a, home_score=gh, away_score=ga,
                            tournament=r.choice(comps), neutral=neutral))
    return pd.DataFrame(rows), attack, defense


def make_elo(attack, defense):
    rows = []
    dates = pd.date_range("2014-01-01", "2026-06-01", freq="MS")
    r = np.random.default_rng(11)
    base = {t: 1500 + (attack[t] - defense[t]) * 250 for t in POOL}
    for t in POOL:
        walk = np.cumsum(r.normal(0, 8, len(dates)))
        for dt, w in zip(dates, walk):
            rows.append(dict(date=dt.strftime("%Y-%m-%d"), team=t,
                            elo=round(base[t] + w, 1)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    hist, attack, defense = generate_history()
    hist.to_csv(RAW / "results.csv", index=False)
    print(f"sample results.csv: {len(hist):,} matches")

    elo = make_elo(attack, defense)
    elo.to_csv(RAW / "elo.csv", index=False)
    print(f"sample elo.csv: {len(elo):,} rows")
    print("\nSample data ready. Replace with real Kaggle + eloratings.net "
          "data when you have it (see README).")
