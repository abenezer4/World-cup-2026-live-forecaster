"""Orchestrator — runs the whole pipeline end to end and saves a reproducible
record of the run (champion probabilities + metadata: seed, params, versions).
"""
from __future__ import annotations
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from config import RUNS, N_SIMS, SEED, SHRINKAGE, HALF_LIFE_DAYS, TODAY
from prepare import build_matches
from features import build_features
from model import fit
from simulate import simulate
from backtest import backtest
from ingest import load_raw


def main():
    print("=" * 60)
    print("World Cup 2026 — live forecast pipeline")
    print(f"as of {TODAY.date()}")
    print("=" * 60)

    print("\n[1/5] Prepare match table ...")
    matches = build_matches()

    print("\n[2/5] Build features (leakage-guarded) ...")
    features = build_features(matches)

    print("\n[3/5] Fit Dixon-Coles match model ...")
    model = fit(features)
    print(f"      home advantage +{model.home_adv:.3f}, rho {model.rho:+.3f}")

    print("\n[4/5] Simulate remaining matches ...")
    live = load_raw()["live"]
    probs = simulate(model, live, n_sims=N_SIMS, seed=SEED)
    print("      champion probabilities (top 8):")
    for _, r in probs.head(8).iterrows():
        print(f"        {r.team:<20s} {r.champion_prob*100:5.1f}%")

    print("\n[5/5] Backtest & calibration ...")
    bt = backtest(features)
    print(f"      log loss  model {bt['model_log_loss']:.3f}  vs  "
          f"Elo {bt['elo_log_loss']:.3f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    probs.to_csv(run_dir / "champion_probs.csv", index=False)

    meta = dict(
        run_id=run_id, as_of=str(TODAY.date()),
        config=dict(n_sims=N_SIMS, seed=SEED, shrinkage=SHRINKAGE,
                   half_life_days=HALF_LIFE_DAYS),
        model=dict(home_adv=model.home_adv, rho=model.rho, base=model.base,
                  n_teams=len(model.teams)),
        backtest=bt,
        versions=dict(python=platform.python_version(),
                     numpy=np.__version__, pandas=pd.__version__,
                     scipy=scipy.__version__),
    )
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nRun saved -> {run_dir}")
    print("  champion_probs.csv + run_metadata.json")


if __name__ == "__main__":
    main()
