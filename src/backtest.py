"""Stage 6 — Backtest & calibration.

Rolling-origin holdout: train on the earliest matches, test on the most
recent slice, compare against an Elo-only baseline on log loss and Brier
score, and plot a reliability curve. Calibration matters more than accuracy
for a probabilistic forecast.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ASSETS
from features import build_features
from model import fit


def _elo_baseline_probs(elo_diff: float):
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    draw = 0.26 * np.exp(-abs(elo_diff) / 300)
    return p_home * (1 - draw), draw, (1 - p_home) * (1 - draw)


def backtest(features: pd.DataFrame | None = None,
            test_fraction: float = 0.15) -> dict:
    if features is None:
        features = build_features()
    f = features.sort_values("date").reset_index(drop=True)
    split = int(len(f) * (1 - test_fraction))
    train, test = f.iloc[:split], f.iloc[split:]

    m = fit(train)

    eps = 1e-12
    model_ll, elo_ll, model_br, elo_br = [], [], [], []
    cal_pred, cal_obs = [], []
    for _, row in test.iterrows():
        h, d, a = m.win_draw_loss(row.home_team, row.away_team,
                                  host=bool(row.is_host_match))
        bh, bd, ba = _elo_baseline_probs(row.elo_diff)
        outcome = 0 if row.home_score > row.away_score else \
                 (2 if row.home_score < row.away_score else 1)
        y = [0, 0, 0]; y[outcome] = 1
        model_ll.append(-np.log([h, d, a][outcome] + eps))
        elo_ll.append(-np.log([bh, bd, ba][outcome] + eps))
        model_br.append(sum((np.array([h, d, a]) - y) ** 2))
        elo_br.append(sum((np.array([bh, bd, ba]) - y) ** 2))
        cal_pred.append(h); cal_obs.append(1 if outcome == 0 else 0)

    result = dict(n_test=len(test),
                 model_log_loss=float(np.mean(model_ll)),
                 elo_log_loss=float(np.mean(elo_ll)),
                 model_brier=float(np.mean(model_br)),
                 elo_brier=float(np.mean(elo_br)))

    ASSETS.mkdir(parents=True, exist_ok=True)
    _calibration_plot(np.array(cal_pred), np.array(cal_obs), result)
    return result


def _calibration_plot(pred, obs, metrics):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pred, bins) - 1
    xs, ys = [], []
    for b in range(10):
        mask = idx == b
        if mask.sum() >= 5:
            xs.append(pred[mask].mean()); ys.append(obs[mask].mean())

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(xs, ys, "o-", lw=2, ms=7, color="#16a34a", label="model")
    ax.set_xlabel("predicted home-win probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration — is 60% really 60%?")
    ax.legend(); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(ASSETS / "calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    r = backtest()
    print("Backtest (rolling-origin holdout):")
    print(f"  test matches : {r['n_test']}")
    print(f"  log loss     : model {r['model_log_loss']:.3f}  |  "
          f"Elo {r['elo_log_loss']:.3f}")
    print(f"  Brier        : model {r['model_brier']:.3f}  |  "
          f"Elo {r['elo_brier']:.3f}")
    print(f"  calibration plot -> assets/calibration.png")
