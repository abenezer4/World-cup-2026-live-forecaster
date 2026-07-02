"""Stage 4 — Match model: weighted Dixon-Coles (bivariate Poisson).

Fits per-team attack/defence strengths plus a host-advantage term by weighted
maximum likelihood, with:
  * sum-to-zero constraint on attack params (identifiability — required)
  * L2 shrinkage toward zero (regularises small-sample teams)
  * the Dixon-Coles rho correction for low-scoring draws

Format-agnostic: this file doesn't know or care about the tournament bracket.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from config import MAX_GOALS, SHRINKAGE
from features import build_features


def _dc_tau(h, a, lam, mu, rho):
    if h == 0 and a == 0: return 1 - lam * mu * rho
    if h == 0 and a == 1: return 1 + lam * rho
    if h == 1 and a == 0: return 1 + mu * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(self, teams, attack, defense, home_adv, rho, base):
        self.teams = teams
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.rho = rho
        self.base = base

    def _rates(self, home, away, host=False):
        adv = self.home_adv if host else 0.0
        a_h = self.attack.get(home, 0.0); d_h = self.defense.get(home, 0.0)
        a_a = self.attack.get(away, 0.0); d_a = self.defense.get(away, 0.0)
        lam = np.exp(self.base + a_h - d_a + adv)
        mu  = np.exp(self.base + a_a - d_h)
        return lam, mu

    def expected_goals(self, home, away, host=False):
        return self._rates(home, away, host)

    def predict_scoreline(self, home, away, host=False):
        lam, mu = self._rates(home, away, host)
        h = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
        a = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
        m = np.outer(h, a)
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= _dc_tau(i, j, lam, mu, self.rho)
        return m / m.sum()

    def win_draw_loss(self, home, away, host=False):
        m = self.predict_scoreline(home, away, host)
        return (float(np.tril(m, -1).sum()), float(np.trace(m)),
                float(np.triu(m, 1).sum()))


def fit(features=None, shrinkage: float = SHRINKAGE) -> DixonColesModel:
    if features is None:
        features = build_features()

    teams = sorted(set(features.home_team) | set(features.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    def unpack(p):
        atk = p[:n] - p[:n].mean()          # sum-to-zero constraint
        dfn = p[n:2 * n]
        return atk, dfn, p[2 * n], p[2 * n + 1], p[2 * n + 2]

    hi = features.home_team.map(idx).values
    ai = features.away_team.map(idx).values
    hg = features.home_score.values.astype(int)
    ag = features.away_score.values.astype(int)
    host = features.is_host_match.values.astype(float)
    w = features.weight.values

    def nll(p):
        atk, dfn, ha, rho, base = unpack(p)
        rho = np.clip(rho, -0.2, 0.2)
        lam = np.exp(base + atk[hi] - dfn[ai] + ha * host)
        mu  = np.exp(base + atk[ai] - dfn[hi])
        ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        tau = np.ones_like(lam)
        m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        ll = ll + np.log(np.clip(tau, 1e-9, None))
        penalty = shrinkage * (np.sum(atk ** 2) + np.sum(dfn ** 2))
        return -(w * ll).sum() + penalty

    p0 = np.concatenate([np.zeros(2 * n), [0.3, -0.1, 0.0]])
    res = minimize(nll, p0, method="L-BFGS-B")
    atk, dfn, ha, rho, base = unpack(res.x)
    return DixonColesModel(teams, dict(zip(teams, atk)), dict(zip(teams, dfn)),
                           float(ha), float(np.clip(rho, -0.2, 0.2)), float(base))


if __name__ == "__main__":
    m = fit()
    print(f"fitted {len(m.teams)} teams | home advantage +{m.home_adv:.3f} "
          f"(log-goals) | rho {m.rho:+.3f}")
    h, d, a = m.win_draw_loss("Brazil", "France")
    print(f"example — Brazil vs France: {h:.1%} / {d:.1%} / {a:.1%}")
