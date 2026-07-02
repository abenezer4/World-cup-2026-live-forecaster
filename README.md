# Who Wins the 2026 World Cup? — A Live, Calibrated Forecast

A bivariate-Poisson (Dixon–Coles) match model feeding a Monte Carlo tournament
simulator, wrapped in a Streamlit dashboard. It re-prices every team's title
chances **as results come in**: played group matches are locked to reality, and
only the unplayed knockout bracket is simulated.



---

## The idea in one paragraph

You don't predict a tournament winner directly — there are far too few finals to
learn from. Instead you predict **goals in a single match** well, then let a
bracket of those matches play out tens of thousands of times and count how often
each team is left standing. The match engine here is a weighted **Dixon–Coles
bivariate Poisson**: it estimates each team's attack and defence strength plus a
host-country effect, then turns any matchup into a full scoreline distribution.
Recent games count more than old ones (180-day half-life) and competitive games
count more than friendlies. After each matchday you update one CSV, re-run, and
the title odds sharpen.

## Why this is more than a toy

- **It's calibrated, not just confident.** On held-out matches the model beats an
  Elo-only baseline on both log loss (**1.043 vs 1.151**) and Brier score
  (**0.623 vs 0.700**). When it says 60%, it means it. See the reliability curve
  in `assets/calibration.png`.
- **No data leakage.** Every feature for a match is computed using only matches
  strictly before it, enforced by an assertion and covered by unit tests.
- **Small-sample teams are regularised.** Attack/defence estimates shrink toward
  the mean so a debutant with three matches can't look like a contender on noise.
- **Reproducible runs.** Every run writes champion probabilities plus a metadata
  file recording seed, sim count, fitted parameters, data hashes and library
  versions.

## What's under the hood

| Stage | File | What it does |
|-------|------|--------------|
| Ingest | `src/ingest.py` | Loads raw sources, canonicalises team names, **fails loudly** on any unmapped name or malformed row |
| Prepare | `src/prepare.py` | Merges history + live results into one chronological table; resolves host-vs-neutral by venue country |
| Features | `src/features.py` | Elo-as-of-date, rolling goal rates, and a per-match weight = time-decay × competition importance, with a leakage guard |
| Model | `src/model.py` | Weighted-MLE Dixon–Coles: sum-to-zero constraint for identifiability, L2 shrinkage, rho low-score-draw correction |
| Simulate | `src/simulate.py` | Locks in actual group results; vectorised Monte Carlo over the remaining knockout bracket |
| Backtest | `src/backtest.py` | Rolling-origin holdout; log loss, Brier, calibration plot vs Elo baseline |
| App | `app.py` | Streamlit dashboard: title race, match predictor, standings, calibration |

## Modelling choices, briefly

- **Bivariate Poisson over a classifier.** Modelling goals directly is the
  principled route to win/draw/loss *and* exact scorelines, and the Dixon–Coles
  correction fixes Poisson's well-known underestimate of low-scoring draws —
  which matter enormously in knockouts.
- **Time-decay + importance weighting instead of a state-space model.** Weighted
  MLE captures most of the "current form" signal at a fraction of the complexity.
  A Kalman/state-space upgrade is the obvious next step if calibration demands it.
- **Simulate, don't extrapolate.** The champion question is answered by sampling
  the bracket, not by fitting "who wins it all" on a handful of past finals.

## Run it

```bash
pip install -r requirements.txt

# 1. generate the synthetic demo data (so it runs immediately)
python src/make_synthetic.py

# 2. run the whole pipeline once (prepare -> fit -> simulate -> backtest)
python src/run.py

# 3. launch the dashboard
streamlit run app.py
```

Update the forecast each matchday by editing `data/raw/wc2026_live.csv` with the
latest results and re-running. The app caches the heavy steps, so tuning the
sliders stays responsive.

## Using real data

The repo ships with **synthetic** data so it runs out of the box. For a live
forecast, swap in:

- **Match history** — Kaggle `martj42/international-football-results`
- **Elo ratings** — eloratings.net (time-series snapshots)

Keep the same column schema (see `src/make_synthetic.py` for the exact shape),
add any new country spellings to `data/team_aliases.csv`, and re-run.

## Honest limitations

- Synthetic data is shown by default; headline numbers above are from that demo
  and will change with real inputs.
- The bracket logic is a compact 8-group → 16-team knockout for clarity; the real
  48-team format with third-place qualifiers is a straightforward extension.
- Injuries, suspensions and lineup news aren't modelled — a known gap that
  bookmakers exploit.
- This is analysis, **not betting advice**, and isn't affiliated with FIFA.

## Tech

Python · NumPy · pandas · SciPy · Matplotlib · Streamlit

---

*Built as a portfolio project: a live, honest, calibration-first take on the
"predict the World Cup with ML" genre.*
