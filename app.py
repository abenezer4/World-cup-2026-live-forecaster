"""app.py — Streamlit dashboard for the World Cup 2026 live forecaster.

Thin layer: imports and calls src/ functions, never reimplements logic.
Heavy steps are cached so slider nudges stay snappy.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import ingest
import prepare
import model as model_mod
import simulate as sim_mod
import backtest as bt_mod
import bracket_view
from config import N_SIMS, SHRINKAGE, SEED, TODAY, ASSETS

st.set_page_config(page_title="World Cup 2026 — Live Forecast", page_icon="⚽",
                   layout="wide")

PITCH, LIME, CHALK, MUTED = "#0B3D2E", "#C6FF4F", "#F2F4F0", "#7E938A"
st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #0E5038 0%,
      {PITCH} 55%, #16241F 100%); color: {CHALK}; }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1, h2, h3 {{ color: {CHALK}; }}
  .eyebrow {{ color:{LIME}; font-weight:700; letter-spacing:.15em;
      text-transform:uppercase; font-size:.75rem; }}
  div[data-testid="stMetricValue"] {{ color:{LIME}; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                     "savefig.facecolor": "none", "text.color": CHALK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": CHALK,
                     "xtick.color": CHALK, "ytick.color": CHALK})


@st.cache_data(show_spinner=False)
def load_matches():
    return prepare.build_matches()

@st.cache_resource(show_spinner=False)
def fit_model(shrinkage):
    return model_mod.fit(shrinkage=shrinkage)

@st.cache_data(show_spinner=False)
def run_sim(n_sims, seed, shrinkage):
    live = ingest.load_raw()["live"]
    m = fit_model(shrinkage)
    return sim_mod.simulate(m, live, n_sims=n_sims, seed=seed)

@st.cache_data(show_spinner=False)
def standings_table():
    live = ingest.load_raw()["live"]
    return sim_mod.group_standings(live)

@st.cache_data(show_spinner=False)
def run_backtest_cached():
    return bt_mod.backtest()


with st.sidebar:
    st.markdown("<div class='eyebrow'>Controls</div>", unsafe_allow_html=True)
    n_sims = st.select_slider("Simulations", [5_000, 10_000, 25_000, 50_000,
                              100_000], value=N_SIMS)
    shrinkage = st.slider("Shrinkage", 0.0, 0.30, SHRINKAGE, 0.01)
    seed = st.number_input("Seed", value=SEED, step=1)
    st.divider()
    st.caption(f"Forecast as of **{TODAY.date()}**. Update "
               "`data/raw/wc2026_live.csv` with new results and re-run.")
    if st.button("↻ Re-run forecast", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear()

st.markdown("<div class='eyebrow'>Dixon-Coles · Monte Carlo · live-updating</div>",
           unsafe_allow_html=True)
st.markdown("# Who wins the 2026 World Cup?")
st.caption("Group results and completed knockout matches are locked to reality; "
          "only the remaining bracket is simulated.")

with st.spinner("Fitting model and simulating remaining matches…"):
    matches = load_matches()
    probs = run_sim(int(n_sims), int(seed), float(shrinkage))
    standings = standings_table()
    model = fit_model(float(shrinkage))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches trained on", f"{len(matches):,}")
c2.metric("Simulations", f"{int(n_sims/1000)}k")
c3.metric("Home advantage", f"+{model.home_adv:.2f}")
c4.metric("Favorite's odds", f"{probs.champion_prob.iloc[0]*100:.0f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Title race", "⚔️ Match predictor",
                                        "📊 Groups", "🎯 Calibration", "🗂️ Bracket"])

with tab1:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Champion probability")
        top = probs.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5.4))
        colors = [LIME if i == len(top) - 1 else "#5E7A6B" for i in range(len(top))]
        ax.barh(top.team, top.champion_prob * 100, color=colors, height=.72)
        for y, (v, t) in enumerate(zip(top.champion_prob * 100, top.team)):
            ax.text(v + .4, y, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9)
        ax.set_xlabel("champion probability (%)")
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        st.pyplot(fig, use_container_width=True)
    with right:
        st.subheader("Full table")
        show = probs.copy()
        show["champion_prob"] = (show.champion_prob * 100).round(1)
        show.columns = ["Team", "Title %"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=460)

with tab2:
    st.subheader("Head-to-head predictor")
    teams = sorted(model.teams)
    c1, c2, c3 = st.columns([1, 1, .6])
    home = c1.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = c2.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)
    host = c3.checkbox("Team A is host")
    if home != away:
        h, d, a = model.win_draw_loss(home, away, host=host)
        lam, mu = model.expected_goals(home, away, host=host)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{h*100:.0f}%")
        m2.metric("Draw", f"{d*100:.0f}%")
        m3.metric(f"{away} win", f"{a*100:.0f}%")
        st.caption(f"Expected goals — {home}: {lam:.2f}, {away}: {mu:.2f}")

with tab3:
    st.subheader("Final group standings (real results)")
    show = standings.rename(columns={"team": "Team", "group": "Grp", "rank": "#",
                                     "pts": "Pts", "gf": "GF", "ga": "GA", "gd": "GD"})
    order = [c for c in ["Grp", "#", "Team", "Pts", "GF", "GA", "GD"] if c in show.columns]
    st.dataframe(show.sort_values(["Grp", "#"])[order], use_container_width=True,
                hide_index=True, height=560)

with tab4:
    st.subheader("Is the model calibrated?")
    try:
        res = run_backtest_cached()
        m1, m2, m3 = st.columns(3)
        m1.metric("Log loss (model)", f"{res['model_log_loss']:.3f}")
        m2.metric("Log loss (Elo)", f"{res['elo_log_loss']:.3f}")
        m3.metric("Brier (model)", f"{res['model_brier']:.3f}")
        cal_png = ASSETS / "calibration.png"
        if cal_png.exists():
            st.image(str(cal_png), width=520)
    except Exception as e:
        st.warning(f"Backtest unavailable: {e}")

with tab5:
    st.subheader("Full knockout bracket")
    st.caption("Real results are locked in. Dashed boxes are undetermined "
              "matches — the highlighted team is the model's predicted "
              "winner, with its confidence shown below.")
    bracket_html = bracket_view.build_bracket_html(model)
    st.components.v1.html(bracket_html, height=1560, scrolling=True)

st.caption("Method: weighted Dixon-Coles bivariate Poisson + Monte Carlo "
          "tournament simulation. Analysis only — not betting advice; "
          "not affiliated with FIFA.")
