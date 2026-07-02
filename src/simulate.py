"""Stage 5 — Simulate the real 2026 bracket.

The 48-team format (12 groups -> 32 qualifiers via 24 automatic + 8 best
third-place teams) determines the Round-of-32 pairings through a 495-way
lookup table published by FIFA (Annex C), keyed on *which* eight groups send
a third-place team. Once the group stage is finished, that combinatorial rule
has already resolved itself into one concrete bracket — so rather than
re-implement the 495-way table, we hardcode the actual realised bracket below
and keep it simple to update by hand each tournament (or each time you re-run
this project for a different competition).

Everything already played (group stage + completed knockout matches) is
locked in from the live results file. Only matches that haven't happened yet
are simulated, vectorised across N_SIMS runs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import N_SIMS, SEED
from model import DixonColesModel

# ---------------------------------------------------------------------------
# The actual Round-of-32 bracket, in bracket order (adjacent pairs of 2 feed
# the same Round-of-16 match). A leaf is either:
#   - a plain team name (str)  -> already decided, locked in
#   - a (home, away) tuple     -> not yet played, will be simulated
# Update this list by hand as results come in and re-run.
# ---------------------------------------------------------------------------
BRACKET_R32 = [
    # order matters: each adjacent pair feeds one Round-of-16 match, and the
    # resulting R16 winners must pair up in the exact order the real bracket
    # combines them into quarterfinals and semifinals. Verified against the
    # official bracket (match IDs M73-M104): R16 pairs must appear in the
    # sequence [M89, M90, M93, M94, M91, M92, M95, M96] so that adjacent
    # reduction naturally produces QF = [M97, M98, M99, M100] and
    # SF = [M101(=M97+M98), M102(=M99+M100)] -- an earlier ordering here had
    # M91/M92 adjacent to M89/M90, which would have paired the wrong two
    # quarterfinal winners into the same semifinal.
    "Paraguay", "France",                                  # -> M89
    "Canada", "Morocco",                                   # -> M90
    ("Portugal", "Croatia"), ("Spain", "Austria"),          # M83, M84 -> M93
    "United States", "Belgium",                            # M81, M82 (both decided) -> M94
    "Brazil", "Norway",                                     # -> M91
    "Mexico", "England",                                    # M92 (England already beat DR Congo, M80)
    ("Argentina", "Cape Verde"), ("Australia", "Egypt"),    # M86, M88 -> M95
    ("Switzerland", "Algeria"), ("Colombia", "Ghana"),      # M85, M87 -> M96
]
# host flags for the pending matches, in the same order as the tuples above
BRACKET_R32_HOST = [False] * 6


def group_standings(live: pd.DataFrame) -> pd.DataFrame:
    """Final group tables from the real group-stage results (for display)."""
    rows = []
    for g, gdf in live[live.stage == "group"].groupby("group"):
        tbl = {}
        for _, m in gdf.iterrows():
            for t in (m.home_team, m.away_team):
                tbl.setdefault(t, dict(pts=0, gf=0, ga=0))
            hs, as_ = int(m.home_score), int(m.away_score)
            tbl[m.home_team]["gf"] += hs; tbl[m.home_team]["ga"] += as_
            tbl[m.away_team]["gf"] += as_; tbl[m.away_team]["ga"] += hs
            if hs > as_: tbl[m.home_team]["pts"] += 3
            elif hs < as_: tbl[m.away_team]["pts"] += 3
            else: tbl[m.home_team]["pts"] += 1; tbl[m.away_team]["pts"] += 1
        for t, s in tbl.items():
            rows.append(dict(group=g, team=t, pts=s["pts"], gf=s["gf"],
                             ga=s["ga"], gd=s["gf"] - s["ga"]))
    df = pd.DataFrame(rows).sort_values(["group", "pts", "gd", "gf"],
                                        ascending=[True, False, False, False])
    df["rank"] = df.groupby("group").cumcount() + 1
    return df.reset_index(drop=True)


def _resolve_pair(model: DixonColesModel, home: str, away: str, host: bool,
                  n: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorised: sample n independent results for one fixed matchup."""
    ph, pd_, pa = model.win_draw_loss(home, away, host=host)
    # a drawn knockout tie goes to penalties; split roughly by strength
    p_home_wins_overall = ph + pd_ * (ph / (ph + pa)) if (ph + pa) > 0 else ph + pd_ / 2
    u = rng.random(n)
    return np.where(u < p_home_wins_overall, home, away)


def simulate(model: DixonColesModel, live: pd.DataFrame,
            n_sims: int = N_SIMS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # build the 16-slot Round-of-16 field for every simulation
    n_leaves = len(BRACKET_R32)
    field = np.empty((n_sims, n_leaves), dtype=object)
    pending_idx = 0
    for i, leaf in enumerate(BRACKET_R32):
        if isinstance(leaf, str):
            field[:, i] = leaf                                   # locked
        else:
            home, away = leaf
            host = BRACKET_R32_HOST[pending_idx] if pending_idx < len(BRACKET_R32_HOST) else False
            field[:, i] = _resolve_pair(model, home, away, host, n_sims, rng)
            pending_idx += 1

    # single-elimination the rest of the way: R16 -> QF -> SF -> Final
    cur = field
    while cur.shape[1] > 1:
        nxt = np.empty((n_sims, cur.shape[1] // 2), dtype=object)
        for j in range(nxt.shape[1]):
            home, away = cur[:, 2 * j], cur[:, 2 * j + 1]
            key = np.char.add(np.char.add(home.astype(str), "|"), away.astype(str))
            out = np.empty(n_sims, dtype=object)
            for k in np.unique(key):
                mask = key == k
                hn, an = k.split("|")
                out[mask] = _resolve_pair(model, hn, an, False, int(mask.sum()), rng)
            nxt[:, j] = out
        cur = nxt

    champions = cur[:, 0]
    vals, counts = np.unique(champions, return_counts=True)
    probs = (pd.DataFrame({"team": vals, "champion_prob": counts / n_sims})
            .sort_values("champion_prob", ascending=False)
            .reset_index(drop=True))
    return probs


def predict_bracket_path(model: DixonColesModel, bracket=BRACKET_R32,
                         host_flags=BRACKET_R32_HOST) -> dict:
    """The model's single most-likely winner at every remaining match, walked
    round by round through the real bracket. This is what a *visual* bracket
    should show (one deterministic predicted path) -- distinct from
    `simulate()`, which aggregates thousands of random draws into a champion
    probability distribution. A drawn tie's "winner" here is whichever side
    the model gives the higher combined win+penalty-adjusted probability to.

    Returns a dict keyed by round name -> list of match dicts, each with
    home/away/winner/confidence/played (bool).
    """
    def pick(home, away, host=False):
        ph, pd_, pa = model.win_draw_loss(home, away, host=host)
        p_home = ph + pd_ * (ph / (ph + pa)) if (ph + pa) > 0 else ph + pd_ / 2
        return (home, p_home) if p_home >= 0.5 else (away, 1 - p_home)

    rounds = {}
    # Round of 32 (only the still-pending ties; already-decided ones pass through)
    r32_matches = []
    pending_idx = 0
    field = []
    for leaf in bracket:
        if isinstance(leaf, str):
            field.append(leaf)
        else:
            home, away = leaf
            host = host_flags[pending_idx] if pending_idx < len(host_flags) else False
            winner, conf = pick(home, away, host)
            r32_matches.append(dict(home=home, away=away, winner=winner,
                                    confidence=conf, played=False))
            field.append(winner)
            pending_idx += 1
    rounds["Round of 32 (remaining)"] = r32_matches

    round_names = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
    cur = field
    for rname in round_names:
        nxt = []
        matches = []
        for i in range(0, len(cur), 2):
            home, away = cur[i], cur[i + 1]
            winner, conf = pick(home, away)
            matches.append(dict(home=home, away=away, winner=winner,
                                confidence=conf, played=False))
            nxt.append(winner)
        rounds[rname] = matches
        cur = nxt

    rounds["champion"] = cur[0]
    return rounds


if __name__ == "__main__":
    from ingest import load_raw
    from model import fit

    live = load_raw()["live"]
    print("Final group standings:")
    st = group_standings(live)
    print(st.head(4).to_string(index=False))

    m = fit()
    probs = simulate(m, live)
    print("\nChampion probabilities (top 10):")
    print(probs.head(10).to_string(index=False))

    path = predict_bracket_path(m)
    print(f"\nPredicted path -> champion: {path['champion']}")
    for rname in ["Round of 32 (remaining)", "Round of 16", "Quarterfinals",
                  "Semifinals", "Final"]:
        print(f"\n{rname}:")
        for mt in path[rname]:
            print(f"  {mt['home']} vs {mt['away']}  ->  {mt['winner']} "
                  f"({mt['confidence']*100:.0f}%)")
