"""Global configuration — single source of truth for every tunable knob."""
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
RAW         = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed"
ASSETS      = ROOT / "assets"
RUNS        = ROOT / "runs"

# ── model ──────────────────────────────────────────────────────────────────
HALF_LIFE_DAYS    = 730.0        # 2 years -- international teams play too few
                                 # matches/year for a short half-life; validated
                                 # against backtest in the full-workflow notebook
MAX_GOALS         = 8            # Poisson truncation
SHRINKAGE         = 0.30         # validated against backtest -- 0.05 let
                                 # sparse-data teams' parameters run away
SEED              = 7

# ── simulation ─────────────────────────────────────────────────────────────
N_SIMS            = 50_000

# ── feature weights ────────────────────────────────────────────────────────
IMPORTANCE = {
    "Friendly":      0.30,
    "Qualifier":     0.70,
    "Continental":   0.85,
    "World Cup":     1.00,
}
DEFAULT_IMPORTANCE = 0.50

# ── tournament ─────────────────────────────────────────────────────────────
TODAY = pd.Timestamp.now(tz="UTC").normalize()

HOST_COUNTRIES = {"United States", "Canada", "Mexico"}

# the 48 teams at the 2026 World Cup, grouped by their actual draw group
WC_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Switzerland", "Canada", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Uruguay", "Saudi Arabia"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Colombia", "Portugal", "DR Congo", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_WC_TEAMS = sorted({t for g in WC_GROUPS.values() for t in g})
