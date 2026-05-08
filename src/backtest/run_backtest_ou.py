import pandas as pd
import numpy as np

from src.backtest.backtest_ou import run_backtest

# 1) Load data
df = pd.read_csv("data/odds/MEX.csv")

# 2) Feature engineering
df["goals_total"] = df["HG"] + df["AG"]

# ⚠️ temporal (hasta tener odds reales)
df["odds_over"] = np.random.uniform(1.8, 2.2, len(df))
df["odds_under"] = np.random.uniform(1.8, 2.2, len(df))

df["lambda_calibrated"] = 2.5

# 3) Thresholds
thresholds = {
    "min_ev": 0.03,
    "min_odds": 1.6,
    "max_odds": 3.2,
    "kelly_fraction": 0.25,
    "kelly_cap": 0.05,
}

# 4) Run
results, bets = run_backtest(df, thresholds)

print("RESULTS:")
print(results)

print("\nEV DISTRIBUTION:")
print(bets["ev"].describe())