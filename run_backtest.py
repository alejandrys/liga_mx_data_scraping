import pandas as pd

from src.hist_odds import load_hist_odds
from src.backtest import run_backtest, evaluate
from collections import defaultdict
from src.elo import update_elo, get_result

df = load_hist_odds("data/odds/MEX.csv")

print("\n=== SAMPLE DATA ===")
print(df.head())

print("\n=== ODDS STATS ===")
print(df[["odds_home", "odds_draw", "odds_away"]].describe())

df = df.sort_values("date")

elo = defaultdict(lambda: 1500)

history, n_bets, records = run_backtest(df)

df_calib = pd.DataFrame(records)
df_calib.to_csv("output/calibration.csv", index=False)

print("\n=== DEBUG HISTORY ===")
print(type(history))
print(len(history))
print(history[:5])

metrics = evaluate(history)

print("\n=== BACKTEST RESULTS ===")
print(metrics)
print(f"n_bets: {n_bets}")


