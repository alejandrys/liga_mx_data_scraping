import numpy as np
import pandas as pd
from src.betting.selector_ou import select_bets


def settle_bets(bets: pd.DataFrame):
    """
    Determine win/loss and returns.
    """

    # Resultado real
    goals = bets["goals_total"].values

    is_over = bets["side"] == "Over"
    is_under = bets["side"] == "Under"

    win = (
        (is_over & (goals > 2.5)) |
        (is_under & (goals <= 2.5))
    )

    odds = bets["odds"].values

    # Profit por unidad apostada
    returns = np.where(win, odds - 1.0, -1.0)

    bets["win"] = win
    bets["return"] = returns

    return bets


def run_backtest(df: pd.DataFrame, thresholds=None):
    """
    Full backtest pipeline.
    """

    # 1) Selección de apuestas
    bets = select_bets(df, thresholds)

    if bets.empty:
        return {
            "n_bets": 0,
            "roi": 0.0,
            "avg_ev": 0.0,
            "ev_std": 0.0
        }

    # 2) Settlement
    bets = settle_bets(bets)

    # 3) Métricas
    n_bets = len(bets)
    total_return = bets["return"].sum()
    roi = total_return / n_bets

    avg_ev = bets["ev"].mean()
    ev_std = bets["ev"].std()

    results = {
        "n_bets": n_bets,
        "roi": roi,
        "avg_ev": avg_ev,
        "ev_std": ev_std,
    }

    return results, bets