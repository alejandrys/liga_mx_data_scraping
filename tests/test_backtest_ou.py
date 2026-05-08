def test_backtest_runs():
    from src.backtest.backtest_ou import run_backtest
    import pandas as pd

    df = pd.DataFrame({
        "lambda_calibrated": [2.5, 2.2],
        "odds_over": [2.0, 2.1],
        "odds_under": [1.8, 1.9],
        "goals_total": [3, 1]
    })

    results, _ = run_backtest(df)

    assert results["n_bets"] >= 0