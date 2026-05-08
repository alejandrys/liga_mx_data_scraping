import numpy as np
import pandas as pd
from scipy.stats import poisson


# =========================
# Core metrics
# =========================

def ev(prob, odds):
    prob = np.asarray(prob, dtype=float)
    odds = np.asarray(odds, dtype=float)

    value = prob * odds - 1.0
    value = np.where(
        np.isfinite(prob) & np.isfinite(odds) & (odds > 1.0),
        value,
        np.nan
    )

    return value


def kelly(prob, odds):
    prob = np.asarray(prob, dtype=float)
    odds = np.asarray(odds, dtype=float)

    b = odds - 1.0

    fraction = np.divide(
        prob * b - (1.0 - prob),
        b,
        out=np.zeros_like(prob),
        where=(b > 0) & np.isfinite(prob)
    )

    return np.clip(fraction, 0.0, 1.0)


# =========================
# Main selector
# =========================

def select_bets(df, thresholds=None):
    """
    Select Over/Under bets based on EV + risk filters.

    Required columns:
        - lambda_calibrated
        - odds_over
        - odds_under

    Optional:
        - p_over_calibrated
        - p_under_calibrated
    """

    cfg = {
        "min_ev": 0.03,
        "min_odds": 1.6,
        "max_odds": 3.2,
        "kelly_fraction": 0.25,
        "kelly_cap": 0.05,
    }
    if thresholds:
        cfg.update(thresholds)

    out = df.copy()

    # -------------------------
    # 1) Get probabilities
    # -------------------------
    if "p_over_calibrated" in out.columns and "p_under_calibrated" in out.columns:
        p_over = pd.to_numeric(out["p_over_calibrated"], errors="coerce").values
        p_under = pd.to_numeric(out["p_under_calibrated"], errors="coerce").values
    else:
        lambdas = pd.to_numeric(out["lambda_calibrated"], errors="coerce").clip(0.5, 5.0)
        p_over = poisson.sf(2, lambdas)
        p_under = poisson.cdf(2, lambdas)

    probs = np.column_stack([p_over, p_under])

    # -------------------------
    # 2) Odds
    # -------------------------
    odds_over = pd.to_numeric(out["odds_over"], errors="coerce").values
    odds_under = pd.to_numeric(out["odds_under"], errors="coerce").values

    odds = np.column_stack([odds_over, odds_under])

    # -------------------------
    # 3) EV
    # -------------------------
    ev_matrix = ev(probs, odds)

    # -------------------------
    # 4) Select best side ONLY
    # -------------------------
    best_idx = np.nanargmax(ev_matrix, axis=1)
    best_ev = ev_matrix[np.arange(len(ev_matrix)), best_idx]
    best_prob = probs[np.arange(len(probs)), best_idx]
    best_odds = odds[np.arange(len(odds)), best_idx]

    side = np.where(best_idx == 0, "Over", "Under")

    # -------------------------
    # 5) Filters
    # -------------------------
    mask = (
        np.isfinite(best_ev)
        & (best_ev >= cfg["min_ev"])
        & (best_odds >= cfg["min_odds"])
        & (best_odds <= cfg["max_odds"])
    )

    # -------------------------
    # 6) Kelly sizing
    # -------------------------
    kelly_raw = kelly(best_prob, best_odds)
    kelly_final = np.minimum(
        kelly_raw * cfg["kelly_fraction"],
        cfg["kelly_cap"]
    )

    # -------------------------
    # 7) Build result
    # -------------------------
    result = pd.DataFrame({
        "side": side,
        "prob": best_prob,
        "odds": best_odds,
        "ev": best_ev,
        "kelly": kelly_final,
    }, index=out.index)

    result = result[mask].copy()

    # opcional: merge con columnas originales
    result = result.join(out, how="left")

    return result