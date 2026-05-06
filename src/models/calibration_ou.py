import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson

PROB_COLUMNS = ["p_over", "p_under"]


def global_lambda_scale(
    actual_goals,
    predicted_lambda_total,
    *,
    min_scale=0.5,
    max_scale=1.5,
):
    """
    Estimate one multiplicative scale for total-goals lambdas.
    """

    actual = np.asarray(actual_goals, dtype=float)
    predicted = np.asarray(predicted_lambda_total, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("actual_goals and predicted_lambda_total must have the same shape")

    mask = np.isfinite(actual) & np.isfinite(predicted) & (predicted > 0)
    if not mask.any():
        return 1.0

    scale = actual[mask].sum() / predicted[mask].sum()
    return float(np.clip(scale, min_scale, max_scale))


def apply_global_lambda_scale(lambda_total, scale):
    """
    Apply a global multiplicative scale to total-goals lambdas.
    """

    scaled = np.asarray(lambda_total, dtype=float) * float(scale)
    scaled = np.clip(scaled, 1e-6, None)

    if np.isscalar(lambda_total):
        return float(scaled)

    return pd.Series(scaled, index=_preserve_index(lambda_total, len(scaled)), name="lambda_total_scaled")


def market_ou_probs(odds_over, odds_under=None):
    """
    Convert Over/Under decimal odds to no-vig market probabilities.
    """

    odds = _coerce_ou_odds(odds_over, odds_under)

    valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    inv = np.full_like(odds, np.nan, dtype=float)
    inv[valid] = 1.0 / odds[valid]

    totals = inv.sum(axis=1)
    probs = np.full_like(inv, np.nan, dtype=float)
    valid_totals = valid & np.isfinite(totals) & (totals > 0)
    probs[valid_totals] = inv[valid_totals] / totals[valid_totals, None]

    if _is_single_market(odds_over, odds_under):
        return probs[0]

    return pd.DataFrame(
        probs,
        index=_preserve_index(odds_over, len(probs)),
        columns=PROB_COLUMNS,
    )


def infer_lambda_from_market(
    market_over,
    market_under=None,
    *,
    min_lambda=0.5,
    max_lambda=5.0,
):
    """
    Infer Poisson total-goals lambda from market Over 2.5 prices or probabilities.
    """

    p_over, index, is_scalar = _coerce_market_over_prob(market_over, market_under)

    lambdas = np.array(
        [
            _invert_over_probability(p, min_lambda, max_lambda)
            for p in p_over
        ],
        dtype=float,
    )

    if is_scalar:
        return float(lambdas[0])

    return pd.Series(lambdas, index=index, name="lambda_total_market")


def blend_lambda(lambda_model, lambda_market, alpha):
    model = np.asarray(lambda_model, dtype=float)
    market = np.asarray(lambda_market, dtype=float)
    model, market = np.broadcast_arrays(model, market)

    weight = float(np.clip(alpha, 0.0, 1.0))

    blended = np.where(
        np.isfinite(model) & np.isfinite(market),
        (1.0 - weight) * model + weight * market,
        np.where(np.isfinite(model), model, market)
    )

    blended = np.clip(blended, 0.5, 5.0)

    if blended.size == 1:
        return float(blended[0])

    if isinstance(lambda_model, pd.Series):
        index = lambda_model.index
    elif isinstance(lambda_market, pd.Series):
        index = lambda_market.index
    else:
        index = None

    return pd.Series(blended.ravel(), index=index, name="lambda_total_blend")


def _coerce_ou_odds(odds_over, odds_under):
    if odds_under is None:
        odds = np.asarray(odds_over, dtype=float)
        if odds.ndim == 1:
            if odds.shape[0] != 2:
                raise ValueError("odds_over must contain [odds_over, odds_under]")
            return odds.reshape(1, 2)
        if odds.ndim == 2 and odds.shape[1] == 2:
            return odds
        raise ValueError("odds_over must be a two-column array or odds_under must be provided")

    over = np.asarray(odds_over, dtype=float)
    under = np.asarray(odds_under, dtype=float)
    over, under = np.broadcast_arrays(over, under)
    return np.column_stack([over.ravel(), under.ravel()])


def _coerce_market_over_prob(market_over, market_under):
    if isinstance(market_over, pd.DataFrame):
        if "p_over" not in market_over.columns:
            raise ValueError("market_over DataFrame must contain p_over")
        return market_over["p_over"].to_numpy(dtype=float), market_over.index, False

    if isinstance(market_over, pd.Series) and market_under is None:
        return market_over.to_numpy(dtype=float), market_over.index, False

    if market_under is None:
        values = np.asarray(market_over, dtype=float)
        if values.ndim == 0:
            return np.array([float(values)]), None, True
        if values.ndim == 1:
            if values.shape[0] == 2 and (values > 1.0).all():
                probs = market_ou_probs(values)
                return np.array([probs[0]]), None, True
            if (
                values.shape[0] == 2
                and np.isfinite(values).all()
                and (values >= 0.0).all()
                and (values <= 1.0).all()
                and np.isclose(values.sum(), 1.0)
            ):
                return np.array([values[0]]), None, True
            return values, _preserve_index(market_over, len(values)), False
        if values.ndim == 2 and values.shape[1] == 2:
            probs = market_ou_probs(values)
            return probs["p_over"].to_numpy(dtype=float), _preserve_index(market_over, len(probs)), False
        raise ValueError("market_over must be p_over values, [odds_over, odds_under], or a two-column odds array")

    over = np.asarray(market_over, dtype=float)
    under = np.asarray(market_under, dtype=float)
    over, under = np.broadcast_arrays(over, under)
    is_scalar = over.ndim == 0

    over_flat = over.ravel()
    under_flat = under.ravel()
    if ((over_flat > 1.0) & (under_flat > 1.0)).all():
        probs = market_ou_probs(over_flat, under_flat)
        return probs["p_over"].to_numpy(dtype=float), _preserve_index(market_over, len(over_flat)), is_scalar

    denom = over_flat + under_flat
    p_over = np.divide(
        over_flat,
        denom,
        out=np.full_like(over_flat, np.nan, dtype=float),
        where=denom > 0,
    )
    return p_over, _preserve_index(market_over, len(p_over)), is_scalar


def _invert_over_probability(p_over_market, min_lambda, max_lambda):
    lower = 0.5
    upper = 5.0
    eps = 1e-10

    if not np.isfinite(p_over_market):
        return np.nan

    p_over_market = float(np.clip(p_over_market, 1e-12, 1.0 - 1e-12))

    def objective(lambda_total):
        return poisson.sf(2, lambda_total) - p_over_market

    try:
        lower_value = objective(lower)
        upper_value = objective(upper)

        if abs(lower_value) < eps:
            return lower
        if abs(upper_value) < eps:
            return upper

        if lower_value * upper_value > 0.0:
            p_lower = poisson.sf(2, lower)
            p_upper = poisson.sf(2, upper)

            if p_over_market <= p_lower:
                return lower
            if p_over_market >= p_upper:
                return upper

            return lower  # fallback conservador

        lambda_total = brentq(objective, lower, upper)
        return float(np.clip(lambda_total, lower, upper))

    except (ValueError, RuntimeError, OverflowError):
        approx = 2.5 + 2.0 * (p_over_market - 0.5)
        return float(np.clip(approx, lower, upper))
    


def _is_single_market(odds_over, odds_under):
    if odds_under is None:
        odds = np.asarray(odds_over, dtype=float)
        return odds.ndim == 1
    return np.isscalar(odds_over) and np.isscalar(odds_under)


def _preserve_index(values, expected_length):
    index = getattr(values, "index", None)
    if callable(index) or index is None or len(index) != expected_length:
        return None
    return index
