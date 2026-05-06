import numpy as np
import pandas as pd
from scipy.stats import poisson

PROB_COLUMNS = ["p_over", "p_under"]


def compute_ou_probs(lambda_total):
    """
    Compute Over/Under 2.5 probabilities using Poisson distribution.
    """

    lambdas = np.asarray(lambda_total, dtype=float)

    # 🔒 estabilidad numérica
    lambdas = np.clip(lambdas, 1e-6, None)

    p_under = poisson.cdf(2, lambdas)
    p_over = poisson.sf(2, lambdas)

    return pd.DataFrame(
        {
            "p_over": p_over,
            "p_under": p_under,
        },
        index=_preserve_index(lambda_total, len(lambdas)),
        columns=PROB_COLUMNS,
    )


def validate_probs(df: pd.DataFrame, rtol=1e-12, atol=1e-12):
    """
    Validate that probabilities sum to 1.
    """

    missing = [col for col in PROB_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    probs = df[PROB_COLUMNS].to_numpy(dtype=float)

    if not np.all(np.isfinite(probs)):
        raise ValueError("Non-finite probabilities detected")

    totals = probs.sum(axis=1)

    if not np.allclose(totals, 1.0, rtol=rtol, atol=atol):
        raise ValueError("Probabilities do not sum to 1")

    return df


def _preserve_index(lambda_total, expected_length):
    index = getattr(lambda_total, "index", None)
    if index is None or len(index) != expected_length:
        return None
    return index