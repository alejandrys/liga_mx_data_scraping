import numpy as np


def shin_probs(odds):
    odds = np.array(odds, dtype=float)

    # protección básica
    if any(o <= 1 for o in odds):
        return np.array([1/3, 1/3, 1/3])

    inv = 1 / odds
    total = inv.sum()

    # fallback simple si shin falla
    probs = inv / total

    return probs