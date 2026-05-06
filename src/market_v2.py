import numpy as np

def odds_to_probs(odds):
    inv = np.array([1/o for o in odds], dtype=float)
    return inv / inv.sum()

def blend_market_model(p_market, p_model, alpha=0.25):
    """
    p_final = p_market + alpha * (p_model - p_market)
    alpha pequeño = edge conservador
    """
    p = np.array(p_market) + alpha * (np.array(p_model) - np.array(p_market))
    return p / p.sum()

