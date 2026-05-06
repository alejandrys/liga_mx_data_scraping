import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

def dixon_coles_correction(i, j, lam, mu, rho):
    if i == 0 and j == 0:
        return 1 - (lam * mu * rho)
    elif i == 0 and j == 1:
        return 1 + (lam * rho)
    elif i == 1 and j == 0:
        return 1 + (mu * rho)
    elif i == 1 and j == 1:
        return 1 - rho
    return 1


def dc_prob_matrix(lam, mu, rho, max_goals=10):
    mat = np.zeros((max_goals+1, max_goals+1))

    for i in range(max_goals+1):
        for j in range(max_goals+1):
            base = poisson.pmf(i, lam) * poisson.pmf(j, mu)
            mat[i, j] = base * dixon_coles_correction(i, j, lam, mu, rho)

    return mat / mat.sum()


def dc_outcomes(lam, mu, rho=0.0):
    mat = dc_prob_matrix(lam, mu, rho)

    pH = np.tril(mat, -1).sum()
    pD = np.trace(mat)
    pA = np.triu(mat, 1).sum()

    return pH, pD, pA