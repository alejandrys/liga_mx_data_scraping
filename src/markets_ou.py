import math

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def prob_under_25(lam_h, lam_a, max_goals=10):
    p = 0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            if i + j <= 2:
                p += poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
    return p

def prob_over_25(lam_h, lam_a):
    return 1 - prob_under_25(lam_h, lam_a)