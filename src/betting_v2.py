def ev(p, odds):
    return p * (odds - 1) - (1 - p)

def kelly(p, odds, frac=0.25, cap=0.03):
    b = odds - 1
    f = (p * b - (1 - p)) / b if b > 0 else 0
    f = max(0, f) * frac
    return min(f, cap)

def dyn_threshold(disagreement, base=0.02):
    # exige más edge si hay más desacuerdo
    return base + 0.08 * disagreement