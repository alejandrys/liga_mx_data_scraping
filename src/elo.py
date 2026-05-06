def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, result_a, k=20):
    ea = expected_score(ra, rb)
    return ra + k * (result_a - ea)

def get_result(home_goals, away_goals):
    if home_goals > away_goals: return 1
    if home_goals == away_goals: return 0.5
    return 0