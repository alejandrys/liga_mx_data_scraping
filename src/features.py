import numpy as np
from src.mapping import TEAM_MAP



def compute_lambda(
    df,
    team,
    is_home,
    as_of,
    window=20,
    decay_alpha=0.003,
):


    df = df[df["date"] < as_of].copy()

    if is_home:
        df = df[df["home"] == team]
    else:
        df = df[df["away"] == team]

    if df.empty:
        return 1.3

    df = df.sort_values("date").tail(window)

    if is_home:
        goals = df["home_goals"]
    else:
        goals = df["away_goals"]

    days_diff = (as_of - df["date"]).dt.days
    weights = np.exp(-decay_alpha * days_diff)

    goals = goals.values
    weights = weights.values

    if len(goals) == 0 or len(goals) != len(weights):
        return df["home_goals"].mean() if len(df) > 0 else 1.3

    return np.average(goals, weights=weights)


import numpy as np

def compute_lambda_advanced(df, team, opponent, is_home, as_of):
    df = df[df["date"] < as_of].copy()

    # últimos N partidos
    df = df.sort_values("date").tail(200)

    # stats equipo
    team_home = df[df["home"] == team]
    team_away = df[df["away"] == team]

    team_scored = np.concatenate([
        team_home["home_goals"].values,
        team_away["away_goals"].values
    ])

    team_conceded = np.concatenate([
        team_home["away_goals"].values,
        team_away["home_goals"].values
    ])

    # stats rival
    opp_home = df[df["home"] == opponent]
    opp_away = df[df["away"] == opponent]

    opp_conceded = np.concatenate([
        opp_home["away_goals"].values,
        opp_away["home_goals"].values
    ])

    if len(team_scored) < 5 or len(opp_conceded) < 5:
        return None

    attack = np.mean(team_scored)
    defense_opp = np.mean(opp_conceded)

    # liga baseline
    league_avg = np.mean(df["home_goals"])

    lam = (attack * defense_opp) / league_avg

    # home advantage simple
    if is_home:
        lam *= 1.1

    return lam