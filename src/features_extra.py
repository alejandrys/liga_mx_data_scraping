def rolling_form(df, team, n=5):
    df_t = df[(df["home"] == team) | (df["away"] == team)].tail(n)
    pts = 0
    for _, r in df_t.iterrows():
        if r["home"] == team:
            if r["home_goals"] > r["away_goals"]: pts += 3
            elif r["home_goals"] == r["away_goals"]: pts += 1
        else:
            if r["away_goals"] > r["home_goals"]: pts += 3
            elif r["away_goals"] == r["home_goals"]: pts += 1
    return pts / (3 * n) if n > 0 else 0.5