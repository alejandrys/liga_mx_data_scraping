import pandas as pd
from src.mapping import map_team


def load_hist_odds(path="data/odds/MEX.csv"):
    df = pd.read_csv(path)

    # ---- columnas base ----
    df = df[[
        "Date", "Home", "Away",
        "HG", "AG",
        "B365CH", "B365CD", "B365CA",
        "AvgCH", "AvgCD", "AvgCA"
    ]].copy()

    df.columns = [
        "date", "home", "away",
        "home_goals", "away_goals",
        "b365_home", "b365_draw", "b365_away",
        "avg_home", "avg_draw", "avg_away"
    ]

    # ---- fechas ----
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # ---- elegir odds ----
    # PRIORIDAD:
    # 1) B365 (si existe)
    # 2) Avg (fallback)
    df["odds_home"] = df["b365_home"].fillna(df["avg_home"])
    df["odds_draw"] = df["b365_draw"].fillna(df["avg_draw"])
    df["odds_away"] = df["b365_away"].fillna(df["avg_away"])

    # ---- convertir a float (por si vienen como string) ----
    for col in ["odds_home", "odds_draw", "odds_away"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- limpiar filas inválidas ----
    df = df.dropna(subset=[
        "home_goals", "away_goals",
        "odds_home", "odds_draw", "odds_away"
    ])

    # ---- normalizar equipos ----
    df["home"] = df["home"].apply(map_team)
    df["away"] = df["away"].apply(map_team)

    return df