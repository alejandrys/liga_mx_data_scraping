import requests
import pandas as pd
from src.mapping import map_team


def normalize_team(s):
    if not isinstance(s, str):
        return s
    
    try:
        s = s.encode('latin1').decode('utf-8')
    except:
        pass

    s = s.lower().strip()

    #quitar acentos
    import unicodedata
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

    return s

def get_odds(api_key, league="soccer_mexico_ligamx"):
    url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"

    params = {
        "apiKey": api_key,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    rows = []

    for match in data:
        home = map_team(match["home_team"])
        away = map_team(match["away_team"])

        if not match["bookmakers"]:
            continue

        book = match["bookmakers"][0]

        market = next(
            (m for m in book["markets"] if m["key"] == "h2h"),
            None
        )

        if not market:
            continue

        outcomes = {o["name"]: o["price"] for o in market["outcomes"]}

        rows.append({
            "home_team": home,
            "away_team": away,
            "odds_home_win": outcomes.get(match["home_team"]),
            "odds_away_win": outcomes.get(match["away_team"]),
            "odds_draw": outcomes.get("Draw")
        })

    return pd.DataFrame(rows)


def load_hist(path):
    import pandas as pd
    from src.mapping import map_team

    df = pd.read_csv(path)

    # fechas
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].dt.tz_localize(None)

    # normalización de equipos
    df["home"] = df["home"].apply(map_team)
    df["away"] = df["away"].apply(map_team)

    # 🔥 FILTRO DE EQUIPOS ACTUALES (AQUÍ VA)
    VALID_TEAMS = {
        'america', 'atlas', 'cruz azul', 'guadalajara', 'juarez',
        'mazatlan', 'necaxa', 'pachuca', 'puebla', 'san luis',
        'santos laguna', 'tigres', 'tijuana', 'toluca', 'pumas'
    }

    df = df[
        df["home"].isin(VALID_TEAMS) &
        df["away"].isin(VALID_TEAMS)
    ]

    #Filtro temporal
    #df = df[df["date"] > "2019-01-01"]

    return df