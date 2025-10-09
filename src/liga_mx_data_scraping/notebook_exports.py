# Auto-generated from 03_liga_mx_data_scraping.ipynb

# --- Cell 1 ---


# Configuración Inicial
import requests
import pandas as pds
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from scipy.stats import poisson
from glob import glob

ODDS_API_KEY = "d27a39583e5f94530090dc01afd2bddc"  # ← REEMPLÁZALO CON TU CLAVE
LEAGUE = "Liga MX"
REGION = "mx"



# --- Cell 2 ---
#pip install python-dotenv

# --- Cell 3 ---

pds.set_option('display.max_rows', None)

# --- Cell 4 ---
def get_sports():
    url = "https://api.the-odds-api.com/v4/sports"
    params = {
        "apiKey": ODDS_API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"Error {r.status_code}: {r.text}")
    return pds.json_normalize(r.json())

sports_df = get_sports()
#sports_df[["key", "title", "active"]]

sports_df

# --- Cell 5 ---
# ============================================
# 1. 🔌 Conexión a OddsAPI para obtener cuotas
# ============================================
def get_odds(league="soccer_mexico_ligamx"):
    url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    return pds.json_normalize(response.json())

odds_df = get_odds()
odds_df[["home_team", "away_team", "bookmakers"]].head()



# --- Cell 6 ---
print(odds_df.columns.tolist())


# --- Cell 7 ---
# Extraer promedios genéricos para ejemplo (luego se pueden conectar a estadísticas reales)
def estimate_avg_goals(team_name):
    team_name = team_name.lower()
    if "américa" in team_name:
        return 2.0
    elif "tigres" in team_name:
        return 1.7
    elif "chivas" in team_name:
        return 1.3
    elif "pumas" in team_name:
        return 1.4
    else:
        return 1.5

matches = []
for _, row in odds_df.iterrows():
    try:
        home = row["home_team"]
        away = row["away_team"]
        odds = row["bookmakers"][0]["markets"][0]["outcomes"]
        odds_map = {o["name"]: o["price"] for o in odds}
        matches.append({
            "home_team": home,
            "away_team": away,
            "avg_goals_home": estimate_avg_goals(home),
            "avg_goals_away": estimate_avg_goals(away),
            "odds_home_win": odds_map.get(home, None),
            "odds_draw": odds_map.get("Draw", None),
            "odds_away_win": odds_map.get(away, None)
        })
    except Exception as e:
        continue

df_matches = pds.DataFrame(matches).dropna()

df_matches.head()




# --- Cell 8 ---
## Modelo Poisson + Probabilidades
def poisson_matrix(lam_home, lam_away, max_goals=6):
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i][j] = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
    return mat

def poisson_probabilities(lam_home, lam_away):
    matrix = poisson_matrix(lam_home, lam_away)
    prob_home = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away = np.sum(np.triu(matrix, 1))
    return prob_home, prob_draw, prob_away

def expected_value(prob, odds):
    return prob * odds - 1




# --- Cell 9 ---

## 4. Calcular resultados y recomendación
results = []
for _, row in df_matches.iterrows():
    ph, pd, pa = poisson_probabilities(row["avg_goals_home"], row["avg_goals_away"])
    ev_home = expected_value(ph, row["odds_home_win"])
    ev_draw = expected_value(pd, row["odds_draw"])
    ev_away = expected_value(pa, row["odds_away_win"])

    best_ev = max(ev_home, ev_draw, ev_away)
    decision = "Home" if best_ev == ev_home else "Draw" if best_ev == ev_draw else "Away"

    results.append({
        "Match": f"{row['home_team']} vs {row['away_team']}",
        "Prob_H": round(ph, 3),
        "Prob_D": round(pd, 3),
        "Prob_A": round(pa, 3),
        "EV_H": round(ev_home, 2),
        "EV_D": round(ev_draw, 2),
        "EV_A": round(ev_away, 2),
        "Best_Bet": decision,
    })

output = pds.DataFrame(results)
display(output)


# --- Cell 10 ---
# ============================================
# 3. 📡 Scraping Sofascore para métricas reales
# ============================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_sofascore_match_urls():
    base_url = "https://www.sofascore.com/es/futbol/mexico/liga-mx/34"
    r = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("a", href=True)
    match_urls = [l["href"] for l in links if "/mx/" in l["href"] and "/resumen" in l["href"]]
    return list(set(["https://www.sofascore.com" + u for u in match_urls]))

def extract_match_metrics(url):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    raw = soup.get_text().lower()
    # Aquí puedes incluir patrones para xG, tiros, etc.
    print(f"Extrayendo métricas de: {url}")
    print(raw[:1000])  # Recorte para revisión
    return {}

urls = get_sofascore_match_urls()
for url in urls[:]:  # limitar a 2 partidos para pruebas
    extract_match_metrics(url)




# --- Cell 11 ---
#pip install scraperFC

# --- Cell 12 ---


import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime



# --- Cell 13 ---
from ScraperFC.sofascore import Sofascore  

# --- Cell 14 ---
from scipy.special import gammaln
from scipy.optimize import minimize
from math import log, exp

# --- Cell 15 ---


# 1) Lee todos los CSV descargados (uno por temporada) desde Football-Data
files = glob("C:/Users/aleja/Downloads/MEX.csv")  # <-- ajusta la ruta a tu carpeta
if not files:
    raise FileNotFoundError("No encontré CSV en data/mx/*.csv. ¿Ya los descargaste?")

dfs = []
for f in files:
    df = pd.read_csv(f)
    # Football-Data: columnas típicas -> Date, HomeTeam, AwayTeam, FTHG, FTAG
    # (a veces hay encabezados distintos por temporada; si cambian, ajusta aquí)
    rename_map = {
        "Home":"home", "Away":"away",
        "HG":"home_goals", "AG":"away_goals",
        "Date":"Date"
    }
    # algunas temporadas usan 'Div','Date','HomeTeam','AwayTeam', 'FTHG','FTAG',...
    df = df.rename(columns=rename_map)
    # 2) Parseo de fecha (Football-Data suele usar day-first)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # 3) Subconjunto mínimo para tu pipeline
    base = df[["date","home","away","home_goals","away_goals"]].dropna().copy()

    # (Opcional) Si quieres incluir cuotas históricas de cierre (ej. Pinnacle/mercado):
    # Columnas comunes: PSH/PSD/PSA (Pinnacle), B365H/B365D/B365A (Bet365), MaxH/MaxD/MaxA (máxima), AvgH/AvgD/AvgA (promedio)
    for col in ["PSH","PSD","PSA","AvgH","AvgD","AvgA","MaxH","MaxD","MaxA","B365H","B365D","B365A"]:
        if col in df.columns:
            base[col] = df[col]

    dfs.append(base)

# 4) Unifica temporadas y ordena por fecha
hist = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)

# 5) Crea columnas xG vacías (hasta que tengamos xG)
hist["xG_home"] = pd.NA
hist["xG_away"] = pd.NA

print(hist.head())
print("Partidos en histórico:", len(hist))


# --- Cell 16 ---
# =========================
# PARCHE: normalizar fechas a tz-naive
# =========================
import pandas as pd
import numpy as np

def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        # si tiene tz (aware) -> quita tz
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except AttributeError:
        try:
            ts = ts.tz_localize(None)
        except Exception:
            pass
    return ts

# 1) hist.date -> naive
hist = hist.copy()
hist["date"] = _to_naive_series(hist["date"])
if "year" not in hist.columns:
    hist["year"] = hist["date"].dt.year

# 2) df_matches.date -> naive (si existe)
if "df_matches" in globals() and isinstance(df_matches, pd.DataFrame):
    if "date" in df_matches.columns:
        df_matches["date"] = _to_naive_series(df_matches["date"])

# 3) NOW -> naive
NOW_NAIVE = _to_naive_ts(pd.Timestamp.now(tz="America/Mexico_City"))

# 4) Reemplaza tu rolling_lambda_asof para comparar siempre naive vs naive
def rolling_lambda_asof(hist_df, team, side, as_of):
    as_of_naive = _to_naive_ts(as_of)
    dates_naive = _to_naive_series(hist_df["date"])
    df = hist_df.loc[dates_naive < as_of_naive]

    use_xg = ("xG_home" in df.columns and df["xG_home"].notna().any())
    if side == "home":
        mask = df["home"] == team
        series = df.loc[mask, "xG_home" if use_xg else "home_goals"]
    else:
        mask = df["away"] == team
        series = df.loc[mask, "xG_away" if use_xg else "away_goals"]

    if series.empty:
        return 1.3

    WINDOW = globals().get("WINDOW", 10)
    DECAY_ALPHA = globals().get("DECAY_ALPHA", 0.003)
    SEASON_DECAY = globals().get("SEASON_DECAY", 1.0)

    series = series.tail(WINDOW)
    dates  = df.loc[mask, "date"].tail(WINDOW)
    years  = (df.loc[mask, "year"].tail(WINDOW)
              if "year" in df.columns else dates.dt.year)

    delta_days = (as_of_naive - dates).dt.days.clip(lower=0)
    w_time   = np.exp(-DECAY_ALPHA * delta_days)
    w_season = (SEASON_DECAY ** (as_of_naive.year - years)).astype(float)
    w = w_time * w_season
    lam = float(np.average(series, weights=w))
    return max(lam, 0.2)

# 5) Si construyes as_of así, asegúrate de forzarlo a naive:
#    as_of_raw puede venir de df_matches['date'] (naive) o NaT -> usa NOW_NAIVE
def _pick_as_of(as_of_raw):
    return _to_naive_ts(as_of_raw) if pd.notna(as_of_raw) else NOW_NAIVE


# --- Cell 17 ---
hist = hist.copy()
hist["date"] = _to_naive_series(hist["date"])

# --- Cell 18 ---
# ============================================
# PIPELINE ÚNICO: λ rolling+decay + Dixon–Coles + Odds/Hist + EV + Kelly
# Requisitos previos: variable `hist` ya cargada con columnas:
#   date, home, away, home_goals, away_goals
#   (opcional) xG_home, xG_away, AvgH/AvgD/AvgA o PSH/PSD/PSA
# ============================================
import math, warnings, requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize, Bounds

warnings.filterwarnings("ignore")

# ---------- Configuración ----------
# Rolling λ
WINDOW = 10                # últimos N partidos del equipo
DECAY_ALPHA = 0.003        # decaimiento temporal ~ exp(-alpha * días)
SEASON_DECAY = 0.85        # penalización por año adicional (0.85^Δaños); 1.0 = sin penalizar
USE_XG = True              # si existen xG_home/xG_away, se usan; si no, se usan goles

# OddsAPI (opcional; si no la usas, se toman odds del propio hist)
ODDS_API_KEY = "d27a39583e5f94530090dc01afd2bddc"          # <-- pon aquí tu key si quieres odds en vivo
SPORT_KEY = "soccer_mexico_ligamx"
REGION    = "uk"           # 'uk'/'us'/'eu' (válidos en OddsAPI)

# Kelly / banca
BANKROLL = 10000.0
KELLY_FRACTION = 0.5       # 0.25–0.50 recomendado
MAX_STAKE_PCT = 0.03       # 3% de la banca como tope por apuesta

# ---------- Sanitización de `hist` ----------
required_cols = {"date","home","away","home_goals","away_goals"}
missing = required_cols - set(hist.columns)
if missing:
    raise ValueError(f"`hist` no tiene las columnas requeridas: {missing}")

# Fechas a datetime + columna año
hist = hist.copy()
hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
hist = hist.dropna(subset=["date","home","away","home_goals","away_goals"]).sort_values("date").reset_index(drop=True)
hist["year"] = hist["date"].dt.year.astype(int)

# Asegura columnas xG si no existen
for c in ["xG_home","xG_away"]:
    if c not in hist.columns:
        hist[c] = pd.NA

# ---------- Funciones de λ (rolling + decays) ----------
def rolling_lambda_asof(hist_df, team, side, as_of):
    df = hist_df[hist_df["date"] < as_of]
    if side == "home":
        mask = df["home"] == team
        series = df.loc[mask, "xG_home" if (USE_XG and df["xG_home"].notna().any()) else "home_goals"]
    else:
        mask = df["away"] == team
        series = df.loc[mask, "xG_away" if (USE_XG and df["xG_away"].notna().any()) else "away_goals"]

    if series.empty:
        return 1.3  # valor conservador

    # Últimos N + pesos por tiempo/temporada
    series = series.tail(WINDOW)
    dates  = df.loc[mask, "date"].tail(WINDOW)
    years  = df.loc[mask, "year"].tail(WINDOW)
    delta_days = (as_of - dates).dt.days.clip(lower=0)
    w_time   = np.exp(-DECAY_ALPHA * delta_days)
    w_season = (SEASON_DECAY ** (as_of.year - years)).astype(float)
    w = w_time * w_season
    lam = float(np.average(series, weights=w))
    return max(lam, 0.2)

# ---------- Dixon–Coles ----------
def tau_dc(x, y, lam_h, lam_a, rho):
    if x==0 and y==0: return max(1 - lam_h*lam_a*rho, 1e-8)
    if x==0 and y==1: return max(1 + lam_a*rho,       1e-8)
    if x==1 and y==0: return max(1 + lam_h*rho,       1e-8)
    if x==1 and y==1: return max(1 - rho,             1e-8)
    return 1.0

def score_matrix_dc(lh, la, rho, max_goals=10):
    I = np.arange(0, max_goals+1)
    J = np.arange(0, max_goals+1)
    M = np.outer(poisson.pmf(I, lh), poisson.pmf(J, la))
    for x in (0,1):
        for y in (0,1):
            M[x,y] *= tau_dc(x,y,lh,la,rho)
    s = M.sum()
    if s>0: M /= s
    return M

def probs_HDA_dc(lh, la, rho, max_goals=10):
    M = score_matrix_dc(lh, la, rho, max_goals=max_goals)
    pH = np.tril(M, -1).sum(); pD = np.trace(M); pA = np.triu(M, 1).sum()
    s = pH + pD + pA
    return float(pH/s), float(pD/s), float(pA/s)

def log_p_pois(k, lam):
    if lam <= 0: return -1e9
    return k*np.log(lam) - lam - math.lgamma(k+1)

def estimate_rho(hist_df, sample_size=800):
    H = hist_df.dropna(subset=["home_goals","away_goals"]).copy().sort_values("date")
    if len(H) == 0:
        return 0.0
    H = H.tail(min(sample_size, len(H)))
    lamH, lamA, X, Y = [], [], [], []
    for _, r in H.iterrows():
        as_of = r["date"]
        lh = rolling_lambda_asof(hist_df, r["home"], "home", as_of)
        la = rolling_lambda_asof(hist_df, r["away"], "away", as_of)
        lamH.append(lh); lamA.append(la)
        X.append(int(r["home_goals"])); Y.append(int(r["away_goals"]))
    lamH = np.array(lamH); lamA = np.array(lamA)
    X = np.array(X, dtype=int); Y = np.array(Y, dtype=int)

    def neg_ll(rho):
        if rho < -0.3 or rho > 0.3: return 1e9
        ll = 0.0
        for i in range(len(X)):
            ll += log_p_pois(X[i], lamH[i]) + log_p_pois(Y[i], lamA[i]) + np.log(tau_dc(X[i], Y[i], lamH[i], lamA[i], rho))
        return -ll

    res = minimize(lambda v: neg_ll(v[0]), x0=np.array([0.05]), method="L-BFGS-B", bounds=Bounds([-0.3],[0.3]))
    return float(res.x[0]) if res.success else 0.0

# ---------- EV & Kelly ----------
def expected_value(prob, odds):
    return prob*odds - 1.0

def kelly_fraction_decimal(prob, odds):
    b = max(odds - 1.0, 0.0); q = 1.0 - prob
    if b <= 0: return 0.0
    f = (b*prob - q) / b
    return max(f, 0.0)

# ---------- OddsAPI (opcional) ----------
def get_odds_oddsapi():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": "h2h", "oddsFormat": "decimal"}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise Exception(f"OddsAPI error {r.status_code}: {r.text}")
    data = r.json()
    rows = []
    for ev in data:
        home, away = ev.get("home_team"), ev.get("away_team")
        h2h = None
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") == "h2h":
                    h2h = m; break
            if h2h: break
        if not h2h: continue
        price_home = price_draw = price_away = None
        for o in h2h.get("outcomes", []):
            nm, pr = o.get("name"), o.get("price")
            if nm == home: price_home = pr
            elif nm == "Draw": price_draw = pr
            elif nm == away: price_away = pr
        if home and away and price_home and price_draw and price_away:
            rows.append({
                "home_team": home, "away_team": away,
                "odds_home_win": float(price_home),
                "odds_draw": float(price_draw),
                "odds_away_win": float(price_away)
            })
    return pd.DataFrame(rows)

# ---------- Preparación de df_matches ----------
def build_df_matches_from_hist(hist_df, n=20):
    """Si no hay OddsAPI, usa odds históricas del propio hist (últimos n partidos con odds)."""
    cols_sets = [
        ("AvgH","AvgD","AvgA"),  # promedio mercado
        ("PSH","PSD","PSA"),     # Pinnacle
        ("B365H","B365D","B365A"),
        ("MaxH","MaxD","MaxA")
    ]
    base = None
    for (h,d,a) in cols_sets:
        if {h,d,a}.issubset(hist_df.columns):
            tmp = hist_df.dropna(subset=[h,d,a]).copy()
            if len(tmp) > 0:
                tmp = tmp.tail(n)
                tmp = tmp.rename(columns={
                    "home":"home_team","away":"away_team",
                    h:"odds_home_win", d:"odds_draw", a:"odds_away_win"
                })
                tmp = tmp[["date","home_team","away_team","odds_home_win","odds_draw","odds_away_win"]]
                base = tmp
                break
    if base is None:
        # Sin odds históricas: crea un ejemplo sintético con odds 2.4/3.2/3.0 (NO PARA APUESTAS REALES)
        print("⚠️ No encontré columnas de odds en `hist`; crea df_matches sintético para continuar el flujo.")
        sample = hist_df.tail(n).copy()
        base = sample.rename(columns={"home":"home_team","away":"away_team"})[["date","home_team","away_team"]]
        base["odds_home_win"] = 2.40; base["odds_draw"] = 3.20; base["odds_away_win"] = 3.00
    return base.reset_index(drop=True)

# ===================================================
#                 EJECUCIÓN
# ===================================================
print("1) Estimando ρ (Dixon–Coles) con lambdas 'as-of'…")
rho_hat = estimate_rho(hist, sample_size=800)
print(f"   ρ estimado = {rho_hat:.4f}")

print("2) Preparando df_matches…")
df_matches = pd.DataFrame()
if ODDS_API_KEY and ODDS_API_KEY.strip() and "TU_API_KEY" not in ODDS_API_KEY:
    try:
        df_matches = get_odds_oddsapi()
        if df_matches.empty:
            raise ValueError("OddsAPI devolvió 0 partidos.")
        print(f"   OddsAPI listo: {len(df_matches)} partidos.")
    except Exception as e:
        print("   Advertencia OddsAPI → usar histórico:", e)
        df_matches = build_df_matches_from_hist(hist)
else:
    df_matches = build_df_matches_from_hist(hist)

# ¿Fecha de evaluación "as-of"?
NOW_MX = pd.Timestamp.now(tz="America/Mexico_City")
as_of_series = df_matches["date"] if "date" in df_matches.columns else pd.Series([NOW_MX]*len(df_matches))

print("3) Calculando λ, probabilidades DC, EV y Kelly…")
rows = []
for i, m in df_matches.reset_index(drop=True).iterrows():
    as_of = _pick_as_of(as_of_series.iloc[i]) if "as_of_series" in locals() else NOW_NAIVE
    home = m["home_team"]; away = m["away_team"]
    lh = rolling_lambda_asof(hist, home, "home", as_of)
    la = rolling_lambda_asof(hist, away, "away", as_of)
    pH, pD, pA = probs_HDA_dc(lh, la, rho_hat, max_goals=10)

    evH = expected_value(pH, m["odds_home_win"])
    evD = expected_value(pD, m["odds_draw"])
    evA = expected_value(pA, m["odds_away_win"])

    cand = {"Home": (pH, m["odds_home_win"], evH),
            "Draw": (pD, m["odds_draw"],    evD),
            "Away": (pA, m["odds_away_win"],evA)}
    best_name, (p_best, o_best, ev_best) = max(cand.items(), key=lambda kv: kv[1][2])

    k_full = kelly_fraction_decimal(p_best, o_best)
    k_frac = KELLY_FRACTION * k_full
    k_cap  = min(k_frac, MAX_STAKE_PCT)
    stake  = round(BANKROLL * k_cap, 2)

    rows.append({
        "Match": f"{home} vs {away}",
        "as_of": pd.to_datetime(as_of).strftime("%Y-%m-%d"),
        "λ_home": round(lh,3), "λ_away": round(la,3),
        "Prob_H": round(pH,3), "Prob_D": round(pD,3), "Prob_A": round(pA,3),
        "Odds_H": round(float(m["odds_home_win"]),3),
        "Odds_D": round(float(m["odds_draw"]),3),
        "Odds_A": round(float(m["odds_away_win"]),3),
        "EV_H": round(evH,3), "EV_D": round(evD,3), "EV_A": round(evA,3),
        "Best": best_name, "Best_EV": round(ev_best,3),
        "Kelly_full": round(k_full,3), "Kelly_frac": round(k_frac,3),
        "Stake_cap": round(k_cap,3), "Stake_suggested": stake
    })

out = pd.DataFrame(rows).sort_values("Best_EV", ascending=False).reset_index(drop=True)

pd.set_option("display.max_columns", None)
print("\n=== Parámetros del modelo ===")
print(f"WINDOW={WINDOW} | DECAY_ALPHA={DECAY_ALPHA} | SEASON_DECAY={SEASON_DECAY} | USE_XG={USE_XG}")
print(f"ρ (Dixon–Coles) = {rho_hat:.4f}")

print("\n=== Pronósticos + EV + Kelly ===")
print(out.to_string(index=False))

pos = out[(out["Best_EV"]>0) & (out["Stake_suggested"]>0)].copy()
print("\n=== Recomendaciones con valor (EV>0) ===")
print(pos.to_string(index=False) if not pos.empty else "No hay valor positivo ahora mismo.")


# --- Cell 19 ---
# =========================================
# OPERACIONALIZACIÓN: guardado, TOP-K, budget y log
# Requiere: `out` DataFrame del pipeline (con columnas: Match, Prob_H/D/A, Odds_*, EV_*, Best, Best_EV, Stake_suggested)
#           y `hist` (tu histórico ya cargado)
# =========================================
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Parámetros de operación ---
TOP_K = 5                    # toma los mejores K por EV
MAX_TOTAL_RISK_PCT = 0.10    # p.ej., arriesgar como máximo 10% de la banca total en la jornada
BANKROLL = 10000.0           # debe coincidir con el usado en el pipeline

# --- Carpetas de salida ---
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1) Guarda histórico para cacheo (evita volver a descargar/armar)
hist_path = Path("output/hist_mx.parquet")
try:
    hist.to_parquet(hist_path, index=False)
    print(f"✓ Histórico guardado en {hist_path}")
except Exception as e:
    print("⚠️ No se pudo guardar 'hist' como Parquet:", e)

# 2) Selección de picks con EV>0 y stake>0
picks = out[(out["Best_EV"]>0) & (out["Stake_suggested"]>0)].copy()
picks = picks.sort_values("Best_EV", ascending=False).head(TOP_K).reset_index(drop=True)

if picks.empty:
    print("No hay picks con EV positivo en este momento.")
else:
    # 3) Respeta presupuesto total por jornada
    budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
    sum_stakes = picks["Stake_suggested"].sum()
    if sum_stakes > budget_max:
        scale = budget_max / sum_stakes
        picks["Stake_final"] = (picks["Stake_suggested"] * scale).round(2)
        picks["Stake_scale"] = round(scale, 3)
    else:
        picks["Stake_final"] = picks["Stake_suggested"]
        picks["Stake_scale"] = 1.0

    # 4) Exporta pronósticos completos y picks
    all_pred_path = Path(f"output/predictions_{ts}.csv")
    picks_path    = Path(f"output/picks_{ts}.csv")
    out.to_csv(all_pred_path, index=False, encoding="utf-8")
    picks.to_csv(picks_path, index=False, encoding="utf-8")
    print(f"✓ Pronósticos completos: {all_pred_path}")
    print(f"✓ Picks seleccionados:   {picks_path}")

    # 5) Log incremental (para reconciliar resultados después)
    log_path = Path("output/run_log.parquet")
    run_meta = picks.copy()
    run_meta.insert(0, "run_ts", ts)
    run_meta.insert(1, "bankroll_start", BANKROLL)
    run_meta["risk_budget"] = budget_max
    # Campos útiles para reconciliar:
    # - Resultado real (a completar después): result (Win/Lose/Push)
    # - Retorno: pnl
    for col in ["result","pnl"]:
        if col not in run_meta.columns:
            run_meta[col] = pd.NA
    try:
        if log_path.exists():
            old = pd.read_parquet(log_path)
            pd.concat([old, run_meta], ignore_index=True).to_parquet(log_path, index=False)
        else:
            run_meta.to_parquet(log_path, index=False)
        print(f"✓ Log actualizado: {log_path}")
    except Exception as e:
        print("⚠️ No se pudo actualizar el log:", e)

    # 6) Vista rápida de picks
    display_cols = ["Match","Best","Best_EV","Prob_H","Prob_D","Prob_A","Odds_H","Odds_D","Odds_A","Stake_final","Stake_scale"]
    print("\n=== Picks operativos (TOP-K con budget) ===")
    print(picks[display_cols].to_string(index=False))


# --- Cell 20 ---
# ============================================
# Construcción de OUT con NO-VIG + EDGES integrados
# y guardado de PICKS incluyendo edges/fair odds
# ============================================
import numpy as np
import pandas as pd
import math
from pathlib import Path
from datetime import datetime

# ----- helpers no-vig -----
def fair_probs_proportional(odds_triplet):
    r = np.array([1.0/odds_triplet[0], 1.0/odds_triplet[1], 1.0/odds_triplet[2]], dtype=float)
    p = r / r.sum()
    return p

def fair_probs_shin(odds_triplet, max_s=0.25, iters=60):
    r = np.array([1.0/odds_triplet[0], 1.0/odds_triplet[1], 1.0/odds_triplet[2]], dtype=float)
    q = r / r.sum()
    def f(s):
        return (q / (s + (1.0 - s)*q)).sum() - 1.0
    lo, hi = 0.0, max_s
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    s_hat = 0.5*(lo+hi)
    p = q / (s_hat + (1.0 - s_hat)*q)
    p = p / p.sum()
    return p, float(s_hat)

def expected_value(prob, odds):
    return prob*odds - 1.0

def kelly_fraction_decimal(prob, odds):
    b = max(odds - 1.0, 0.0); q = 1.0 - prob
    if b <= 0: return 0.0
    f = (b*prob - q) / b
    return max(f, 0.0)

# ----- prob as-of series (naive) -----
as_of_series = df_matches["date"] if "date" in df_matches.columns else pd.Series([pd.Timestamp.now()] * len(df_matches))
def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except AttributeError:
        try: ts = ts.tz_localize(None)
        except Exception: pass
    return ts
def _pick_as_of(x):
    return _to_naive_ts(x) if pd.notna(x) else _to_naive_ts(pd.Timestamp.now())

# ----- prepara rho (si no lo tienes ya) -----
try:
    rho_hat
except NameError:
    rho_hat = estimate_rho(hist, sample_size=800)

# ===== LOOP principal: modelo + EV + Kelly + no-vig + edges =====
rows = []
for i, m in df_matches.reset_index(drop=True).iterrows():
    as_of = _pick_as_of(as_of_series.iloc[i])
    home = m["home_team"]; away = m["away_team"]

    lh = rolling_lambda_asof(hist, home, "home", as_of)
    la = rolling_lambda_asof(hist, away, "away", as_of)
    pH, pD, pA = probs_HDA_dc(lh, la, rho_hat, max_goals=10)

    oH, oD, oA = float(m["odds_home_win"]), float(m["odds_draw"]), float(m["odds_away_win"])
    evH, evD, evA = expected_value(pH, oH), expected_value(pD, oD), expected_value(pA, oA)

    # no-vig (proporcional y Shin)
    p_prop = fair_probs_proportional([oH, oD, oA])
    p_shin, s_hat = fair_probs_shin([oH, oD, oA])

    # edges vs mercado no-vig (elige Shin como principal)
    edge_H = pH - p_shin[0]
    edge_D = pD - p_shin[1]
    edge_A = pA - p_shin[2]
    best_edge_name = np.array(["Home","Draw","Away"])[np.argmax([edge_H, edge_D, edge_A])]
    best_edge_val  = float(np.max([edge_H, edge_D, edge_A]))

    # kelly (sobre el mejor EV)
    cand = {"Home": (pH, oH, evH), "Draw": (pD, oD, evD), "Away": (pA, oA, evA)}
    best_name, (p_best, o_best, ev_best) = max(cand.items(), key=lambda kv: kv[1][2])
    k_full = kelly_fraction_decimal(p_best, o_best)
    k_frac = KELLY_FRACTION * k_full
    k_cap  = min(k_frac, MAX_STAKE_PCT)
    stake  = round(BANKROLL * k_cap, 2)

    rows.append({
        "Match": f"{home} vs {away}",
        "as_of": pd.to_datetime(as_of).strftime("%Y-%m-%d"),
        "λ_home": round(lh,3), "λ_away": round(la,3),

        "Prob_H": round(pH,4), "Prob_D": round(pD,4), "Prob_A": round(pA,4),
        "Odds_H": round(oH,3), "Odds_D": round(oD,3), "Odds_A": round(oA,3),

        "EV_H": round(evH,3), "EV_D": round(evD,3), "EV_A": round(evA,3),
        "Best": best_name, "Best_EV": round(ev_best,3),
        "Kelly_full": round(k_full,3), "Kelly_frac": round(k_frac,3),
        "Stake_cap": round(k_cap,3), "Stake_suggested": stake,

        # Mercado sin margen (proporcional)
        "Mkt_noVig_H_prop": round(p_prop[0],4),
        "Mkt_noVig_D_prop": round(p_prop[1],4),
        "Mkt_noVig_A_prop": round(p_prop[2],4),
        "FairOdds_H_prop": round(1.0/p_prop[0],3),
        "FairOdds_D_prop": round(1.0/p_prop[1],3),
        "FairOdds_A_prop": round(1.0/p_prop[2],3),

        # Mercado sin margen (Shin)
        "Mkt_noVig_H_shin": round(p_shin[0],4),
        "Mkt_noVig_D_shin": round(p_shin[1],4),
        "Mkt_noVig_A_shin": round(p_shin[2],4),
        "FairOdds_H_shin": round(1.0/p_shin[0],3),
        "FairOdds_D_shin": round(1.0/p_shin[1],3),
        "FairOdds_A_shin": round(1.0/p_shin[2],3),
        "Shin_s": round(s_hat,4),

        # Edges (modelo vs mercado no-vig Shin)
        "Edge_H": round(edge_H,4),
        "Edge_D": round(edge_D,4),
        "Edge_A": round(edge_A,4),
        "Best_by_Edge": best_edge_name,
        "Best_Edge": round(best_edge_val,4),
    })

out = pd.DataFrame(rows).sort_values(["Best_EV","Best_Edge"], ascending=False).reset_index(drop=True)
pd.set_option("display.max_columns", None)

print("\n=== Pronósticos + EV + Kelly + No-Vig + Edges ===")
display_cols = [
    "Match","as_of","λ_home","λ_away",
    "Prob_H","Prob_D","Prob_A",
    "Odds_H","Odds_D","Odds_A",
    "EV_H","EV_D","EV_A","Best","Best_EV",
    "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin","Shin_s",
    "FairOdds_H_shin","FairOdds_D_shin","FairOdds_A_shin",
    "Edge_H","Edge_D","Edge_A","Best_by_Edge","Best_Edge",
    "Kelly_full","Kelly_frac","Stake_cap","Stake_suggested"
]
print(out[display_cols].to_string(index=False))

# ===========================
# Guardado de PICKS con edges
# ===========================
TOP_K = 5
MAX_TOTAL_RISK_PCT = 0.10    # máx 10% banca por jornada
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Reglas de selección: EV>0 en el "Best" y Edge>0 en el "Best_by_Edge"
# (si quieres, puedes exigir que coincidan Best y Best_by_Edge; lo dejo opcional)
picks = out.copy()
picks = picks[(picks["Best_EV"] > 0) & (picks["Best_Edge"] > 0)]

# (Opcional) exigir coherencia entre Best y Best_by_Edge:
# picks = picks[picks["Best"] == picks["Best_by_Edge"]]

picks = picks.sort_values(["Best_Edge","Best_EV"], ascending=False).head(TOP_K).reset_index(drop=True)

if picks.empty:
    print("\nNo hay picks con EV>0 y Edge>0 ahora mismo.")
else:
    budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
    sum_stakes = picks["Stake_suggested"].sum()
    if sum_stakes > budget_max:
        scale = budget_max / sum_stakes
        picks["Stake_final"] = (picks["Stake_suggested"] * scale).round(2)
        picks["Stake_scale"] = round(scale, 3)
    else:
        picks["Stake_final"] = picks["Stake_suggested"]
        picks["Stake_scale"] = 1.0

    # Guardar todo + picks (ambos con edges/no-vig)
    all_pred_path = Path(f"output/predictions_{ts}.csv")
    picks_path    = Path(f"output/picks_{ts}.csv")
    out.to_csv(all_pred_path, index=False, encoding="utf-8")
    picks.to_csv(picks_path, index=False, encoding="utf-8")
    print(f"\n✓ Pronósticos completos: {all_pred_path}")
    print(f"✓ Picks seleccionados:   {picks_path}")

    print("\n=== Picks operativos (EV>0 & Edge>0, TOP-K con budget) ===")
    show_cols = [
        "Match","Best","Best_EV","Best_by_Edge","Best_Edge",
        "Prob_H","Prob_D","Prob_A",
        "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin",
        "Odds_H","Odds_D","Odds_A",
        "FairOdds_H_shin","FairOdds_D_shin","FairOdds_A_shin",
        "Stake_suggested","Stake_final","Stake_scale"
    ]
    print(picks[show_cols].to_string(index=False))


# --- Cell 21 ---
# ============================================================
# OUT ROBUSTO: mediana de bookmakers + saneado + normalización + λ regularizado + no-vig + edges + QC flags
# Requiere: hist, probs_HDA_dc, estimate_rho y tu config (WINDOW, DECAY_ALPHA, SEASON_DECAY, BANKROLL, KELLY_FRACTION, MAX_STAKE_PCT)
# ============================================================
import pandas as pd, numpy as np, unicodedata, re, math, requests
from pathlib import Path
from datetime import datetime

# ---------- Config local ----------
SPORT_KEY = globals().get("SPORT_KEY", "soccer_mexico_liga_mx")
REGION    = globals().get("REGION", "uk")
ODDS_API_KEY = globals().get("ODDS_API_KEY", "")
BANKROLL = globals().get("BANKROLL", 10000.0)
KELLY_FRACTION = globals().get("KELLY_FRACTION", 0.5)
MAX_STAKE_PCT = globals().get("MAX_STAKE_PCT", 0.03)

# ---------- Utilidades ----------
def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def normalize_team_name(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip().lower()
    s = _strip_accents(s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    TOKENS_DROP = {'fc','cf','club','de','la','el','los','las','cd','udg'}
    parts = [t for t in s.split() if t not in TOKENS_DROP]
    s = " ".join(parts)

    ALIASES = {
        "america": "club america", "club america": "club america",
        "tigres uanl":"tigres","uanl tigres":"tigres","tigres":"tigres",
        "pumas unam":"pumas","unam pumas":"pumas","pumas":"pumas",
        "club leon":"leon","leon":"leon",
        "queretaro":"queretaro",
        "atl san luis":"atletico san luis","san luis":"atletico san luis",
        "atletico san luis":"atletico san luis",
        "juarez":"juarez","mazatlan":"mazatlan","santos laguna":"santos laguna",
        "tijuana":"club tijuana","club tijuana":"club tijuana",
        "guadalajara":"chivas","chivas":"chivas",
        "pachuca":"pachuca","necaxa":"necaxa","toluca":"toluca",
        "monterrey":"monterrey","cruz azul":"cruz azul","puebla":"puebla","atlas":"atlas",
    }
    return ALIASES.get(s, s)

def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except AttributeError:
        try: ts = ts.tz_localize(None)
        except Exception: pass
    return ts

# ---------- 1) Agregador robusto de OddsAPI (mediana por mercado) ----------
def get_odds_oddsapi_median():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": "h2h", "oddsFormat": "decimal"}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    rows = []
    for ev in data:
        home, away = ev.get("home_team"), ev.get("away_team")
        commence = ev.get("commence_time") or ev.get("commenceTime")
        prices = []
        for b in ev.get("bookmakers", []):
            h2h = next((m for m in b.get("markets", []) if m.get("key")=="h2h"), None)
            if not h2h: continue
            ph = pd.NA; pdw = pd.NA; pa = pd.NA
            for o in h2h.get("outcomes", []):
                nm, pr = o.get("name"), o.get("price")
                if nm == home: ph = pr
                elif nm == "Draw": pdw = pr
                elif nm == away: pa = pr
            if pd.notna(ph) and pd.notna(pdw) and pd.notna(pa):
                prices.append([float(ph), float(pdw), float(pa)])
        if prices:
            arr = np.array(prices, dtype=float)
            med = np.median(arr, axis=0)
            rows.append({
                "home_team": home, "away_team": away,
                "odds_home_win": float(med[0]),
                "odds_draw": float(med[1]),
                "odds_away_win": float(med[2]),
                "book_count": len(prices),
                "commence_time": commence
            })
    df = pd.DataFrame(rows)
    if "commence_time" in df.columns:
        df["date"] = _to_naive_series(pd.to_datetime(df["commence_time"], utc=True))
    return df

# ---------- 2) Saneado de cuotas ----------
def implied_sum(row):
    return (1.0/row["odds_home_win"]) + (1.0/row["odds_draw"]) + (1.0/row["odds_away_win"])

def sanitize_df_matches(df, min_odds=1.20, max_odds=15.0, margin_range=(1.02, 1.14), min_books=3):
    df = df.copy()
    for c in ["odds_home_win","odds_draw","odds_away_win"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["book_count"] = pd.to_numeric(df.get("book_count", pd.Series(index=df.index, data=np.nan)), errors="coerce")
    df = df.dropna(subset=["odds_home_win","odds_draw","odds_away_win"])
    # Clip suave
    df["odds_home_win"] = df["odds_home_win"].clip(lower=min_odds, upper=max_odds)
    df["odds_draw"]     = df["odds_draw"].clip(lower=min_odds, upper=max_odds)
    df["odds_away_win"] = df["odds_away_win"].clip(lower=min_odds, upper=max_odds)
    # Margen implícito
    imp = df.apply(implied_sum, axis=1)
    df = df[(imp >= margin_range[0]) & (imp <= margin_range[1])]
    # Exige al menos N books
    if "book_count" in df.columns:
        df = df[df["book_count"] >= min_books]
    return df

# ---------- 3) Normalización + (opcional) fuzzy matching ----------
def try_fuzzy_map_names(df_matches_std, hist_std_set, threshold=90):
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        # sin rapidfuzz: devolvemos tal cual
        return {t: t for t in df_matches_std}
    mapping = {}
    for t in df_matches_std:
        if t in hist_std_set:
            mapping[t] = t
        else:
            cand = process.extractOne(t, list(hist_std_set), scorer=fuzz.WRatio)
            if cand and cand[1] >= threshold:
                mapping[t] = cand[0]
            else:
                mapping[t] = t
    return mapping

# ---------- 4) λ regularizado (cap + shrink a media de liga) ----------
def league_means(hist_df):
    h = hist_df["home_goals"].mean()
    a = hist_df["away_goals"].mean()
    # por si acaso
    h = 1.3 if pd.isna(h) else float(h)
    a = 1.1 if pd.isna(a) else float(a)
    return h, a

def rolling_lambda_asof_regularized(hist_df, team, side, as_of, cap=(0.4, 2.4), shrink=0.30):
    # --- setup y columnas canónicas ---
    as_of = _to_naive_ts(as_of)
    df = hist_df.copy()
    df["date"] = _to_naive_series(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    team_std = team  # aquí ya esperas nombres canónicos *_std
    home_col = "home_std" if "home_std" in df.columns else "home"
    away_col = "away_std" if "away_std" in df.columns else "away"

    # ¿usamos xG?
    use_xg = ("xG_home" in df.columns and "xG_away" in df.columns 
              and (df["xG_home"].notna().any() or df["xG_away"].notna().any()))
    if side == "home":
        value_col = "xG_home" if use_xg else "home_goals"
        mask_team = (df[home_col] == team_std)
    else:
        value_col = "xG_away" if use_xg else "away_goals"
        mask_team = (df[away_col] == team_std)

    # --- filtra hasta as_of, últimos WINDOW y quita NaN en valores ---
    WINDOW = globals().get("WINDOW", 10)
    DECAY_ALPHA = globals().get("DECAY_ALPHA", 0.003)
    SEASON_DECAY = globals().get("SEASON_DECAY", 1.0)

    tmp = df.loc[(df["date"] < as_of) & mask_team, ["date", "year", value_col]].tail(WINDOW).copy()
    tmp = tmp.dropna(subset=[value_col])

    if tmp.empty:
        # fallback suave si no hay histórico utilizable
        lam_raw = 1.3
    else:
        # --- pesos por decaimiento temporal y por temporada ---
        delta_days = (as_of - tmp["date"]).dt.days.clip(lower=0)
        w_time   = np.exp(-DECAY_ALPHA * delta_days.values.astype(float))
        w_season = (SEASON_DECAY ** (as_of.year - tmp["year"]).values.astype(float))
        w = (w_time * w_season).astype(float)

        vals = tmp[value_col].astype(float).values

        # Asegurar misma longitud (por si algo raro queda desalineado)
        if len(vals) != len(w):
            n = min(len(vals), len(w))
            vals = vals[-n:]
            w    = w[-n:]

        # Si los pesos se vuelven degenerados, usa uniformes
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones_like(vals, dtype=float)

        lam_raw = float(np.average(vals, weights=w))

    # --- regularización: cap + shrink hacia la media de liga ---
    mean_h = float(df["home_goals"].mean()) if df["home_goals"].notna().any() else 1.3
    mean_a = float(df["away_goals"].mean()) if df["away_goals"].notna().any() else 1.1
    prior  = mean_h if side == "home" else mean_a

    lam_raw = max(min(lam_raw, cap[1]), cap[0])  # cap
    lam = (1.0 - shrink) * lam_raw + shrink * prior
    return max(lam, 0.2)

# ---------- 5) No-vig, EV, Kelly ----------
def fair_probs_proportional(odds_triplet):
    r = np.array([1.0/odds_triplet[0], 1.0/odds_triplet[1], 1.0/odds_triplet[2]], dtype=float)
    return r / r.sum()

def fair_probs_shin(odds_triplet, max_s=0.25, iters=60):
    r = np.array([1.0/odds_triplet[0], 1.0/odds_triplet[1], 1.0/odds_triplet[2]], dtype=float)
    q = r / r.sum()
    def f(s): return (q / (s + (1.0 - s)*q)).sum() - 1.0
    lo, hi = 0.0, max_s
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    s_hat = 0.5*(lo+hi)
    p = q / (s_hat + (1.0 - s_hat)*q)
    p = p / p.sum()
    return p, float(s_hat)

def expected_value(prob, odds): return prob*odds - 1.0
def kelly_fraction_decimal(prob, odds):
    b = max(odds - 1.0, 0.0); q = 1.0 - prob
    if b <= 0: return 0.0
    return max((b*prob - q)/b, 0.0)

# ---------- 6) Preparar histórico canónico ----------
hist = hist.copy()
hist["date"] = _to_naive_series(hist["date"])
hist["home_std"] = hist["home"].apply(normalize_team_name)
hist["away_std"] = hist["away"].apply(normalize_team_name)
if "year" not in hist.columns:
    hist["year"] = hist["date"].dt.year

# ---------- 7) Traer odds (mediana) + saneado ----------
assert ODDS_API_KEY and ODDS_API_KEY.strip(), "Falta ODDS_API_KEY"
dfm = get_odds_oddsapi_median()
dfm["home_team_std"] = dfm["home_team"].apply(normalize_team_name)
dfm["away_team_std"] = dfm["away_team"].apply(normalize_team_name)
dfm["date"] = _to_naive_series(dfm["date"])
dfm = sanitize_df_matches(dfm, min_odds=1.20, max_odds=15.0, margin_range=(1.02, 1.14), min_books=3)
if dfm.empty:
    raise RuntimeError("Tras saneado no quedaron partidos. Relaja filtros o revisa SPORT_KEY/REGION.")

# ---------- 8) Fuzzy mapping (si hace falta) ----------
hist_names = set(pd.concat([hist["home_std"], hist["away_std"]]).unique())
mapping_home = try_fuzzy_map_names(dfm["home_team_std"].unique(), hist_names, threshold=90)
mapping_away = try_fuzzy_map_names(dfm["away_team_std"].unique(), hist_names, threshold=90)
dfm["home_team_std"] = dfm["home_team_std"].map(mapping_home)
dfm["away_team_std"] = dfm["away_team_std"].map(mapping_away)

# ---------- 9) Estimar rho (si no existe) ----------
try:
    rho_hat
except NameError:
    rho_hat = estimate_rho(hist, sample_size=800)

# ---------- 10) LOOP principal con λ regularizado ----------
rows = []
for _, m in dfm.reset_index(drop=True).iterrows():
    as_of = _to_naive_ts(m.get("date", pd.Timestamp.now()))
    AS_OF = pd.Timestamp.now(tz="America/Mexico_City").tz_convert(None)
    home_std = m["home_team_std"]; away_std = m["away_team_std"]

    #lh = rolling_lambda_asof_regularized(hist, home_std, "home", as_of, cap=(0.4, 2.4), shrink=0.30)
    #la = rolling_lambda_asof_regularized(hist, away_std, "away", as_of, cap=(0.4, 2.4), shrink=0.30)
    pH, pD, pA = probs_HDA_dc(lh, la, rho_hat, max_goals=10)

    oH, oD, oA = float(m["odds_home_win"]), float(m["odds_draw"]), float(m["odds_away_win"])
    evH, evD, evA = expected_value(pH, oH), expected_value(pD, oD), expected_value(pA, oA)

    p_prop = fair_probs_proportional([oH, oD, oA])
    p_shin, s_hat = fair_probs_shin([oH, oD, oA])

    edge_H, edge_D, edge_A = pH - p_shin[0], pD - p_shin[1], pA - p_shin[2]
    best_edge_name = np.array(["Home","Draw","Away"])[np.argmax([edge_H, edge_D, edge_A])]
    best_edge_val  = float(np.max([edge_H, edge_D, edge_A]))

    cand = {"Home": (pH, oH, evH), "Draw": (pD, oD, evD), "Away": (pA, oA, evA)}
    best_name, (p_best, o_best, ev_best) = max(cand.items(), key=lambda kv: kv[1][2])
    k_full = kelly_fraction_decimal(p_best, o_best)
    k_frac = KELLY_FRACTION * k_full
    k_cap  = min(k_frac, MAX_STAKE_PCT)
    stake  = round(BANKROLL * k_cap, 2)

    # QC flags
    imp_sum = (1.0/oH + 1.0/oD + 1.0/oA)
    qc = {
        "QC_shin_high": (s_hat > 0.18),
        "QC_low_books": (m.get("book_count", 0) < 3),
        "QC_margin_off": not (1.02 <= imp_sum <= 1.14),
        "QC_lambda_fallback": (abs(lh-1.3)<1e-6 or abs(la-1.3)<1e-6),
    }

    rows.append({
        "Match": f"{m['home_team']} vs {m['away_team']}",
        "as_of": as_of.strftime("%Y-%m-%d"),
        "λ_home": round(lh,3), "λ_away": round(la,3),
        "Prob_H": round(pH,4), "Prob_D": round(pD,4), "Prob_A": round(pA,4),
        "Odds_H": round(oH,3), "Odds_D": round(oD,3), "Odds_A": round(oA,3),
        "EV_H": round(evH,3), "EV_D": round(evD,3), "EV_A": round(evA,3),
        "Best": best_name, "Best_EV": round(ev_best,3),
        "Kelly_full": round(k_full,3), "Kelly_frac": round(k_frac,3),
        "Stake_cap": round(k_cap,3), "Stake_suggested": stake,
        # no-vig (prop)
        "Mkt_noVig_H_prop": round(p_prop[0],4),
        "Mkt_noVig_D_prop": round(p_prop[1],4),
        "Mkt_noVig_A_prop": round(p_prop[2],4),
        "FairOdds_H_prop": round(1.0/p_prop[0],3),
        "FairOdds_D_prop": round(1.0/p_prop[1],3),
        "FairOdds_A_prop": round(1.0/p_prop[2],3),
        # no-vig (Shin)
        "Mkt_noVig_H_shin": round(p_shin[0],4),
        "Mkt_noVig_D_shin": round(p_shin[1],4),
        "Mkt_noVig_A_shin": round(p_shin[2],4),
        "FairOdds_H_shin": round(1.0/p_shin[0],3),
        "FairOdds_D_shin": round(1.0/p_shin[1],3),
        "FairOdds_A_shin": round(1.0/p_shin[2],3),
        "Shin_s": round(s_hat,4),
        "Best_by_Edge": best_edge_name, "Best_Edge": round(best_edge_val,4),
        # flags
        **qc
    })

out = pd.DataFrame(rows).sort_values(["Best_Edge","Best_EV"], ascending=False).reset_index(drop=True)
print("✅ OUT robusto (mediana por book + saneado + nombres canónicos + λ regularizado).")
print(out.head(10).to_string(index=False))

# ---------- Picks con filtros de calidad ----------
TOP_K = 5
MAX_TOTAL_RISK_PCT = globals().get("MAX_TOTAL_RISK_PCT", 0.10)
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

picks = out[(out["Best_EV"]>0) & (out["Best_Edge"]>0)].copy()
# descarta eventos con señales de mala calidad
picks = picks[~(picks["QC_shin_high"] | picks["QC_margin_off"] | picks["QC_lambda_fallback"])]

picks = picks.sort_values(["Best_Edge","Best_EV"], ascending=False).head(TOP_K).reset_index(drop=True)

if not picks.empty:
    budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
    sum_stakes = picks["Stake_suggested"].sum()
    if sum_stakes > budget_max:
        scale = budget_max / sum_stakes
        picks["Stake_final"] = (picks["Stake_suggested"] * scale).round(2)
        picks["Stake_scale"] = round(scale, 3)
    else:
        picks["Stake_final"] = picks["Stake_suggested"]
        picks["Stake_scale"] = 1.0

    all_pred_path = Path(f"output/predictions_{ts}.csv")
    picks_path    = Path(f"output/picks_{ts}.csv")
    out.to_csv(all_pred_path, index=False, encoding="utf-8")
    picks.to_csv(picks_path, index=False, encoding="utf-8")
    print(f"✓ Pronósticos completos: {all_pred_path}")
    print(f"✓ Picks seleccionados:   {picks_path}")

    show_cols = ["Match","Best","Best_EV","Best_by_Edge","Best_Edge",
                 "Prob_H","Prob_D","Prob_A",
                 "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin","Shin_s",
                 "Odds_H","Odds_D","Odds_A",
                 "FairOdds_H_shin","FairOdds_D_shin","FairOdds_A_shin",
                 "Stake_suggested","Stake_final","Stake_scale",
                 "QC_shin_high","QC_margin_off","QC_lambda_fallback"]
    print("\n=== Picks (filtros de calidad aplicados) ===")
    print(picks[show_cols].to_string(index=False))
else:
    print("No hay picks con EV>0 & Edge>0 tras filtros de calidad.")


# --- Cell 22 ---


# 1) Elige prob. de mercado sin margen por evento (híbrido)
def choose_mkt_novig_row(row, s_cut=0.22):
    # si Shin no satura (s < s_cut), usa Shin; si no, proporcional
    if float(row.get("Shin_s", 0.25)) < s_cut:
        return np.array([row["Mkt_noVig_H_shin"], row["Mkt_noVig_D_shin"], row["Mkt_noVig_A_shin"]], dtype=float), "shin"
    else:
        return np.array([row["Mkt_noVig_H_prop"], row["Mkt_noVig_D_prop"], row["Mkt_noVig_A_prop"]], dtype=float), "prop"

# 2) Recalcula edges vs mercado híbrido y aplica penalización por desacuerdo para el stake
def rebuild_edges_and_stakes(out_df, delta_cut=0.18, penalty=True):
    df = out_df.copy()
    edges = []; best_edge = []; best_name = []; mkt_method=[]; deltas=[]
    stake_final = []
    for _, r in df.iterrows():
        p_mod = np.array([r["Prob_H"], r["Prob_D"], r["Prob_A"]], dtype=float)
        p_mkt, mth = choose_mkt_novig_row(r, s_cut=0.22)
        mkt_method.append(mth)
        e = p_mod - p_mkt
        edges.append(e)
        idx = int(np.argmax(e))
        best_edge.append(float(e[idx]))
        best_name.append(["Home","Draw","Away"][idx])
        deltas.append(float(np.max(np.abs(e))))

        # penalización del stake por desacuerdo grande
        stake = float(r.get("Stake_suggested", 0.0))
        if penalty and deltas[-1] > delta_cut:
            scale = float(np.exp(- (deltas[-1]/delta_cut)**2))
            stake *= scale
        stake_final.append(round(stake, 2))

    E = np.vstack(edges)
    df["Edge_H_hybrid"] = np.round(E[:,0],4)
    df["Edge_D_hybrid"] = np.round(E[:,1],4)
    df["Edge_A_hybrid"] = np.round(E[:,2],4)
    df["Best_by_Edge_hybrid"] = best_name
    df["Best_Edge_hybrid"] = np.round(best_edge,4)
    df["Mkt_noVig_method"] = mkt_method
    df["Disagreement_max"] = np.round(deltas,4)
    df["Stake_final_pen"] = stake_final
    return df

# 3) Aplica reconstrucción y vuelve a seleccionar picks
out_h = rebuild_edges_and_stakes(out, delta_cut=0.18, penalty=True)

# Filtros QC: mantenemos márgen y libros y lambda_fallback; NO bloqueamos por Shin alto (solo lo marcamos)
qc_block = (out_h["QC_margin_off"] | out_h["QC_low_books"] | out_h["QC_lambda_fallback"])
candidates = out_h[~qc_block].copy()

# Reglas de selección:
#   - EV>0 en el "Best" original (sobre cuota ofertada),
#   - y Edge_hybrid>0 (modelo > mercado no-vig híbrido) en el mismo resultado.
mask_ev = candidates["Best_EV"] > 0
mask_edge = (
    ((candidates["Best"]=="Home") & (candidates["Edge_H_hybrid"]>0)) |
    ((candidates["Best"]=="Draw") & (candidates["Edge_D_hybrid"]>0)) |
    ((candidates["Best"]=="Away") & (candidates["Edge_A_hybrid"]>0))
)
picks = candidates[mask_ev & mask_edge].copy()

# Ordena por Edge híbrido y EV; TOP_K con presupuesto
TOP_K = 5
picks = picks.sort_values(["Best_Edge_hybrid","Best_EV"], ascending=False).head(TOP_K).reset_index(drop=True)

# re-cap de banca total (si lo usabas antes)
BANKROLL = globals().get("BANKROLL", 10000.0)
MAX_TOTAL_RISK_PCT = globals().get("MAX_TOTAL_RISK_PCT", 0.10)
budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
sum_stakes = picks["Stake_final_pen"].sum() if not picks.empty else 0.0
if sum_stakes > 0 and sum_stakes > budget_max:
    scale = budget_max / sum_stakes
    picks["Stake_final_pen"] = (picks["Stake_final_pen"] * scale).round(2)
    picks["Stake_scale_pen"] = round(scale, 3)
else:
    picks["Stake_scale_pen"] = 1.0

# === Mostrar tabla híbrida (OUT) sin Stake_scale_pen ===
display_cols_out = [
    "Match","as_of","Best","Best_EV",
    "Mkt_noVig_method","Disagreement_max",
    "Prob_H","Prob_D","Prob_A",
    "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin","Shin_s",
    "Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop",
    "Edge_H_hybrid","Edge_D_hybrid","Edge_A_hybrid",
    "Best_by_Edge_hybrid","Best_Edge_hybrid",
    "Odds_H","Odds_D","Odds_A",
    "Stake_suggested",
    "QC_shin_high","QC_low_books","QC_margin_off","QC_lambda_fallback",
]
cols_out = [c for c in display_cols_out if c in out_h.columns]
print("\n=== Tabla con mercado HÍBRIDO + penalización por desacuerdo ===")
print(out_h[cols_out].to_string(index=False))

# === Guardado ===
from pathlib import Path
from datetime import datetime
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_h.to_csv(f"output/predictions_hybrid_{ts}.csv", index=False, encoding="utf-8")

# === Mostrar/guardar picks (aquí sí existe Stake_scale_pen) ===
if not picks.empty:
    picks_cols = [
        "Match","as_of","Best","Best_EV",
        "Mkt_noVig_method","Disagreement_max",
        "Prob_H","Prob_D","Prob_A",
        "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin","Shin_s",
        "Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop",
        "Edge_H_hybrid","Edge_D_hybrid","Edge_A_hybrid","Best_by_Edge_hybrid","Best_Edge_hybrid",
        "Odds_H","Odds_D","Odds_A",
        "Stake_suggested","Stake_final_pen","Stake_scale_pen"
    ]
    picks_cols = [c for c in picks_cols if c in picks.columns]
    print("\n=== Picks (híbrido, penalización aplicada) ===")
    print(picks[picks_cols].to_string(index=False))
    picks.to_csv(f"output/picks_hybrid_{ts}.csv", index=False, encoding="utf-8")
    print(f"\n✓ Picks híbridos guardados: output/picks_hybrid_{ts}.csv")
else:
    print("\n(No hay picks tras aplicar reglas híbridas + penalización)")


# --- Cell 23 ---
# ================== A) Agregador con dispersión + saneado estricto ==================
import numpy as np, pandas as pd, requests

SPORT_KEY = globals().get("SPORT_KEY", "soccer_mexico_liga_mx")
REGION    = globals().get("REGION", "uk")
ODDS_API_KEY = globals().get("ODDS_API_KEY", "")

def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
    return s

def _implied_sum(row):
    return (1/row["odds_home_win"]) + (1/row["odds_draw"]) + (1/row["odds_away_win"])

def get_odds_oddsapi_median_dispersion():
    """
    Devuelve mediana por mercado y medidas de dispersión (p5, p95 y dispersión relativa).
    Columnas clave:
      - odds_* (medianas)
      - book_count
      - disp_*_rel (p95/p5 - 1)
      - disp_max_rel (max de las 3)
      - date
    """
    assert ODDS_API_KEY, "Falta ODDS_API_KEY"
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": "h2h", "oddsFormat": "decimal"}
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    data = r.json()

    rows = []
    for ev in data:
        home, away = ev.get("home_team"), ev.get("away_team")
        commence = ev.get("commence_time") or ev.get("commenceTime")
        tri = []
        for b in ev.get("bookmakers", []):
            h2h = next((m for m in b.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h: continue
            ph = pd.NA; pdw = pd.NA; pa = pd.NA
            for o in h2h.get("outcomes", []):
                nm, pr = o.get("name"), o.get("price")
                if nm == home: ph = pr
                elif nm == "Draw": pdw = pr
                elif nm == away: pa = pr
            if pd.notna(ph) and pd.notna(pdw) and pd.notna(pa):
                tri.append([float(ph), float(pdw), float(pa)])
        if not tri: 
            continue
        arr = np.array(tri, dtype=float)
        med = np.median(arr, axis=0)
        p5  = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr,95, axis=0)
        disp_rel = (p95 / np.maximum(p5, 1e-9)) - 1.0

        rows.append({
            "home_team": home, "away_team": away,
            "odds_home_win": float(med[0]),
            "odds_draw": float(med[1]),
            "odds_away_win": float(med[2]),
            "p5_home": float(p5[0]), "p5_draw": float(p5[1]), "p5_away": float(p5[2]),
            "p95_home": float(p95[0]), "p95_draw": float(p95[1]), "p95_away": float(p95[2]),
            "disp_H_rel": float(disp_rel[0]), "disp_D_rel": float(disp_rel[1]), "disp_A_rel": float(disp_rel[2]),
            "disp_max_rel": float(np.max(disp_rel)),
            "book_count": int(len(arr)),
            "date": commence
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = _to_naive_series(df["date"])
    return df

def sanitize_df_matches_plus(df,
                             min_odds=1.20, max_odds=15.0,
                             margin_range=(1.02, 1.12),
                             min_books=5,
                             disp_thresh=0.12,
                             disp_thresh_heavy_fav=0.08,
                             heavy_fav_cut=1.50):
    """ 
    Saneado estricto:
      - clip cuotras [min_odds, max_odds]
      - margen implícito en [1.02, 1.12]
      - al menos min_books casas
      - dispersión relativa ≤ disp_thresh (p95/p5 - 1); si hay cuota ≤ heavy_fav_cut en algún lado, exige disp ≤ disp_thresh_heavy_fav
    """
    if df.empty: 
        return df
    df = df.copy()
    for c in ["odds_home_win","odds_draw","odds_away_win"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=min_odds, upper=max_odds)
    df["book_count"] = pd.to_numeric(df["book_count"], errors="coerce")

    # margen
    imp = df.apply(_implied_sum, axis=1)
    ok_margin = (imp >= margin_range[0]) & (imp <= margin_range[1])

    # libros
    ok_books = df["book_count"] >= min_books

    # dispersión
    ok_disp = (df["disp_H_rel"] <= disp_thresh) & (df["disp_D_rel"] <= disp_thresh) & (df["disp_A_rel"] <= disp_thresh)

    # más estricto si hay favorito muy pesado en algún lado
    heavy_fav = (df[["odds_home_win","odds_draw","odds_away_win"]].min(axis=1) <= heavy_fav_cut)
    ok_disp_heavy = (~heavy_fav) | (
        (df["disp_H_rel"] <= disp_thresh_heavy_fav) &
        (df["disp_D_rel"] <= disp_thresh_heavy_fav) &
        (df["disp_A_rel"] <= disp_thresh_heavy_fav)
    )

    df = df[ ok_margin & ok_books & ok_disp & ok_disp_heavy ].copy()
    return df

# === úsalo en tu pipeline en lugar del antiguo agregador/saneador:
dfm_raw = get_odds_oddsapi_median_dispersion()
dfm = sanitize_df_matches_plus(dfm_raw, min_books=5, margin_range=(1.02,1.12))
print(f"Odds saneadas: {dfm.shape[0]} partidos (de {dfm_raw.shape[0] if dfm_raw is not None else 0})")


# --- Cell 24 ---
# ================== B) Selección final de picks endurecida ==================
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

# -- utilidades híbrido (por si no existen en tu sesión) --
def choose_mkt_novig_row(row, s_cut=0.22):
    if float(row.get("Shin_s", 0.25)) < s_cut:
        return np.array([row["Mkt_noVig_H_shin"], row["Mkt_noVig_D_shin"], row["Mkt_noVig_A_shin"]], dtype=float), "shin"
    else:
        return np.array([row["Mkt_noVig_H_prop"], row["Mkt_noVig_D_prop"], row["Mkt_noVig_A_prop"]], dtype=float), "prop"

def rebuild_edges_and_stakes(out_df, delta_cut=0.18, penalty=True):
    df = out_df.copy()
    edges = []; best_edge = []; best_name = []; mkt_method=[]; deltas=[]
    stake_final = []
    for _, r in df.iterrows():
        p_mod = np.array([r["Prob_H"], r["Prob_D"], r["Prob_A"]], dtype=float)
        p_mkt, mth = choose_mkt_novig_row(r, s_cut=0.22)
        mkt_method.append(mth)
        e = p_mod - p_mkt
        edges.append(e)
        idx = int(np.argmax(e))
        best_edge.append(float(e[idx]))
        best_name.append(["Home","Draw","Away"][idx])
        deltas.append(float(np.max(np.abs(e))))
        stake_final.append(float(r.get("Stake_suggested", 0.0)))  # penalización ya aplicada en tu paso previo

    E = np.vstack(edges)
    df["Edge_H_hybrid"] = np.round(E[:,0],4)
    df["Edge_D_hybrid"] = np.round(E[:,1],4)
    df["Edge_A_hybrid"] = np.round(E[:,2],4)
    df["Best_by_Edge_hybrid"] = best_name
    df["Best_Edge_hybrid"] = np.round(best_edge,4)
    df["Mkt_noVig_method"] = mkt_method
    df["Disagreement_max"] = np.round(deltas,4)
    df["Stake_final_pen"] = np.round(stake_final, 2)
    return df

# -- si no existe out_h, lo construyo a partir de out --
if 'out_h' not in globals():
    out_h = rebuild_edges_and_stakes(out, delta_cut=0.18, penalty=True)

# === reglas endurecidas ===
BANKROLL = globals().get("BANKROLL", 10000.0)
MAX_TOTAL_RISK_PCT = globals().get("MAX_TOTAL_RISK_PCT", 0.10)

EDGE_MIN = 0.06          # edge mínimo vs mercado no-vig híbrido
DISAGREE_MAX = 0.18      # desacuerdo máximo permitido
MARGIN_OK = (1.02, 1.12) # ya filtrado en A), lo rechecamos por si acaso
BOOKS_MIN = 5            # mínimo de casas
PICK_CAP = 0.015         # 1.5% bank por pick (cap duro de ejecución)

def implied_sum(row): 
    return (1/row["Odds_H"]) + (1/row["Odds_D"]) + (1/row["Odds_A"])

cand = out_h.copy()
# filtros QC “duros”
qc_block = (cand["QC_margin_off"] | cand["QC_low_books"] | cand["QC_lambda_fallback"])
cand = cand[~qc_block].copy()

# margen rechecado
imp = cand.apply(implied_sum, axis=1)
cand = cand[(imp >= MARGIN_OK[0]) & (imp <= MARGIN_OK[1])]
# libros (si la columna existe)
if "book_count" in cand.columns:
    cand = cand[cand["book_count"] >= BOOKS_MIN]

# condición EV>0 del best y edge híbrido > EDGE_MIN, con desacuerdo limitado
mask_ev = cand["Best_EV"] > 0
mask_edge = (
    ((cand["Best"]=="Home") & (cand["Edge_H_hybrid"]>EDGE_MIN)) |
    ((cand["Best"]=="Draw") & (cand["Edge_D_hybrid"]>EDGE_MIN)) |
    ((cand["Best"]=="Away") & (cand["Edge_A_hybrid"]>EDGE_MIN))
)
mask_dis = cand["Disagreement_max"] <= DISAGREE_MAX

picks_hardened = cand[mask_ev & mask_edge & mask_dis].copy()

# aplica cap 1.5% bank a stake final
picks_hardened["Stake_final_exec"] = np.minimum(
    picks_hardened.get("Stake_final_pen", picks_hardened.get("Stake_suggested", 0.0)) / BANKROLL,
    PICK_CAP
) * BANKROLL
picks_hardened["Stake_final_exec"] = picks_hardened["Stake_final_exec"].round(2)

# presupuesto total
budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
sum_stakes = picks_hardened["Stake_final_exec"].sum() if not picks_hardened.empty else 0.0
if sum_stakes > budget_max > 0:
    scale = budget_max / sum_stakes
    picks_hardened["Stake_final_exec"] = (picks_hardened["Stake_final_exec"] * scale).round(2)
    picks_hardened["Exec_scale"] = round(scale, 3)
else:
    picks_hardened["Exec_scale"] = 1.0

# salida
show_cols = [
    "Match","as_of","Best","Best_EV","Best_by_Edge_hybrid","Best_Edge_hybrid",
    "Mkt_noVig_method","Disagreement_max",
    "Prob_H","Prob_D","Prob_A",
    "Odds_H","Odds_D","Odds_A",
    "Stake_suggested","Stake_final_pen","Stake_final_exec","Exec_scale"
]
print("\n=== PICKS (reglas endurecidas) ===")
print(picks_hardened[[c for c in show_cols if c in picks_hardened.columns]].to_string(index=False))

# guardado
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
picks_hardened.to_csv(f"output/picks_hardened_{ts}.csv", index=False, encoding="utf-8")
print(f"\n✓ Picks endurecidos guardados: output/picks_hardened_{ts}.csv")


# --- Cell 25 ---
AS_OF = pd.Timestamp.now(tz="America/Mexico_City").tz_convert(None)

# --- Cell 26 ---
import numpy as np, pandas as pd
from scipy.optimize import minimize
from datetime import datetime

# ---- utils de tiempo (usa las que ya tengas si existen) ----
def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except AttributeError:
        try: ts = ts.tz_localize(None)
        except Exception: pass
    return ts

# ---- Prob H/D/A con Dixon–Coles (reusa tu probs_HDA_dc si ya existe) ----
from math import exp
from scipy.stats import poisson

def probs_HDA_dc(lh, la, rho, max_goals=10):
    # matriz independiente
    home = np.arange(0, max_goals+1)
    away = np.arange(0, max_goals+1)
    P = np.outer(poisson.pmf(home, lh), poisson.pmf(away, la))

    # ajuste DC en celdas bajas (0–1 goles)
    # (ver paper Dixon–Coles 1997)
    P_adj = P.copy()
    # parejas (0,0), (0,1), (1,0), (1,1)
    P_adj[0,0] *= (1 - (lh*la*rho))
    if max_goals >= 1:
        P_adj[0,1] *= (1 + rho)
        P_adj[1,0] *= (1 + rho)
        P_adj[1,1] *= (1 - rho)

    # re-normaliza suavemente
    s = P_adj.sum()
    if s <= 0: 
        P_adj = P
        s = P_adj.sum()
    P_adj /= s

    pH = np.tril(P_adj, -1).sum()
    pD = np.trace(P_adj)
    pA = np.triu(P_adj, 1).sum()
    return float(pH), float(pD), float(pA)

# ---- Modelo DC de fuerzas (ataque/defensa + HFA) ----
class DCForcesModel:
    def __init__(self, teams, att, deff, hfa, use_xg, decay_alpha, season_decay, ridge):
        self.teams = teams
        self.t2i = {t:i for i,t in enumerate(teams)}
        self.att = att  # tamaño n
        self.deff = deff
        self.hfa = hfa
        self.use_xg = use_xg
        self.decay_alpha = decay_alpha
        self.season_decay = season_decay
        self.ridge = ridge

    def lam(self, hist_df, home_team, away_team, as_of):
        """Devuelve (λ_home, λ_away) as-of con fuerzas actuales."""
        as_of = _to_naive_ts(as_of)
        h = self.t2i.get(home_team)
        a = self.t2i.get(away_team)
        if h is None or a is None:
            # si aparece un equipo nuevo, usa medias de liga
            mean_h = float(hist_df["home_goals"].mean()) if hist_df["home_goals"].notna().any() else 1.3
            mean_a = float(hist_df["away_goals"].mean()) if hist_df["away_goals"].notna().any() else 1.1
            return max(mean_h,0.2), max(mean_a,0.2)
        mu_h = np.exp(self.hfa + self.att[h] - self.deff[a])
        mu_a = np.exp(self.att[a] - self.deff[h])
        # cap suave por robustez
        return float(np.clip(mu_h, 0.2, 3.5)), float(np.clip(mu_a, 0.2, 3.5))

def fit_dc_forces(hist_df, as_of,
                  use_xg=True, decay_alpha=0.003, season_decay=0.85,
                  ridge=1.0, maxiter=800):
    """
    Ajusta ataque/defensa + HFA con MLE (Poisson) + Ridge y pesos por tiempo/temporada.
    Identificación: sum(att)=0 y sum(def)=0 (reparametrización).
    """
    df = hist_df.copy()
    df["date"] = _to_naive_series(df["date"])
    as_of = _to_naive_ts(as_of)
    df = df[df["date"] < as_of]

    # métrica de goles
    use_xg_ok = use_xg and ("xG_home" in df.columns) and ("xG_away" in df.columns) and \
                (df["xG_home"].notna().any() or df["xG_away"].notna().any())
    yh = (df["xG_home"] if use_xg_ok else df["home_goals"]).astype(float)
    ya = (df["xG_away"] if use_xg_ok else df["away_goals"]).astype(float)
    dfe = df[["home_std","away_std","date"]].copy()
    dfe = dfe[(~yh.isna()) & (~ya.isna())].copy()
    yh = yh.loc[dfe.index].values
    ya = ya.loc[dfe.index].values

    teams = sorted(pd.unique(pd.concat([dfe["home_std"], dfe["away_std"]])))
    n = len(teams)
    t2i = {t:i for i,t in enumerate(teams)}

    # índices
    hi = dfe["home_std"].map(t2i).values
    ai = dfe["away_std"].map(t2i).values

    # pesos por tiempo y temporada
    delta_days = (as_of - dfe["date"]).dt.days.clip(lower=0).values.astype(float)
    w_time   = np.exp(-decay_alpha * delta_days)
    years    = pd.to_datetime(dfe["date"]).dt.year.values
    w_season = np.power(season_decay, (as_of.year - years).astype(float))
    w = (w_time * w_season).astype(float)

    # Parámetros: att[0..n-2], def[0..n-2], hfa  (el último es dependiente para sum=0)
    p0 = np.zeros(2*(n-1) + 1, dtype=float)

    def unpack_params(p):
        att_free = p[:(n-1)]
        def_free = p[(n-1):2*(n-1)]
        hfa      = p[-1]
        att = np.concatenate([att_free, [-att_free.sum()]])   # sum(att)=0
        deff = np.concatenate([def_free, [-def_free.sum()]])  # sum(def)=0
        return att, deff, hfa

    def nll(p):
        att, deff, hfa = unpack_params(p)
        log_mu_h = hfa + att[hi] - deff[ai]
        log_mu_a =       att[ai] - deff[hi]
        mu_h = np.exp(np.clip(log_mu_h, -7, 7))
        mu_a = np.exp(np.clip(log_mu_a, -7, 7))
        # Poisson NLL sin constantes: mu - y*log(mu)
        ll = w * (mu_h - yh*np.log(mu_h + 1e-12) + mu_a - ya*np.log(mu_a + 1e-12))
        # regularización ridge
        pen = ridge*(np.sum(att**2) + np.sum(deff**2) + hfa**2)
        return float(ll.sum() + pen)

    res = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": maxiter})
    att, deff, hfa = unpack_params(res.x)
    return DCForcesModel(teams, att, deff, hfa, use_xg_ok, decay_alpha, season_decay, ridge)

# ---- Helper: pronóstico con DC-fuerzas + rho ----
def predict_match_dc_forces(model, hist_df, home_std, away_std, as_of, rho_hat, max_goals=10):
    lh, la = model.lam(hist_df, home_std, away_std, as_of)
    pH, pD, pA = probs_HDA_dc(lh, la, rho_hat, max_goals=max_goals)
    return lh, la, pH, pD, pA


# --- Cell 27 ---
dc_model = fit_dc_forces(hist, AS_OF, use_xg=True, decay_alpha=0.003, season_decay=0.85, ridge=1.0)

# --- Cell 28 ---
lh, la = dc_model.lam(hist, home_std, away_std, as_of)  

# --- Cell 29 ---
# ============================================
# OUT_DC + PICKS_DC con modelo Dixon–Coles de fuerzas
# ============================================
import numpy as np, pandas as pd, re, unicodedata
from datetime import datetime

# ---------- Utiles de tiempo ----------
def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
    return s
def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        ts = ts.tz_localize(None)
    except Exception:
        pass
    return ts

# ---------- Normalización de nombres ----------
STOPWORDS = {"FC","CF","CLUB","DE","DEL","DEPORTIVO","ATL","ATLETICO","ATLÉTICO","CD","UD","SC","AC"}
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
def canon(s):
    s = str(s).upper().strip()
    s = _strip_accents(s)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(toks)

def build_alias_map_from_hist(hist_df):
    canons = {}
    for col in ["home_std","away_std","home","away"]:
        if col in hist_df.columns:
            for t in hist_df[col].dropna().unique():
                canons[canon(t)] = t  # se queda el último; está bien
    return canons

def map_team_to_hist(name, alias_map):
    key = canon(name)
    return alias_map.get(key, name)

# ---------- Mercado no-vig ----------
def fair_probs_proportional(odds_tuple):
    oH, oD, oA = map(float, odds_tuple)
    raw = np.array([1/oH, 1/oD, 1/oA], dtype=float)
    s = raw.sum()
    if s <= 0: 
        return np.array([1/3,1/3,1/3], dtype=float)
    return raw / s

# Para simplificar, usaremos siempre 'prop' como método híbrido por defecto,
# y dejaremos columnas de Shin duplicadas (con s=0.25) para mantener layout.
def compute_market_probs(odds_tuple):
    p_prop = fair_probs_proportional(odds_tuple)
    shin_s = 0.25
    p_shin = p_prop.copy()
    return p_prop, p_shin, shin_s, "prop"

# ---------- Kelly ----------
def kelly_fraction(p, o):
    # fracción óptima (si negativa => 0)
    b = o - 1.0
    f = (p*o - (1-p)) / b if b > 0 else 0.0
    return max(float(f), 0.0)

# ---------- Prob H/D/A con DC (usa la que ya definiste) ----------
try:
    probs_HDA_dc
except NameError:
    from scipy.stats import poisson
    def probs_HDA_dc(lh, la, rho, max_goals=10):
        home = np.arange(0, max_goals+1)
        away = np.arange(0, max_goals+1)
        P = np.outer(poisson.pmf(home, lh), poisson.pmf(away, la))
        P_adj = P.copy()
        P_adj[0,0] *= (1 - (lh*la*rho))
        if max_goals >= 1:
            P_adj[0,1] *= (1 + rho)
            P_adj[1,0] *= (1 + rho)
            P_adj[1,1] *= (1 - rho)
        s = P_adj.sum()
        if s <= 0: P_adj = P; s = P_adj.sum()
        P_adj /= s
        pH = np.tril(P_adj, -1).sum()
        pD = np.trace(P_adj)
        pA = np.triu(P_adj, 1).sum()
        return float(pH), float(pD), float(pA)

# ---------- Chequeos previos ----------
if 'dc_model' not in globals():
    raise RuntimeError("Falta dc_model (ajústalo con fit_dc_forces(hist, AS_OF, ...))")
if 'rho_hat' not in globals():
    raise RuntimeError("Falta rho_hat (estimado DC)")

# Toma df de cuotas
if 'dfm' in globals() and isinstance(dfm, pd.DataFrame) and not dfm.empty:
    df_matches = dfm.copy()
elif 'df_matches' in globals() and isinstance(df_matches, pd.DataFrame) and not df_matches.empty:
    df_matches = df_matches.copy()
else:
    raise RuntimeError("No encuentro dfm / df_matches con cuotas.")

# Asegura columnas clave
need_cols = {"home_team","away_team","odds_home_win","odds_draw","odds_away_win"}
missing = [c for c in need_cols if c not in df_matches.columns]
if missing:
    raise RuntimeError(f"df_matches le faltan columnas: {missing}")

# Fecha as_of del evento (si existe), sino ahora MX
try:
    tz_mx = "America/Mexico_City"
    NOW_MX = pd.Timestamp.now(tz=tz_mx).tz_convert(None)
except Exception:
    NOW_MX = pd.Timestamp.now()

if "date" in df_matches.columns:
    df_matches["as_of"] = _to_naive_series(df_matches["date"])
else:
    df_matches["as_of"] = NOW_MX

# ---------- Mapeo a nombres canónicos del histórico ----------
alias_map = build_alias_map_from_hist(hist)
df_matches["home_team_std"] = df_matches["home_team"].apply(lambda s: map_team_to_hist(s, alias_map))
df_matches["away_team_std"] = df_matches["away_team"].apply(lambda s: map_team_to_hist(s, alias_map))

# ---------- Parámetros de stake/ejecución ----------
BANKROLL = globals().get("BANKROLL", 10000.0)
KELLY_FRAC = globals().get("KELLY_FRAC", 0.5)
STAKE_CAP = globals().get("STAKE_CAP", 0.03)    # 3% bank por pick (sugerido previo)
MAX_TOTAL_RISK_PCT = globals().get("MAX_TOTAL_RISK_PCT", 0.10)
EDGE_MIN = globals().get("EDGE_MIN", 0.06)
DISAGREE_MAX = globals().get("DISAGREE_MAX", 0.18)
PICK_CAP = globals().get("PICK_CAP", 0.015)     # 1.5% bank ejecución final
MARGIN_OK = (1.02, 1.12)
BOOKS_MIN = globals().get("BOOKS_MIN", 5)
DELTA_CUT = 0.18  # p/penalización de stake por desacuerdo

# ---------- Construcción de OUT_DC ----------
rows = []
fallback_flags = []
for _, m in df_matches.iterrows():
    home_std = m["home_team_std"]; away_std = m["away_team_std"]
    as_of = _to_naive_ts(m["as_of"])
    # λ(as-of) con modelo DC-fuerzas
    h_known = home_std in dc_model.teams
    a_known = away_std in dc_model.teams
    lh, la = dc_model.lam(hist, home_std, away_std, as_of)
    fallback_flags.append(not (h_known and a_known))

    # Probs H/D/A con DC
    pH, pD, pA = probs_HDA_dc(lh, la, rho_hat, max_goals=10)

    # Cuotas medianas (ya saneadas)
    oH, oD, oA = float(m["odds_home_win"]), float(m["odds_draw"]), float(m["odds_away_win"])

    # EV por signo
    evH = pH*oH - 1.0
    evD = pD*oD - 1.0
    evA = pA*oA - 1.0
    best_ev = max([("Home", evH), ("Draw", evD), ("Away", evA)], key=lambda x:x[1])

    # Kelly fracc. sobre el mejor (para stake sugerido)
    best_prob = {"Home":pH,"Draw":pD,"Away":pA}[best_ev[0]]
    best_odds = {"Home":oH,"Draw":oD,"Away":oA}[best_ev[0]]
    k_full = kelly_fraction(best_prob, best_odds)
    k_frac = KELLY_FRAC * k_full
    stake_sugg = round(min(BANKROLL*STAKE_CAP, BANKROLL*k_frac), 2)

    # Mercado no-vig (híbrido: prop por defecto; Shin duplicado)
    p_prop, p_shin, shin_s, mth = compute_market_probs((oH,oD,oA))
    # Edges vs mercado híbrido (tomamos prop)
    eH = pH - p_prop[0]; eD = pD - p_prop[1]; eA = pA - p_prop[2]
    best_edge = max([("Home", eH), ("Draw", eD), ("Away", eA)], key=lambda x: x[1])

    # Desacuerdo máximo
    disagree_max = float(np.max(np.abs(np.array([pH,pD,pA]) - p_prop)))

    # Penalización de stake por desacuerdo
    scale_pen = float(np.exp(- (disagree_max/DELTA_CUT)**2)) if disagree_max > DELTA_CUT else 1.0
    stake_pen = round(stake_sugg*scale_pen, 2)

    # Fair odds (prop y “shin” duplicado)
    fair_prop = (1.0/np.maximum(p_prop, 1e-9)).tolist()
    fair_shin = (1.0/np.maximum(p_shin, 1e-9)).tolist()

    rows.append({
        "Match": f"{m['home_team']} vs {m['away_team']}",
        "as_of": as_of,
        "λ_home": round(lh,3), "λ_away": round(la,3),
        "Prob_H": round(pH,4), "Prob_D": round(pD,4), "Prob_A": round(pA,4),
        "Odds_H": oH, "Odds_D": oD, "Odds_A": oA,
        "EV_H": round(evH,3), "EV_D": round(evD,3), "EV_A": round(evA,3),
        "Best": best_ev[0], "Best_EV": round(best_ev[1],3),
        "Kelly_full": round(k_full,3), "Kelly_frac": round(k_frac,3),
        "Stake_cap": STAKE_CAP, "Stake_suggested": stake_sugg,
        "Mkt_noVig_H_prop": round(p_prop[0],4),
        "Mkt_noVig_D_prop": round(p_prop[1],4),
        "Mkt_noVig_A_prop": round(p_prop[2],4),
        "FairOdds_H_prop": round(fair_prop[0],3),
        "FairOdds_D_prop": round(fair_prop[1],3),
        "FairOdds_A_prop": round(fair_prop[2],3),
        "Mkt_noVig_H_shin": round(p_shin[0],4),
        "Mkt_noVig_D_shin": round(p_shin[1],4),
        "Mkt_noVig_A_shin": round(p_shin[2],4),
        "FairOdds_H_shin": round(fair_shin[0],3),
        "FairOdds_D_shin": round(fair_shin[1],3),
        "FairOdds_A_shin": round(fair_shin[2],3),
        "Shin_s": shin_s,
        "Mkt_noVig_method": mth,
        "Edge_H_hybrid": round(eH,4), "Edge_D_hybrid": round(eD,4), "Edge_A_hybrid": round(eA,4),
        "Best_by_Edge_hybrid": best_edge[0], "Best_Edge_hybrid": round(best_edge[1],4),
        "Disagreement_max": round(disagree_max,4),
        # QC flags
        "QC_shin_high": shin_s >= 0.22,
        "QC_low_books": (int(m["book_count"]) if "book_count" in m else 0) < BOOKS_MIN,
        "QC_margin_off": not ( (1/oH + 1/oD + 1/oA) >= MARGIN_OK[0] and (1/oH + 1/oD + 1/oA) <= MARGIN_OK[1] ),
        "QC_lambda_fallback": (not (h_known and a_known)),
        # Stake penalizado por desacuerdo (útil para reporting)
        "Stake_final_pen": stake_pen
    })

out_dc = pd.DataFrame(rows)
# Orden bonito
front_cols = [
    "Match","as_of","λ_home","λ_away","Prob_H","Prob_D","Prob_A",
    "Odds_H","Odds_D","Odds_A","EV_H","EV_D","EV_A","Best","Best_EV",
    "Kelly_full","Kelly_frac","Stake_cap","Stake_suggested",
]
market_cols = [
    "Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop",
    "FairOdds_H_prop","FairOdds_D_prop","FairOdds_A_prop",
    "Mkt_noVig_H_shin","Mkt_noVig_D_shin","Mkt_noVig_A_shin",
    "FairOdds_H_shin","FairOdds_D_shin","FairOdds_A_shin",
    "Shin_s","Mkt_noVig_method",
]
edge_cols = [
    "Edge_H_hybrid","Edge_D_hybrid","Edge_A_hybrid","Best_by_Edge_hybrid","Best_Edge_hybrid","Disagreement_max"
]
qc_cols = ["QC_shin_high","QC_low_books","QC_margin_off","QC_lambda_fallback","Stake_final_pen"]
out_dc = out_dc[front_cols + market_cols + edge_cols + qc_cols]

print("\n=== OUT_DC (Dixon–Coles fuerzas) ===")
print(out_dc.head(10).to_string(index=False))

# ---------- Selección de PICKS_DC (reglas endurecidas) ----------
def implied_sum_row(r):
    return (1/r["Odds_H"]) + (1/r["Odds_D"]) + (1/r["Odds_A"])

cand = out_dc.copy()
qc_block = (cand["QC_margin_off"] | cand["QC_low_books"] | cand["QC_lambda_fallback"])
cand = cand[~qc_block].copy()

imp = cand.apply(implied_sum_row, axis=1)
cand = cand[(imp >= MARGIN_OK[0]) & (imp <= MARGIN_OK[1])]

# Edge mínimo y desacuerdo máximo
mask_ev = cand["Best_EV"] > 0
mask_edge = (
    ((cand["Best"]=="Home") & (cand["Edge_H_hybrid"]>EDGE_MIN)) |
    ((cand["Best"]=="Draw") & (cand["Edge_D_hybrid"]>EDGE_MIN)) |
    ((cand["Best"]=="Away") & (cand["Edge_A_hybrid"]>EDGE_MIN))
)
mask_dis = cand["Disagreement_max"] <= DISAGREE_MAX
picks_dc = cand[mask_ev & mask_edge & mask_dis].copy()

# Stake final de ejecución (cap 1.5% bank)
picks_dc["Stake_final_exec"] = np.minimum(
    picks_dc.get("Stake_final_pen", picks_dc["Stake_suggested"]) / BANKROLL,
    PICK_CAP
) * BANKROLL
picks_dc["Stake_final_exec"] = picks_dc["Stake_final_exec"].round(2)

# Cap de presupuesto total
budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
sum_stakes = picks_dc["Stake_final_exec"].sum() if not picks_dc.empty else 0.0
if sum_stakes > budget_max > 0:
    scale = budget_max / sum_stakes
    picks_dc["Stake_final_exec"] = (picks_dc["Stake_final_exec"] * scale).round(2)
    picks_dc["Exec_scale"] = round(scale, 3)
else:
    picks_dc["Exec_scale"] = 1.0

show_cols = [
    "Match","as_of","Best","Best_EV","Best_by_Edge_hybrid","Best_Edge_hybrid",
    "Mkt_noVig_method","Disagreement_max",
    "Prob_H","Prob_D","Prob_A",
    "Odds_H","Odds_D","Odds_A",
    "Stake_suggested","Stake_final_pen","Stake_final_exec","Exec_scale"
]
print("\n=== PICKS_DC (reglas endurecidas con DC-fuerzas) ===")
if not picks_dc.empty:
    print(picks_dc[[c for c in show_cols if c in picks_dc.columns]].to_string(index=False))
else:
    print("(No hay picks tras filtros endurecidos)")

# ---------- Guardado ----------
from pathlib import Path
Path("output").mkdir(exist_ok=True, parents=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dc.to_csv(f"output/predictions_dc_{ts}.csv", index=False, encoding="utf-8")
if not picks_dc.empty:
    picks_dc.to_csv(f"output/picks_dc_{ts}.csv", index=False, encoding="utf-8")
    print(f"\n✓ Guardados:\n  - output/predictions_dc_{ts}.csv\n  - output/picks_dc_{ts}.csv")
else:
    print(f"\n✓ Guardado:\n  - output/predictions_dc_{ts}.csv")


# --- Cell 30 ---
# ============================================
# COMPARADOR out (rolling λ) vs out_dc (DC-fuerzas)
# ============================================
import pandas as pd, numpy as np, re, unicodedata
from datetime import datetime
from pathlib import Path

# ---- helpers de normalización de nombres ----
STOPWORDS = {"FC","CF","CLUB","DE","DEL","DEPORTIVO","ATL","ATLETICO","ATLÉTICO","CD","UD","SC","AC"}
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
def canon(s):
    s = str(s).upper().strip()
    s = _strip_accents(s)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(toks)

def split_match(df, col="Match"):
    home = []; away=[]
    for m in df[col].astype(str):
        if " vs " in m:
            h,a = m.split(" vs ",1)
        elif " VS " in m:
            h,a = m.split(" VS ",1)
        else:
            # fallback: intenta por guion
            parts = re.split(r"\s+vs\.?\s+|\s+-\s+", m, flags=re.I)
            h,a = (parts[0], parts[1]) if len(parts)>=2 else (m, "")
        home.append(h.strip()); away.append(a.strip())
    df = df.copy()
    df["home_name"] = home
    df["away_name"] = away
    df["home_can"] = df["home_name"].apply(canon)
    df["away_can"] = df["away_name"].apply(canon)
    return df

def ensure_market_props(df):
    """ Asegura columnas de mercado no-vig proporcional y edges_hybrid si faltan. """
    d = df.copy()
    # mercado prop
    have_prop = {"Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop"}.issubset(d.columns)
    if not have_prop and {"Odds_H","Odds_D","Odds_A"}.issubset(d.columns):
        inv = pd.DataFrame({
            "H": 1/d["Odds_H"].astype(float),
            "D": 1/d["Odds_D"].astype(float),
            "A": 1/d["Odds_A"].astype(float),
        })
        s = inv.sum(axis=1).replace(0,np.nan)
        d["Mkt_noVig_H_prop"] = (inv["H"]/s).fillna(1/3)
        d["Mkt_noVig_D_prop"] = (inv["D"]/s).fillna(1/3)
        d["Mkt_noVig_A_prop"] = (inv["A"]/s).fillna(1/3)
    # edges híbridos (usamos prop como híbrido si no existen)
    have_edge = {"Edge_H_hybrid","Edge_D_hybrid","Edge_A_hybrid"}.issubset(d.columns)
    have_prob = {"Prob_H","Prob_D","Prob_A","Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop"}.issubset(d.columns)
    if not have_edge and have_prob:
        d["Edge_H_hybrid"] = d["Prob_H"] - d["Mkt_noVig_H_prop"]
        d["Edge_D_hybrid"] = d["Prob_D"] - d["Mkt_noVig_D_prop"]
        d["Edge_A_hybrid"] = d["Prob_A"] - d["Mkt_noVig_A_prop"]
        # best_by_edge si falta
        def _best_row(r):
            arr = np.array([r["Edge_H_hybrid"], r["Edge_D_hybrid"], r["Edge_A_hybrid"]], float)
            return ["Home","Draw","Away"][int(np.nanargmax(arr))]
        if "Best_by_Edge_hybrid" not in d.columns:
            d["Best_by_Edge_hybrid"] = d.apply(_best_row, axis=1)
        if "Best_Edge_hybrid" not in d.columns:
            d["Best_Edge_hybrid"] = d[["Edge_H_hybrid","Edge_D_hybrid","Edge_A_hybrid"]].max(axis=1)
    return d

# ---- chequeos ----
if 'out' not in globals() or out is None or out.empty:
    raise RuntimeError("No encuentro `out` (pipeline rolling λ). Ejecuta tu flujo base primero.")
if 'out_dc' not in globals() or out_dc is None or out_dc.empty:
    raise RuntimeError("No encuentro `out_dc` (DC-fuerzas). Ejecuta la celda DC previa.")

# ---- preparar dataframes clave ----
o = out.copy()
d = out_dc.copy()

# homogeneizar columnas de prob/odds si nombres difieren
rename_map = {
    "odds_home_win":"Odds_H","odds_draw":"Odds_D","odds_away_win":"Odds_A",
}
for k,v in rename_map.items():
    if k in o.columns and v not in o.columns:
        o[v] = o[k]
    if k in d.columns and v not in d.columns:
        d[v] = d[k]

# asegurar mercado prop y edges
o = ensure_market_props(o)
d = ensure_market_props(d)

# extraer home/away canónicos para el join
o = split_match(o, "Match")
d = split_match(d, "Match")

# ---- join por equipos (home_can & away_can); si hay duplicados, intentamos con fecha cercana ----
key_cols = ["home_can","away_can"]
cand = o.merge(
    d, on=key_cols, how="inner", suffixes=("_out","_dc")
)

if cand.empty:
    # intento laxo: join por set sin localía (no recomendado, pero útil en torneos con ida/vuelta pegados)
    o["pair_key"] = o.apply(lambda r: " :: ".join(sorted([r["home_can"], r["away_can"]])), axis=1)
    d["pair_key"] = d.apply(lambda r: " :: ".join(sorted([r["home_can"], r["away_can"]])), axis=1)
    cand = o.merge(d, on="pair_key", how="inner", suffixes=("_out","_dc"))

if cand.empty:
    raise RuntimeError("No pude emparejar filas entre out y out_dc. Revisa nombres de equipos en ambos DataFrames.")

# ---- métricas de comparación ----
def _safe(c, sfx):
    return c + sfx if c + sfx in cand.columns else None

def col(c): return c in cand.columns

# deltas de lambda y prob
cand["d_lambda_home"] = cand["λ_home_dc"] - cand["λ_home_out"] if col("λ_home_dc") and col("λ_home_out") else np.nan
cand["d_lambda_away"] = cand["λ_away_dc"] - cand["λ_away_out"] if col("λ_away_dc") and col("λ_away_out") else np.nan

for p in ["H","D","A"]:
    c_out = f"Prob_{p}_out"; c_dc = f"Prob_{p}_dc"
    if col(c_out) and col(c_dc):
        cand[f"d_Prob_{p}"] = cand[c_dc] - cand[c_out]

# deltas de edge y EV
if col("Best_Edge_hybrid_dc") and col("Best_Edge_hybrid_out"):
    cand["d_Best_Edge_hybrid"] = cand["Best_Edge_hybrid_dc"] - cand["Best_Edge_hybrid_out"]
if col("Best_EV_dc") and col("Best_EV_out"):
    cand["d_Best_EV"] = cand["Best_EV_dc"] - cand["Best_EV_out"]

# cambio de pick por señal (edge) y por EV
cand["Switch_pick_edge"] = (cand["Best_by_Edge_hybrid_out"] != cand["Best_by_Edge_hybrid_dc"]) if col("Best_by_Edge_hybrid_out") and col("Best_by_Edge_hybrid_dc") else False
cand["Switch_pick_ev"]   = (cand["Best_out"] != cand["Best_dc"]) if col("Best_out") and col("Best_dc") else False

# desacuerdo con mercado
if col("Disagreement_max_dc") and col("Disagreement_max_out"):
    cand["d_Disagreement"] = cand["Disagreement_max_dc"] - cand["Disagreement_max_out"]

# stake (usa final si existe; si no, sugerido)
stake_out = "Stake_final_exec_out" if col("Stake_final_exec_out") else ("Stake_final_pen_out" if col("Stake_final_pen_out") else ("Stake_suggested_out" if col("Stake_suggested_out") else None))
stake_dc  = "Stake_final_exec_dc"  if col("Stake_final_exec_dc")  else ("Stake_final_pen_dc"  if col("Stake_final_pen_dc")  else ("Stake_suggested_dc"  if col("Stake_suggested_dc")  else None))
if stake_out and stake_dc:
    cand["d_Stake"] = cand[stake_dc] - cand[stake_out]

# resumen numérico
summary = {}
def _mad(x): 
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x)>0 else np.nan

summary["n_matches_compared"] = int(cand.shape[0])
summary["mean|Δλ_home|"] = _mad(cand.get("d_lambda_home"))
summary["mean|Δλ_away|"] = _mad(cand.get("d_lambda_away"))
for p in ["H","D","A"]:
    if f"d_Prob_{p}" in cand.columns:
        summary[f"mean|ΔProb_{p}|"] = _mad(cand[f"d_Prob_{p}"])
if "d_Best_Edge_hybrid" in cand.columns:
    summary["mean|ΔBest_Edge|"] = _mad(cand["d_Best_Edge_hybrid"])
if "d_Disagreement" in cand.columns:
    summary["mean|ΔDisagreement|"] = _mad(cand["d_Disagreement"])
if "Switch_pick_edge" in cand.columns:
    summary["share_switch_edge"] = float(cand["Switch_pick_edge"].mean())
if "Switch_pick_ev" in cand.columns:
    summary["share_switch_ev"] = float(cand["Switch_pick_ev"].mean())

print("\n=== RESUMEN OUT vs OUT_DC ===")
for k,v in summary.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# top diferencias de probabilidad
cols_show = [
    "Match_out","as_of_out","Match_dc","as_of_dc",
    "λ_home_out","λ_away_out","λ_home_dc","λ_away_dc",
    "Prob_H_out","Prob_D_out","Prob_A_out","Prob_H_dc","Prob_D_dc","Prob_A_dc",
    "d_lambda_home","d_lambda_away","d_Prob_H","d_Prob_D","d_Prob_A",
    "Best_by_Edge_hybrid_out","Best_by_Edge_hybrid_dc","d_Best_Edge_hybrid",
    "Best_out","Best_dc","d_Best_EV",
    "Disagreement_max_out","Disagreement_max_dc","d_Disagreement"
]
cols_show = [c for c in cols_show if c in cand.columns]

print("\n=== TOP 10 |ΔProb| (H o A) ===")
cand["_max_abs_dProb"] = np.nanmax(np.abs(cand[[c for c in ["d_Prob_H","d_Prob_A"] if c in cand.columns]].values), axis=1) if any(c in cand.columns for c in ["d_Prob_H","d_Prob_A"]) else 0
print(cand.sort_values("_max_abs_dProb", ascending=False)[cols_show].head(10).to_string(index=False))

if "Switch_pick_edge" in cand.columns:
    sw = cand[cand["Switch_pick_edge"]==True]
    if not sw.empty:
        print("\n=== CAMBIOS DE PICK (por edge) ===")
        print(sw[cols_show].to_string(index=False))
    else:
        print("\n(No hubo cambios de pick por edge)")

# guardar CSV de comparación completa
Path("output").mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cmp_path = f"output/compare_out_vs_dc_{ts}.csv"
cand.drop(columns=["_max_abs_dProb"], errors="ignore").to_csv(cmp_path, index=False, encoding="utf-8")
print(f"\n✓ Comparación completa guardada en: {cmp_path}")

# (opcional) comparación de picks si ya tienes picks y picks_dc en memoria
if 'picks' in globals() and isinstance(picks, pd.DataFrame) and not picks.empty:
    sp = split_match(picks, "Match")
    sp["key"] = sp["home_can"] + " vs " + sp["away_can"]
    picks_set = set(sp["key"])
else:
    picks_set = set()

if 'picks_dc' in globals() and isinstance(picks_dc, pd.DataFrame) and not picks_dc.empty:
    spd = split_match(picks_dc, "Match")
    spd["key"] = spd["home_can"] + " vs " + spd["away_can"]
    picks_dc_set = set(spd["key"])
else:
    picks_dc_set = set()

if picks_set or picks_dc_set:
    only_out   = sorted(picks_set - picks_dc_set)
    only_dc    = sorted(picks_dc_set - picks_set)
    both       = sorted(picks_set & picks_dc_set)
    print("\n=== COMPARACIÓN DE LISTAS DE PICKS ===")
    print(f"Solo OUT : {len(only_out)} -> {only_out[:10]}")
    print(f"Solo OUT_DC: {len(only_dc)} -> {only_dc[:10]}")
    print(f"En ambos : {len(both)} -> {both[:10]}")


# --- Cell 31 ---
# ============================================================
# GRID-SEARCH de hiperparámetros para DC-fuerzas (rápido)
# ============================================================
import numpy as np, pandas as pd, unicodedata, re
from datetime import datetime
from pathlib import Path

# ---------- helpers de tiempo ----------
def _to_naive_series(s):
    s = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
    return s
def _to_naive_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try: ts = ts.tz_localize(None)
    except Exception: pass
    return ts

# ---------- normalización nombres ----------
STOPWORDS = {"FC","CF","CLUB","DE","DEL","DEPORTIVO","ATL","ATLETICO","ATLÉTICO","CD","UD","SC","AC"}
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
def canon(s):
    s = str(s).upper().strip()
    s = _strip_accents(s)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(toks)

def build_alias_map_from_hist(hist_df):
    canons = {}
    for col in ["home_std","away_std","home","away"]:
        if col in hist_df.columns:
            for t in hist_df[col].dropna().unique():
                canons[canon(t)] = t
    return canons

def map_team_to_hist(name, alias_map):
    key = canon(name)
    return alias_map.get(key, name)

# ---------- mercado no-vig (prop) ----------
def fair_probs_proportional(odds_tuple):
    oH,oD,oA = map(float, odds_tuple)
    raw = np.array([1/oH,1/oD,1/oA], dtype=float)
    s = raw.sum()
    return raw/s if s>0 else np.array([1/3,1/3,1/3], dtype=float)

def compute_market_probs(odds_tuple):
    p_prop = fair_probs_proportional(odds_tuple)
    shin_s = 0.25  # placeholder para mantener layout
    p_shin = p_prop.copy()
    return p_prop, p_shin, shin_s, "prop"

# ---------- prob H/D/A con DC (usa la que ya tengas) ----------
try:
    probs_HDA_dc
except NameError:
    from scipy.stats import poisson
    def probs_HDA_dc(lh, la, rho, max_goals=10):
        home = np.arange(0, max_goals+1)
        away = np.arange(0, max_goals+1)
        P = np.outer(poisson.pmf(home, lh), poisson.pmf(away, la))
        P_adj = P.copy()
        P_adj[0,0] *= (1 - (lh*la*rho))
        if max_goals >= 1:
            P_adj[0,1] *= (1 + rho)
            P_adj[1,0] *= (1 + rho)
            P_adj[1,1] *= (1 - rho)
        s = P_adj.sum()
        if s <= 0: P_adj = P; s = P_adj.sum()
        P_adj /= s
        pH = np.tril(P_adj, -1).sum()
        pD = np.trace(P_adj)
        pA = np.triu(P_adj, 1).sum()
        return float(pH), float(pD), float(pA)

# ---------- chequeos previos ----------
if 'hist' not in globals() or hist is None or hist.empty:
    raise RuntimeError("Falta `hist`.")
if 'rho_hat' not in globals():
    raise RuntimeError("Falta `rho_hat`.")
if 'fit_dc_forces' not in globals():
    raise RuntimeError("Falta `fit_dc_forces` (celda DC-fuerzas).")

# tomamos cuotas
if 'dfm' in globals() and isinstance(dfm, pd.DataFrame) and not dfm.empty:
    matches_base = dfm.copy()
elif 'df_matches' in globals() and isinstance(df_matches, pd.DataFrame) and not df_matches.empty:
    matches_base = df_matches.copy()
else:
    raise RuntimeError("No encuentro dfm/df_matches con cuotas.")

# asegurar columnas
need_cols = {"home_team","away_team","odds_home_win","odds_draw","odds_away_win"}
missing = [c for c in need_cols if c not in matches_base.columns]
if missing:
    raise RuntimeError(f"df de cuotas carece de columnas: {missing}")

# as_of
if "date" in matches_base.columns:
    matches_base["as_of"] = _to_naive_series(matches_base["date"])
else:
    try:
        NOW_MX = pd.Timestamp.now(tz="America/Mexico_City").tz_convert(None)
    except Exception:
        NOW_MX = pd.Timestamp.now()
    matches_base["as_of"] = NOW_MX

# mapeo a canónicos
alias_map = build_alias_map_from_hist(hist)
matches_base["home_team_std"] = matches_base["home_team"].apply(lambda s: map_team_to_hist(s, alias_map))
matches_base["away_team_std"] = matches_base["away_team"].apply(lambda s: map_team_to_hist(s, alias_map))

# ---------- hiperparámetros a explorar ----------
GRID = {
    "use_xg":       [True, False],
    "ridge":        [0.5, 1.0, 2.0],
    "decay_alpha":  [0.002, 0.003, 0.0045],
    "season_decay": [0.80, 0.85, 0.90],
}
# puedes reducir el grid si quieres velocidad:
# GRID["use_xg"] = [True]
# GRID["ridge"] = [1.0]
# etc.

# ---------- umbrales de ejecución (usa los de tu flujo) ----------
BANKROLL   = globals().get("BANKROLL", 10000.0)
KELLY_FRAC = globals().get("KELLY_FRAC", 0.5)
STAKE_CAP  = globals().get("STAKE_CAP", 0.03)
PICK_CAP   = globals().get("PICK_CAP", 0.015)
MAX_TOTAL_RISK_PCT = globals().get("MAX_TOTAL_RISK_PCT", 0.10)
EDGE_MIN   = globals().get("EDGE_MIN", 0.06)
DISAGREE_MAX = globals().get("DISAGREE_MAX", 0.18)
MARGIN_OK  = (1.02, 1.12)
DELTA_CUT  = 0.18
BOOKS_MIN  = globals().get("BOOKS_MIN", 5)

# ---------- función para construir out con un modelo dado ----------
def build_out_from_model(dc_model, matches_df, rho_hat):
    rows=[]
    for _, m in matches_df.iterrows():
        home_std = m["home_team_std"]; away_std = m["away_team_std"]
        as_of    = _to_naive_ts(m["as_of"])
        lh, la   = dc_model.lam(hist, home_std, away_std, as_of)
        pH,pD,pA = probs_HDA_dc(lh, la, rho_hat, max_goals=10)
        oH,oD,oA = float(m["odds_home_win"]), float(m["odds_draw"]), float(m["odds_away_win"])
        evH, evD, evA = pH*oH-1, pD*oD-1, pA*oA-1
        best_ev = max([("Home",evH),("Draw",evD),("Away",evA)], key=lambda x:x[1])
        # Kelly
        best_prob = {"Home":pH,"Draw":pD,"Away":pA}[best_ev[0]]
        best_odds = {"Home":oH,"Draw":oD,"Away":oA}[best_ev[0]]
        b = best_odds-1
        k_full = max((best_prob*best_odds - (1-best_prob))/b, 0) if b>0 else 0
        k_frac = KELLY_FRAC * k_full
        stake_sugg = round(min(BANKROLL*STAKE_CAP, BANKROLL*k_frac), 2)
        # mercado no-vig (prop)
        p_prop, p_shin, shin_s, method = compute_market_probs((oH,oD,oA))
        eH,eD,eA = pH-p_prop[0], pD-p_prop[1], pA-p_prop[2]
        best_edge = max([("Home",eH),("Draw",eD),("Away",eA)], key=lambda x:x[1])
        disagree = float(np.max(np.abs(np.array([pH,pD,pA])-p_prop)))
        scale_pen = float(np.exp(- (disagree/DELTA_CUT)**2)) if disagree > DELTA_CUT else 1.0
        stake_pen = round(stake_sugg*scale_pen, 2)
        rows.append({
            "Match": f"{m['home_team']} vs {m['away_team']}",
            "as_of": as_of,
            "λ_home": round(lh,3), "λ_away": round(la,3),
            "Prob_H": round(pH,4), "Prob_D": round(pD,4), "Prob_A": round(pA,4),
            "Odds_H": oH, "Odds_D": oD, "Odds_A": oA,
            "EV_H": round(evH,3), "EV_D": round(evD,3), "EV_A": round(evA,3),
            "Best": best_ev[0], "Best_EV": round(best_ev[1],3),
            "Edge_H_hybrid": round(eH,4), "Edge_D_hybrid": round(eD,4), "Edge_A_hybrid": round(eA,4),
            "Best_by_Edge_hybrid": best_edge[0], "Best_Edge_hybrid": round(best_edge[1],4),
            "Disagreement_max": round(disagree,4),
            "Mkt_noVig_H_prop": round(p_prop[0],4),
            "Mkt_noVig_D_prop": round(p_prop[1],4),
            "Mkt_noVig_A_prop": round(p_prop[2],4),
            "Stake_suggested": stake_sugg, "Stake_final_pen": stake_pen,
            # QC básicos (si tienes book_count y margen en matches_df, se checan luego)
            "book_count": int(m["book_count"]) if "book_count" in m and pd.notna(m["book_count"]) else 0
        })
    return pd.DataFrame(rows)

# ---------- función de evaluación ----------
def implied_sum_row(r): 
    return (1/r["Odds_H"]) + (1/r["Odds_D"]) + (1/r["Odds_A"])

def evaluate_out(out_df):
    if out_df.empty:
        return {"n_matches":0, "mean_disagreement":np.inf, "mean_abs_dprob":np.inf,
                "picks_count":0, "mean_best_ev":-np.inf, "sum_stake_exec":0.0}
    df = out_df.copy()
    # margen + libros
    imp = df.apply(implied_sum_row, axis=1)
    ok_margin = (imp >= MARGIN_OK[0]) & (imp <= MARGIN_OK[1])
    ok_books  = (df["book_count"] >= BOOKS_MIN) if "book_count" in df.columns else True
    df = df[ ok_margin & ok_books ]
    if df.empty:
        return {"n_matches":0, "mean_disagreement":np.inf, "mean_abs_dprob":np.inf,
                "picks_count":0, "mean_best_ev":-np.inf, "sum_stake_exec":0.0}
    # métrica de desacuerdo/distrib
    mean_disagree = float(df["Disagreement_max"].mean())
    # distancia media a mercado (promedio de |Δ| sobre H/D/A)
    dprob = np.abs(df[["Prob_H","Prob_D","Prob_A"]].values - df[["Mkt_noVig_H_prop","Mkt_noVig_D_prop","Mkt_noVig_A_prop"]].values).mean()
    mean_abs_dprob = float(dprob)

    # selección endurecida
    mask_ev = df["Best_EV"] > 0
    mask_edge = (
        ((df["Best"]=="Home") & (df["Edge_H_hybrid"]>EDGE_MIN)) |
        ((df["Best"]=="Draw") & (df["Edge_D_hybrid"]>EDGE_MIN)) |
        ((df["Best"]=="Away") & (df["Edge_A_hybrid"]>EDGE_MIN))
    )
    mask_dis = df["Disagreement_max"] <= DISAGREE_MAX
    picks = df[mask_ev & mask_edge & mask_dis].copy()

    # stake final con cap 1.5% bank
    if not picks.empty:
        picks["Stake_final_exec"] = np.minimum(
            picks["Stake_final_pen"] / BANKROLL, PICK_CAP
        ) * BANKROLL
        picks["Stake_final_exec"] = picks["Stake_final_exec"].round(2)
        # cap de presupuesto total
        budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
        sum_stakes = picks["Stake_final_exec"].sum()
        if sum_stakes > budget_max > 0:
            scale = budget_max / sum_stakes
            picks["Stake_final_exec"] = (picks["Stake_final_exec"] * scale).round(2)

    return {
        "n_matches": int(df.shape[0]),
        "mean_disagreement": mean_disagree,
        "mean_abs_dprob": mean_abs_dprob,
        "picks_count": int(picks.shape[0]),
        "mean_best_ev": float(df["Best_EV"].mean()),
        "sum_stake_exec": float(picks["Stake_final_exec"].sum() if not picks.empty else 0.0)
    }

# ---------- función objetivo ----------
# Minimiza desacuerdo, pero bonifica tener picks (no trivial). Ajusta pesos si gustas.
WEIGHTS = {"w_dis":1.0, "w_dprob":0.6, "w_picks":0.08}
PICKS_TARGET = 2   # objetivo deseable de picks ejecutables
def objective(m):
    if m["n_matches"] == 0 or np.isinf(m["mean_disagreement"]):
        return np.inf
    pen = WEIGHTS["w_dis"]*m["mean_disagreement"] + WEIGHTS["w_dprob"]*m["mean_abs_dprob"]
    bonus = WEIGHTS["w_picks"] * min(m["picks_count"], PICKS_TARGET)
    return float(pen - bonus)

# ---------- grid loop ----------
try:
    AS_OF = pd.Timestamp.now(tz="America/Mexico_City").tz_convert(None)
except Exception:
    AS_OF = pd.Timestamp.now()

results = []
best_tuple = None
best_score = np.inf
best_out = None
best_model = None

grid_list = []
for uxg in GRID["use_xg"]:
    for rg in GRID["ridge"]:
        for da in GRID["decay_alpha"]:
            for sd in GRID["season_decay"]:
                grid_list.append((uxg,rg,da,sd))

print(f"Probando {len(grid_list)} combinaciones…")
for (uxg, rg, da, sd) in grid_list:
    # fit
    model = fit_dc_forces(hist, AS_OF, use_xg=uxg, decay_alpha=da, season_decay=sd, ridge=rg)
    # out temporal
    out_tmp = build_out_from_model(model, matches_base, rho_hat)
    # eval
    metrics = evaluate_out(out_tmp)
    score = objective(metrics)
    results.append({
        "use_xg":uxg, "ridge":rg, "decay_alpha":da, "season_decay":sd,
        **metrics, "score":score
    })
    if score < best_score:
        best_score = score
        best_tuple = (uxg, rg, da, sd)
        best_out   = out_tmp.copy()
        best_model = model

res = pd.DataFrame(results).sort_values("score", ascending=True)
print("\n=== TOP 10 combinaciones por score (↓ mejor) ===")
print(res.head(10).to_string(index=False))

# ------------ aplicar la mejor (opcional) ------------
APPLY_BEST = True  # pon False si solo quieres ver resultados

if APPLY_BEST and best_model is not None:
    dc_model = best_model
    out_dc = best_out
    # reconstruimos picks_dc con las reglas endurecidas (idénticas a evaluate_out pero guardando filas)
    df = out_dc.copy()
    imp = df.apply(implied_sum_row, axis=1)
    ok_margin = (imp >= MARGIN_OK[0]) & (imp <= MARGIN_OK[1])
    ok_books  = (df["book_count"] >= BOOKS_MIN) if "book_count" in df.columns else True
    df = df[ ok_margin & ok_books ]
    mask_ev = df["Best_EV"] > 0
    mask_edge = (
        ((df["Best"]=="Home") & (df["Edge_H_hybrid"]>EDGE_MIN)) |
        ((df["Best"]=="Draw") & (df["Edge_D_hybrid"]>EDGE_MIN)) |
        ((df["Best"]=="Away") & (df["Edge_A_hybrid"]>EDGE_MIN))
    )
    mask_dis = df["Disagreement_max"] <= DISAGREE_MAX
    picks_dc = df[mask_ev & mask_edge & mask_dis].copy()
    if not picks_dc.empty:
        picks_dc["Stake_final_exec"] = np.minimum(
            picks_dc["Stake_final_pen"]/BANKROLL, PICK_CAP
        )*BANKROLL
        picks_dc["Stake_final_exec"] = picks_dc["Stake_final_exec"].round(2)
        budget_max = BANKROLL * MAX_TOTAL_RISK_PCT
        sum_stakes = picks_dc["Stake_final_exec"].sum()
        if sum_stakes > budget_max > 0:
            scale = budget_max / sum_stakes
            picks_dc["Stake_final_exec"] = (picks_dc["Stake_final_exec"]*scale).round(2)
            picks_dc["Exec_scale"] = round(scale,3)
        else:
            picks_dc["Exec_scale"] = 1.0

    # guardar
    Path("output").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res.to_csv(f"output/param_search_dc_{ts}.csv", index=False, encoding="utf-8")
    out_dc.to_csv(f"output/predictions_dc_best_{ts}.csv", index=False, encoding="utf-8")
    if not picks_dc.empty:
        picks_dc.to_csv(f"output/picks_dc_best_{ts}.csv", index=False, encoding="utf-8")
        print(f"\n✓ Guardados:\n  - output/param_search_dc_{ts}.csv\n  - output/predictions_dc_best_{ts}.csv\n  - output/picks_dc_best_{ts}.csv")
    else:
        print(f"\n✓ Guardados:\n  - output/param_search_dc_{ts}.csv\n  - output/predictions_dc_best_{ts}.csv\n(No hubo picks tras filtros)")

    print("\n=== Mejor combinación aplicada ===")
    uxg, rg, da, sd = best_tuple
    print(f"use_xg={uxg} | ridge={rg} | decay_alpha={da} | season_decay={sd} | score={best_score:.4f}")
else:
    print("\nNo se aplicó la mejor combinación automáticamente (APPLY_BEST=False).")


