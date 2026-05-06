import numpy as np
from collections import defaultdict
import pandas as pd

from src.features import compute_lambda_advanced
from src.features_extra import rolling_form
from src.model import dc_outcomes
from src.market_v2 import odds_to_probs
from src.betting_v2 import kelly
from src.elo import update_elo, get_result
from src.residual_model import load_residual_model

import joblib
import os


# -----------------
# CALIBRATION
# -----------------
iso = None
if os.path.exists("output/iso_home.pkl"):
    iso = joblib.load("output/iso_home.pkl")
    print("Calibración activada")
else:
    print("Calibración NO encontrada")

# -----------------
# RESIDUAL MODEL
# -----------------
model_residual = None
if os.path.exists("output/residual_model.pkl"):
    model_residual = load_residual_model()
    print("Residual model activo")
else:
    print("Residual model NO encontrado")


def run_backtest(df):

    bankroll = 1.0
    history = []
    n_bets = 0
    records = []
    all_edges = []

    elo = defaultdict(lambda: 1500)

    for i in range(50, len(df)):

        if n_bets > 300:
            break

        row = df.iloc[i]
        past = df.iloc[:i]

        h = row["home"]
        a = row["away"]

        # -----------------
        # ELO
        # -----------------
        ra, rb = elo[h], elo[a]
        elo_diff = ra - rb

        # -----------------
        # LAMBDA
        # -----------------
        lam_h = compute_lambda_advanced(past, h, a, True, row["date"])
        lam_a = compute_lambda_advanced(past, a, h, False, row["date"])

        if lam_h is None or lam_a is None:
            continue

        # ELO adjustment
        lam_h *= (1 + 0.002 * elo_diff)
        lam_a *= (1 - 0.002 * elo_diff)

        # FORM
        form_h = rolling_form(past, h)
        form_a = rolling_form(past, a)

        # -----------------
        # MARKET
        # -----------------
        odds = [
            row["odds_home"],
            row["odds_draw"],
            row["odds_away"]
        ]

        p_market = odds_to_probs(odds)

        # -----------------
        # POISSON
        # -----------------
        pH, pD, pA = dc_outcomes(lam_h, lam_a, rho=-0.1)

        if iso is not None:
            pH = iso.predict([pH])[0]

        p_model = [pH, pD, pA]

        # -----------------
        # FEATURES
        # -----------------
        row_features = {
            "p_market_H": p_market[0],
            "elo_diff": elo_diff,
            "form_diff": form_h - form_a,
            "lam_diff": lam_h - lam_a,
            "lam_sum": lam_h + lam_a,
            "diff_model_market": p_model[0] - p_market[0]
        }

        # -----------------
        # RESIDUAL MODEL
        # -----------------
        if model_residual is not None:
            X = pd.DataFrame([row_features])
            delta = model_residual.predict_proba(X)[0, 1]


            #pH = p_market[0] + 1.0 * (delta - p_market[0])
            pH = delta

            # redistribuir proporcionalmente
            scale = (1 - pH) / (p_market[1] + p_market[2])
            pD = p_market[1] * scale
            pA = p_market[2] * scale

            # normalización
            total = pH + pD + pA
            pH /= total
            pD /= total
            pA /= total

            p_model = [pH, pD, pA]

        # -----------------
        # FINAL PROBS
        # -----------------
        alpha = 0.25

        pH = p_market[0] + alpha * (p_model[0] - p_market[0])
        pD = p_market[1] + alpha * (p_model[1] - p_market[1])
        pA = p_market[2] + alpha * (p_model[2] - p_market[2])

        # normalizar
        total = pH + pD + pA
        pH /= total
        pD /= total
        pA /= total

        p_final = [pH, pD, pA]

        # -----------------
        # FILTRO MODELO vs MERCADO
        # -----------------
        if abs(p_model[0] - p_market[0]) < 0.002:
            continue

        # -----------------
        # OUTCOME
        # -----------------
        outcome = (
            0 if row["home_goals"] > row["away_goals"]
            else 1 if row["home_goals"] == row["away_goals"]
            else 2
        )

        # -----------------
        # RECORDS
        # -----------------
        records.append({
            "p_market_H": p_market[0],
            "p_market_D": p_market[1],
            "p_market_A": p_market[2],
            "elo_diff": elo_diff,
            "form_diff": form_h - form_a,
            "lam_diff": lam_h - lam_a,
            "lam_sum": lam_h + lam_a,
            "diff_model_market": p_model[0] - p_market[0],
            "outcome": outcome
        })

        # -----------------
        # EDGE (EV REAL)
        # -----------------
        edges = [
            p_final[0] * odds[0] - 1,
            p_final[1] * odds[1] - 1,
            p_final[2] * odds[2] - 1,
        ]

        # cap de edges (evitar outliers irreales)
        edges = [min(e, 0.12) for e in edges]


        best_i = max(range(3), key=lambda i: edges[i])

        if i < 200:
            print("edge:", edges[best_i], "odds:", odds[best_i])

        if edges[best_i] >= 0.015:
            all_edges.append(edges[best_i])

        # -----------------
        # FILTROS
        # -----------------
        if edges[best_i] < 0.02:
            continue

        if odds[best_i] < 1.5:
            continue

        # -----------------
        # BET SIZE
        # -----------------
        edge = edges[best_i]

        if edge > 0.08:
            stake = 0.3 * kelly(p_final[best_i], odds[best_i])
        elif edge > 0.05:
            stake = 0.2 * kelly(p_final[best_i], odds[best_i])
        else:
            stake = 0.05 * kelly(p_final[best_i], odds[best_i])

        if stake <= 0:
            continue

        # ✔ FIX n_bets
        n_bets += 1

        # -----------------
        # RESULT
        # -----------------
        if best_i == outcome:
            profit = stake * (odds[best_i] - 1)
        else:
            profit = -stake

        bankroll += profit
        history.append(float(bankroll))

        # -----------------
        # UPDATE ELO
        # -----------------
        res = get_result(row["home_goals"], row["away_goals"])
        elo[h] = update_elo(ra, rb, res)
        elo[a] = update_elo(rb, ra, 1 - res)

    print("\n=== TOP EDGES ===")
    top_edges = sorted(all_edges, reverse=True)[:20]
    for e in top_edges:
        print(e)

    print("\n=== EDGE STATS ===")
    if len(all_edges) > 0:
        print("mean:", np.mean(all_edges))
        print("max:", np.max(all_edges))
        print("count:", len(all_edges))
    else:
        print("No edges")

    return history, n_bets, records


def evaluate(history):

    history = np.array(history, dtype=float)

    if len(history) < 2:
        return {"ROI": 0, "Sharpe": 0, "MaxDD": 0}

    returns = np.diff(history)

    roi = history[-1] - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
    max_dd = np.max(np.maximum.accumulate(history) - history)

    return {
        "ROI": roi,
        "Sharpe": sharpe,
        "MaxDD": max_dd
    }