from datetime import datetime
import pandas as pd

from src.config import ODDS_API_KEY
from src.ingestion import get_odds, load_hist
from src.features import compute_lambda
from src.model import dc_outcomes
from src.market import shin_probs
from src.betting import detect_mispricing


def run():
    hist = load_hist("data/hist.csv")
    odds = get_odds(ODDS_API_KEY)

    print("\n=== TEAMS FROM ODDS ===")
    print(sorted(set(odds["home_team"])))

    odds = odds.drop_duplicates(subset=["home_team", "away_team"])

    rows = []
    now = datetime.now()

    for _, m in odds.iterrows():
        h = m["home_team"]
        a = m["away_team"]

        odds_triplet = [
            m["odds_home_win"],
            m["odds_draw"],
            m["odds_away_win"]
        ]

        # validación de odds
        if any(o is None or o <= 1 for o in odds_triplet):
            continue

        lam_h = compute_lambda(hist, h, True, now)
        lam_a = compute_lambda(hist, a, False, now)

        if lam_h == 1.3 or lam_a == 1.3:
            print(f"fallback: {h} vs {a}")

        pH, pD, pA = dc_outcomes(lam_h, lam_a, rho=-0.1)

        market_probs = shin_probs(odds_triplet)
        model_probs = [pH, pD, pA]

        bets = detect_mispricing(
            model_probs,
            market_probs,
            odds_triplet,
            threshold=0.03
        )

        best_bet = bets[0] if bets else None

        rows.append({
            "match": f"{h} vs {a}",

            "lambda_home": lam_h,
            "lambda_away": lam_a,

            "pH": pH,
            "pD": pD,
            "pA": pA,

            "market_pH": market_probs[0],
            "market_pD": market_probs[1],
            "market_pA": market_probs[2],

            "odds_home": odds_triplet[0],
            "odds_draw": odds_triplet[1],
            "odds_away": odds_triplet[2],

            "n_bets": len(bets),

            "best_side": best_bet["side"] if best_bet else None,
            "best_edge": best_bet["edge"] if best_bet else None,
            "best_ev": best_bet["ev"] if best_bet else None,
            "best_odds": best_bet["odds"] if best_bet else None
        })

    df = pd.DataFrame(rows)

    df.to_csv("output/predictions.csv", index=False)

    picks = df[df["best_edge"].notna()]
    picks.to_csv("output/picks.csv", index=False)

    print("Pipeline ejecutado")
    print(f"Matches procesados: {len(df)}")
    print(f"Picks detectados: {len(picks)}")