import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


# FEATURES CONSISTENTES (train + predict)
FEATURES = [
    "p_market_H",
    "elo_diff",
    "form_diff",
    "lam_diff",
    "lam_sum",
    "diff_model_market"
]


def train_residual_model(df):
    # Target: home gana
    y = (df["outcome"] == 0).astype(int)

    # Validación mínima
    df = df.dropna(subset=FEATURES + ["outcome"])

    X = df[FEATURES]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, "output/residual_model.pkl")

    print("✔ residual model entrenado")
    print("Features usadas:", FEATURES)
    print("Samples:", len(X))


def load_residual_model():
    return joblib.load("output/residual_model.pkl")


def predict_home_prob(model, row):
    """
    row debe contener TODAS las FEATURES
    """

    X = pd.DataFrame([{
        "p_market_H": row["p_market_H"],
        "elo_diff": row["elo_diff"],
        "form_diff": row["form_diff"],
        "lam_diff": row["lam_diff"],
        "lam_sum": row["lam_sum"],
        "diff_model_market": row["diff_model_market"]
    }])

    return model.predict_proba(X)[0, 1]