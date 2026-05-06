import pandas as pd
from sklearn.isotonic import IsotonicRegression


df = pd.read_csv("output/calibration.csv")

# binario: home win
y = (df["outcome"] == 0).astype(int)
p = df["pH"]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p, y)

# guardar modelo
import joblib
joblib.dump(iso, "output/iso_home.pkl")

print("Modelo calibrado guardado")