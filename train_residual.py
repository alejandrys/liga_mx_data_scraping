import pandas as pd
from src.residual_model import train_residual_model

df = pd.read_csv("output/calibration.csv")

train_residual_model(df)