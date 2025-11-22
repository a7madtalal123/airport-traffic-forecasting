import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_PATH = "./data/flights_RUH.csv"

df = pd.read_csv(DATA_PATH)

# convert datetime
for col in ["movement.scheduledTime.local", "movement.scheduledTime.utc"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def pick_dt(row):
    if pd.notna(row.get("movement.scheduledTime.local")):
        return row["movement.scheduledTime.local"]
    return row.get("movement.scheduledTime.utc")

df["scheduled_dt"] = df.apply(pick_dt, axis=1)
df = df[df["scheduled_dt"].notna()].copy()
df["date_hour"] = df["scheduled_dt"].dt.floor("H")

df = df[df["movement.terminal"].notna()].copy()

ts = df.groupby(["movement.terminal", "date_hour"]).size().reset_index(name="y")
ts = ts.sort_values(["movement.terminal", "date_hour"])

# time features
ts["hour"] = ts["date_hour"].dt.hour
ts["weekday"] = ts["date_hour"].dt.weekday
ts["month"] = ts["date_hour"].dt.month
ts["is_weekend"] = ts["weekday"].isin([4, 5]).astype(int)

# lags
ts["lag1"] = ts.groupby("movement.terminal")["y"].shift(1)
ts["lag3"] = ts.groupby("movement.terminal")["y"].shift(3)
ts["lag6"] = ts.groupby("movement.terminal")["y"].shift(6)

# rolling means
ts["ma3"] = ts.groupby("movement.terminal")["y"].rolling(3).mean().reset_index(level=0, drop=True).shift(1)
ts["ma6"] = ts.groupby("movement.terminal")["y"].rolling(6).mean().reset_index(level=0, drop=True).shift(1)

ts = ts.dropna()

# split
ts = ts.sort_values("date_hour").reset_index(drop=True)
split = int(len(ts) * 0.8)
train, test = ts.iloc[:split], ts.iloc[split:]

# naive baseline
test_naive = test[test["lag1"].notna()]
y_true = test_naive["y"]
y_pred_naive = test_naive["lag1"]

# MAE / RMSE / sMAPE
def smape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    denom[denom == 0] = 1e-6
    return np.mean(np.abs(y - yhat) / denom) * 100

baseline_results = {
    "MAE_naive": mean_absolute_error(y_true, y_pred_naive),
    "RMSE_naive": sqrt(mean_squared_error(y_true, y_pred_naive)),
    "sMAPE_naive": smape(y_true, y_pred_naive),
}

print("Baseline Results:")
for k, v in baseline_results.items():
    print(k, " = ", round(v, 4))

os.makedirs("./outputs", exist_ok=True)
ts.to_csv("./outputs/hourly_features.csv", index=False)
print("Saved feature file to outputs/hourly_features.csv")
