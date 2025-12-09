# src/04_train_xgboost_with_graph.py
import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

HOURLY_FEATS = "./outputs/hourly_features.csv"          #  run_pipeline.py
EMB_CSV      = "./outputs/node2vec_embeddings.csv"      #  03_graph_embedding.py
DATA_PATH    = "./data/flights_RUH.csv"               

assert os.path.exists(HOURLY_FEATS), "Run src/run_pipeline.py first"
assert os.path.exists(EMB_CSV), "Run src/03_graph_embedding.py first"

ts = pd.read_csv(HOURLY_FEATS, parse_dates=["date_hour"])
ts["terminal_id"] = ts["movement.terminal"].astype("category").cat.codes
emb = pd.read_csv(EMB_CSV)


df = pd.read_csv(DATA_PATH)
for col in ["movement.scheduledTime.local", "movement.scheduledTime.utc"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def pick_dt(row):
    return row["movement.scheduledTime.local"] if pd.notna(row.get("movement.scheduledTime.local")) else row.get("movement.scheduledTime.utc")

df["scheduled_dt"] = df.apply(pick_dt, axis=1)
df = df[df["scheduled_dt"].notna()].copy()
df["date_hour"] = df["scheduled_dt"].dt.floor("h")  

RUH_IATA, RUH_ICAO = "RUH", "OERK"
is_origin_RUH = (df.get("origin_airport_iata","").astype(str).str.upper()==RUH_IATA) | \
                (df.get("origin_airport_icao","").astype(str).str.upper()==RUH_ICAO)

outbound = df[is_origin_RUH].copy()
hour_dest = (outbound
             .groupby(["movement.terminal","date_hour","destination_airport_iata"])
             .size().reset_index(name="cnt"))
idx = hour_dest.groupby(["movement.terminal","date_hour"])["cnt"].idxmax()
top1 = hour_dest.loc[idx, ["movement.terminal","date_hour","destination_airport_iata"]] \
                .rename(columns={"destination_airport_iata":"dominant_dest"})


ts = ts.merge(top1, on=["movement.terminal","date_hour"], how="left")
ts["dominant_dest"] = ts["dominant_dest"].astype(str)
ts = ts.merge(emb, left_on="dominant_dest", right_on="airport", how="left").drop(columns=["airport"])


vec_cols = [c for c in ts.columns if c.startswith("n2v_")]
features = ["terminal_id","hour","weekday","month","is_weekend","lag1","lag3","lag6","ma3","ma6"] + vec_cols
target   = "y"

data = ts.dropna(subset=features + [target]).sort_values("date_hour").reset_index(drop=True)


split = int(len(data)*0.8)
train, test = data.iloc[:split], data.iloc[split:]
X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# XGBoost
xgb = XGBRegressor(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)

def smape(y, yhat):
    y, yhat = np.array(y), np.array(pred)
    denom = (np.abs(y)+np.abs(yhat))/2
    denom[denom==0] = 1e-6
    return np.mean(np.abs(y-yhat)/denom)*100

report = {
    "MAE_XGB":   round(mean_absolute_error(y_test, pred), 3),
    "RMSE_XGB":  round(sqrt(mean_squared_error(y_test, pred)), 3),
    "sMAPE_XGB": round(smape(y_test, pred), 2),
    "test_hours": len(y_test),
    "n_features": len(features)
}
print(" XGBoost+Node2Vec report:", report)


os.makedirs("./outputs", exist_ok=True)
outpath = "./outputs/xgb_node2vec_predictions.csv"
pd.DataFrame({"date_hour": test["date_hour"], "actual": y_test, "pred": pred}).to_csv(outpath, index=False)
print("Saved predictions to", outpath)
