# src/04b_train_classic_ml.py
import os
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- مسارات الإدخال/الإخراج ----
FEATS = "./outputs/hourly_features.csv"          # من run_pipeline.py
EMB   = "./outputs/node2vec_embeddings.csv"      # من 03_graph_embedding.py
DATA  = "./data/flights_RUH.csv"                 # ملف البيانات الأصلي
OUT_DIR = "./outputs"

assert os.path.exists(FEATS), "Run src/run_pipeline.py first"
assert os.path.exists(EMB),   "Run src/03_graph_embedding.py first"
os.makedirs(OUT_DIR, exist_ok=True)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-6
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

# ---- تحميل الميزات الأساسية بالساعة ----
ts = pd.read_csv(FEATS, parse_dates=["date_hour"])
# نتأكد من وجود terminal_id لأن ملف hourly_features.csv قد لا يحتويه
if "terminal_id" not in ts.columns:
    ts["movement.terminal"] = ts["movement.terminal"].astype(str)
    ts["terminal_id"] = ts["movement.terminal"].astype("category").cat.codes

# ---- حساب الوجهة المسيطرة لكل ساعة (للرحلات الصادرة من RUH) ----
df = pd.read_csv(DATA)
for col in ["movement.scheduledTime.local", "movement.scheduledTime.utc"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def pick_dt(row):
    return row["movement.scheduledTime.local"] if pd.notna(row.get("movement.scheduledTime.local")) else row.get("movement.scheduledTime.utc")

df["scheduled_dt"] = df.apply(pick_dt, axis=1)
df = df[df["scheduled_dt"].notna()].copy()
df["date_hour"] = df["scheduled_dt"].dt.floor("h")

RUH_IATA, RUH_ICAO = "RUH","OERK"
is_origin_RUH = (df.get("origin_airport_iata","").astype(str).str.upper()==RUH_IATA) | \
                (df.get("origin_airport_icao","").astype(str).str.upper()==RUH_ICAO)
outbound = df[is_origin_RUH].copy()

hour_dest = (outbound
             .groupby(["movement.terminal","date_hour","destination_airport_iata"])
             .size().reset_index(name="cnt"))
idx = hour_dest.groupby(["movement.terminal","date_hour"])["cnt"].idxmax()
top1 = hour_dest.loc[idx, ["movement.terminal","date_hour","destination_airport_iata"]] \
                .rename(columns={"destination_airport_iata":"dominant_dest"})

# ---- دمج Node2Vec ----
emb = pd.read_csv(EMB)
ts = ts.merge(top1, on=["movement.terminal","date_hour"], how="left")
ts["dominant_dest"] = ts["dominant_dest"].astype(str)
ts = ts.merge(emb, left_on="dominant_dest", right_on="airport", how="left").drop(columns=["airport"])

# ---- استكمال ميزات التأخر والمتوسطات لو ناقصة بعد الدمج ----
for col in ["lag1","lag3","lag6","ma3","ma6"]:
    if col not in ts.columns:
        if col.startswith("lag"):
            ts[col] = ts.groupby("movement.terminal")["y"].shift(int(col[-1]))
        else:
            w = int(col[-1])
            ts[col] = ts.groupby("movement.terminal")["y"].rolling(w).mean().reset_index(level=0,drop=True).shift(1)

# إسقاط أي صفوف ناقصة
ts = ts.dropna().sort_values("date_hour").reset_index(drop=True)

# ---- تعريف قائمة الميزات والهدف ----
vec_cols = [c for c in ts.columns if c.startswith("n2v_")]
base_feats = ["terminal_id","hour","weekday","month","is_weekend","lag1","lag3","lag6","ma3","ma6"]
features = base_feats + vec_cols
target = "y"

# ---- تقسيم زمني 80/20 ----
split = int(len(ts) * 0.8)
train, test = ts.iloc[:split], ts.iloc[split:]
X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# ---- تعريف النماذج ----
# LR: نستخدم Ridge لاستقرار أفضل على ميزات كثيرة، مع StandardScaler
models = {
    "LR(Ridge)": Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ]),
    "SVR(RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
    ]),
    "RF": RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42, min_samples_leaf=1
    ),
}

# ---- تدريب وتقييم وحفظ التوقعات ----
all_reports = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rep = {
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "RMSE": round(sqrt(mean_squared_error(y_test, pred)), 3),
        "sMAPE": round(smape(y_test, pred), 2),
        "n_features": len(features),
        "test_hours": len(y_test),
    }
    all_reports.append(rep)

    # حفظ التوقعات لكل نموذج
    out_pred = os.path.join(OUT_DIR, f"{name.replace('(','_').replace(')','').replace('/','-').replace(' ','_').lower()}_predictions.csv")
    pd.DataFrame({"date_hour": test["date_hour"], "actual": y_test, "pred": pred}).to_csv(out_pred, index=False)
    print(f"Saved predictions: {out_pred}")

# ---- حفظ تقرير المقاييس ----
rep_df = pd.DataFrame(all_reports).sort_values("RMSE")
rep_csv = os.path.join(OUT_DIR, "classic_ml_metrics.csv")
rep_df.to_csv(rep_csv, index=False)
print("\n✅ Classic ML report:")
print(rep_df.to_string(index=False))
print(f"Saved metrics: {rep_csv}")
