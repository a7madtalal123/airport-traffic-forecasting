
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------
# 1) Configure input path
# -------------------------------
# Change this if your file lives elsewhere
FEATURES_PATH = "./outputs/hourly_features.csv"

# Candidate names for the target column (next-hour movements)
TARGET_CANDIDATES = ["y_t_plus_1", "y_next", "target", "y", "y_t1"]

# Columns to drop if present (non-predictive identifiers / timestamps, etc.)
DROP_COL_HINTS = [
    "timestamp", "time", "date", "datetime",
    "flight", "flight_id", "id",
    "origin", "destination", "icao", "iata",
    "terminal_name", "status"
]

TOPK = 20  # how many top features to plot in the single-model charts

# -------------------------------
# 2) Load data
# -------------------------------
if not os.path.exists(FEATURES_PATH):
    sys.exit(f"[ERROR] Could not find file: {FEATURES_PATH}\n"
             f"Please adjust FEATURES_PATH at the top of features.py.")

df = pd.read_csv(FEATURES_PATH)

# Detect target column
target_col = None
for cand in TARGET_CANDIDATES:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    # fallback: if y_t exists use it (will reflect current hour)
    if "y_t" in df.columns:
        target_col = "y_t"
    else:
        sys.exit("[ERROR] Could not infer target column. "
                 "Rename your target to one of "
                 f"{TARGET_CANDIDATES + ['y_t']}.")

# -------------------------------
# 3) Select features
# -------------------------------
# numeric only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != target_col]

# drop obvious non-predictive columns by keyword
filtered = []
for c in feature_cols:
    lc = c.lower()
    if any(h in lc for h in DROP_COL_HINTS):
        continue
    filtered.append(c)

if len(filtered) == 0:
    sys.exit("[ERROR] No numeric features left after filtering. "
             "Please review your CSV columns.")

feature_cols = filtered
X = df[feature_cols].values
y = df[target_col].values

# Keep chronological order if already sorted (do not shuffle)
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# 4) Train models (regressors)
# -------------------------------
# XGBoost settings aligned with paper
xgb = XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    reg_alpha=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)

# -------------------------------
# 5) Compute importances
# -------------------------------
# XGB returns dict with keys f0, f1, ...
raw_imp = xgb.get_booster().get_score(importance_type="gain")

def map_xgb_to_names(xgb_dict, names):
    out = {}
    for k, v in xgb_dict.items():
        if k.startswith("f"):
            idx = int(k[1:])
            if 0 <= idx < len(names):
                out[names[idx]] = float(v)
    return out

xgb_imp_named = map_xgb_to_names(raw_imp, feature_cols)
xgb_df = pd.DataFrame(
    sorted(xgb_imp_named.items(), key=lambda kv: kv[1], reverse=True),
    columns=["feature", "importance_gain"]
)

rf_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_.astype(float)
}).sort_values("importance", ascending=False)

# Save CSVs for the paper artifacts
xgb_df.to_csv("feature_importance_xgb.csv", index=False)
rf_df.to_csv("feature_importance_rf.csv", index=False)

# -------------------------------
# 6) Plotting helpers (no seaborn, single plots, default colors)
# -------------------------------
def plot_topk(df_imp, val_col, title, out_path, topk=TOPK):
    top = df_imp.head(topk).iloc[::-1]  # reverse for horizontal bar
    plt.figure(figsize=(8, max(6, 0.4 * len(top))))
    plt.barh(top["feature"], top[val_col])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# Single-model charts
plot_topk(xgb_df, "importance_gain", "XGBoost Feature Importance (gain)", "feature_importance_xgb.png")
plot_topk(rf_df, "importance", "Random Forest Feature Importance", "feature_importance_rf.png")

# Side-by-side comparison on the union of topK from both
top_x = xgb_df.head(TOPK).copy()
top_r = rf_df.head(TOPK).copy()
union_feats = list(dict.fromkeys(list(top_x["feature"]) + list(top_r["feature"])))

def align(vals_features, df_imp, col_name):
    mp = dict(zip(df_imp["feature"], df_imp[col_name]))
    return [mp.get(f, 0.0) for f in vals_features]

x_vals_xgb = align(union_feats, xgb_df, "importance_gain")
x_vals_rf  = align(union_feats, rf_df, "importance")

fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.4 * len(union_feats))), sharey=True)
axes[0].barh(union_feats[::-1], x_vals_xgb[::-1])
axes[0].set_title("XGBoost (gain)")
axes[0].set_xlabel("Importance")

axes[1].barh(union_feats[::-1], x_vals_rf[::-1])
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance_side_by_side.png", dpi=300)
plt.close()

print("Saved figures:")
print(" - feature_importance_xgb.png")
print(" - feature_importance_rf.png")
print(" - feature_importance_side_by_side.png")
print("Also saved CSVs: feature_importance_xgb.csv, feature_importance_rf.csv")
