import pandas as pd
import numpy as np
import json
import os
import time
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#NUM_PCA_FEATURES = 20
NUM_PCA_FEATURES_TO_REPORT = 20
TOP_NUM_PCA_WEIGHTS_TO_REPORT = 20  # adjust for readability
NUM_FOLDS = 10
TOP_VARS_PER_COMPONENT = 1000
SAVE_TOP_VARS_CSV = "pca_vars_tract.csv"


# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--use_voting", action="store_true", help="Include margin_dem (voting) in the model")
parser.add_argument("--target", choices=["trash_avg", "trash_slope", "fire_avg", "fire_slope"])
parser.add_argument("--max_pca", type=int, default=1000, help="Maximum number of PCA components to evaluate")

args = parser.parse_args()

NUM_PCA_FEATURES = args.max_pca

if args.target == "trash_avg":
    target_column = "avg_score"
elif args.target == "trash_slope":
    target_column = "trash_score_slope"
elif args.target == "fire_avg":
    target_column = "avg_fire_distance_km"
elif args.target == "fire_slope":
    target_column = "fire_distance_slope"
else:
    raise ValueError(f"❌ Unknown target: {args.target}")

import argparse

def normalize_var(var):
    # Convert B19313G_E001 → B19313G_001E
    if "_E" in var:
        prefix, suffix = var.split("_E")
        return f"{prefix}_{suffix}E"
    return var

VARS_JSON_PATH = "./vars.json"
def load_var_metadata():
    if os.path.exists(VARS_JSON_PATH):
        with open(VARS_JSON_PATH, "r") as f:
            metadata = json.load(f).get("variables", {})
            return {
                k: {
                    "label": v.get("label", ""),
                    "concept": v.get("concept", "")
                }
                for k, v in metadata.items()
            }
    else:
        print(f"⚠️ Warning: {VARS_JSON_PATH} not found. Metadata will be omitted.")
        return {}


def describe_pca_components(pca, feature_names, metadata):
    print("\n🔍 PCA Component Descriptions (Top Loadings):")
    components = pca.components_
    for i, comp in enumerate(components[:NUM_PCA_FEATURES_TO_REPORT]):
        print(f"\n🧭 PCA Component {i+1}")
        sorted_idx = np.argsort(np.abs(comp))[::-1]

        labels = []
        for rank in range(TOP_NUM_PCA_WEIGHTS_TO_REPORT):
            idx = sorted_idx[rank]
            var = feature_names[idx]
            normalized_var = normalize_var(var)
            label = metadata.get(normalized_var, {}).get("label", "N/A")
            labels.append(label)
        print(f"  Labels: {' | '.join(labels)}")

        print("  Top Weights:")
        for rank in range(TOP_NUM_PCA_WEIGHTS_TO_REPORT):
            idx = sorted_idx[rank]
            var = feature_names[idx]
            weight = comp[idx]
            normalized_var = normalize_var(var)
            label = metadata.get(normalized_var, {}).get("label", "N/A")
            print(f"    {rank+1:2d}. {var:30s} weight = {weight:+.4f} | {label}")

# === Config ===
CSV_PATH = "demographics_with_trash_score_tract.parquet"
IGNORED_COLS = {
    "NAME", "state", "county", "state_fips", "county_fips",
    "tract_fips", "tract_geoid", "avg_score", "std_score", "num_samples"
}
if not args.use_voting:
    IGNORED_COLS.add("margin_dem")

# === Load Data ===
df = pd.read_parquet(CSV_PATH)
y = df[target_column].dropna()
drop_cols = [col for col in IGNORED_COLS if col in df.columns]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
X = X.loc[y.index].fillna(0)
# === Automatically exclude any features related to the target
target_keywords = {
    "trash_avg": ["trash", "avg_score"],
    "trash_slope": ["trash", "slope"],
    "fire_avg": ["fire", "avg_fire_distance"],
    "fire_slope": ["fire", "slope"],
}
for keyword in target_keywords[args.target]:
    X = X.drop(columns=[col for col in X.columns if keyword in col], errors="ignore")

# Automatically exclude direct predictors of the target to prevent leakage
if target_column in ("trash_avg", "trash_score_slope"):
    X = X.drop(columns=[col for col in X.columns if "trash" in col], errors="ignore")

if target_column in ("fire_avg", "fire_slope"):
    X = X.drop(columns=[col for col in X.columns if "fire" in col], errors="ignore")


# === Standardize and PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=NUM_PCA_FEATURES)
X_pca = pca.fit_transform(X_scaled)

# === Describe PCA components ===
metadata = load_var_metadata()
describe_pca_components(pca, X.columns.tolist(), metadata)


# === Save top PCA variable loadings to CSV ===
print("\n💾 Saving top PCA variable loadings...")
loading_df = []
loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f"pca_{i+1}" for i in range(NUM_PCA_FEATURES)])
for i in range(NUM_PCA_FEATURES):
    row = loadings.iloc[i]
    abs_sorted = row.abs().sort_values(ascending=False)[:TOP_VARS_PER_COMPONENT]
    for var in abs_sorted.index:
        loading_df.append({
            "component": f"pca_{i+1}",
            "variable": var,
            "weight": row[var]
        })
load_df = pd.DataFrame(loading_df)
load_df.to_csv(SAVE_TOP_VARS_CSV, index=False)
print(f"✅ Saved: {SAVE_TOP_VARS_CSV} with {len(load_df)} rows")

# === CV to Select Best k ===
print("\n🔍 Running 10-fold CV to select optimal number of PCA components...")
cv_r2_scores = []
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for k in range(1, NUM_PCA_FEATURES + 1):
    r2_scores = []
    start = time.time()
    for train_idx, test_idx in kf.split(X_pca):
        X_train, X_test = X_pca[train_idx, :k], X_pca[test_idx, :k]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LinearRegression().fit(X_train, y_train)
        r2_scores.append(r2_score(y_test, model.predict(X_test)))
    mean_r2 = np.mean(r2_scores)
    cv_r2_scores.append(mean_r2)
    elapsed = time.time() - start
    remaining = (NUM_PCA_FEATURES - k) * elapsed
    eta = time.strftime('%H:%M:%S', time.gmtime(remaining))
    print(f"  k={k:2d}: CV R² = {mean_r2:.12f} | ⏱ {elapsed:.3f}s | ETA: {eta}")

# === Refit with Best k ===
best_k = np.argmax(cv_r2_scores) + 1
print(f"\n✅ Best number of PCA components: {best_k} (CV R² = {cv_r2_scores[best_k-1]:.12f})")

model_final = LinearRegression().fit(X_pca[:, :best_k], y)
r2_final = model_final.score(X_pca[:, :best_k], y)
coefs = model_final.coef_

print(f"\n📈 Final refit on full data: R² = {r2_final:.12f}")
print("\n📊 Coefficients:")
for i, c in enumerate(coefs):
    print(f"  pca_{i+1:2d}: coef = {c:+.12e}")

# === Save PCA regression coefficients ===
coef_df = pd.DataFrame({
    "component": [f"pca_{i+1}" for i in range(best_k)],
    "coef": coefs
})
coef_df.to_csv("pca_tract_coefs.csv", index=False)
print("✅ Saved PCA regression coefficients to pca_tract_coefs.csv")


# === Save predictions by GEOID ===
preds = model_final.predict(X_pca[:, :best_k])
df_out = df.loc[y.index, ["tract_geoid"]].copy()

df_out["predicted_score"] = preds
df_out["tract_geoid"] = df_out["tract_geoid"].astype(str).str.zfill(11)
pred_out = f"predicted_score_by_tract_{args.target}.csv"
df_out.to_csv(pred_out, index=False)
print(f"✅ Saved predicted scores to {pred_out}")



print("✅ Saved predicted scores to predicted_score_by_tract.csv")
