"""
migrate_model.py – Re-serialise the insurance model with the current sklearn.

Run once before starting the backend:
    python migrate_model.py
"""

import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = Path(__file__).parent

# ── 1. Load original model (may warn about version mismatch) ──
print("Loading original model (version warnings are expected)…")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        old_model = joblib.load(BASE / "insurance_model .pkl")
        print("  Original model loaded successfully.")
        original_loaded = True
    except Exception as e:
        print(f"  Could not load original model: {e}")
        original_loaded = False

# ── 2. Load dataset ───────────────────────────────────────────
df = pd.read_csv(BASE / "insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]
print(f"  Dataset: {len(df)} rows, columns: {list(X.columns)}")

# ── 3. If original loaded, try using it directly ──────────────
if original_loaded:
    try:
        preds = old_model.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"  Original model works! R²={r2:.4f}, MAE={mae:.0f}, RMSE={rmse:.0f}")

        # Re-save with current version
        out = BASE / "insurance_model_compat.pkl"
        joblib.dump(old_model, out)
        print(f"  ✅ Saved as {out.name}")
        exit(0)
    except Exception as e:
        print(f"  Original model predict failed: {e}. Will retrain.")

# ── 4. Retrain a fresh pipeline ───────────────────────────────
print("\nRetraining a fresh RandomForest pipeline…")

cat_cols = ["sex", "smoker", "region"]
num_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X, y)
preds = pipeline.predict(X)
r2 = r2_score(y, preds)
mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
print(f"  Retrained model: R²={r2:.4f}, MAE={mae:.0f}, RMSE={rmse:.0f}")

out = BASE / "insurance_model_compat.pkl"
joblib.dump(pipeline, out)
print(f"  ✅ Saved as {out.name}")
