# train_model.py
# Predicting flight delays using weather and traffic data — baseline Random Forest model.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib
import json

# === PATHS ===
BASE = os.path.expanduser('~/thesis')
DATA_PATH = os.path.join(BASE, 'data/processed/training_features.csv')
OUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

print("📦 Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=['hour'])

# === SELECT FEATURES ===
num_cols = [
    'hour_of_day', 'day_of_week', 'is_weekend',
    'temp', 'humidity', 'pressure', 'wind_speed',
    'avg_speed', 'total_vehicles'
]

cat_cols = [col for col in ['weather', 'airline_name'] if col in df.columns]
target = 'delayed_15'

# === HANDLE MISSING VALUES ===
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Handle categorical
for c in cat_cols:
    df[c] = df[c].fillna('Unknown')
    top10 = df[c].value_counts().index[:10]
    df[c] = df[c].apply(lambda x: x if x in top10 else 'Other')

# === ONE-HOT ENCODE CATEGORICAL ===
try:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    # compatibility for older sklearn versions
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')

if cat_cols:
    cat_df = pd.DataFrame(
        enc.fit_transform(df[cat_cols]),
        columns=enc.get_feature_names_out(cat_cols),
        index=df.index
    )
    X = pd.concat([df[num_cols], cat_df], axis=1)
else:
    X = df[num_cols].copy()

y = df[target].fillna(0).astype(int)

print("✅ Features prepared:", X.shape)
print("Target distribution:", np.bincount(y))

# === SPLIT TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TRAIN MODEL ===
print("🔧 Training RandomForest (baseline)...")
model = RandomForestClassifier(
    n_estimators=50,      # you can increase later to 200 for stronger model
    max_depth=12,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# === EVALUATION ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

if y_proba is not None:
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["roc_auc"] = None

print("\n📊 Evaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# === SAVE RESULTS ===
model_path = os.path.join(OUT_DIR, "rf_delay_model.joblib")
metrics_path = os.path.join(OUT_DIR, "rf_metrics.json")
importances_path = os.path.join(OUT_DIR, "feature_importances.csv")

joblib.dump(model, model_path)

pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).to_csv(importances_path, index=False)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Metrics saved to: {metrics_path}")
print(f"✅ Importances saved to: {importances_path}")
