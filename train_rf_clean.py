#!/usr/bin/env python3
"""
train_rf_final.py

Train a RandomForest classifier on the prepared features while removing
leaking columns (dep_scheduled, dep_actual, delay_min) and the flight_number
identifier. Saves model, metrics, feature importances and plots to results/clean_rf/.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Paths
# -------------------------
BASE = Path.home() / "thesis"
DATA_IN = BASE / "data" / "processed" / "training_features.csv"
OUT_DIR = BASE / "results" / "clean_rf"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Config
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200

# Columns to remove because they leak future info (or are identifiers)
LEAK_COLS = ["dep_scheduled", "dep_actual", "delay_min", "flight_number", "hour", "collected_at"]

# Target column
TARGET = "delayed_15"

# -------------------------
# Load data
# -------------------------
print("📦 Loading feature file:", DATA_IN)
if not DATA_IN.exists():
    raise SystemExit(f"Missing features file: {DATA_IN}")

df = pd.read_csv(DATA_IN)

print("Initial shape:", df.shape)
# ensure target exists
if TARGET not in df.columns:
    raise SystemExit(f"Target column '{TARGET}' not found in features file.")

# -------------------------
# Clean / drop leakage
# -------------------------
present_leaks = [c for c in LEAK_COLS if c in df.columns]
if present_leaks:
    print("✅ Dropping leakage / identifier columns:", present_leaks)
    df = df.drop(columns=present_leaks)

# Drop rows with missing target
n_before = len(df)
df = df.dropna(subset=[TARGET])
print(f"Dropped {n_before - len(df)} rows with missing target. New shape: {df.shape}")

# -------------------------
# Prepare X, y
# -------------------------
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# Keep only useful numeric/time features.
# Convert any datetime-like columns that remained to numeric features if needed.
# But prefer to drop 'hour' earlier; if there is any other datetime-like column, drop it.
for c in X.columns:
    if "time" in c.lower() or "date" in c.lower():
        # avoid complex timezone-aware merging issues in models
        print(f"-> Dropping datetime-like column: {c}")
        X = X.drop(columns=[c])

# Identify categorical columns to one-hot encode (object dtype)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns to encode:", cat_cols)

# One-hot encode categoricals with drop_first=False (keeps interpretability).
# We will use pd.get_dummies to avoid label-encoding pitfalls.
X_enc = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

print("Feature matrix after encoding:", X_enc.shape)

# Ensure all column dtypes are numeric
# Convert boolean to int
for c in X_enc.select_dtypes(include=["bool"]).columns:
    X_enc[c] = X_enc[c].astype(int)

# Replace infinite values (if any)
X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)

# -------------------------
# Train / Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Target distribution (train):", np.bincount(y_train))
print("Target distribution (test):", np.bincount(y_test))

# -------------------------
# Train RandomForest
# -------------------------
print("🔧 Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"  # helps with class imbalance
)
rf.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = rf.predict(X_test)
try:
    y_proba = rf.predict_proba(X_test)[:, 1]
except Exception:
    # fallback if predict_proba unavailable
    y_proba = None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "roc_auc": float(roc) if roc is not None else None,
    "confusion_matrix": cm.tolist(),
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

print("\n📊 Evaluation metrics:")
for k, v in metrics.items():
    if k != "confusion_matrix":
        print(f"  {k}: {v}")
print("  confusion_matrix:\n", cm)

# -------------------------
# Save model + metrics
# -------------------------
model_path = OUT_DIR / "rf_clean_model.joblib"
metrics_path = OUT_DIR / "rf_clean_metrics.json"
fi_path = OUT_DIR / "feature_importances.csv"

joblib.dump(rf, model_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# feature importances (map back to feature names)
importances = pd.Series(rf.feature_importances_, index=X_enc.columns)
importances = importances.sort_values(ascending=False)
importances.to_csv(fi_path, header=["importance"])

print("✅ Model, metrics, and importances saved to:", OUT_DIR)

# -------------------------
# Plots
# -------------------------
# 1) Feature importances barplot (top 15)
plt.figure(figsize=(10, 6))
top_n = 15
sns.barplot(x=importances.values[:top_n], y=importances.index[:top_n], palette="viridis")
plt.title(f"Top {top_n} Feature Importances (Clean RF)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importances.png", dpi=150)
plt.close()
print("✅ Saved feature importances →", PLOTS_DIR / "feature_importances.png")

# 2) Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred On Time", "Pred Delayed"],
            yticklabels=["Actual On Time", "Actual Delayed"])
plt.title("Confusion Matrix - Clean Random Forest")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("✅ Saved confusion matrix →", PLOTS_DIR / "confusion_matrix.png")

# 3) ROC curve if probabilities exist
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Clean Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print("✅ Saved ROC curve →", PLOTS_DIR / "roc_curve.png")

print("🎉 All outputs saved. Done.")