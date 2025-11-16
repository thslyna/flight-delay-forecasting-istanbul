import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# === SETUP ===
BASE = os.path.expanduser('~/thesis')
PLOT_DIR = os.path.join(BASE, 'results', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# === 1️⃣ FEATURE IMPORTANCES ===
feat_path = os.path.join(BASE, 'results', 'feature_importances.csv')
df_feat = pd.read_csv(feat_path).sort_values('importance', ascending=True).tail(12)

plt.figure(figsize=(9, 6))
plt.barh(df_feat['feature'], df_feat['importance'], color='skyblue', edgecolor='black')
plt.title('Top 12 Feature Importances – Flight Delay Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_importances.png'), dpi=300)
plt.close()

print(f"✅ Saved feature importances → {PLOT_DIR}/feature_importances.png")

# === 2️⃣ CONFUSION MATRIX ===
metrics_path = os.path.join(BASE, 'results', 'rf_metrics.json')
with open(metrics_path) as f:
    m = json.load(f)

cm = pd.DataFrame(
    m["confusion_matrix"],
    index=["Actual On Time", "Actual Delayed"],
    columns=["Predicted On Time", "Predicted Delayed"]
)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix – Random Forest Flight Delay Model')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

print(f"✅ Saved confusion matrix → {PLOT_DIR}/confusion_matrix.png")

# === 3️⃣ ROC CURVE ===
# Load model and training features
model_path = os.path.join(BASE, 'results', 'rf_delay_model.joblib')
model = joblib.load(model_path)

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'training_features.csv'))

# numerical columns we used
num_cols = [
    'hour_of_day', 'day_of_week', 'is_weekend',
    'temp', 'humidity', 'pressure', 'wind_speed',
    'avg_speed', 'total_vehicles'
]

# categorical columns we used in training
cat_cols = [c for c in ['weather', 'airline_name'] if c in df.columns]
target = 'delayed_15'

# fill numerical missing with medians
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
    else:
        # if a numeric column doesn't exist, create it as zeros
        df[c] = 0.0

# fill categorical missing
for c in cat_cols:
    df[c] = df[c].fillna('Unknown')

# build categorical one-hot (BUT DON'T trust column names; we'll align with model.feature_names_in_)
try:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')

if cat_cols:
    cat_df = pd.DataFrame(enc.fit_transform(df[cat_cols]),
                          columns=enc.get_feature_names_out(cat_cols),
                          index=df.index)
    X_full = pd.concat([df[num_cols], cat_df], axis=1)
else:
    X_full = df[num_cols].copy()

y = df[target].fillna(0).astype(int)

# Now align X_full columns with what the model expects
if hasattr(model, 'feature_names_in_'):
    expected = list(model.feature_names_in_)
    # For columns the model expects but are missing now -> add zeros
    for col in expected:
        if col not in X_full.columns:
            X_full[col] = 0.0
    # For extra cols in X_full not expected -> drop them
    X = X_full.reindex(columns=expected, fill_value=0.0)
else:
    # fallback if model has no feature_names_in_
    X = X_full.copy()

# train/test split (same method as training script)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# predict probabilities and plot ROC
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Flight Delay Model')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'roc_curve.png'), dpi=300)
plt.close()

print(f"✅ Saved ROC curve → {PLOT_DIR}/roc_curve.png")
print("\n🎨 All plots successfully generated and saved!")
