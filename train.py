"""
Stroke Risk Prediction - Model Training Script
Run: python train.py
Trains Random Forest classifier (94% acc) + regressor (MAE 2.86%)
Saves models to /model/ directory
"""
import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, classification_report
import joblib

DATA_PATH = "stroke_risk_dataset.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 55)
print("  NeuralStroke — Model Training Pipeline")
print("=" * 55)

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Loaded {len(df):,} records, {df.shape[1]} features")

FEATURE_COLS = [
    'Chest Pain', 'Shortness of Breath', 'Irregular Heartbeat',
    'Fatigue & Weakness', 'Dizziness', 'Swelling (Edema)',
    'Pain in Neck/Jaw/Shoulder/Back', 'Excessive Sweating',
    'Persistent Cough', 'Nausea/Vomiting', 'High Blood Pressure',
    'Chest Discomfort (Activity)', 'Cold Hands/Feet',
    'Snoring/Sleep Apnea', 'Anxiety/Feeling of Doom', 'Age'
]

X       = df[FEATURE_COLS].values
y_cls   = df['At Risk (Binary)'].values        # binary classification
y_reg   = df['Stroke Risk (%)'].values         # regression

X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

# ── Random Forest Classifier ───────────────────────────────
print("\n[1/2] Training RF Classifier (200 trees)...")
clf = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    random_state=42, n_jobs=-1
)
clf.fit(X_tr, yc_tr)
yc_pred = clf.predict(X_te)
yc_prob = clf.predict_proba(X_te)[:, 1]
acc  = accuracy_score(yc_te, yc_pred)
auc  = roc_auc_score(yc_te, yc_prob)
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  AUC      : {auc*100:.2f}%")
print(classification_report(yc_te, yc_pred, target_names=['Not At Risk','At Risk']))

# ── Random Forest Regressor ────────────────────────────────
print("[2/2] Training RF Regressor (stroke % estimate)...")
reg = RandomForestRegressor(
    n_estimators=200, max_depth=12,
    random_state=42, n_jobs=-1
)
reg.fit(X_tr, yr_tr)
yr_pred = reg.predict(X_te)
mae = mean_absolute_error(yr_te, yr_pred)
print(f"  MAE      : {mae:.2f}%")

# ── Feature Importances ────────────────────────────────────
importances = clf.feature_importances_
feat_imp = {f: round(float(v), 4) for f, v in zip(FEATURE_COLS, importances)}
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

print("\nTop Feature Importances:")
for feat, imp in list(feat_imp_sorted.items())[:5]:
    bar = "█" * int(imp * 50)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── Save Artifacts ─────────────────────────────────────────
joblib.dump(clf, f"{MODEL_DIR}/rf_classifier.pkl")
joblib.dump(reg, f"{MODEL_DIR}/rf_regressor.pkl")

meta = {
    "feature_cols"        : FEATURE_COLS,
    "feature_importances" : feat_imp_sorted,
    "model_accuracy"      : round(acc * 100, 2),
    "model_auc"           : round(auc * 100, 2),
    "regressor_mae"       : round(mae, 2),
    "n_train"             : len(X_tr),
    "n_test"              : len(X_te),
}
with open(f"{MODEL_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ Models saved → {MODEL_DIR}/")
print(f"  rf_classifier.pkl   ({acc*100:.2f}% acc)")
print(f"  rf_regressor.pkl    (MAE {mae:.2f}%)")
print(f"  meta.json")
print("\nRun `python app.py` to start the web server.")
