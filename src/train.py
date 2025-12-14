import pandas as pd
import numpy as np
import os
import json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# ========================================================
# PATH SETUP
# ========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "universal_signal_model.xgb")
SCALER_PATH = os.path.join(MODEL_DIR, "universal_scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ========================================================
# LOAD ALL LABELED DATA
# ========================================================
def load_dataset():
    print("[INFO] Loading labeled feature files...")
    files = glob(os.path.join(DATA_PATH, "labeled_feat_*.csv"))

    if not files:
        raise Exception("[ERROR] No labeled feature files found!")

    df_list = []
    for f in files:
        df = pd.read_csv(f)

        # Skip too-small datasets
        if len(df) < 100:
            print(f"[SKIP] {os.path.basename(f)} too small ({len(df)} rows) — skipping")
            continue

        print(f"[✔] Loaded {os.path.basename(f)} ({len(df)} rows)")
        df_list.append(df)

    if not df_list:
        raise Exception("[ERROR] No valid datasets to train on!")

    merged = pd.concat(df_list, ignore_index=True)
    print(f"\n[INFO] MERGED TOTAL ROWS → {len(merged)}\n")

    return merged


# ========================================================
# PREPARE FEATURES FOR TRAINING
# ========================================================
def prepare_data(df):
    if "LABEL" not in df.columns:
        raise Exception("[ERROR] LABEL column is missing in dataset!")

    y = df["LABEL"]

    # Select only numeric columns except LABEL
    X = df.select_dtypes(include=[np.number]).drop(columns=["LABEL"], errors="ignore")

    # Remove NaNs
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # SCALE DATA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save feature names (for predict.py)
    feature_names = list(X.columns.astype(str))

    print("[INFO] Number of Features:", len(feature_names))
    print("[INFO] Class distribution:")
    print(y.value_counts(), "\n")

    return X_scaled, y, scaler, feature_names


# ========================================================
# TRAIN XGBOOST MODEL
# ========================================================
def train_model():
    df = load_dataset()
    X, y, scaler, feature_names = prepare_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y
    )

    print("[INFO] Training XGBoost Model...\n")

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"
    }

    evals = [(dtrain, "train"), (dtest, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # ======================================================
    # SAVE MODEL + SCALER + FEATURE_NAMES
    # ======================================================
    model.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_names, f, indent=4)

    print(f"\n[✔] MODEL SAVED → {MODEL_PATH}")
    print(f"[✔] SCALER SAVED → {SCALER_PATH}")
    print(f"[✔] FEATURE NAMES SAVED → {FEATURE_PATH}")

    # ======================================================
    # EVALUATE MODEL
    # ======================================================
    preds = model.predict(dtest)
    accuracy = (preds == y_test).mean()

    print(f"\n[✔] TEST ACCURACY: {accuracy:.4f}")

    # Show top features
    importance = model.get_score(importance_type="gain")
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n[TOP IMPORTANT FEATURES]")
    for k, v in importance_sorted[:20]:
        print(f"{k:20s} → {v:.4f}")

    return model


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    print(">> STARTING TRAINING PIPELINE...\n")
    train_model()
