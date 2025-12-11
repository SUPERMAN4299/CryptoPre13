import os
import pandas as pd
import yfinance as yf
import ta
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# CONFIG
# -------------------------------
COINS = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
    "ADA-USD","AVAX-USD","DOGE-USD","DOT-USD","TRX-USD"
]

INTERVAL = "1h"
PERIOD = "730d"
FEATURES = ["Close","Volume","rsi","ema_9","ema_21","macd","atr"]


# -------------------------------
# STEP 1 — DOWNLOAD
# -------------------------------
def download_all():
    os.makedirs("data/raw", exist_ok=True)
    for coin in COINS:
        print(f"[DOWNLOADING] {coin}...")
        df = yf.download(coin, interval=INTERVAL, period=PERIOD)
        df = df.dropna()
        df.to_csv(f"data/raw/{coin}.csv")
    print("[✔] DOWNLOAD COMPLETE\n")



def add_features_to_all():
    os.makedirs("data/processed", exist_ok=True)
    for coin in COINS:
        file = f"data/raw/{coin}.csv"
        print(f"[FEATURES] {coin}...")

        df = pd.read_csv(file, index_col=0, parse_dates=True)

        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        df["ema_9"] = ta.trend.EMAIndicator(df["Close"], 9).ema_indicator()
        df["ema_21"] = ta.trend.EMAIndicator(df["Close"], 21).ema_indicator()
        df["macd"] = ta.trend.MACD(df["Close"]).macd()
        df["atr"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()

        df = df.dropna()
        df.to_csv(f"data/processed/feat_{coin}.csv")

    print("[✔] FEATURES COMPLETE\n")


def label_all(horizon=3, buy_th=0.003, sell_th=-0.003):
    all_df = []
    print("[LABELING] Creating BUY/HOLD/SELL labels...\n")

    for coin in COINS:
        file = f"data/processed/feat_{coin}.csv"
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1

        df["label"] = 1  # default HOLD
        df.loc[df["future_return"] <= sell_th, "label"] = 0  # SELL
        df.loc[df["future_return"] >= buy_th, "label"] = 2  # BUY

        df = df.dropna()
        all_df.append(df)

    merged = pd.concat(all_df)
    merged.to_csv("data/processed/all_coins_labeled.csv")
    print("[✔] LABELING COMPLETE\n")
    return merged


# -------------------------------
# STEP 4 — TRAIN MODEL
# -------------------------------
def train_model():
    print("[TRAINING] Loading dataset...")
    df = pd.read_csv("data/processed/all_coins_labeled.csv")

    X = df[FEATURES]
    y = df["label"]

    print("[TRAINING] Training XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/universal_signal_model.pkl", "wb"))

    print("[✔] MODEL TRAINED AND SAVED → models/universal_signal_model.pkl\n")


# -------------------------------
# RUN EVERYTHING
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 STARTING FULL TRAINING PIPELINE...\n")

    download_all()
    add_features_to_all()
    label_all()
    train_model()

    print("🎯 TRAINING PIPELINE FINISHED SUCCESSFULLY!")
