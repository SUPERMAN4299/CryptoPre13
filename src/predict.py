import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
import json
from difflib import get_close_matches
from xgboost import XGBClassifier


# ------------------------------
# MODEL PATHS
# ------------------------------
MODEL_PATH = "CryptoPre13/models/universal_signal_model.xgb"
SCALER_PATH = "CryptoPre13/models/universal_scaler.pkl"
FEATURE_PATH = "CryptoPre13/models/feature_names.json"


API_BASE = "https://api.binance.com"
API_URL = API_BASE + "/api/v3/klines"
EXCHANGE_INFO = API_BASE + "/api/v3/exchangeInfo"
INTERVAL = "1m"
LIMIT = 200


# ------------------------------
# FETCH VALID SYMBOLS FROM BINANCE
# ------------------------------
def get_binance_symbols():
    try:
        r = requests.get(EXCHANGE_INFO, timeout=5)
        data = r.json()
        return [s["symbol"] for s in data["symbols"]]
    except:
        return []


BINANCE_SYMBOLS = get_binance_symbols()


# ------------------------------
# AUTO SYMBOL DETECTION & CORRECTION
# ------------------------------
def normalize_symbol(symbol):
    symbol = symbol.upper()

    # Exact match → good
    if symbol in BINANCE_SYMBOLS:
        return symbol, None

    # Convert USD → USDT
    if symbol.endswith("USD"):
        guess = symbol.replace("USD", "USDT")
        if guess in BINANCE_SYMBOLS:
            return guess, f"{symbol} not found. Using {guess}."

    # If only coin is given (e.g., BTC → BTCUSDT)
    if len(symbol) <= 5:
        guess = symbol + "USDT"
        if guess in BINANCE_SYMBOLS:
            return guess, f"{symbol} is incomplete. Using {guess}."

    # Fuzzy match for typo correction
    close = get_close_matches(symbol, BINANCE_SYMBOLS, n=1, cutoff=0.6)
    if close:
        return close[0], f"{symbol} not found. Did you mean {close[0]}?"

    # Nothing found
    return None, f"Symbol {symbol} is invalid on Binance."


# ------------------------------
# LOAD MODEL, SCALER, FEATURES
# ------------------------------
def load_assets():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    return model, scaler, features


# ------------------------------
# TECHNICAL INDICATORS
# ------------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_hist(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line - signal

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ------------------------------
# FETCH MARKET DATA
# ------------------------------
def get_live_data(symbol):
    params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
    r = requests.get(API_URL, params=params, timeout=5)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time","Open","High","Low","Close","Volume",
        "_1","_2","_3","_4","_5","_6"
    ])

    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
    return df


# ------------------------------
# BUILD FEATURES
# ------------------------------
def build_features(df):
    df["return"] = df["Close"].pct_change()
    df["regime"] = (df["return"] > 0).astype(int)

    df["ema_9"] = ema(df["Close"], 9)
    df["ema_21"] = ema(df["Close"], 21)
    df["ema_50"] = ema(df["Close"], 50)
    df["ema_100"] = ema(df["Close"], 100)

    df["rsi"] = rsi(df["Close"], 14)
    df["macd_hist"] = macd_hist(df["Close"])

    df["atr"] = atr(df)
    df["atr_pct"] = df["atr"] / df["Close"]

    df["future_close"] = 0
    df["future_return"] = 0

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df


# ------------------------------
# SAFE SCALING
# ------------------------------
def safe_scale(scaler, X):
    required = scaler.feature_names_in_
    for col in required:
        if col not in X.columns:
            X[col] = 0.0
    return scaler.transform(X[required])


# ------------------------------
# TRADING LOGIC
# ------------------------------
def trade_levels(signal, price):
    atr_val = price * 0.003

    if signal == "BUY":
        return price, price - 2*atr_val, price + 4*atr_val, 2.0
    if signal == "SELL":
        return price, price + 2*atr_val, price - 4*atr_val, 2.0
    return price, "-", "-", "-"

def trend_strength(df):
    score = 0
    c = df["Close"].iloc[-1]

    # EMAs
    if c > df["ema_9"].iloc[-1]: score += 1
    if c > df["ema_21"].iloc[-1]: score += 1
    if c > df["ema_50"].iloc[-1]: score += 1
    if c > df["ema_100"].iloc[-1]: score += 1

    # RSI
    r = df["rsi"].iloc[-1]
    if r > 55: score += 1
    if r < 45: score -= 1

    # MACD histogram
    if df["macd_hist"].iloc[-1] > 0: score += 1
    else: score -= 1

    return score

def trend_description(signal, score):
    if signal == "BUY":
        return f"Uptrend detected with positive momentum (Trend Score: {score})."
    if signal == "SELL":
        return f"Downtrend pressure increasing (Trend Score: {score})."
    return f"Market neutral; no strong trend (Trend Score: {score})."


# ------------------------------
# PREDICT SIGNAL
# ------------------------------
def predict_signal(symbol):
    model, scaler, FEATURES = load_assets()

    df = get_live_data(symbol)
    df = build_features(df)

    X = df[FEATURES].tail(1).copy()
    X_scaled = safe_scale(scaler, X)

    pred = int(model.predict(X_scaled)[0])
    prob = model.predict_proba(X_scaled)[0]
    conf = float(max(prob)) * 100
    price = float(df["Close"].iloc[-1])

    mapping = {0: "SELL", 1: "BUY", 2: "HOLD"}
    signal = mapping[pred]

    entry, sl, tp, rr = trade_levels(signal, price)
    strength = trend_strength(df)
    desc = trend_description(signal, strength)

    return signal, conf, price, entry, sl, tp, rr, desc


# ------------------------------
# OUTPUT HANDLER
# ------------------------------
def run_predict(symbol):
    fixed_symbol, note = normalize_symbol(symbol)

    if fixed_symbol is None:
        print(f"\n❌ {note}\n")
        return

    if note:
        print(f"\n⚠️  {note}")

    (signal, conf, price,
     entry, sl, tp, rr, desc) = predict_signal(fixed_symbol)

    print("\n========== SIGNAL ==========")
    print(f"Symbol        : {fixed_symbol}")
    print(f"Price         : {price}")
    print(f"Signal        : {signal}")
    print(f"Confidence    : {conf:.2f}%")
    print("----------------------------------")
    print(f"Entry         : {entry}")
    print(f"Stop Loss     : {sl}")
    print(f"Take Profit   : {tp}")
    print(f"R/R           : {rr}")
    print("----------------------------------")
    print(f"Description   : {desc}")
    print("===================================\n")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    symbol = input("Enter crypto symbol (e.g., BTCUSDT or BTCUSD or BTC): ")
    run_predict(symbol)
