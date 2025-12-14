import pandas as pd
import numpy as np
import ta
import requests
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Fetch full historical klines (Binance unlimited downloader)
# -------------------------------------------------------
def fetch_full_history(symbol, interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"

    all_rows = []
    last_end_time = None

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if last_end_time:
            params["endTime"] = last_end_time

        data = requests.get(url, params=params).json()
        if not data:
            break

        all_rows.extend(data)

        last_end_time = data[0][0] - 1  # go backwards

        if len(data) < limit:
            break

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ])

    df["Open"] = df["open"].astype(float)
    df["High"] = df["high"].astype(float)
    df["Low"] = df["low"].astype(float)
    df["Close"] = df["close"].astype(float)
    df["Volume"] = df["volume"].astype(float)

    df = df[["Open","High","Low","Close","Volume"]]
    return df.dropna().reset_index(drop=True)

# -------------------------------------------------------
# Add indicators for all historical rows
# -------------------------------------------------------
def add_indicators(df):
    df["ema_9"] = ta.trend.EMAIndicator(df["Close"],9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["Close"],21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"],50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(df["Close"],100).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd_hist"] = ta.trend.MACD(df["Close"]).macd_diff()

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["Close"]

    return df.dropna().reset_index(drop=True)

# -------------------------------------------------------
# Build full dataset for a symbol
# -------------------------------------------------------
def build_full_features(symbol="BTCUSDT"):
    print(f"[FEATURES] Building features for {symbol}...")

    df = fetch_full_history(symbol, "1h")
    df = add_indicators(df)

    out = os.path.join(OUT_DIR, f"feat_{symbol}.csv")
    df.to_csv(out, index=False)

    print(f"[✔] Saved feature file → {out} ({len(df)} rows)")
    return df

# -------------------------------------------------------
# Run for top coins
# -------------------------------------------------------
if __name__ == "__main__":
    coins = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","DOTUSDT","TRXUSDT"]
    for c in coins:
        build_full_features(c)
