import requests
import pandas as pd
import time

def get_binance_klines(symbol, interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if isinstance(data, dict) and "code" in data:
            print(f"[BINANCE ERROR] {data}")
            return None

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df = df[["open", "high", "low", "close", "volume"]]
        
        print(f"[✔] Binance fetched: {symbol}")
        return df

    except Exception as e:
        print(f"[ERROR] Binance download failed for {symbol}: {e}")
        return None
