import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ==============================
# CORRECT PROJECT PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)

# Binance optional fallback (API-free)
BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{year}-{month}.zip"


# ==============================
# DOWNLOAD USING YAHOO FINANCE
# ==============================
def download_yahoo(symbol):
    print(f"[YAHOO] Downloading {symbol}...")

    try:
        df = yf.download(
            symbol,
            interval="1h",
            period="730d",
            auto_adjust=True
        )
        if df is None or df.empty:
            print(f"[YAHOO] Empty data for {symbol}")
            return None

        df.reset_index(inplace=True)
        print(f"[✔] Yahoo data OK for {symbol}")
        return df

    except Exception as e:
        print(f"[YAHOO ERROR] {symbol}: {e}")
        return None


# ==============================
# OPTIONAL BINANCE FALLBACK
# ==============================
def download_binance(symbol):
    print(f"[BINANCE] Fallback for {symbol} not implemented (optional).")
    return None


# ==============================
# SAVE DOWNLOADED DATA
# ==============================
def save_csv(symbol, df):
    path = os.path.join(RAW_DIR, f"raw_{symbol}.csv")
    df.to_csv(path, index=False)
    print(f"[✔] SAVED: {path}")


# ==============================
# MAIN DOWNLOAD FUNCTION
# ==============================
def download_single_symbol(symbol):
    df = download_yahoo(symbol)

    if df is None:
        print(f"[WARN] Yahoo failed → Trying Binance…")
        df = download_binance(symbol)

    if df is None:
        print(f"[ERROR] {symbol} could NOT be downloaded.")
        return None

    save_csv(symbol, df)
    return df


# ==============================
# BATCH DOWNLOAD FOR ALL TOP 10
# ==============================
def download_all():
    symbols = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "TRX-USD"
    ]

    print("\n=========================================")
    print("🔥 STARTING FULL DATA DOWNLOAD")
    print("Saving to:", RAW_DIR)
    print("=========================================\n")

    for sym in symbols:
        download_single_symbol(sym)
        print("")

    print("\n[✔] DOWNLOAD COMPLETE")


if __name__ == "__main__":
    download_all()
