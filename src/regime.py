import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# -------------------------------------------------------
# FIXED REGIME DETECTOR (MATCHES YOUR FEATURES)
# -------------------------------------------------------
def detect_regime(df):

    ema_short = df["ema_9"]
    ema_mid   = df["ema_21"]
    ema_long  = df["ema_100"]

    atr_pct = df["atr_pct"]

    regimes = []

    for i in range(len(df)):

        # 1 → Uptrend
        if ema_short[i] > ema_mid[i] > ema_long[i]:
            regimes.append(1)
            continue

        # 0 → Downtrend
        if ema_short[i] < ema_mid[i] < ema_long[i]:
            regimes.append(0)
            continue

        # 2 → High volatility chop
        if atr_pct[i] > 0.015:
            regimes.append(2)
            continue

        # 3 → Low volatility ranging
        regimes.append(3)

    df["regime"] = regimes
    return df


# -------------------------------------------------------
# APPLY TO ALL FEATURE FILES
# -------------------------------------------------------
def add_regimes_to_all():
    print("[INFO] Adding market regimes to all feature files...")

    files = [f for f in os.listdir(DATA_PATH) if f.startswith("feat_")]

    if not files:
        print("[ERROR] No feature files found.")
        return

    for file in files:
        print(f"[PROCESS] {file}")

        df = pd.read_csv(os.path.join(DATA_PATH, file))

        required = ["ema_9", "ema_21", "ema_100", "atr_pct"]

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] Missing: {missing}")
            continue

        df = detect_regime(df)

        df.to_csv(os.path.join(DATA_PATH, file), index=False)
        print(f"[✔] Regime added → {file}")

    print("\n[✔] MARKET REGIME ENGINE COMPLETE")


if __name__ == "__main__":
    add_regimes_to_all()
