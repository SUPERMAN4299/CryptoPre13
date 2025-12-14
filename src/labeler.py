import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


# -----------------------------------------------------
# SIMPLE + STABLE LABEL ENGINE (ALWAYS PRODUCES DATA)
# -----------------------------------------------------
def apply_labels(df, future_step=1, threshold=0.002):
    """
    future_step  = predict next candle (1 hour later)
    threshold    = 0.2% change required for buy/sell signal
    """

    df["future_close"] = df["Close"].shift(-future_step)

    df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"]

    # BUY = 2
    df["LABEL"] = 0  # default: HOLD

    df.loc[df["future_return"] > threshold, "LABEL"] = 2
    df.loc[df["future_return"] < -threshold, "LABEL"] = 1

    df = df.dropna()

    return df


# -----------------------------------------------------
# PROCESS ALL FEATURE FILES
# -----------------------------------------------------
def label_all():
    print("[INFO] Labeling all feature files...")

    files = [f for f in os.listdir(DATA_PATH) if f.startswith("feat_")]

    if not files:
        print("[ERROR] No feature files found!")
        return

    for file in files:
        print(f"[PROCESS] {file}")

        df = pd.read_csv(os.path.join(DATA_PATH, file))

        if "Close" not in df.columns:
            print("[SKIP] Missing Close column.")
            continue

        df = apply_labels(df)

        if len(df) < 100:
            print(f"[WARN] {file} produced very small dataset ({len(df)} rows).")
        else:
            print(f"[✔] {file} labeled successfully ({len(df)} rows).")

        out = f"labeled_{file}"
        df.to_csv(os.path.join(DATA_PATH, out), index=False)

    print("\n[✔] LABELING COMPLETE")


if __name__ == "__main__":
    label_all()
