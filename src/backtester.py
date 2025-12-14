import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
RESULTS_PATH = os.path.join(BASE_DIR, "backtest_results")

os.makedirs(RESULTS_PATH, exist_ok=True)

LABEL = "label"  # your model output (-1,0,1)

# ----------------------------------------------
# ATR-Based SL/TP Helper
# ----------------------------------------------
def compute_levels(row):
    price = row["Close"]
    atr = row["atr"]

    if row[LABEL] == 2:  # BUY (mapped model output)
        sl = price - (1.5 * atr)
        tp = price + (2.0 * atr)
    elif row[LABEL] == 0:  # SELL
        sl = price + (1.5 * atr)
        tp = price - (2.0 * atr)
    else:
        return None, None  # HOLD = no trade

    return sl, tp


# ----------------------------------------------
# RUN BACKTEST ON SINGLE COIN
# ----------------------------------------------
def backtest_coin(filepath):
    df = pd.read_csv(filepath)

    # Only valid labeled rows
    df = df.dropna()

    initial_balance = 1000
    balance = initial_balance
    equity_curve = [balance]

    wins = 0
    losses = 0
    trades = 0

    trade_log = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        signal = row[LABEL]  # 0=sell,1=hold,2=buy

        if signal == 1:
            continue  # HOLD = skip

        sl, tp = compute_levels(row)
        if sl is None:
            continue

        entry = row["Close"]
        low_future = df.iloc[i+1]["Low"] if i+1 < len(df) else entry
        high_future = df.iloc[i+1]["High"] if i+1 < len(df) else entry

        trades += 1

        # BUY BACKTEST
        if signal == 2:
            if low_future <= sl:   # SL hit
                balance *= 0.985
                losses += 1
                outcome = "SL"
            elif high_future >= tp:  # TP hit
                balance *= 1.02
                wins += 1
                outcome = "TP"
            else:
                outcome = "NONE"

        # SELL BACKTEST
        if signal == 0:
            if high_future >= sl:
                balance *= 0.985
                losses += 1
                outcome = "SL"
            elif low_future <= tp:
                balance *= 1.02
                wins += 1
                outcome = "TP"
            else:
                outcome = "NONE"

        equity_curve.append(balance)
        trade_log.append([i, entry, sl, tp, outcome])

    # Metrics
    accuracy = (wins / trades) * 100 if trades > 0 else 0
    profit_factor = wins / losses if losses > 0 else wins
    max_drawdown = (initial_balance - min(equity_curve)) / initial_balance * 100

    summary = {
        "Total Trades": trades,
        "Wins": wins,
        "Losses": losses,
        "Accuracy %": accuracy,
        "Profit Factor": profit_factor,
        "Max Drawdown %": max_drawdown,
        "Final Balance": balance,
        "Return %": (balance - initial_balance) / initial_balance * 100,
    }

    return summary, trade_log, equity_curve


# ----------------------------------------------
# RUN BACKTEST FOR ALL LABELED FILES
# ----------------------------------------------
def run_all_backtests():
    files = [f for f in os.listdir(DATA_PATH) if f.startswith("labeled_")]

    results = {}

    for file in files:
        print(f"[BACKTEST] {file}...")

        filepath = os.path.join(DATA_PATH, file)
        summary, log, curve = backtest_coin(filepath)

        results[file] = summary

        # Save summary
        pd.DataFrame([summary]).to_csv(
            os.path.join(RESULTS_PATH, f"summary_{file}.csv"),
            index=False
        )

        # Save trade log
        pd.DataFrame(log, columns=["Index", "Entry", "SL", "TP", "Outcome"]).to_csv(
            os.path.join(RESULTS_PATH, f"log_{file}.csv"),
            index=False
        )

        # Save equity curve plot
        plt.figure(figsize=(10, 5))
        plt.plot(curve)
        plt.title(f"Equity Curve - {file}")
        plt.xlabel("Trades")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_PATH, f"equity_{file}.png"))
        plt.close()

        print(f"[✔] Completed: {file}\n")

    print("\nAll backtests complete!")
    return results


if __name__ == "__main__":
    run_all_backtests()
