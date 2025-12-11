import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

# Helper: Run a script and print output
def run_step(name, file):
    print(f"\n\n==============================")
    print(f"🚀 Starting Step: {name}")
    print(f"==============================\n")
    time.sleep(1)

    result = subprocess.run([sys.executable, os.path.join(BASE, file)])
    
    if result.returncode != 0:
        print(f"\n❌ ERROR in {name}! Stopping pipeline.")
        sys.exit(1)
    
    print(f"\n✔ COMPLETED: {name}")
    time.sleep(1)


def main():
    print("===========================================")
    print("🔥 TRADE AI — FULL AUTOMATED PIPELINE STARTED")
    print("===========================================\n")

    # 1️⃣ Download market data
    #run_step("Download Raw Data", "downloader.py")

    # 2️⃣ Generate features
    #run_step("Feature Engineering", "features.py")

    # 3️⃣ Market Regime Detection
    run_step("Market Regime Engine", "regime.py")

    # 4️⃣ Labeling
    run_step("Dynamic Multi-Step Labeling", "labeler.py")

    # 5️⃣ Ensemble Training
    run_step("Ensemble Model Training", "train_ensemble.py")

    print("\n\n===========================================")
    print("🎉 PIPELINE COMPLETE — ALL MODELS TRAINED SUCCESSFULLY")
    print("===========================================\n")
    print("Next: Run prediction with → python predict.py")
    print("Example: BTC-USD  |  ETH-USD  |  SOL-USD\n")


if __name__ == "__main__":
    main()
