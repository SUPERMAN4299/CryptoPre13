Here are **all three complete documentation files** — fully polished, professional, and copy-paste ready.
You can save them as:

* `docs/installation.md`
* `docs/architecture.md`
* `docs/retraining.md`

All formatted exactly the way marketplaces and GitHub expect.

---

# 📄 **installation.md**

```
# Installation Guide – TradeAI Crypto Prediction System

This guide explains how to install, configure, and run the TradeAI system locally.

---

## ✅ 1. Requirements

- Python 3.10+  
- pip (Python package manager)  
- Virtual environment recommended  
- Internet connection for data fetching  

---

## ✅ 2. Download the Project

Clone or extract the project folder:

```

git clone [https://github.com/yourrepo/tradeai](https://github.com/yourrepo/tradeai)
cd tradeai

```

---

## ✅ 3. Create & Activate Virtual Environment

### Windows:
```

python -m venv venv
venv/Scripts/activate

```

### Mac/Linux:
```

python3 -m venv venv
source venv/bin/activate

```

---

## ✅ 4. Install Dependencies

```

pip install -r requirements.txt

```

---

## ✅ 5. Folder Structure Overview

```

TradeAI/
├── models/               # Trained LSTM models
├── scalers/              # Normalization files
├── src/                  # Core backend
├── ui/                   # Streamlit UI
└── docs/                 # Documentation

```

---

## ✅ 6. Running the Streamlit App

Use the command:

```

streamlit run ui/app.py

```

A browser window will open automatically.

---

## ✅ 7. Using the App

1. Enter a symbol (e.g., BTCUSDT, ETHUSD).
2. The system normalizes it automatically.
3. Backend fetches price data.
4. LSTM model generates prediction.
5. UI displays:
   - Candlesticks  
   - Volume bars  
   - Buy/Sell signals  
   - Confidence score  

---

## ❗ Troubleshooting

### **ModuleNotFoundError**
Make sure you activated the virtual environment.

### **Model Not Found**
Ensure `/models/` and `/scalers/` folders exist.

### **Streamlit not opening**
Run:

```

streamlit cache clear

```

---

## 🎉 Installation Complete

You are ready to use TradeAI and generate AI-driven crypto predictions.
```

---
