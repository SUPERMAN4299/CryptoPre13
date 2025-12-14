# 📄 **architecture.md**

```
# System Architecture – TradeAI Crypto Prediction System

This document explains the internal design, data flow, and technical structure of the TradeAI platform.

---

# 🏗 1. High-Level Overview

TradeAI is made of three core layers:

1. **User Interface (UI)** – Streamlit dashboard  
2. **Prediction Engine (Backend)** – Python-based ML pipeline  
3. **Model Assets** – LSTM models + scalers + feature configs  

---

# 📘 2. Component Breakdown

## 2.1 Streamlit UI (`/ui/`)

Responsible for:

- Receiving crypto symbol from user  
- Visualizing candlestick chart  
- Plotting volume bars  
- Displaying Buy/Sell signals  
- Running prediction on button click  

Files:

```

ui/app.py
ui/components.py

```

---

## 2.2 Backend Prediction System (`/src/`)

This layer processes data, loads models, and generates predictions.

### 🔹 predict.py
- Main prediction interface  
- Orchestrates full pipeline  
- Error-safe wrapper  

### 🔹 preprocess.py
Handles:

- OHLCV data normalization  
- Feature engineering  
- Universal + per-symbol scaling  

### 🔹 model_loader.py
Loads:

- ML model (.pt / .pkl)  
- Scalers  
- Feature lists  

### 🔹 utils.py
General-purpose helpers:

- Logging  
- Symbol normalization  
- Data formatting  

---

# 🔍 3. Data Flow Diagram

```

User Symbol
↓
Symbol Normalizer
↓
Data Fetcher
↓
Preprocessor (features + scaler)
↓
LSTM Model
↓
Prediction Output
↓
Signal Generator
↓
Streamlit UI

```

---

# 🧠 4. Machine Learning Model

- Architecture: **LSTM sequence model**
- Input: OHLCV + engineered features
- Window Size: Configurable (default 60)
- Output: Next price movement (Up/Down)
- Training: Per-asset / universal

---

# 📦 5. Model Files (`/models/`)

Each model contains:

- Weight tensors  
- Architecture metadata  
- Version tag (v3 recommended)  

Example:

```

BTCUSDT_lstm_model.pkl
universal_scaler.pkl

```

---

# 🔧 6. Scaling System (`/scalers/`)

Two types:

1. **Universal scaler** — common features  
2. **Symbol-specific scalers** — unique patterns  

Stored in:

```

scalers/ETHUSDT_scaler.pkl

```

---

# 📊 7. UI Visualization Pipeline

UI uses:

- Plotly candlestick chart  
- Volume histogram  
- Buy/Sell markers  
- Signal summary widgets  
- Error banner  

---

# 🧱 8. Extendability

TradeAI is modular and supports:

- Plug-in models  
- Custom indicators  
- Additional datasets  
- REST API conversion  
- Desktop EXE conversion  

---

# 📘 9. Summary

TradeAI is a clean, production-ready AI system built using:

- Python  
- Streamlit  
- LSTM deep learning  
- Modular architecture  
- Complete documentation  

It is designed for easy integration, training, and commercial use.
```

---

