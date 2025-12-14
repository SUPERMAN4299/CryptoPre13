# 📄 **retraining.md**

```
# Model Retraining Guide – TradeAI

This guide explains how to retrain the LSTM models used by TradeAI using new data or new crypto symbols.

---

# 🎯 1. Why Retrain?

Retraining helps to:

- Improve accuracy  
- Adapt to market shifts  
- Add new symbols  
- Enhance generalization  
- Upgrade model architecture  

---

# 📂 2. Required Files

You need:

```

train.py
src/preprocess.py
models/
scalers/
feature_names.json

```

---

# 🔄 3. Training Command

Basic example:

```

python train.py --symbol BTCUSDT --epochs 50

```

Arguments:

| Flag | Meaning |
|------|---------|
| `--symbol` | Which crypto pair to train on |
| `--epochs` | Training duration |
| `--batch` | Batch size |
| `--lr` | Learning rate |

---

# 📥 4. Data Collection

The trainer retrieves:

- OHLCV historical data  
- Technical indicator values  
- Lag sequences  

You can plug in custom data by modifying:

```

src/preprocess.py

```

---

# 🧠 5. LSTM Structure

Model includes:

- Input layer  
- LSTM block  
- Dropout  
- Dense output  

Default loss: **MSE**  
Optimizer: **Adam**

---

# ⚙️ 6. Saving New Models

After training, system saves:

```

models/SYMBOL_lstm_model.pkl
scalers/SYMBOL_scaler.pkl

```

No manual work required.

---

# 🧪 7. Testing a Trained Model

Run prediction:

```

python src/predict.py

```

Enter your new symbol, e.g.:

```

ETHUSDT

```

You should see:

- Predicted change  
- Buy/Sell signal  
- Confidence score  

---

# 🔧 8. Training Recommendations

### For better accuracy:
- Increase epochs (50 → 100)  
- Add more historical data  
- Add custom features (RSI, MACD, SMA)  
- Increase sequence window size  

### For faster training:
- Use GPU  
- Reduce window length  
- Reduce model layers  

---

# 📌 9. Troubleshooting

### **Loss not decreasing**
Try lowering learning rate.

### **Model overfitting**
Increase dropout.

### **Prediction always same**
Check scaler + normalization.

### **Training crashes**
Check input feature shape.

---

# 🎉 Retraining Complete

You now have a fully updated LSTM model ready to plug into the TradeAI system.
```

---
