import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.subplots as sp

from src.predict import predict_signal, normalize_symbol


# ======================================
# FETCH BINANCE HISTORICAL DATA (CHART)
# ======================================
def get_binance_klines(symbol, interval="1h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "_1","_2","_3","_4","_5","_6"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df


# ======================================
# INDICATORS
# ======================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series):
    e12 = ema(series, 12)
    e26 = ema(series, 26)
    macd_line = e12 - e26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


# ======================================
# MULTI-TIMEFRAME TREND HEATMAP
# ======================================
def get_trend(df):
    return "BUY" if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] else "SELL"

def multi_tf_heatmap(symbol, intervals=["1m","5m","15m","1h","4h","1d"]):
    results = []
    for tf in intervals:
        try:
            df = get_binance_klines(symbol, interval=tf, limit=200)
            df["ema9"] = ema(df["close"], 9)
            df["ema21"] = ema(df["close"], 21)
            trend = get_trend(df)
        except:
            trend = "ERR"

        results.append((tf, trend))

    return pd.DataFrame(results, columns=["Timeframe", "Trend"])


# ======================================
# CRYPTOPANIC NEWS FETCH (FREE)
# ======================================
def get_crypto_news(symbol, limit=7):
    base = "https://cryptopanic.com/api/v1/posts/?auth_token=&kind=news"
    coin = symbol.replace("USDT", "")
    url = f"{base}&currencies={coin}&filter=hot"

    try:
        r = requests.get(url, timeout=5)
        posts = r.json().get("results", [])
    except:
        return []

    news_list = []
    for p in posts[:limit]:
        title = p.get("title", "No title")
        src = p.get("source", {}).get("title", "Unknown")
        link = p.get("url", "")
        published = p.get("published_at", "")[:10]

        votes = p.get("votes", {})
        bull = votes.get("positive", 0)
        bear = votes.get("negative", 0)

        if bull > bear:
            sentiment = "🟢 Bullish"
        elif bear > bull:
            sentiment = "🔴 Bearish"
        else:
            sentiment = "🟡 Neutral"

        news_list.append({
            "title": title,
            "source": src,
            "url": link,
            "published": published,
            "sentiment": sentiment
        })

    return news_list


# ======================================
# PRO CANDLE CHART (EMA + RSI + MACD)
# ======================================
def build_full_chart(df, entry, sl, tp):

    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)

    df["rsi"] = rsi(df["close"])
    macd_line, macd_signal, macd_hist = macd(df["close"])

    fig = sp.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Candles"
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema9"], name="EMA9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema21"], name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema100"], name="EMA100"), row=1, col=1)

    # TP / SL / Entry lines
    if entry:
        fig.add_hline(y=entry, line_color="blue", annotation_text="Entry")
    if sl != "-":
        fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    if tp != "-":
        fig.add_hline(y=tp, line_color="green", annotation_text="TP")

    # MACD
    fig.add_trace(go.Bar(x=df["time"], y=macd_hist, name="MACD Hist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=macd_line, name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=macd_signal, name="Signal"), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_color="green", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


# ======================================
# STREAMLIT UI — PRO DASHBOARD
# ======================================
st.title("⚡ TradeAI — Pro Trading Dashboard")

symbol_input = st.text_input("Enter Symbol (BTCUSDT, ETHUSD, etc.):", "BTCUSDT")
interval = st.selectbox("Chart Interval", ["1m","5m","15m","1h","4h","1d"], index=3)

if st.button("Generate Signal"):

    # Normalize symbol
    fixed_symbol, note = normalize_symbol(symbol_input)
    if note:
        st.warning(note)

    if fixed_symbol is None:
        st.error("Invalid symbol.")
        st.stop()

    # Run model prediction
    signal, conf, price, entry, sl, tp, rr, desc = predict_signal(fixed_symbol)

    st.subheader(f"{fixed_symbol} — {signal} ({conf:.2f}%)")
    st.write(desc)

    # Multi-TF Heatmap
    st.markdown("### 🔥 Multi-Timeframe Trend Heatmap")
    heatmap = multi_tf_heatmap(fixed_symbol)
    st.dataframe(heatmap.set_index("Timeframe"))

    # Get chart data
    df = get_binance_klines(fixed_symbol, interval=interval, limit=500)

    # Build chart
    fig = build_full_chart(df, entry, sl, tp)
    st.plotly_chart(fig, use_container_width=True)

    # Trade Levels
    st.markdown("### 🧾 Trade Levels")
    c1, c2, c3 = st.columns(3)
    c1.info(f"Entry: {entry}")
    c2.success(f"TP: {tp}")
    c3.error(f"SL: {sl}")

    st.markdown(f"### 🎯 Risk/Reward Ratio: **{rr}**")

    # ======================================
    # NEWS SECTION
    # ======================================
    st.markdown("### 📰 Latest Crypto News")

    news = get_crypto_news(fixed_symbol)

    if not news:
        st.info("No news found for this asset.")
    else:
        for item in news:
            st.markdown(f"""
            <div style="padding:12px; border-radius:10px; background-color:#111827; margin-bottom:10px;">
                <h4>{item['sentiment']} — {item['title']}</h4>
                <p style="color:#9ca3af;">{item['source']} — {item['published']}</p>
                <a href="{item['url']}" target="_blank">Read More</a>
            </div>
            """, unsafe_allow_html=True)
