import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Crypto Dashboard Predictivo", layout="wide")

CRYPTOS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Polkadot": "DOT-USD",
    "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD",
    "Polygon": "MATIC-USD"
}

@st.cache_data
def cargar_datos(ticker):
    df = yf.download(ticker, period="5y", interval="1wk")

    # Asegurar que Close es una Serie 1D
    close = df["Close"].squeeze()

    data = pd.DataFrame({
        "Close": close
    })

    data["RSI"] = RSIIndicator(data["Close"]).rsi()
    data["EMA20"] = EMAIndicator(data["Close"], window=20).ema_indicator()

    data.dropna(inplace=True)
    return data


def predecir(df, semanas=12):
    df = df.copy()
    df["t"] = np.arange(len(df))

    X = df[["t"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(df), len(df) + semanas).reshape(-1, 1)
    preds = model.predict(future_t)

    return preds

# ============================
# DASHBOARD
# ============================

st.title("📊 Dashboard Predictivo de Criptomonedas")

crypto_name = st.selectbox("Selecciona una criptomoneda", list(CRYPTOS.keys()))
ticker = CRYPTOS[crypto_name]

df = cargar_datos(ticker)
pred = predecir(df, semanas=12)

precio_actual = df["Close"].iloc[-1]
precio_pred = pred[-1]
variacion = (precio_pred / precio_actual - 1) * 100

# ALERTAS
if variacion >= 25:
    st.error(f"🚀 ALERTA SUBIDA: +{variacion:.2f}% previsto")
elif variacion <= -5:
    st.warning(f"⚠️ ALERTA CAÍDA: {variacion:.2f}% previsto")
else:
    st.success("🟢 Sin alertas significativas")

# GRAFICA
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df["Close"], label="Histórico")
future_dates = pd.date_range(df.index[-1], periods=12, freq="W")
ax.plot(future_dates, pred, label="Predicción", linestyle="--")
ax.legend()
ax.grid()

st.pyplot(fig)

# INFO
st.metric("Precio actual (USD)", f"{precio_actual:.2f}")
st.metric("Precio previsto 12 semanas", f"{precio_pred:.2f}")
st.metric("Variación prevista", f"{variacion:.2f}%")
