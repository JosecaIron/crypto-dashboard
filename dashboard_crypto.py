import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crypto Predictive Dashboard", layout="wide")

CRYPTO_TICKERS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Polkadot": "DOT-USD",
    "BNB": "BNB-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD",
    "Polygon": "MATIC-USD"
}

FORECAST_WEEKS = 1
BOOTSTRAP_MODELS = 50

# ---------------- DATA ----------------
@st.cache_data
def cargar_datos(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    df = df[["Close", "Volume"]].dropna()

    df["return_1d"] = df["Close"].pct_change()
    df["return_7d"] = df["Close"].pct_change(7)
    df["volatility_7d"] = df["return_1d"].rolling(7).std()

    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_7"] = SMAIndicator(df["Close"], 7).sma_indicator()
    df["SMA_21"] = SMAIndicator(df["Close"], 21).sma_indicator()
    df["MA_ratio"] = df["SMA_7"] / df["SMA_21"]

    df["volume_change"] = df["Volume"].pct_change()

    # Target: retorno futuro semanal
    df["target"] = df["Close"].shift(-7) / df["Close"] - 1

    return df.dropna()

# ---------------- MODEL ----------------
def entrenar_y_predecir(df):
    features = [
        "return_1d", "return_7d", "volatility_7d",
        "RSI", "MA_ratio", "volume_change"
    ]

    X = df[features].values
    y = df["target"].values

    preds = []

    for seed in range(BOOTSTRAP_MODELS):
        model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=seed
        )

        model.fit(X[:-7], y[:-7])
        pred = model.predict(X[-1].reshape(1, -1))[0]
        preds.append(pred)

    return np.array(preds)

# ---------------- UI ----------------
st.title("📊 Crypto Predictive Dashboard (Modelo Robusto)")

crypto = st.selectbox(
    "Selecciona una criptomoneda",
    list(CRYPTO_TICKERS.keys())
)

ticker = CRYPTO_TICKERS[crypto]
df = cargar_datos(ticker)

preds = entrenar_y_predecir(df)

mean_pred = preds.mean()
p5 = np.percentile(preds, 5)
p95 = np.percentile(preds, 95)

# ---------------- ALERTS ----------------
if p95 > 0.25:
    st.success(f"📈 ALERTA SUBIDA FUERTE: posible +{p95*100:.2f}%")
elif p5 < -0.05:
    st.warning(f"📉 ALERTA CAÍDA: posible {p5*100:.2f}%")
else:
    st.info("🟢 Sin alertas significativas")

# ---------------- CHART ----------------
st.subheader("Histórico y proyección semanal")

last_price = df["Close"].iloc[-1]
future_price = last_price * (1 + mean_pred)

dates = pd.concat([
    df.index[-90:],
    pd.date_range(df.index[-1], periods=2, freq="7D")
])

prices = list(df["Close"].iloc[-90:]) + [future_price]

plt.figure(figsize=(10, 4))
plt.plot(dates[:-1], prices[:-1], label="Histórico")
plt.plot(dates[-2:], prices[-2:], "--", label="Predicción")

plt.legend()
plt.grid()
st.pyplot(plt)

# ---------------- METRICS ----------------
st.subheader("Predicción semanal")

col1, col2, col3 = st.columns(3)

col1.metric("Predicción media", f"{mean_pred*100:.2f}%")
col2.metric("Escenario pesimista (5%)", f"{p5*100:.2f}%")
col3.metric("Escenario optimista (95%)", f"{p95*100:.2f}%")

st.caption("Modelo basado en retornos + Gradient Boosting + Bootstrap")
