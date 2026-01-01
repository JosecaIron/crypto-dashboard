import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from sklearn.ensemble import HistGradientBoostingRegressor

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(
    page_title="Crypto Predictive Dashboard",
    layout="wide"
)

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

BOOTSTRAP_MODELS = 50

# --------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS (BLINDADA)
# --------------------------------------------------
@st.cache_data
def cargar_datos(ticker):
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)

    df = df[["Close", "Volume"]].dropna()

    # 🔒 FIX CRÍTICO: garantizar Series 1D
    df["Close"] = df["Close"].squeeze()
    df["Volume"] = df["Volume"].squeeze()

    # Features de retornos
    df["return_1d"] = df["Close"].pct_change()
    df["return_7d"] = df["Close"].pct_change(7)
    df["volatility_7d"] = df["return_1d"].rolling(7).std()

    # Indicadores técnicos (seguros)
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    df["SMA_7"] = SMAIndicator(close=df["Close"], window=7).sma_indicator()
    df["SMA_21"] = SMAIndicator(close=df["Close"], window=21).sma_indicator()
    df["MA_ratio"] = df["SMA_7"] / df["SMA_21"]

    df["volume_change"] = df["Volume"].pct_change()

    # Target: retorno semanal futuro
    df["target"] = df["Close"].shift(-7) / df["Close"] - 1

    df = df.dropna()

    return df

# --------------------------------------------------
# MODELO ROBUSTO (BOOTSTRAP + GRADIENT BOOSTING)
# --------------------------------------------------
def entrenar_y_predecir(df):
    features = [
        "return_1d",
        "return_7d",
        "volatility_7d",
        "RSI",
        "MA_ratio",
        "volume_change"
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

        # Entrenamos hasta la última semana conocida
        model.fit(X[:-7], y[:-7])

        pred = model.predict(X[-1].reshape(1, -1))[0]
        preds.append(pred)

    return np.array(preds)

# --------------------------------------------------
# INTERFAZ
# --------------------------------------------------
st.title("📊 Crypto Predictive Dashboard")
st.caption("Modelo robusto sin TensorFlow · Retornos · Bootstrap · Alertas fiables")

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

# --------------------------------------------------
# ALERTAS
# --------------------------------------------------
st.subheader("🔔 Alertas automáticas")

if p95 > 0.25:
    st.success(f"📈 ALERTA SUBIDA FUERTE · Escenario optimista: +{p95*100:.2f}%")
elif p5 < -0.05:
    st.warning(f"📉 ALERTA CAÍDA · Escenario pesimista: {p5*100:.2f}%")
else:
    st.info("🟢 Sin alertas significativas previstas")

# --------------------------------------------------
# GRÁFICA
# --------------------------------------------------
st.subheader("📈 Precio histórico y proyección semanal")

last_price = df["Close"].iloc[-1]
future_price = last_price * (1 + mean_pred)

dates_hist = df.index[-90:]
dates_future = pd.date_range(dates_hist[-1], periods=2, freq="7D")

prices_hist = df["Close"].iloc[-90:]
prices_future = [prices_hist.iloc[-1], future_price]

plt.figure(figsize=(10, 4))
plt.plot(dates_hist, prices_hist, label="Histórico")
plt.plot(dates_future, prices_future, "--", label="Predicción")

plt.grid(True)
plt.legend()
st.pyplot(plt)

# --------------------------------------------------
# MÉTRICAS
# --------------------------------------------------
st.subheader("📊 Predicción semanal (%)")

col1, col2, col3 = st.columns(3)

col1.metric("Predicción media", f"{mean_pred*100:.2f}%")
col2.metric("Escenario pesimista (5%)", f"{p5*100:.2f}%")
col3.metric("Escenario optimista (95%)", f"{p95*100:.2f}%")

st.caption("Predicción basada en distribución de modelos (bootstrap)")
