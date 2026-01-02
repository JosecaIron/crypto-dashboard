import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

# --------------------------------------------------
# CONFIGURACIÓN
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
MIN_ROWS = 120  # mínimo de datos limpios para predecir

# --------------------------------------------------
# RSI MANUAL (ROBUSTO)
# --------------------------------------------------
def calcular_rsi(close, window=14):
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# --------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# --------------------------------------------------
@st.cache_data
def cargar_datos(ticker):
    df = yf.download(ticker, period="5y", interval="1d")

    # Asegurar columnas planas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close", "Volume"]].dropna()

    # Forzar Series 1D reales
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    volume = pd.Series(df["Volume"].values.flatten(), index=df.index)

    df["Close"] = close
    df["return_1d"] = close.pct_change()
    df["return_7d"] = close.pct_change(7)
    df["volatility_7d"] = df["return_1d"].rolling(7).std()

    df["RSI"] = calcular_rsi(close)

    df["SMA_7"] = close.rolling(7).mean()
    df["SMA_21"] = close.rolling(21).mean()
    df["MA_ratio"] = df["SMA_7"] / df["SMA_21"]

    df["volume_change"] = volume.pct_change()

    # Target: retorno semanal futuro
    df["target"] = close.shift(-7) / close - 1

    # Limpieza final
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

# --------------------------------------------------
# MODELO Y PREDICCIÓN
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

    data = df[features + ["target"]].copy()

    if len(data) < MIN_ROWS:
        return None

    X = data[features].values
    y = data["target"].values

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

# --------------------------------------------------
# INTERFAZ
# --------------------------------------------------
st.title("📊 Crypto Predictive Dashboard")
st.caption("Modelo robusto sin TensorFlow · Producción estable")

crypto = st.selectbox(
    "Selecciona una criptomoneda",
    list(CRYPTO_TICKERS.keys())
)

ticker = CRYPTO_TICKERS[crypto]

df = cargar_datos(ticker)

if df.empty or len(df) < MIN_ROWS:
    st.error("❌ No hay suficientes datos históricos limpios para generar una predicción fiable.")
    st.stop()

preds = entrenar_y_predecir(df)

if preds is None:
    st.error("❌ No se pudo entrenar el modelo con los datos actuales.")
    st.stop()

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
    st.info("🟢 Sin alertas relevantes previstas")

# --------------------------------------------------
# GRÁFICA
# --------------------------------------------------
st.subheader("📈 Precio histórico y proyección semanal")

last_price = df["Close"].iloc[-1]
future_price = last_price * (1 + mean_pred)

plt.figure(figsize=(10, 4))
plt.plot(df.index[-90:], df["Close"].iloc[-90:], label="Histórico")
plt.plot(
    [df.index[-1], df.index[-1] + pd.Timedelta(days=7)],
    [last_price, future_price],
    "--",
    label="Predicción"
)

plt.legend()
plt.grid(True)
st.pyplot(plt)

# --------------------------------------------------
# MÉTRICAS
# --------------------------------------------------
st.subheader("📊 Predicción semanal (%)")

c1, c2, c3 = st.columns(3)
c1.metric("Predicción media", f"{mean_pred*100:.2f}%")
c2.metric("Escenario pesimista (5%)", f"{p5*100:.2f}%")
c3.metric("Escenario optimista (95%)", f"{p95*100:.2f}%")

st.caption("Predicción basada en distribución bootstrap · ML clásico estable")
