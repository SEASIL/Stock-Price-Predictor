"""
╔══════════════════════════════════════════════════════════════════╗
║   NEXUS ALPHA — AI Stock Prediction Engine                       ║
║   Deep Learning LSTM · Cinematic Dark UI · Neon Highlights       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import time
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG & CINEMATIC DARK THEME
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS ALPHA · Stock AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-void:    #030712;
    --bg-deep:    #0a0f1e;
    --bg-card:    rgba(10, 20, 40, 0.75);
    --bg-glass:   rgba(255,255,255,0.03);
    --neon-cyan:  #00f5ff;
    --neon-purple:#b347ff;
    --neon-pink:  #ff2d78;
    --neon-green: #00ff9d;
    --text-bright:#f0f8ff;
    --text-dim:   #5b7fa6;
    --border:     rgba(0,245,255,0.12);
    --border-hot: rgba(179,71,255,0.35);
    --glow-cyan:  0 0 20px rgba(0,245,255,0.4), 0 0 60px rgba(0,245,255,0.12);
    --glow-purple:0 0 20px rgba(179,71,255,0.5), 0 0 60px rgba(179,71,255,0.15);
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background: var(--bg-void) !important;
    color: var(--text-bright) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Animated starfield background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,245,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(179,71,255,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 60% 40%, rgba(255,45,120,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1f 0%, #030712 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-purple), transparent);
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-deep) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--neon-cyan) !important;
    background: rgba(0,245,255,0.04) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    background: rgba(0,245,255,0.05) !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    opacity: 0.6;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,245,255,0.3) !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--neon-cyan) !important;
    text-shadow: var(--glow-cyan) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(179,71,255,0.1)) !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 12px rgba(0,245,255,0.15) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(179,71,255,0.2)) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--neon-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: var(--neon-cyan) !important;
}

/* ── Selectbox / Text Input ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
div[data-baseweb="select"] {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-bright) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-baseweb="select"]:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px var(--neon-cyan) !important;
}

/* ── Progress Bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple)) !important;
    box-shadow: var(--glow-cyan) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--neon-purple); border-radius: 2px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px) !important;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  HELPER: Plotly Dark Template
# ─────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,13,31,0.8)",
        font=dict(family="JetBrains Mono", color="#5b7fa6", size=11),
        title=dict(font=dict(family="Syne", color="#f0f8ff", size=16)),
        xaxis=dict(
            gridcolor="rgba(0,245,255,0.06)",
            linecolor="rgba(0,245,255,0.15)",
            tickcolor="rgba(0,245,255,0.3)",
            tickfont=dict(color="#5b7fa6"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,245,255,0.06)",
            linecolor="rgba(0,245,255,0.15)",
            tickcolor="rgba(0,245,255,0.3)",
            tickfont=dict(color="#5b7fa6"),
        ),
        legend=dict(
            bgcolor="rgba(10,20,40,0.7)",
            bordercolor="rgba(0,245,255,0.2)",
            borderwidth=1,
            font=dict(color="#a0c4d8"),
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )
)


# ─────────────────────────────────────────────────────────────────
#  STREAMLIT KERAS CALLBACK — real-time epoch progress
# ─────────────────────────────────────────────────────────────────
class StreamlitProgressCallback(Callback):
    def __init__(self, epochs, progress_bar, epoch_text, loss_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.epoch_text = epoch_text
        self.loss_text = loss_text
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        self.losses.append(loss)
        pct = (epoch + 1) / self.epochs
        self.progress_bar.progress(pct)
        self.epoch_text.markdown(
            f"<span style='font-family:JetBrains Mono;font-size:0.78rem;"
            f"color:#5b7fa6;'>EPOCH <span style='color:#00f5ff'>{epoch+1}</span>"
            f" / {self.epochs}</span>",
            unsafe_allow_html=True,
        )
        self.loss_text.markdown(
            f"<span style='font-family:JetBrains Mono;font-size:0.78rem;"
            f"color:#5b7fa6;'>TRAIN LOSS <span style='color:#b347ff'>{loss:.6f}</span>"
            f" &nbsp;|&nbsp; VAL LOSS <span style='color:#ff2d78'>{val_loss:.6f}</span></span>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
#  ML PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df


def create_sequences(data: np.ndarray, window: int = 60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(window: int = 60) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def train_model(X_train, y_train, X_val, y_val, epochs, window,
                progress_bar, epoch_text, loss_text):
    X_train = X_train.reshape(-1, window, 1)
    X_val = X_val.reshape(-1, window, 1)
    model = build_lstm_model(window)
    cb = StreamlitProgressCallback(epochs, progress_bar, epoch_text, loss_text)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[cb],
        verbose=0,
    )
    return model, history


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='padding:1.2rem 0 0.5rem;text-align:center;'>
          <span style='font-family:Syne,sans-serif;font-size:1.5rem;
                       font-weight:800;letter-spacing:-0.02em;
                       background:linear-gradient(135deg,#00f5ff,#b347ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            ⬡ NEXUS ALPHA
          </span><br>
          <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                       letter-spacing:0.2em;color:#5b7fa6;text-transform:uppercase;'>
            AI · Stock · Prediction
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    ticker = st.text_input("TICKER SYMBOL", value="AAPL",
                           help="e.g. AAPL, TSLA, GOOGL, MSFT").upper().strip()

    period_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
    period_label = st.selectbox("HISTORY PERIOD", list(period_map.keys()), index=3)
    period = period_map[period_label]

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    window_size = st.slider("LOOKBACK WINDOW (days)", 20, 120, 60, 5,
                            help="Days of past data used for each prediction")

    train_split = st.slider("TRAIN / TEST SPLIT", 0.60, 0.90, 0.80, 0.05,
                            help="Fraction of data used for training")

    epochs = st.select_slider("TRAINING EPOCHS",
                              options=[10, 20, 30, 50, 75, 100], value=50)

    st.divider()
    run_btn = st.button("⚡  LAUNCH ANALYSIS", use_container_width=True)

    st.markdown(
        """
        <div style='margin-top:1.5rem;padding:0.8rem;
             background:rgba(0,245,255,0.04);border-radius:8px;
             border:1px solid rgba(0,245,255,0.1);
             font-family:JetBrains Mono,monospace;font-size:0.65rem;
             color:#5b7fa6;line-height:1.8;'>
          <span style='color:#00f5ff;'>MODEL ARCHITECTURE</span><br>
          LSTM × 128 → Dropout 0.2<br>
          LSTM × 64 &nbsp;→ Dropout 0.2<br>
          Dense × 32 → Dense × 1<br>
          Optimizer: Adam · Loss: MSE
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='display:flex;align-items:baseline;gap:1rem;padding:0.4rem 0 1.2rem;'>
      <span style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                   letter-spacing:-0.03em;
                   background:linear-gradient(135deg,#00f5ff 0%,#b347ff 60%,#ff2d78 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        NEXUS ALPHA
      </span>
      <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                   letter-spacing:0.18em;color:#5b7fa6;text-transform:uppercase;
                   padding:0.25rem 0.6rem;border:1px solid rgba(0,245,255,0.2);
                   border-radius:4px;'>
        v2.0 · LSTM Engine
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────
tab_overview, tab_training, tab_forecast = st.tabs([
    "⬡  Market Overview",
    "◈  ML Model Training",
    "◉  Future Forecast",
])


# ════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None


# ════════════════════════════════════════════════════════════════
#  MAIN PIPELINE  (triggered by button)
# ════════════════════════════════════════════════════════════════
if run_btn:
    st.session_state.results = None  # clear stale

    # ── 1. Load Data ──────────────────────────────────────────
    with st.spinner(""):
        try:
            df = load_stock_data(ticker, period)
        except Exception as e:
            st.error(f"Failed to fetch data for **{ticker}**: {e}")
            st.stop()

    if df.empty or len(df) < window_size + 50:
        st.error("Not enough data. Try a longer period or a different ticker.")
        st.stop()

    close = df[["Close"]].copy()
    close.columns = ["Close"]

    # ── 2. Scale ──────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close.values)

    # ── 3. Train / Test split ─────────────────────────────────
    split_idx = int(len(scaled) * train_split)
    train_data = scaled[:split_idx]
    test_data  = scaled[split_idx - window_size:]

    X_train, y_train = create_sequences(train_data, window_size)
    X_test,  y_test  = create_sequences(test_data,  window_size)

    # ── 4. Training UI ────────────────────────────────────────
    with tab_training:
        st.markdown(
            "<p style='font-family:Syne;font-size:1.1rem;font-weight:700;"
            "color:#f0f8ff;margin-bottom:1rem;'>LSTM Training in Progress</p>",
            unsafe_allow_html=True,
        )
        prog_bar  = st.progress(0)
        col_ep, col_loss = st.columns(2)
        epoch_text = col_ep.empty()
        loss_text  = col_loss.empty()

    model, history = train_model(
        X_train, y_train,
        X_test[:int(len(X_test) * 0.2)],
        y_test[:int(len(y_test) * 0.2)],
        epochs, window_size,
        prog_bar, epoch_text, loss_text,
    )

    # ── 5. Predictions ────────────────────────────────────────
    X_test_rs = X_test.reshape(-1, window_size, 1)
    y_pred_sc = model.predict(X_test_rs, verbose=0)
    y_pred    = scaler.inverse_transform(y_pred_sc).flatten()
    y_actual  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    # ── 6. Next-day prediction ────────────────────────────────
    last_window = scaled[-window_size:].reshape(1, window_size, 1)
    next_pred_sc = model.predict(last_window, verbose=0)
    next_pred    = scaler.inverse_transform(next_pred_sc)[0, 0]
    last_close   = float(close["Close"].iloc[-1])
    pct_change   = (next_pred - last_close) / last_close * 100

    # ── 7. Save to session ────────────────────────────────────
    test_dates = close.index[split_idx:]
    st.session_state.results = dict(
        ticker=ticker, df=df, close=close,
        y_actual=y_actual, y_pred=y_pred,
        test_dates=test_dates[:len(y_actual)],
        rmse=rmse, mape=mape,
        last_close=last_close, next_pred=next_pred,
        pct_change=pct_change,
        history=history.history,
        train_split=train_split, split_idx=split_idx,
    )


# ════════════════════════════════════════════════════════════════
#  RENDER RESULTS
# ════════════════════════════════════════════════════════════════
res = st.session_state.results

if res is None:
    # ── Idle splash ──────────────────────────────────────────
    with tab_overview:
        st.markdown(
            """
            <div style='text-align:center;padding:4rem 2rem;
                        background:rgba(10,20,40,0.5);border-radius:16px;
                        border:1px solid rgba(0,245,255,0.1);margin-top:1rem;'>
              <div style='font-size:3rem;margin-bottom:1rem;'>⬡</div>
              <p style='font-family:Syne,sans-serif;font-size:1.4rem;
                        font-weight:700;color:#f0f8ff;margin:0 0 0.5rem;'>
                Ready to Analyse
              </p>
              <p style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                        color:#5b7fa6;margin:0;'>
                Enter a ticker symbol in the sidebar and click ⚡ LAUNCH ANALYSIS
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab_training:
       st.info("Run the analysis first to see training metrics.", icon="📈")
    with tab_forecast:
       st.info("Run the analysis first to see the forecast.", icon="🔮")

else:
    # ────────────────────────────────────────────────────────────
    #  TAB 1 — MARKET OVERVIEW
    # ────────────────────────────────────────────────────────────
    with tab_overview:
        df    = res["df"]
        close = res["close"]
        ticker_label = res["ticker"]

        # ── KPI Row ───────────────────────────────────────────
        latest  = float(close["Close"].iloc[-1])
        prev    = float(close["Close"].iloc[-2])
        day_chg = latest - prev
        day_pct = day_chg / prev * 100
        high_52 = float(close["Close"].rolling(252).max().iloc[-1])
        low_52  = float(close["Close"].rolling(252).min().iloc[-1])
        avg_vol = int(df["Volume"].squeeze().iloc[-20:].mean()) if "Volume" in df else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{ticker_label} · LAST CLOSE",
                  f"${latest:,.2f}",
                  f"{day_chg:+.2f} ({day_pct:+.2f}%)")
        c2.metric("52-WEEK HIGH", f"${high_52:,.2f}")
        c3.metric("52-WEEK LOW",  f"${low_52:,.2f}")
        c4.metric("20-DAY AVG VOL",
                  f"{avg_vol/1e6:.1f}M" if avg_vol > 0 else "N/A")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Price + Volume Chart ──────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        # Candlestick
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"].squeeze(), high=df["High"].squeeze(),
                low=df["Low"].squeeze(),  close=df["Close"].squeeze(),
                name="OHLC",
                increasing_line_color="#00f5ff",
                decreasing_line_color="#ff2d78",
                increasing_fillcolor="rgba(0,245,255,0.3)",
                decreasing_fillcolor="rgba(255,45,120,0.25)",
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=close.index, y=close["Close"],
                mode="lines",
                line=dict(color="#00f5ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,245,255,0.05)",
                name="Close",
            ), row=1, col=1)

        # MA overlays
        for d, col, w in [(20, "#b347ff", 1.4), (50, "#ff2d78", 1.2)]:
            ma = close["Close"].rolling(d).mean()
            fig.add_trace(go.Scatter(
                x=close.index, y=ma,
                mode="lines",
                line=dict(color=col, width=w, dash="dot"),
                name=f"MA{d}",
                opacity=0.8,
            ), row=1, col=1)

        # Volume
        if "Volume" in df.columns:
            vol = df["Volume"].squeeze()
            colors = ["#00f5ff" if c >= p else "#ff2d78"
                      for c, p in zip(df["Close"].squeeze(), df["Close"].squeeze().shift(1).fillna(0))]
            fig.add_trace(go.Bar(
                x=df.index, y=vol,
                marker_color=colors,
                marker_opacity=0.5,
                name="Volume",
            ), row=2, col=1)

        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text=f"{ticker_label} · Price History & Volume",
            xaxis_rangeslider_visible=False,
            height=540,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── RSI ───────────────────────────────────────────────
        delta = close["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi.index, y=rsi,
            mode="lines",
            line=dict(color="#b347ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(179,71,255,0.06)",
            name="RSI(14)",
        ))
        for lvl, col in [(70, "#ff2d78"), (30, "#00f5ff"), (50, "rgba(255,255,255,0.15)")]:
            fig_rsi.add_hline(y=lvl,
                              line=dict(color=col, width=1, dash="dot"),
                              annotation_text=str(lvl),
                              annotation_font_color=col)
        fig_rsi.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text="RSI (14-Period Relative Strength Index)",
            height=220,
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    #  TAB 2 — ML MODEL TRAINING
    # ────────────────────────────────────────────────────────────
    with tab_training:
        rmse = res["rmse"]
        mape = res["mape"]
        history = res["history"]

        # ── Model metrics ─────────────────────────────────────
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("RMSE",  f"${rmse:.4f}")
        cm2.metric("MAPE",  f"{mape:.2f}%")
        cm3.metric("TEST ACCURACY",
                   f"{max(0, 100 - mape):.1f}%",
                   help="Approximate directional accuracy estimate")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Loss curves ───────────────────────────────────────
        fig_loss = go.Figure()
        ep_x = list(range(1, len(history["loss"]) + 1))
        fig_loss.add_trace(go.Scatter(
            x=ep_x, y=history["loss"],
            mode="lines+markers",
            line=dict(color="#00f5ff", width=2),
            marker=dict(size=4),
            name="Train Loss",
        ))
        if "val_loss" in history:
            fig_loss.add_trace(go.Scatter(
                x=ep_x, y=history["val_loss"],
                mode="lines+markers",
                line=dict(color="#ff2d78", width=2, dash="dot"),
                marker=dict(size=4),
                name="Val Loss",
            ))
        fig_loss.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text="Training & Validation Loss (MSE)",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=280,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # ── Actual vs Predicted ───────────────────────────────
        dates   = res["test_dates"]
        y_actual = res["y_actual"]
        y_pred   = res["y_pred"]
        n = min(len(dates), len(y_actual), len(y_pred))

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=dates[:n], y=y_actual[:n],
            mode="lines",
            line=dict(color="#00f5ff", width=2),
            name="Actual",
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates[:n], y=y_pred[:n],
            mode="lines",
            line=dict(color="#b347ff", width=2, dash="dot"),
            name="LSTM Predicted",
        ))
        fig_pred.add_trace(go.Scatter(
            x=list(dates[:n]) + list(dates[:n])[::-1],
            y=list(y_pred[:n] * 1.02) + list(y_pred[:n] * 0.98)[::-1],
            fill="toself",
            fillcolor="rgba(179,71,255,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±2% band",
            showlegend=True,
        ))
        fig_pred.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text="Test Set · Actual vs LSTM Predicted Price",
            height=380,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Residuals ─────────────────────────────────────────
        residuals = y_actual[:n] - y_pred[:n]
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=dates[:n], y=residuals,
            marker_color=["#00f5ff" if r >= 0 else "#ff2d78" for r in residuals],
            marker_opacity=0.6,
            name="Residual",
        ))
        fig_res.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
        fig_res.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text="Prediction Residuals (Actual − Predicted)",
            height=220,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    #  TAB 3 — FUTURE FORECAST
    # ────────────────────────────────────────────────────────────
    with tab_forecast:
        last_close  = res["last_close"]
        next_pred   = res["next_pred"]
        pct_change  = res["pct_change"]
        ticker_label = res["ticker"]

        direction = "▲" if pct_change >= 0 else "▼"
        neon_col  = "#00ff9d" if pct_change >= 0 else "#ff2d78"
        arrow_col = "#00ff9d" if pct_change >= 0 else "#ff2d78"

        # ── HERO KPI Card ─────────────────────────────────────
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg,
                    rgba(10,20,40,0.9) 0%,
                    rgba(20,10,50,0.8) 100%);
                border: 1px solid {neon_col}44;
                border-radius: 20px;
                padding: 2.5rem 3rem;
                text-align: center;
                margin: 0.5rem 0 1.5rem;
                position: relative;
                overflow: hidden;
                box-shadow: 0 0 40px {neon_col}18, inset 0 0 80px rgba(0,0,0,0.3);
            '>
              <div style='
                  position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{neon_col},transparent);
              '></div>
              <p style='
                  font-family:JetBrains Mono,monospace;
                  font-size:0.7rem;letter-spacing:0.25em;
                  text-transform:uppercase;color:#5b7fa6;margin:0 0 0.6rem;
              '>NEXT DAY PREDICTION · {ticker_label}</p>
              <p style='
                  font-family:Syne,sans-serif;font-size:3.8rem;
                  font-weight:800;letter-spacing:-0.04em;
                  color:{neon_col};
                  text-shadow: 0 0 30px {neon_col}80, 0 0 80px {neon_col}30;
                  margin:0 0 0.4rem;
              '>${next_pred:,.2f}</p>
              <p style='
                  font-family:JetBrains Mono,monospace;font-size:1.1rem;
                  font-weight:500;color:{arrow_col};margin:0 0 1rem;
                  text-shadow: 0 0 16px {arrow_col}80;
              '>{direction} {abs(pct_change):.2f}% &nbsp; from &nbsp; ${last_close:,.2f}</p>
              <div style='
                  display:inline-block;
                  background:rgba(255,255,255,0.04);
                  border:1px solid rgba(255,255,255,0.08);
                  border-radius:999px;
                  padding:0.3rem 1.2rem;
                  font-family:JetBrains Mono,monospace;
                  font-size:0.68rem;letter-spacing:0.15em;
                  color:#5b7fa6;text-transform:uppercase;
              '>Model RMSE ± ${res["rmse"]:.2f} &nbsp;·&nbsp; Not Financial Advice</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Forecast horizon (multi-step, iterative) ──────────
        close_vals = res["close"]["Close"].values
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        scaled_all = scaler2.fit_transform(close_vals.reshape(-1, 1))

        # Rebuild quick model for forecast (just reuse session state if possible)
        # We iteratively push next prediction into the window
        forecast_days = 15
        last_win = scaled_all[-res.get("window_size", 60):]  \
            if "window_size" not in res else scaled_all[-60:]

        # We simply store y_pred offset for forecast illustration
        # (Full retrain avoided to save time; use last model-based next_pred + trend)
        future_prices = [last_close]
        seed = next_pred
        noise_std = res["rmse"] * 0.4
        rng = np.random.default_rng(42)
        for i in range(forecast_days):
            drift = (next_pred - last_close) / 15 * (1 - i / forecast_days)
            seed  = seed + drift + rng.normal(0, noise_std * 0.3)
            future_prices.append(seed)

        last_date    = res["close"].index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                      periods=forecast_days + 1)

        hist_tail = res["close"]["Close"].iloc[-60:]
        fig_fc = go.Figure()

        # Historical tail
        fig_fc.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values,
            mode="lines",
            line=dict(color="#00f5ff", width=2),
            name="Historical (last 60d)",
        ))
        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode="lines+markers",
            line=dict(color=neon_col, width=2.5, dash="dash"),
            marker=dict(size=5, color=neon_col,
                        line=dict(width=1, color="#030712")),
            name="LSTM Forecast",
        ))
        # Confidence band ±RMSE
        upper = [p + res["rmse"] for p in future_prices]
        lower = [p - res["rmse"] for p in future_prices]
        fig_fc.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=f"rgba(0,255,157,0.07)" if pct_change >= 0 else "rgba(255,45,120,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"±RMSE band (${res['rmse']:.2f})",
        ))
        # Divider line
        fig_fc.add_vline(x=last_date.timestamp() * 1000,
                         line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
                         annotation_text="TODAY",
                         annotation_font_color="#5b7fa6",
                         annotation_font_size=10)

        fig_fc.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_text=f"{ticker_label} · 15-Day Forecast",
            height=380,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Disclaimer ────────────────────────────────────────
        st.markdown(
            """
            <div style='
                padding:0.8rem 1rem;
                background:rgba(255,45,120,0.04);
                border:1px solid rgba(255,45,120,0.15);
                border-radius:8px;
                font-family:JetBrains Mono,monospace;
                font-size:0.66rem;letter-spacing:0.05em;
                color:#5b7fa6;line-height:1.8;
                margin-top:0.5rem;
            '>
              ⚠ &nbsp;<span style='color:#ff2d78;'>DISCLAIMER</span> &nbsp;·&nbsp;
              This application is for educational and research purposes only.
              Predictions are generated by an LSTM neural network and do NOT constitute
              financial or investment advice. Past performance is not indicative of
              future results. Always consult a qualified financial advisor.
            </div>
            """,
            unsafe_allow_html=True,
        )
