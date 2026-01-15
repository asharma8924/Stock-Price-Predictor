import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Paths (project-root safe)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Default file produced by your backtest.py
DEFAULT_BACKTEST = DATA_DIR / "backtest_all.csv"


# -----------------------------
# Helpers
# -----------------------------
def load_backtest_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def sharpe_ratio(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return float("nan")
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Stock Price Predictor Dashboard", layout="wide")

st.title("📈 Stock Price Predictor Dashboard")
st.caption("Tech indicators + (optional) news sentiment → model predictions → backtest results.")

st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload a backtest CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    source_name = "Uploaded CSV"
else:
    if not DEFAULT_BACKTEST.exists():
        st.error(
            f"Could not find default backtest file at:\n{DEFAULT_BACKTEST}\n\n"
            "Run this first from the project root:\n"
            "  python -m src.backtest\n"
            "Then refresh this page."
        )
        st.stop()
    df = load_backtest_df(DEFAULT_BACKTEST)
    source_name = str(DEFAULT_BACKTEST)

st.sidebar.success(f"Loaded: {source_name}")

# If you only have AAPL right now, this will just show AAPL
if "ticker" in df.columns:
    tickers = sorted(df["ticker"].astype(str).unique())
    ticker = st.sidebar.selectbox("Ticker", tickers, index=0)
    view = df[df["ticker"] == ticker].sort_values("date").copy()
else:
    ticker = "N/A"
    view = df.sort_values("date").copy()

st.sidebar.header("Strategy Settings")
threshold = st.sidebar.slider("Probability Threshold (UP)", 0.40, 0.80, 0.55, 0.01)

# Recompute signals/equity in-app (so the slider changes behavior)
if "proba_up" in view.columns and "target_return" in view.columns:
    view["signal"] = (view["proba_up"] >= threshold).astype(int)
    view["strategy_return"] = view["signal"] * view["target_return"]
    view["buyhold_return"] = view["target_return"]

    view["equity_strategy"] = (1 + view["strategy_return"]).cumprod()
    view["equity_buyhold"] = (1 + view["buyhold_return"]).cumprod()
else:
    st.warning(
        "This CSV does not contain the expected columns ('proba_up', 'target_return'). "
        "Equity curves may not update with the threshold slider."
    )

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

if "equity_strategy" in view.columns and len(view) > 0:
    strat_total = view["equity_strategy"].iloc[-1] - 1
    bh_total = view["equity_buyhold"].iloc[-1] - 1 if "equity_buyhold" in view.columns else float("nan")
    strat_sharpe = sharpe_ratio(view.get("strategy_return", pd.Series(dtype=float)))
    hit_rate = (view.get("strategy_return", pd.Series(dtype=float)) > 0).mean()

    col1.metric("Strategy Total Return", f"{strat_total:.2%}")
    col2.metric("Buy & Hold Total Return", f"{bh_total:.2%}" if pd.notna(bh_total) else "N/A")
    col3.metric("Strategy Sharpe", f"{strat_sharpe:.2f}" if pd.notna(strat_sharpe) else "N/A")
    col4.metric("Win Rate", f"{hit_rate:.2%}" if pd.notna(hit_rate) else "N/A")
else:
    col1.metric("Strategy Total Return", "N/A")
    col2.metric("Buy & Hold Total Return", "N/A")
    col3.metric("Strategy Sharpe", "N/A")
    col4.metric("Win Rate", "N/A")


# -----------------------------
# Charts
# -----------------------------
st.subheader(f"📊 Equity Curve — {ticker}")

fig, ax = plt.subplots(figsize=(12, 5))
if "equity_strategy" in view.columns:
    ax.plot(view["date"], view["equity_strategy"], label="Strategy")
if "equity_buyhold" in view.columns:
    ax.plot(view["date"], view["equity_buyhold"], label="Buy & Hold")
ax.set_title("Equity Curves")
ax.set_xlabel("Date")
ax.set_ylabel("Equity (Starting at 1.0)")
ax.legend()
st.pyplot(fig)


st.subheader("📉 Predicted Probability vs Actual Next-Day Return")

fig2, ax2 = plt.subplots(figsize=(12, 5))
if "proba_up" in view.columns:
    ax2.plot(view["date"], view["proba_up"], label="Predicted P(Up)")
if "target_return" in view.columns:
    ax2.plot(view["date"], view["target_return"], label="Actual Next-Day Return")
ax2.axhline(threshold, linestyle="--", linewidth=2, label=f"Threshold {threshold:.2f}")
ax2.set_title("Model Signal vs Real Return")
ax2.set_xlabel("Date")
ax2.legend()
st.pyplot(fig2)


# -----------------------------
# Data preview
# -----------------------------
st.subheader("🧾 Data Preview")
st.dataframe(view.tail(50))
