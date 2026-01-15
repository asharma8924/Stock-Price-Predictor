import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Paths (project-root safe)
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")


# -----------------------------
# Feature engineering
# -----------------------------
def make_price_panel(tickers, start="2020-01-01", end=None):
    """
    Download prices and return a long 'panel' dataframe:
    columns: date, ticker, close
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    # Extract close prices into a wide DataFrame with columns = tickers
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex columns: (Ticker, Field) in your setup
        if "Close" in df.columns.get_level_values(1):
            close = df.xs("Close", axis=1, level=1)
        else:
            raise KeyError("Could not find 'Close' in downloaded data (MultiIndex).")
    else:
        # Single ticker case
        if "Close" in df.columns:
            close = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError("Could not find 'Close' in downloaded data.")

    # Convert wide -> long
    tmp = close.stack().reset_index()

    # Robust column renaming (handles Date/index/level_0 and ticker column variations)
    cols = list(tmp.columns)

    date_col = "Date" if "Date" in cols else ("index" if "index" in cols else "level_0")
    ticker_col = "level_1" if "level_1" in cols else ("Ticker" if "Ticker" in cols else cols[1])
    value_col = 0 if 0 in cols else cols[-1]

    panel = tmp.rename(columns={date_col: "date", ticker_col: "ticker", value_col: "close"})

    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel



def add_technical_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical features per ticker.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = panel.groupby("ticker", group_keys=False)

    # daily return
    panel["return_1d"] = g["close"].pct_change()

    # moving averages
    panel["sma_5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    panel["sma_20"] = g["close"].transform(lambda s: s.rolling(20).mean())

    # volatility of returns
    panel["volatility_20"] = g["return_1d"].transform(lambda s: s.rolling(20).std())

    # target: next-day return, and up/down label
    panel["target_return"] = g["return_1d"].shift(-1)
    panel["target_up"] = (panel["target_return"] > 0).astype(int)

    return panel


# -----------------------------
# Backtest
# -----------------------------
def backtest_strategy(
    panel: pd.DataFrame,
    model,
    ticker: str = "AAPL",
    proba_threshold: float = 0.5,
    fee_bps: float = 0.0,
):
    """
    Simple long/cash strategy:
    - If model predicts UP with prob >= threshold -> long next day
    - else stay in cash
    Strategy return = position * next-day return - fees
    """
    feats = ["return_1d", "sma_5", "sma_20", "volatility_20", "sentiment_mean", "headline_count"]

    df = panel[panel["ticker"] == ticker].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Make sure required columns exist
    for c in feats + ["target_return"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Drop rows where features/target are NaN (rolling windows cause NaNs early)
    df = df.dropna(subset=feats + ["target_return"]).reset_index(drop=True)

    X = df[feats]

    # Predict probabilities for UP class
    proba_up = model.predict_proba(X)[:, 1]
    pred_up = (proba_up >= proba_threshold).astype(int)

    df["proba_up"] = proba_up
    df["pred_up"] = pred_up

    # position for next day based on today's signal (we assume signal known end-of-day)
    df["position"] = df["pred_up"]

    # Fees (optional): apply fee when position changes
    fee = fee_bps / 10000.0
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["fee"] = df["trade"] * fee

    df["strategy_return"] = df["position"] * df["target_return"] - df["fee"]
    df["buyhold_return"] = df["target_return"]

    df["equity_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["equity_buyhold"] = (1 + df["buyhold_return"]).cumprod()

    # basic stats
    total_strategy = df["equity_strategy"].iloc[-1] - 1
    total_buyhold = df["equity_buyhold"].iloc[-1] - 1

    sharpe = np.nan
    if df["strategy_return"].std() != 0:
        sharpe = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(252)

    stats = {
        "rows_used": len(df),
        "total_return_strategy": total_strategy,
        "total_return_buyhold": total_buyhold,
        "sharpe_strategy": sharpe,
        "avg_daily_return_strategy": df["strategy_return"].mean(),
        "hit_rate": (df["strategy_return"] > 0).mean(),
        "trades": int(df["trade"].sum()),
    }

    return df, stats


def main():
    # 1) Load model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # 2) Choose tickers and make panel
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]
    panel = make_price_panel(tickers, start="2020-01-01")

    # 3) Add technical features
    panel = add_technical_features(panel)

    # 4) If you don't have sentiment for older years, set defaults (works for demo)
    #    Later, you can merge your real sentiment dataframe instead.
    if "sentiment_mean" not in panel.columns:
        panel["sentiment_mean"] = 0.0
    if "headline_count" not in panel.columns:
        panel["headline_count"] = 0.0

    # 5) Backtest ALL tickers and combine
    all_bt = []
    all_stats = []

    for t in tickers:
        bt_df, stats = backtest_strategy(
            panel,
            model,
            ticker=t,
            proba_threshold=0.55,
            fee_bps=2.0,
        )
        all_bt.append(bt_df.assign(ticker=t))
        stats["ticker"] = t
        all_stats.append(stats)

    bt_all = pd.concat(all_bt, ignore_index=True)
    stats_df = pd.DataFrame(all_stats)

    print("\n=== Backtest Stats (All Tickers) ===")
    print(
        stats_df[["ticker", "rows_used", "total_return_strategy", "total_return_buyhold", "sharpe_strategy", "trades"]])

    # 6) Save results
    out_path = os.path.join(PROJECT_ROOT, "data", "backtest_all.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bt_all.to_csv(out_path, index=False)
    print(f"\nSaved backtest results to: {out_path}")

    stats_path = os.path.join(PROJECT_ROOT, "data", "backtest_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved backtest stats to: {stats_path}")


if __name__ == "__main__":
    main()
