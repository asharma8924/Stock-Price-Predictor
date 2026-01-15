# Stock Price Predictor (Multi-Ticker + News Sentiment + Backtesting)

End-to-end pipeline that predicts next-day stock direction and returns across multiple tickers using:
- Technical indicators from historical prices (yfinance)
- Optional NLP sentiment signals from news headlines
- Scikit-learn models (Random Forest)
- Strategy backtesting vs. buy-and-hold
- Streamlit dashboard to visualize predictions and performance

**Tickers:** AAPL, MSFT, NVDA, AMZN, GOOGL, SPY

---

## Features (Engineered)

From price history:
- `return_1d`
- `sma_5`, `sma_20`
- `volatility_20`

From news (optional / pipeline-ready):
- `sentiment_mean`
- `headline_count`

Targets:
- `target_up` (classification: up/down)
- `target_return` (regression / evaluation)

---

## Project Structure

