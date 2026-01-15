Stock Price Predictor (Multi-Ticker + News Sentiment + Backtesting)
End-to-end pipeline that predicts next-day stock direction and returns across multiple tickers using:

Technical indicators from historical prices (yfinance)
Optional NLP sentiment signals from news headlines
Scikit-learn models (Random Forest)
Strategy backtesting vs. buy-and-hold
Streamlit dashboard to visualize predictions and performance
Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL, SPY

Features (Engineered)
From price history:

return_1d
sma_5, sma_20
volatility_20
From news (optional / pipeline-ready):

sentiment_mean
headline_count
Targets:

target_up (classification: up/down)
target_return (regression / evaluation)

Project Structure
stock-price-predictor/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/                  # cached yfinance pulls, raw headlines (optional)
│  ├─ processed/            # merged features/targets per ticker
│  └─ outputs/              # backtest results, metrics, charts
├─ src/
│  ├─ config.py             # tickers list, date ranges, feature flags (sentiment on/off)
│  ├─ data/
│  │  ├─ fetch_prices.py    # yfinance download, cleaning, corporate actions handling
│  │  ├─ fetch_news.py      # headlines pull (optional), caching
│  │  └─ make_dataset.py    # join prices + sentiment, build targets
│  ├─ features/
│  │  ├─ technical.py       # return_1d, sma_5/20, volatility_20, etc.
│  │  └─ sentiment.py       # sentiment_mean, headline_count (optional)
│  ├─ models/
│  │  ├─ train.py           # time-series split, fit RF classifier, persist model
│  │  ├─ predict.py         # next-day preds per ticker, batch inference
│  │  └─ evaluate.py        # accuracy, AUC (optional), regression metrics on returns
│  ├─ backtest/
│  │  ├─ strategy.py        # rules: long if p(up)>thresh, else cash (or short optional)
│  │  └─ backtest.py        # simulate daily portfolio, compare vs buy-and-hold
│  └─ utils/
│     ├─ metrics.py         # CAGR, Sharpe, max drawdown, hit rate, etc.
│     └─ io.py              # saving/loading data, models, results
├─ app/
│  └─ streamlit_app.py      # dashboard UI: ticker select, charts, metrics
└─ notebooks/               # experiments (optional, keep separate from src)
