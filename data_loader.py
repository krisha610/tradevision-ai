import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(stock_name):
    start_date = "2017-01-01"
    end_date   = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    ticker = yf.Ticker(stock_name)
    data   = ticker.history(start=start_date, end=end_date, auto_adjust=False)

    if data is None or data.empty:
        return None, {}

    # Use Adj Close if available
    if 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    # ── Data Quality Fixes ────────────────────────────────────────
    # 1. Remove rows where Close is 0 or NaN
    data = data[data['Close'] > 0].copy()
    data.dropna(subset=['Close'], inplace=True)

    # 2. Fill missing Volume with forward fill
    if 'Volume' in data.columns:
        data['Volume'] = data['Volume'].replace(0, np.nan)
        data['Volume'].fillna(method='ffill', inplace=True)
        data['Volume'].fillna(0, inplace=True)

    # 3. Remove obvious outliers — price jumps >70% in one day
    pct_chg = data['Close'].pct_change().abs()
    data = data[pct_chg < 0.70].copy()

    # 4. Sort index ascending, drop duplicates
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep='last')]

    # 5. Need at least 200 rows for meaningful training
    if len(data) < 200:
        return None, {}

    # ── Fetch info safely ─────────────────────────────────────────
    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    try:
        news = ticker.news
        info['news'] = news if news else []
    except Exception:
        info['news'] = []

    return data, info