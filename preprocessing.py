import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════════════════════════════
#  FEATURE 1 — News Sentiment Score
# ══════════════════════════════════════════════════════════════════
def get_news_sentiment_series(ticker_symbol, date_index):
    """
    Fetch recent news headlines for stock, score each as +1/0/-1
    using keyword matching (no API needed).
    Returns a pd.Series aligned to date_index.
    """
    POSITIVE_WORDS = [
        "surge","rally","gain","rise","jump","high","record","profit",
        "beat","growth","strong","upgrade","buy","bullish","positive",
        "revenue","earnings beat","outperform","dividend","acquisition"
    ]
    NEGATIVE_WORDS = [
        "fall","drop","crash","loss","decline","weak","downgrade","sell",
        "bearish","negative","lawsuit","fraud","miss","cut","layoff",
        "recession","default","penalty","probe","investigation"
    ]

    # Fetch headlines from Google News RSS
    scores = []
    try:
        short = ticker_symbol.replace(".NS","").replace(".BO","")
        url   = f"https://news.google.com/rss/search?q={urllib.request.quote(short)}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        req   = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            tree = ET.parse(r)
        items = tree.getroot().findall(".//item")[:20]
        for item in items:
            title = (item.findtext("title") or "").lower()
            score = 0
            for w in POSITIVE_WORDS:
                if w in title: score += 1
            for w in NEGATIVE_WORDS:
                if w in title: score -= 1
            scores.append(max(-1, min(1, score)))
    except:
        pass

    # Average score → single float, fill entire series with it
    avg_score = float(np.mean(scores)) if scores else 0.0

    sentiment_series = pd.Series(avg_score, index=date_index)
    return sentiment_series


# ══════════════════════════════════════════════════════════════════
#  FEATURE 2 — Earnings Calendar Flag
# ══════════════════════════════════════════════════════════════════
def get_earnings_flag_series(ticker_symbol, date_index):
    """
    Returns 1 on earnings date ±2 days, else 0.
    Model learns earnings dates = high volatility.
    """
    flags = pd.Series(0.0, index=date_index)
    try:
        tk   = yf.Ticker(ticker_symbol)
        cal  = tk.calendar
        if cal is not None and not cal.empty:
            # Earnings Date row
            if "Earnings Date" in cal.index:
                earn_dates = cal.loc["Earnings Date"]
                if not isinstance(earn_dates, pd.Series):
                    earn_dates = pd.Series([earn_dates])
                for ed in earn_dates:
                    if pd.notna(ed):
                        ed = pd.Timestamp(ed).tz_localize(None)
                        for delta in range(-2, 3):
                            target = ed + timedelta(days=delta)
                            # Find closest date in index
                            idx_pos = date_index.searchsorted(target)
                            if idx_pos < len(date_index):
                                flags.iloc[idx_pos] = 1.0
    except:
        pass
    return flags


# ══════════════════════════════════════════════════════════════════
#  FEATURE 3 — Fear & Greed Index
# ══════════════════════════════════════════════════════════════════
def compute_fear_greed_series(data):
    """
    Custom Fear & Greed Index from:
    - RSI (momentum)
    - Price vs 125-day MA (trend)
    - BB width (volatility — wide = fear)
    - Volume ratio (high vol = panic or euphoria)
    All normalized to 0-1. Higher = Greed, Lower = Fear.
    """
    df    = data.copy()
    close = df['Close'].squeeze()

    # RSI component (0-1)
    delta = close.diff()
    gain  = delta.where(delta>0,0.0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0.0)).rolling(14).mean()
    rsi   = 100 - (100/(1+gain/(loss+1e-10)))
    rsi_norm = rsi / 100.0

    # Price vs 125-day MA (above = greed)
    ma125    = close.rolling(125).mean()
    trend    = ((close - ma125) / (ma125 + 1e-10)).clip(-1, 1)
    trend_norm = (trend + 1) / 2.0

    # Bollinger Band Width (high = fear)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_width = (2 * bb_std) / (bb_mid + 1e-10)
    bb_norm  = 1 - (bb_width / (bb_width.rolling(252).max() + 1e-10))

    # Volume ratio
    if 'Volume' in df.columns:
        vol_ma   = df['Volume'].rolling(20).mean()
        vol_rat  = df['Volume'] / (vol_ma + 1e-10)
        vol_norm = (vol_rat / (vol_rat.rolling(252).max() + 1e-10)).clip(0,1)
    else:
        vol_norm = pd.Series(0.5, index=df.index)

    # Weighted average
    fg = (0.35*rsi_norm + 0.30*trend_norm + 0.20*bb_norm + 0.15*vol_norm)
    fg = fg.fillna(0.5)
    return fg


# ══════════════════════════════════════════════════════════════════
#  FEATURE 4 — Global Market Correlation
# ══════════════════════════════════════════════════════════════════
def get_global_market_series(date_index, start_date="2017-01-01"):
    """
    Fetch S&P500 and Nikkei daily returns.
    US market today → Indian market tomorrow (lag effect).
    Returns normalized daily returns aligned to date_index.
    """
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    sp500_ret  = pd.Series(0.0, index=date_index)
    nikkei_ret = pd.Series(0.0, index=date_index)

    for sym, series in [("^GSPC", sp500_ret), ("^N225", nikkei_ret)]:
        try:
            df_g = yf.Ticker(sym).history(start=start_date, end=end_date)
            if df_g.empty: continue
            ret = df_g['Close'].pct_change().shift(1)  # lag 1 day
            ret.index = ret.index.tz_localize(None)
            # Align to our date_index
            aligned = ret.reindex(date_index, method='ffill').fillna(0)
            # Normalize to [-1, 1]
            aligned = (aligned / (aligned.abs().rolling(252).max() + 1e-10)).clip(-1,1)
            if sym == "^GSPC":
                sp500_ret  = aligned
            else:
                nikkei_ret = aligned
        except:
            pass

    return sp500_ret, nikkei_ret


# ══════════════════════════════════════════════════════════════════
#  CORE TECHNICAL FEATURES
# ══════════════════════════════════════════════════════════════════
def add_technical_features(data, ticker_symbol="", add_external=True):
    df    = data.copy()
    close = df['Close'].squeeze()

    # EMA 9, 21, 50
    df['EMA9']  = close.ewm(span=9,  adjust=False).mean()
    df['EMA21'] = close.ewm(span=21, adjust=False).mean()
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain  = delta.where(delta>0,0.0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0.0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/(loss+1e-10)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume ratio
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20).mean()
        df['Vol_ratio'] = df['Volume'] / (vol_ma + 1e-10)
    else:
        df['Vol_ratio'] = 1.0

    # Price momentum
    df['Mom5']  = close.pct_change(5)
    df['Mom10'] = close.pct_change(10)

    # Bollinger Band width
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_width'] = (2 * bb_std) / (bb_mid + 1e-10)

    # Drop NaN from rolling windows before adding external features
    df.dropna(inplace=True)

    if add_external and len(df) > 50:
        idx = df.index

        # Feature 1 — News Sentiment
        if ticker_symbol:
            df['News_sentiment'] = get_news_sentiment_series(ticker_symbol, idx).values
        else:
            df['News_sentiment'] = 0.0

        # Feature 2 — Earnings Flag
        if ticker_symbol:
            df['Earnings_flag'] = get_earnings_flag_series(ticker_symbol, idx).values
        else:
            df['Earnings_flag'] = 0.0

        # Feature 3 — Fear & Greed
        df['Fear_greed'] = compute_fear_greed_series(df).values

        # Feature 4 — Global markets
        sp_ret, nk_ret = get_global_market_series(idx)
        df['SP500_ret']  = sp_ret.values
        df['Nikkei_ret'] = nk_ret.values

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════
#  SCALE + SPLIT
# ══════════════════════════════════════════════════════════════════
def scale_data(data, window_size=60, use_features=True, ticker_symbol=""):
    if use_features:
        df = add_technical_features(data, ticker_symbol=ticker_symbol, add_external=True)
        feature_cols = [
            'Close','EMA9','EMA21','EMA50',
            'RSI','MACD','MACD_signal',
            'Vol_ratio','Mom5','Mom10','BB_width',
            'News_sentiment','Earnings_flag','Fear_greed',
            'SP500_ret','Nikkei_ret'
        ]
        # Keep only columns that exist (external fetch may fail)
        feature_cols = [c for c in feature_cols if c in df.columns]
        feature_data = df[feature_cols].values
    else:
        df           = data.copy()
        feature_data = df['Close'].values.reshape(-1,1)

    close_prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(feature_data)

    close_scaler = MinMaxScaler(feature_range=(0,1))
    close_scaler.fit_transform(close_prices)

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len]
    test_data  = scaled_data[training_data_len - window_size:]

    return close_prices, scaled_data, train_data, test_data, scaler, close_scaler, training_data_len


def create_sequences(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i])
        y.append(dataset[i, 0])
    X = np.array(X)
    y = np.array(y)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y