-- CrowdWisdomTrading Database Schema

CREATE TABLE IF NOT EXISTS raw_tweets (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    timestamp_est TIMESTAMP NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    reposts INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    author_followers INTEGER DEFAULT 0,
    author_verified BOOLEAN DEFAULT 0,
    source TEXT DEFAULT 'apify'
);

CREATE INDEX IF NOT EXISTS idx_tweets_date ON raw_tweets (ticker, timestamp_utc);

CREATE TABLE IF NOT EXISTS raw_news (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    published_utc TIMESTAMP NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    source TEXT,
    url TEXT,
    relevance_score REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_news_date ON raw_news (ticker, published_utc);

CREATE TABLE IF NOT EXISTS price_bars (
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_bars (ticker, date);

CREATE TABLE IF NOT EXISTS engineered_features (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    close REAL,
    raw_sentiment_mean REAL,
    weighted_sentiment REAL,
    tweet_volume INTEGER,
    tweet_volume_zscore REAL,
    sentiment_dispersion REAL,
    sentiment_momentum_3d REAL,
    sentiment_momentum_7d REAL,
    sentiment_momentum_14d REAL,
    bull_bear_ratio REAL,
    return_lag_1d REAL,
    return_lag_2d REAL,
    return_lag_3d REAL,
    return_lag_5d REAL,
    volatility_5d REAL,
    volatility_21d REAL,
    rsi_14d REAL,
    macd REAL,
    macd_signal REAL,
    sentiment_price_divergence REAL,
    target_price_n0 REAL,
    target_price_n1 REAL,
    target_price_n2 REAL,
    target_price_n3 REAL,
    target_price_n4 REAL,
    target_price_n5 REAL,
    target_ret_n0 REAL,
    target_ret_n1 REAL,
    target_ret_n2 REAL,
    target_ret_n3 REAL,
    target_ret_n4 REAL,
    target_ret_n5 REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    model_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_price REAL NOT NULL,
    actual_price REAL,
    predicted_return REAL NOT NULL,
    actual_return REAL,
    residual REAL,
    PRIMARY KEY (date, ticker, model_name, horizon)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, strategy_name, metric_name)
);
