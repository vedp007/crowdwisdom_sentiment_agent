"""
Database manager for SQLite storage of tweets, news, prices, features, and backtest results.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database connections, schema migrations, and queries."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        """Initializes tables using schema.sql."""
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_file}")
            
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()
            
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
        logger.info(f"Initialized database schema at {self.db_path}")

    def insert_tweets(self, tweets: List[Dict[str, Any]]) -> int:
        """Bulk insert tweets ignoring duplicates."""
        if not tweets:
            return 0
        sql = """
        INSERT OR REPLACE INTO raw_tweets (
            id, ticker, timestamp_utc, timestamp_est, text, likes, reposts,
            views, replies, author_followers, author_verified, source
        ) VALUES (
            :id, :ticker, :timestamp_utc, :timestamp_est, :text, :likes, :reposts,
            :views, :replies, :author_followers, :author_verified, :source
        )
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(sql, tweets)
            conn.commit()
            return cursor.rowcount

    def insert_news(self, news_items: List[Dict[str, Any]]) -> int:
        """Bulk insert news items ignoring duplicates."""
        if not news_items:
            return 0
        sql = """
        INSERT OR IGNORE INTO raw_news (
            id, ticker, published_utc, title, snippet, source, url, relevance_score
        ) VALUES (
            :id, :ticker, :published_utc, :title, :snippet, :source, :url, :relevance_score
        )
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(sql, news_items)
            conn.commit()
            return cursor.rowcount

    def insert_price_bars(self, df_prices: pd.DataFrame, ticker: str) -> int:
        """Inserts or replaces daily OHLCV price bars from DataFrame."""
        if df_prices.empty:
            return 0
            
        df = df_prices.copy()
        if "ticker" not in df.columns:
            df["ticker"] = ticker
            
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        records = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].to_dict(orient="records")
        sql = """
        INSERT OR REPLACE INTO price_bars (
            ticker, date, open, high, low, close, adj_close, volume
        ) VALUES (
            :ticker, :date, :open, :high, :low, :close, :adj_close, :volume
        )
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(sql, records)
            conn.commit()
            return cursor.rowcount

    def save_features(self, df_features: pd.DataFrame) -> int:
        """Saves engineered feature DataFrame to database."""
        if df_features.empty:
            return 0
        with self.get_connection() as conn:
            df_features.to_sql("engineered_features", conn, if_exists="replace", index=False)
        return len(df_features)

    def save_predictions(self, df_preds: pd.DataFrame) -> int:
        """Saves model predictions DataFrame."""
        if df_preds.empty:
            return 0
        with self.get_connection() as conn:
            df_preds.to_sql("model_predictions", conn, if_exists="replace", index=False)
        return len(df_preds)

    def save_metrics(self, run_id: str, strategy_name: str, metrics: Dict[str, float]):
        """Saves evaluation and backtest metrics."""
        records = [
            {"run_id": run_id, "strategy_name": strategy_name, "metric_name": k, "metric_value": float(v)}
            for k, v in metrics.items()
        ]
        sql = """
        INSERT OR REPLACE INTO backtest_metrics (run_id, strategy_name, metric_name, metric_value)
        VALUES (:run_id, :strategy_name, :metric_name, :metric_value)
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()

    def get_tweets_df(self, ticker: str) -> pd.DataFrame:
        """Loads all tweets for a ticker into a DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM raw_tweets WHERE ticker = ? ORDER BY timestamp_utc ASC",
                conn,
                params=(ticker,)
            )

    def get_news_df(self, ticker: str) -> pd.DataFrame:
        """Loads all news items for a ticker into a DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM raw_news WHERE ticker = ? ORDER BY published_utc DESC",
                conn,
                params=(ticker,)
            )

    def get_price_bars_df(self, ticker: str) -> pd.DataFrame:
        """Loads price bars into a DataFrame indexed by date."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM price_bars WHERE ticker = ? ORDER BY date ASC",
                conn,
                params=(ticker,)
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df

    def get_features_df(self, ticker: str) -> pd.DataFrame:
        """Loads engineered features for a ticker."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM engineered_features WHERE ticker = ? ORDER BY date ASC",
                conn,
                params=(ticker,)
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df
