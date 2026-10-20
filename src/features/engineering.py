"""
Quantitative Feature Engineering Module.
Creates strictly time-based features from sentiment and price only,
with multi-horizon target generation (n+0 to n+5) ensuring zero lookahead bias.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Constructs quantitative sentiment, price momentum, volatility, and interaction features."""
    
    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons if horizons is not None else [0, 1, 2, 3, 4, 5]

    def build_feature_dataset(
        self,
        prices_df: pd.DataFrame,
        daily_sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges daily price bars with aggregated sentiment metrics and constructs features.
        All rolling windows look STRICTLY backward to guarantee no future data leakage.
        """
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty")
            
        df_p = prices_df.copy()
        if "date" in df_p.columns:
            df_p["date"] = pd.to_datetime(df_p["date"]).dt.strftime("%Y-%m-%d")
        else:
            df_p["date"] = pd.to_datetime(df_p.index).strftime("%Y-%m-%d")
            
        df_p.set_index("date", inplace=True)
        df_p.sort_index(inplace=True)

        # Merge with daily sentiment
        if daily_sentiment_df is not None and not daily_sentiment_df.empty:
            df_s = daily_sentiment_df.copy()
            df_s["date"] = pd.to_datetime(df_s["date"]).dt.strftime("%Y-%m-%d")
            df_s.set_index("date", inplace=True)
            merged = df_p.join(df_s, how="left")
        else:
            merged = df_p.copy()
            merged["weighted_sentiment"] = 0.0
            merged["raw_sentiment_mean"] = 0.0
            merged["tweet_volume"] = 0
            merged["sentiment_dispersion"] = 0.0
            merged["bull_bear_ratio"] = 1.0

        # Fill missing sentiment on non-trading days/weekends via forward fill and baseline defaults
        merged["weighted_sentiment"] = merged["weighted_sentiment"].ffill().fillna(0.0)
        merged["raw_sentiment_mean"] = merged["raw_sentiment_mean"].ffill().fillna(0.0)
        merged["tweet_volume"] = merged["tweet_volume"].fillna(0).astype(int)
        merged["sentiment_dispersion"] = merged["sentiment_dispersion"].fillna(0.0)
        merged["bull_bear_ratio"] = merged["bull_bear_ratio"].fillna(1.0)

        # -------------------------------------------------------------
        # 1. Price-Only Quantitative Features (Strictly Backward-Looking)
        # -------------------------------------------------------------
        close = merged["close"]
        merged["return_lag_1d"] = close.pct_change(1)
        merged["return_lag_2d"] = close.pct_change(2)
        merged["return_lag_3d"] = close.pct_change(3)
        merged["return_lag_5d"] = close.pct_change(5)
        merged["return_lag_10d"] = close.pct_change(10)

        # Rolling Volatility
        merged["volatility_5d"] = merged["return_lag_1d"].rolling(5).std() * np.sqrt(252)
        merged["volatility_21d"] = merged["return_lag_1d"].rolling(21).std() * np.sqrt(252)

        # Relative Strength Index (RSI 14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        merged["rsi_14d"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12, 26, 9) normalized by close price
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        merged["macd"] = (ema12 - ema26) / close
        merged["macd_signal"] = merged["macd"].ewm(span=9, adjust=False).mean()

        # -------------------------------------------------------------
        # 2. Sentiment-Only Quantitative Features
        # -------------------------------------------------------------
        # Post Volume Z-Score relative to 20-day baseline
        vol_mean_20 = merged["tweet_volume"].rolling(20, min_periods=5).mean()
        vol_std_20 = merged["tweet_volume"].rolling(20, min_periods=5).std().replace(0, 1.0)
        merged["tweet_volume_zscore"] = (merged["tweet_volume"] - vol_mean_20) / (vol_std_20 + 1e-6)

        # Sentiment Momentum over 3d, 7d, 14d
        merged["sentiment_momentum_3d"] = merged["weighted_sentiment"] - merged["weighted_sentiment"].shift(3)
        merged["sentiment_momentum_7d"] = merged["weighted_sentiment"] - merged["weighted_sentiment"].shift(7)
        merged["sentiment_momentum_14d"] = merged["weighted_sentiment"] - merged["weighted_sentiment"].shift(14)
        merged["sentiment_ma_7d"] = merged["weighted_sentiment"].rolling(7, min_periods=3).mean()

        # -------------------------------------------------------------
        # 3. Sentiment x Price Interaction Features
        # -------------------------------------------------------------
        # Divergence between sentiment momentum and price returns
        z_ret_5d = (merged["return_lag_5d"] - merged["return_lag_5d"].rolling(20, min_periods=5).mean()) / (merged["return_lag_5d"].rolling(20, min_periods=5).std() + 1e-6)
        z_sent_5d = (merged["sentiment_momentum_7d"] - merged["sentiment_momentum_7d"].rolling(20, min_periods=5).mean()) / (merged["sentiment_momentum_7d"].rolling(20, min_periods=5).std() + 1e-6)
        merged["sentiment_price_divergence"] = (z_sent_5d - z_ret_5d).fillna(0.0)

        # Sentiment intensity scaled by abnormal tweet volume
        merged["sentiment_volume_surge"] = (merged["weighted_sentiment"] * merged["tweet_volume_zscore"].clip(lower=0)).fillna(0.0)

        # -------------------------------------------------------------
        # 4. Multi-Horizon Targets (n+0, n+1, n+2, n+3, n+4, n+5)
        # -------------------------------------------------------------
        for h in self.horizons:
            # Price at horizon h (shift forward: future price aligned to current row)
            merged[f"target_price_n{h}"] = close.shift(-h)
            # Return from current close to horizon h close
            if h == 0:
                # n+0 represents same-period close price
                merged[f"target_ret_n{h}"] = 0.0
            else:
                merged[f"target_ret_n{h}"] = (close.shift(-h) - close) / close

        merged.reset_index(inplace=True)
        # Drop rows where initial rolling indicators are NaN (first 21 days for warm-up)
        # But preserve recent rows even if target_price_n5 is NaN (for live inference!)
        clean_df = merged.dropna(subset=["rsi_14d", "volatility_21d", "tweet_volume_zscore"]).reset_index(drop=True)
        
        logger.info(f"Engineered feature set constructed: {clean_df.shape[0]} samples with {clean_df.shape[1]} columns.")
        return clean_df

    def get_feature_columns(self) -> List[str]:
        """Returns the list of input predictor column names (sentiment and price only)."""
        return [
            # Sentiment features
            "weighted_sentiment",
            "raw_sentiment_mean",
            "tweet_volume",
            "tweet_volume_zscore",
            "sentiment_dispersion",
            "bull_bear_ratio",
            "sentiment_momentum_3d",
            "sentiment_momentum_7d",
            "sentiment_momentum_14d",
            "sentiment_ma_7d",
            
            # Price features
            "return_lag_1d",
            "return_lag_2d",
            "return_lag_3d",
            "return_lag_5d",
            "return_lag_10d",
            "volatility_5d",
            "volatility_21d",
            "rsi_14d",
            "macd",
            "macd_signal",
            
            # Sentiment-Price Interaction
            "sentiment_price_divergence",
            "sentiment_volume_surge"
        ]
