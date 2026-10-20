"""
Quantitative Sentiment Analysis Engine.
Combines Lexicon/VADER financial adaptations with engagement and follower-weighting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Computes NLP sentiment scores with domain financial adaptations and engagement weighting."""
    
    def __init__(self, engagement_weights: Dict[str, float] = None):
        self.vader = SentimentIntensityAnalyzer()
        self._add_financial_lexicon()
        
        self.weights = engagement_weights or {
            "likes": 1.0,
            "reposts": 2.0,
            "replies": 1.5,
            "views": 0.05,
            "follower_scale": 10.0,
            "verified_multiplier": 1.35
        }

    def _add_financial_lexicon(self):
        """Augments VADER lexicon with high-signal financial market terminology."""
        financial_terms = {
            "bullish": 3.2,
            "breakout": 2.8,
            "outperform": 2.6,
            "upgrade": 2.7,
            "beat": 2.5,
            "all-time high": 3.0,
            "ath": 2.8,
            "calls": 1.8,
            "buy": 2.2,
            "surge": 2.6,
            "rally": 2.5,
            "soar": 2.7,
            "record revenue": 3.2,
            "margin expansion": 2.8,
            "bearish": -3.2,
            "downgrade": -2.7,
            "breakdown": -2.8,
            "miss": -2.6,
            "puts": -1.8,
            "sell": -2.2,
            "plunge": -3.0,
            "dump": -2.8,
            "crash": -3.5,
            "correction": -2.0,
            "dilution": -2.5,
            "bubble": -2.4,
            "insider selling": -2.2,
            "margin contraction": -2.8
        }
        self.vader.lexicon.update(financial_terms)

    def score_text(self, text: str) -> float:
        """Calculates compound polarity score in range [-1.0, 1.0]."""
        if not text:
            return 0.0
        scores = self.vader.polarity_scores(text)
        return float(scores["compound"])

    def compute_engagement_weight(self, row: Dict[str, Any]) -> float:
        """
        Calculates credibility & reach weight based on engagement and follower counts:
        w = ln(1 + likes + 2*reposts + 1.5*replies + 0.05*views) * ln(follower_scale + followers) * (verified_mult)
        """
        likes = float(row.get("likes", 0))
        reposts = float(row.get("reposts", 0))
        replies = float(row.get("replies", 0))
        views = float(row.get("views", 0))
        followers = float(row.get("author_followers", 100))
        is_verified = bool(row.get("author_verified", 0))

        engagement_score = (
            self.weights["likes"] * likes +
            self.weights["reposts"] * reposts +
            self.weights["replies"] * replies +
            self.weights["views"] * views
        )
        
        # Logarithmic saturation to prevent single viral tweet from completely overwhelming daily sample
        eng_factor = np.log1p(engagement_score)
        follower_factor = np.log(self.weights["follower_scale"] + max(0.0, followers))
        verified_boost = self.weights["verified_multiplier"] if is_verified else 1.0

        weight = max(0.05, eng_factor * follower_factor * verified_boost)
        return float(weight)

    def process_tweets_dataframe(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Applies sentiment scoring and engagement weights to each tweet record."""
        if tweets_df.empty:
            return pd.DataFrame()
            
        df = tweets_df.copy()
        logger.info(f"Scoring {len(df)} tweets with financial sentiment analyzer...")
        
        # Calculate raw compound sentiment
        df["sentiment_score"] = df["text"].apply(self.score_text)
        
        # Calculate engagement weights
        records = df.to_dict(orient="records")
        weights = [self.compute_engagement_weight(r) for r in records]
        df["weight"] = weights
        df["weighted_sentiment"] = df["sentiment_score"] * df["weight"]

        # Extract date from EST timestamp for alignment with US trading sessions
        df["est_datetime"] = pd.to_datetime(df["timestamp_est"])
        df["trade_date"] = df["est_datetime"].dt.strftime("%Y-%m-%d")

        return df

    def aggregate_daily_sentiment(self, scored_tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates individual scored tweets into daily quantitative sentiment metrics.
        Produces:
        - raw_sentiment_mean
        - weighted_sentiment
        - tweet_volume
        - sentiment_dispersion (standard deviation)
        - bull_bear_ratio
        - positive_ratio, negative_ratio
        """
        if scored_tweets_df.empty:
            return pd.DataFrame()

        daily_groups = scored_tweets_df.groupby("trade_date")
        aggregated_rows = []

        for date_str, group in daily_groups:
            total_tweets = len(group)
            total_weight = group["weight"].sum()
            
            # Weighted average
            if total_weight > 0:
                weighted_sent = group["weighted_sentiment"].sum() / total_weight
            else:
                weighted_sent = group["sentiment_score"].mean()
                
            raw_sent_mean = group["sentiment_score"].mean()
            sent_std = group["sentiment_score"].std(ddof=1) if total_tweets > 1 else 0.0
            if np.isnan(sent_std):
                sent_std = 0.0

            # Bullish / Bearish counts (threshold +/- 0.15)
            bullish_count = (group["sentiment_score"] > 0.15).sum()
            bearish_count = (group["sentiment_score"] < -0.15).sum()
            bull_bear_ratio = (bullish_count + 1.0) / (bearish_count + 1.0)

            aggregated_rows.append({
                "date": date_str,
                "raw_sentiment_mean": round(float(raw_sent_mean), 4),
                "weighted_sentiment": round(float(weighted_sent), 4),
                "tweet_volume": int(total_tweets),
                "sentiment_dispersion": round(float(sent_std), 4),
                "bull_bear_ratio": round(float(bull_bear_ratio), 4),
                "bullish_count": int(bullish_count),
                "bearish_count": int(bearish_count),
                "total_engagement": int(group["likes"].sum() + group["reposts"].sum())
            })

        daily_df = pd.DataFrame(aggregated_rows).sort_values("date").reset_index(drop=True)
        return daily_df
