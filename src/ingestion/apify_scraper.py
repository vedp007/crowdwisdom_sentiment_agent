"""
Apify-based X (Twitter) post ingestion module.
Uses Apify Client with Actor ID: 61RPP7dywgiy0JPD0 (apidojo/tweet-scraper) or fallback actors.
"""

import os
from datetime import datetime, timezone
import pytz
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ApifyTweetScraper:
    """
    Interacts with Apify Actor API to extract historical X (Twitter) posts,
    engagement metrics, author follower counts, and UTC/EST aligned timestamps.
    """
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        actor_id: str = "61RPP7dywgiy0JPD0",  # apidojo/tweet-scraper
        ticker: str = "NVDA",
        lookback_days: int = 180
    ):
        self.api_token = api_token or os.getenv("APIFY_API_TOKEN", "")
        self.actor_id = actor_id
        self.ticker = ticker.upper()
        self.lookback_days = lookback_days
        self.client = None
        
        if self.api_token and self.api_token != "your_apify_api_token_here":
            try:
                from apify_client import ApifyClient
                self.client = ApifyClient(self.api_token)
                logger.info(f"Initialized ApifyClient with Actor {self.actor_id}")
            except Exception as e:
                logger.warning(f"Could not initialize ApifyClient: {e}")

    def is_configured(self) -> bool:
        """Checks if a valid Apify API token is configured."""
        return self.client is not None and bool(self.api_token)

    def scrape_posts(self, max_posts: int = 2000) -> List[Dict[str, Any]]:
        """
        Executes Apify Actor run to scrape tweets for ticker/cashtags over the lookback window.
        Extracts engagement (likes, reposts, views, replies) and follower counts.
        """
        if not self.is_configured():
            logger.info("Apify API token not configured or blank. Falling back to offline dataset.")
            return []

        search_query = f"${self.ticker} OR #{self.ticker} OR {self.ticker} lang:en -is:retweet"
        logger.info(f"Triggering Apify scraper (Actor: {self.actor_id}) with query: '{search_query}'")

        run_input = {
            "searchTerms": [f"${self.ticker}", f"#{self.ticker}", self.ticker],
            "maxTweets": max_posts,
            "sort": "Latest",
            "tweetLanguage": "en",
            "proxy": {"useApifyProxy": True}
        }

        try:
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            dataset = self.client.dataset(run["defaultDatasetId"])
            
            raw_items = list(dataset.iterate_items())
            logger.info(f"Retrieved {len(raw_items)} items from Apify dataset.")
            return self._normalize_apify_items(raw_items)
        except Exception as e:
            logger.error(f"Apify execution failed: {e}")
            return []

    def _normalize_apify_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalizes heterogeneous tweet schema into uniform structured records with UTC/EST alignment."""
        records = []
        utc_tz = pytz.UTC
        est_tz = pytz.timezone("America/New_York")

        for item in items:
            try:
                tweet_id = str(item.get("id") or item.get("id_str") or item.get("tweetId") or "")
                text = item.get("text") or item.get("full_text") or item.get("tweetText") or ""
                if not text or not tweet_id:
                    continue

                # Parse timestamp
                raw_time = item.get("createdAt") or item.get("created_at") or item.get("timestamp")
                dt_utc = None
                if raw_time:
                    try:
                        # Handle ISO or standard twitter format
                        if "T" in str(raw_time):
                            dt_utc = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                        else:
                            dt_utc = datetime.strptime(str(raw_time), "%a %b %d %H:%M:%S %z %Y")
                    except Exception:
                        dt_utc = datetime.now(timezone.utc)
                else:
                    dt_utc = datetime.now(timezone.utc)

                if dt_utc.tzinfo is None:
                    dt_utc = utc_tz.localize(dt_utc)
                else:
                    dt_utc = dt_utc.astimezone(utc_tz)

                dt_est = dt_utc.astimezone(est_tz)

                # Author stats
                author = item.get("author", {}) or item.get("user", {})
                followers = author.get("followersCount") or author.get("followers_count") or 100
                is_verified = bool(author.get("isVerified") or author.get("verified"))

                # Engagement metrics
                likes = item.get("likeCount") or item.get("favorite_count") or item.get("likes") or 0
                reposts = item.get("retweetCount") or item.get("retweet_count") or item.get("retweets") or 0
                views = item.get("viewCount") or item.get("views") or (likes * 15)  # Realistic proxy if views missing
                replies = item.get("replyCount") or item.get("reply_count") or item.get("replies") or 0

                records.append({
                    "id": tweet_id,
                    "ticker": self.ticker,
                    "timestamp_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_est": dt_est.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text,
                    "likes": int(likes),
                    "reposts": int(reposts),
                    "views": int(views),
                    "replies": int(replies),
                    "author_followers": int(followers),
                    "author_verified": int(is_verified),
                    "source": "apify"
                })
            except Exception as e:
                logger.debug(f"Skipping malformed tweet record: {e}")
                continue

        logger.info(f"Normalized {len(records)} tweets with UTC/EST alignment.")
        return records
