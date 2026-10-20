"""
Configuration & Settings Module for CrowdWisdomTrading Sentiment Price Prediction Agent.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class PipelineConfig:
    """Quantitative configuration parameters."""
    # Target Asset
    ticker: str = "NVDA"
    company_name: str = "NVIDIA Corporation"
    cashtag: str = "$NVDA"
    search_keywords: List[str] = field(default_factory=lambda: [
        "$NVDA", "NVDA", "NVIDIA", "Blackwell", "Jensen Huang", "GPU demand"
    ])
    
    # Time Horizon
    lookback_days: int = 180
    prediction_horizons: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    # Database
    db_path: Path = DATA_DIR / "crowdwisdom.db"
    
    # Apify & Tavily Credentials (from environment or defaults)
    apify_token: str = field(default_factory=lambda: os.getenv("APIFY_API_TOKEN", ""))
    apify_actor_id: str = field(default_factory=lambda: os.getenv("APIFY_TWEET_ACTOR_ID", "61RPP7dywgiy0JPD0"))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    
    # Model & Backtest Parameters
    train_window_days: int = 90  # Initial expanding window training size
    step_size_days: int = 5      # Walk-forward rolling step size
    transaction_cost_bps: float = 10.0  # 10 bps slippage & fees per trade
    risk_free_rate: float = 0.045       # 4.5% annual risk-free rate for Sharpe/Sortino
    
    # Feature Weighting
    engagement_weights: dict = field(default_factory=lambda: {
        "likes": 1.0,
        "reposts": 2.0,
        "views": 0.05,
        "replies": 1.5,
        "follower_scale": 10.0
    })

DEFAULT_CONFIG = PipelineConfig()
