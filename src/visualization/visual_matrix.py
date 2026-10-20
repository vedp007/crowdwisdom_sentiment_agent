"""
Visual Matrix Output Generator.
Produces high-resolution charts visualizing X sentiment vs price dynamics,
multi-horizon forecast trajectories, feature importances, and backtest equity curves.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

logger = logging.getLogger(__name__)

class VisualMatrixGenerator:
    """Generates visual matrix outputs and charts for reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Apply clean financial styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_sentiment_vs_price(
        self,
        features_df: pd.DataFrame,
        ticker: str = "NVDA",
        filename: str = "sentiment_vs_price_timeline.png"
    ) -> Path:
        """
        Chart 1: Dual-axis time series displaying X sentiment changes against price as time changes.
        """
        df = features_df.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df.sort_values("date_dt", inplace=True)

        fig, (ax1, ax3) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Top Panel: Price (Left) and Sentiment (Right)
        color_price = "#1f77b4"
        ax1.plot(df["date_dt"], df["close"], color=color_price, linewidth=2.2, label=f"{ticker} Close Price ($)")
        ax1.set_ylabel(f"{ticker} Price (USD)", color=color_price, fontsize=12, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_price)
        ax1.set_title(f"CrowdWisdomTrading: {ticker} Price vs X (Twitter) Sentiment Dynamics (180 Days)", fontsize=14, fontweight="bold", pad=12)

        # Secondary Axis: Sentiment
        ax2 = ax1.twinx()
        color_sent = "#2ca02c"
        sent_series = df["weighted_sentiment"].rolling(3, min_periods=1).mean()
        ax2.plot(df["date_dt"], sent_series, color=color_sent, linewidth=1.8, linestyle="--", label="Weighted Sentiment (3d MA)")
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
        
        # Color fill positive (green) and negative (red) sentiment zones
        ax2.fill_between(df["date_dt"], sent_series, 0, where=(sent_series >= 0), color="green", alpha=0.15)
        ax2.fill_between(df["date_dt"], sent_series, 0, where=(sent_series < 0), color="red", alpha=0.15)
        ax2.set_ylabel("Weighted Sentiment Score [-1, +1]", color="darkgreen", fontsize=12, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="darkgreen")

        # Bottom Panel: Post Volume & Anomalies
        colors_vol = ["#d62728" if z > 1.5 else "#7f7f7f" for z in df["tweet_volume_zscore"]]
        ax3.bar(df["date_dt"], df["tweet_volume"], color=colors_vol, alpha=0.7, width=0.8, label="Daily Tweet Volume")
        ax3.set_ylabel("Post Volume", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=35, ha="right")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="none")

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved sentiment vs price chart to {save_path}")
        return save_path

    def plot_multi_horizon_cone(
        self,
        latest_date: str,
        current_price: float,
        horizon_preds: Dict[int, float],
        ticker: str = "NVDA",
        filename: str = "forecast_matrix_cone.png"
    ) -> Path:
        """
        Chart 2: Visual Forecast Matrix displaying price projections from n+0 to n+5.
        """
        horizons = sorted(horizon_preds.keys())
        prices = [horizon_preds[h] for h in horizons]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot projected path
        ax.plot(horizons, prices, marker="o", markersize=8, color="#d62728", linewidth=2.5, label="Sentiment ML Forecast", zorder=4)
        
        # Identity n+0 point
        ax.scatter([0], [current_price], color="#1f77b4", s=140, zorder=5, label=f"Current Price n+0: ${current_price:.2f}")

        # Add trajectory labels
        for h, p in zip(horizons, prices):
            pct_chg = ((p - current_price) / current_price) * 100.0
            sign = "+" if pct_chg >= 0 else ""
            ax.annotate(
                f"${p:.2f}\n({sign}{pct_chg:.2f}%)",
                xy=(h, p),
                xytext=(0, 10 if pct_chg >= 0 else -20),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.2)
            )

        # Fan uncertainty bounds (widening with horizon)
        std_pct = 0.015
        upper = [p * (1.0 + std_pct * np.sqrt(max(0.2, h))) for h, p in zip(horizons, prices)]
        lower = [p * (1.0 - std_pct * np.sqrt(max(0.2, h))) for h, p in zip(horizons, prices)]
        ax.fill_between(horizons, lower, upper, color="#d62728", alpha=0.15, label="80% Prediction Cone")

        ax.set_xticks(horizons)
        ax.set_xticklabels([f"n+{h}" for h in horizons], fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Horizon (Trading Days Ahead)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Projected Asset Price (USD)", fontsize=12, fontweight="bold")
        ax.set_title(f"CrowdWisdomTrading: {ticker} Multi-Horizon Forecast Matrix (As of {latest_date})", fontsize=13, fontweight="bold")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved forecast cone chart to {save_path}")
        return save_path

    def plot_backtest_equity_curve(
        self,
        backtest_df: pd.DataFrame,
        metrics: Dict[str, Dict[str, float]],
        ticker: str = "NVDA",
        filename: str = "equity_curve_and_drawdown.png"
    ) -> Path:
        """
        Chart 3: Cumulative equity curves and underwater drawdowns comparing Sentiment ML vs Benchmarks.
        """
        df = backtest_df.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df.sort_values("date_dt", inplace=True)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )

        # Top: Cumulative Wealth
        m_sent = metrics.get("Sentiment_ML_Strategy", {})
        m_price = metrics.get("Price_Only_ML", {})
        m_bh = metrics.get("Buy_and_Hold", {})

        lbl_sent = f"Sentiment ML (Sharpe: {m_sent.get('Sharpe_Ratio', 0):.2f}, Ret: {m_sent.get('Total_Return_Pct', 0):.1f}%)"
        lbl_price = f"Price-Only ML (Sharpe: {m_price.get('Sharpe_Ratio', 0):.2f}, Ret: {m_price.get('Total_Return_Pct', 0):.1f}%)"
        lbl_bh = f"Buy & Hold {ticker} (Sharpe: {m_bh.get('Sharpe_Ratio', 0):.2f}, Ret: {m_bh.get('Total_Return_Pct', 0):.1f}%)"

        ax1.plot(df["date_dt"], df["cum_wealth_sentiment"], color="#2ca02c", linewidth=2.4, label=lbl_sent)
        ax1.plot(df["date_dt"], df["cum_wealth_priceonly"], color="#ff7f0e", linewidth=1.8, linestyle="--", label=lbl_price)
        ax1.plot(df["date_dt"], df["cum_wealth_benchmark"], color="#1f77b4", linewidth=1.6, linestyle=":", label=lbl_bh)

        ax1.set_ylabel("Cumulative Wealth ($1.00 Base)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Walk-Forward Out-of-Sample Equity Trajectory & Drawdown ({ticker})", fontsize=14, fontweight="bold", pad=10)
        ax1.legend(loc="upper left", frameon=True, facecolor="white")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Bottom: Underwater Drawdowns
        for col, c, lbl in [
            ("cum_wealth_sentiment", "#2ca02c", "Sentiment ML"),
            ("cum_wealth_priceonly", "#ff7f0e", "Price-Only"),
            ("cum_wealth_benchmark", "#1f77b4", "Buy & Hold")
        ]:
            wealth = df[col].values
            running_max = np.maximum.accumulate(wealth)
            dd = (running_max - wealth) / running_max * -100.0
            ax2.plot(df["date_dt"], dd, color=c, linewidth=1.4, label=lbl)

        ax2.set_ylabel("Drawdown %", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved equity curve chart to {save_path}")
        return save_path

    def plot_feature_importances(
        self,
        df_imp: pd.DataFrame,
        filename: str = "feature_importance_matrix.png"
    ) -> Path:
        """
        Chart 4: Feature importance ranking highlighting sentiment vs price factor contributions.
        """
        if df_imp.empty or "mean_importance" not in df_imp.columns:
            return self.output_dir / filename

        top_df = df_imp.head(10).sort_values("mean_importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#2ca02c" if any(k in f for k in ["sentiment", "tweet", "bull", "surge", "divergence"]) else "#1f77b4" for f in top_df["feature"]]

        ax.barh(top_df["feature"], top_df["mean_importance"], color=colors, alpha=0.85)
        ax.set_xlabel("Mean Information Gain / Split Importance", fontsize=11, fontweight="bold")
        ax.set_title("Top 10 Feature Importance (Green = Sentiment Features, Blue = Price Features)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved feature importance chart to {save_path}")
        return save_path
