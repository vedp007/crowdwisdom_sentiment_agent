"""
Institutional Quantitative Backtesting Engine.
Calculates annualized Sharpe Ratio, Sortino Ratio, Maximum Drawdown,
Calmar Ratio, Win Rate, and cumulative equity curves with realistic transaction costs.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class QuantitativeBacktester:
    """Executes backtests on out-of-sample predictions and calculates risk-adjusted performance."""
    
    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        annual_risk_free_rate: float = 0.045
    ):
        self.cost_per_trade = transaction_cost_bps / 10000.0
        self.daily_rf = annual_risk_free_rate / 252.0
        self.annual_rf = annual_risk_free_rate

    def run_backtest(
        self,
        df_preds: pd.DataFrame,
        horizon: int = 1,
        signal_threshold: float = 0.0015
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Runs out-of-sample trading simulation comparing:
        1. 'Sentiment_ML_Strategy': Trades on Sentiment+Price ML predictions
        2. 'Price_Only_ML': Trades on Price-Only ML predictions
        3. 'Buy_and_Hold': Benchmark long-only asset holding
        """
        sub = df_preds[df_preds["horizon"] == horizon].sort_values("date").copy()
        sub = sub.dropna(subset=["actual_return", "pred_return_sentiment", "pred_return_priceonly"]).reset_index(drop=True)

        if sub.empty:
            raise ValueError(f"No valid out-of-sample prediction samples for horizon {horizon}")

        # ---------------------------------------------------------
        # 1. Generate Position Signals [1 = Long, 0 = Cash]
        # ---------------------------------------------------------
        pred_sent = sub["pred_return_sentiment"].values
        sig_sent = np.zeros(len(sub))
        sig_sent[pred_sent > signal_threshold] = 1.0

        pred_price = sub["pred_return_priceonly"].values
        sig_price = np.zeros(len(sub))
        sig_price[pred_price > signal_threshold] = 1.0

        # Benchmark (Buy and Hold: 100% long at all times)
        sig_bh = np.ones(len(sub))

        actual_rets = sub["actual_return"].values

        # ---------------------------------------------------------
        # 2. Compute Strategy Returns with Transaction Slippage
        # When in cash (sig == 0), the portfolio earns the risk-free rate
        # ---------------------------------------------------------
        turnover_sent = np.abs(np.diff(np.insert(sig_sent, 0, 0)))
        turnover_price = np.abs(np.diff(np.insert(sig_price, 0, 0)))
        turnover_bh = np.abs(np.diff(np.insert(sig_bh, 0, 0)))

        # Daily net returns
        ret_sent = np.where(sig_sent == 1.0, actual_rets, self.daily_rf) - (turnover_sent * self.cost_per_trade)
        ret_price = np.where(sig_price == 1.0, actual_rets, self.daily_rf) - (turnover_price * self.cost_per_trade)
        ret_bh = actual_rets - (turnover_bh * self.cost_per_trade)

        # Cumulative wealth curves (starting at $1.00)
        sub["strat_sentiment_ret"] = ret_sent
        sub["strat_priceonly_ret"] = ret_price
        sub["benchmark_bh_ret"] = ret_bh

        sub["cum_wealth_sentiment"] = np.cumprod(1.0 + ret_sent)
        sub["cum_wealth_priceonly"] = np.cumprod(1.0 + ret_price)
        sub["cum_wealth_benchmark"] = np.cumprod(1.0 + ret_bh)

        # ---------------------------------------------------------
        # 3. Calculate Risk-Adjusted Metrics
        # ---------------------------------------------------------
        metrics = {
            "Sentiment_ML_Strategy": self._calculate_metrics(ret_sent, sub["cum_wealth_sentiment"].values),
            "Price_Only_ML": self._calculate_metrics(ret_price, sub["cum_wealth_priceonly"].values),
            "Buy_and_Hold": self._calculate_metrics(ret_bh, sub["cum_wealth_benchmark"].values)
        }

        return sub, metrics

    def _calculate_metrics(self, daily_returns: np.ndarray, wealth_curve: np.ndarray) -> Dict[str, float]:
        """Calculates annualized Sharpe, Sortino, Max Drawdown, Calmar, and Win Rate."""
        n_days = len(daily_returns)
        if n_days < 2:
            return {}

        total_return = float(wealth_curve[-1] - 1.0)
        cagr = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
        
        mean_ret = float(np.mean(daily_returns))
        std_ret = float(np.std(daily_returns, ddof=1)) + 1e-9
        annualized_vol = float(std_ret * np.sqrt(252.0))

        # Sharpe Ratio (Annualized)
        excess_mean = mean_ret - self.daily_rf
        sharpe = float((excess_mean / std_ret) * np.sqrt(252.0))

        # Downside Deviation for Sortino (returns below risk-free rate)
        downside_diff = np.minimum(0.0, daily_returns - self.daily_rf)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2))) + 1e-9
        sortino = float((excess_mean / downside_std) * np.sqrt(252.0))

        # Maximum Drawdown (MDD)
        running_max = np.maximum.accumulate(wealth_curve)
        drawdowns = (running_max - wealth_curve) / running_max
        max_drawdown = float(np.max(drawdowns))

        # Calmar Ratio
        calmar = float(cagr / max_drawdown) if max_drawdown > 0 else 0.0

        # Win Rate (% positive return days)
        active_days = daily_returns[daily_returns != 0.0]
        win_rate = float(np.mean(active_days > 0) * 100.0) if len(active_days) > 0 else 0.0

        # Profit Factor
        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum()) + 1e-9
        profit_factor = float(gains / losses)

        return {
            "Total_Return_Pct": round(total_return * 100.0, 2),
            "Annualized_Return_CAGR_Pct": round(cagr * 100.0, 2),
            "Annualized_Volatility_Pct": round(annualized_vol * 100.0, 2),
            "Sharpe_Ratio": round(sharpe, 3),
            "Sortino_Ratio": round(sortino, 3),
            "Max_Drawdown_Pct": round(max_drawdown * 100.0, 2),
            "Calmar_Ratio": round(calmar, 3),
            "Win_Rate_Pct": round(win_rate, 2),
            "Profit_Factor": round(profit_factor, 2),
            "Total_Trading_Days": int(n_days)
        }
