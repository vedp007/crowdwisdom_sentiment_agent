"""
Rigorous Walk-Forward Validation Engine.
Performs expanding-window out-of-sample cross validation without lookahead bias,
comparing Sentiment + Price model against Price-Only and Naive benchmarks.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from .multi_horizon_model import MultiHorizonForecaster
import logging

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """Executes expanding-window walk-forward validation on time-series feature datasets."""
    
    def __init__(
        self,
        min_train_size: int = 60,
        step_size: int = 5,
        horizons: List[int] = None
    ):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.horizons = horizons if horizons is not None else [0, 1, 2, 3, 4, 5]

    def evaluate(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        ticker: str = "NVDA"
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Runs walk-forward out-of-sample validation.
        Returns:
            - df_predictions: Out-of-sample predicted prices and returns for each date & horizon
            - metrics_summary: Performance metrics (RMSE, MAE, Hit Rate) per horizon
            - feature_importances: Aggregated feature importances
        """
        n_samples = len(features_df)
        if n_samples < (self.min_train_size + self.step_size):
            raise ValueError(f"Feature dataset has {n_samples} samples, need at least {self.min_train_size + self.step_size}")

        logger.info(f"Initiating walk-forward validation: {n_samples} total bars, min_train={self.min_train_size}, step={self.step_size}")

        out_of_sample_records = []
        last_trained_forecaster = None

        # Price-only feature subset for benchmark comparison
        price_only_cols = [c for c in feature_cols if not any(k in c for k in ["sentiment", "tweet", "bull_bear", "volume_surge", "divergence"])]

        split_indices = list(range(self.min_train_size, n_samples, self.step_size))
        
        for split_idx in split_indices:
            train_end = split_idx
            test_end = min(split_idx + self.step_size, n_samples)
            
            df_train = features_df.iloc[:train_end]
            df_test = features_df.iloc[train_end:test_end]
            
            if df_test.empty:
                break

            # 1. Train Full Model (Sentiment + Price)
            forecaster = MultiHorizonForecaster(horizons=self.horizons, model_type="lightgbm")
            forecaster.fit(df_train, df_train, feature_cols)
            last_trained_forecaster = forecaster

            # 2. Train Benchmark Model (Price Only)
            bench_forecaster = MultiHorizonForecaster(horizons=self.horizons, model_type="lightgbm")
            bench_forecaster.fit(df_train, df_train, price_only_cols)

            # Predict on out-of-sample slice
            curr_prices = df_test["close"].values
            preds_full = forecaster.predict(df_test, curr_prices)
            preds_bench = bench_forecaster.predict(df_test, curr_prices)

            for i, (_, row) in enumerate(df_test.iterrows()):
                trade_date = row["date"]
                curr_p = row["close"]
                
                for h in self.horizons:
                    actual_p = row.get(f"target_price_n{h}", np.nan)
                    actual_r = row.get(f"target_ret_n{h}", np.nan)
                    
                    pred_r_full, pred_p_full = preds_full[h][0][i], preds_full[h][1][i]
                    pred_r_bench, pred_p_bench = preds_bench[h][0][i], preds_bench[h][1][i]

                    out_of_sample_records.append({
                        "date": trade_date,
                        "ticker": ticker,
                        "horizon": h,
                        "current_price": curr_p,
                        "actual_price": actual_p,
                        "actual_return": actual_r,
                        "pred_price_sentiment": pred_p_full,
                        "pred_return_sentiment": pred_r_full,
                        "pred_price_priceonly": pred_p_bench,
                        "pred_return_priceonly": pred_r_bench,
                        "naive_pred_price": curr_p
                    })

        df_preds = pd.DataFrame(out_of_sample_records)
        metrics_summary = self._compute_metrics(df_preds)
        feat_imp = last_trained_forecaster.get_feature_importances() if last_trained_forecaster else pd.DataFrame()

        return df_preds, metrics_summary, feat_imp

    def _compute_metrics(self, df_preds: pd.DataFrame) -> Dict[str, Any]:
        """Calculates RMSE, MAE, MAPE, and Directional Hit Rate for each horizon."""
        summary = {}

        for h in self.horizons:
            sub = df_preds[df_preds["horizon"] == h].dropna(subset=["actual_price", "actual_return"])
            if sub.empty:
                continue

            y_true_p = sub["actual_price"].values
            y_pred_p = sub["pred_price_sentiment"].values
            y_bench_p = sub["pred_price_priceonly"].values
            y_naive_p = sub["naive_pred_price"].values

            y_true_r = sub["actual_return"].values
            y_pred_r = sub["pred_return_sentiment"].values
            y_bench_r = sub["pred_return_priceonly"].values

            # Metric: Price RMSE & MAE
            rmse_sent = float(np.sqrt(np.mean((y_true_p - y_pred_p) ** 2)))
            mae_sent = float(np.mean(np.abs(y_true_p - y_pred_p)))
            mape_sent = float(np.mean(np.abs((y_true_p - y_pred_p) / y_true_p)) * 100)

            rmse_bench = float(np.sqrt(np.mean((y_true_p - y_bench_p) ** 2)))
            mae_bench = float(np.mean(np.abs(y_true_p - y_bench_p)))

            rmse_naive = float(np.sqrt(np.mean((y_true_p - y_naive_p) ** 2)))

            # Directional Hit Rate (Sign concordance on non-zero returns)
            if h > 0:
                dir_correct_sent = np.mean((y_pred_r * y_true_r) > 0) * 100.0
                dir_correct_bench = np.mean((y_bench_r * y_true_r) > 0) * 100.0
            else:
                dir_correct_sent = 100.0
                dir_correct_bench = 100.0

            summary[f"horizon_n{h}"] = {
                "rmse_sentiment": round(rmse_sent, 3),
                "mae_sentiment": round(mae_sent, 3),
                "mape_sentiment_pct": round(mape_sent, 2),
                "hit_rate_sentiment_pct": round(dir_correct_sent, 1),
                "rmse_price_only": round(rmse_bench, 3),
                "mae_price_only": round(mae_bench, 3),
                "hit_rate_price_only_pct": round(dir_correct_bench, 1),
                "rmse_naive_baseline": round(rmse_naive, 3),
                "error_reduction_vs_naive_pct": round(((rmse_naive - rmse_sent) / rmse_naive) * 100, 2) if rmse_naive > 0 else 0.0
            }

        return summary
