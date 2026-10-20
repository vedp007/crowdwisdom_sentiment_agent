"""
Multi-Horizon Predictive Model (n+0 to n+5).
Predicts future price returns and reconstructs price trajectories using sentiment and price only.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

class MultiHorizonForecaster:
    """
    Direct multi-horizon forecasting system for horizons n+0, n+1, n+2, n+3, n+4, n+5.
    Employs LightGBM regressors with Huber loss for outlier-robust quantitative return forecasting.
    """
    
    def __init__(
        self,
        horizons: List[int] = None,
        model_type: str = "lightgbm",
        random_state: int = 42
    ):
        self.horizons = horizons if horizons is not None else [0, 1, 2, 3, 4, 5]
        self.model_type = model_type
        self.random_state = random_state
        self.models: Dict[int, Any] = {}
        self.feature_names: List[str] = []

    def _create_base_model(self, horizon: int) -> Any:
        """Creates an appropriately regularized estimator based on horizon."""
        if self.model_type == "lightgbm":
            # Deeper horizons have higher noise; increase regularization (min_child_samples, reg_alpha)
            return LGBMRegressor(
                n_estimators=60,
                learning_rate=0.04,
                max_depth=3,
                num_leaves=7,
                min_child_samples=max(5, 10 - horizon),
                reg_alpha=0.1 * (horizon + 1),
                reg_lambda=0.2 * (horizon + 1),
                objective="regression_l1",  # MAE / L1 loss to prevent extreme outlier disruption
                random_state=self.random_state,
                verbose=-1
            )
        else:
            return Ridge(alpha=1.0 * (horizon + 1), random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y_df: pd.DataFrame, feature_names: List[str]):
        """
        Fits an independent estimator for each horizon h in [1..5].
        Horizon n+0 is identity/same-period baseline.
        """
        self.feature_names = feature_names
        X_train = X[self.feature_names].values

        for h in self.horizons:
            if h == 0:
                continue  # Identity horizon (P_{t+0} = P_t)
                
            target_col = f"target_ret_n{h}"
            if target_col not in y_df.columns:
                raise ValueError(f"Target column {target_col} missing from target dataframe")

            y = y_df[target_col].values
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) < 15:
                logger.warning(f"Insufficient training samples for horizon n+{h}")
                continue

            model = self._create_base_model(h)
            model.fit(X_train[valid_mask], y[valid_mask])
            self.models[h] = model

        logger.info(f"Trained multi-horizon forecaster for horizons: {list(self.models.keys())}")

    def predict(self, X: pd.DataFrame, current_prices: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Generates predictions for each horizon.
        Returns dictionary mapping horizon h -> (predicted_returns, predicted_prices).
        """
        X_test = X[self.feature_names].values
        n_samples = len(X_test)
        predictions = {}

        # Horizon 0: current price, 0% forward return
        predictions[0] = (np.zeros(n_samples), np.array(current_prices, copy=True))

        for h in self.horizons:
            if h == 0:
                continue
            if h in self.models:
                pred_ret = self.models[h].predict(X_test)
                # Cap unrealistic single-week forecast returns to realistic market bounds [-25%, +25%]
                pred_ret = np.clip(pred_ret, -0.25, 0.25)
                pred_price = current_prices * (1.0 + pred_ret)
                predictions[h] = (pred_ret, pred_price)
            else:
                # Fallback to persistence
                predictions[h] = (np.zeros(n_samples), np.array(current_prices, copy=True))

        return predictions

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts averaged feature importances across all horizon models."""
        if self.model_type != "lightgbm" or not self.models:
            return pd.DataFrame()

        importance_data = {"feature": self.feature_names}
        for h, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance_data[f"horizon_n{h}"] = model.feature_importances_

        df_imp = pd.DataFrame(importance_data)
        numeric_cols = [c for c in df_imp.columns if c.startswith("horizon_")]
        df_imp["mean_importance"] = df_imp[numeric_cols].mean(axis=1)
        return df_imp.sort_values("mean_importance", ascending=False).reset_index(drop=True)
