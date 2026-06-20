"""
Weighted ensemble combiner: Prophet (60%) + ARIMA (40%) by default.

Interface:
  EnsembleForecast.fit(df, series)   — fits both sub-models
  EnsembleForecast.predict(periods)  — returns weighted-average forecast DataFrame
"""
from __future__ import annotations

import logging

import pandas as pd

from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster

logger = logging.getLogger(__name__)


class EnsembleForecast:
    """
    Weighted average ensemble of Prophet and ARIMA.

    Parameters
    ----------
    prophet_weight : weight assigned to Prophet predictions (default 0.6)
    arima_weight   : weight assigned to ARIMA predictions  (default 0.4)
    """

    def __init__(
        self,
        prophet_weight: float = 0.6,
        arima_weight: float = 0.4,
        prophet_kwargs: dict | None = None,
        arima_order: tuple | None = None,
    ) -> None:
        if abs(prophet_weight + arima_weight - 1.0) > 1e-9:
            raise ValueError(
                f"Weights must sum to 1.0; got prophet={prophet_weight}, arima={arima_weight}"
            )
        self.prophet_weight = prophet_weight
        self.arima_weight = arima_weight

        self._prophet = ProphetForecaster(**(prophet_kwargs or {}))
        self._arima = ARIMAForecaster(order=arima_order)
        self._fitted = False

    def fit(self, prophet_df: pd.DataFrame, arima_series: pd.Series) -> "EnsembleForecast":
        """
        Fit both sub-models.

        Parameters
        ----------
        prophet_df    : DataFrame with 'ds' (datetime) and 'y' (float)
        arima_series  : DatetimeIndex-indexed pd.Series
        """
        logger.info("Fitting Prophet sub-model...")
        self._prophet.fit(prophet_df)

        logger.info("Fitting ARIMA sub-model...")
        self._arima.fit(arima_series)

        self._fitted = True
        logger.info(
            "Ensemble fitted (Prophet w=%.2f, ARIMA w=%.2f).",
            self.prophet_weight, self.arima_weight,
        )
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        """
        Produce ensemble forecast for `periods` steps ahead.

        Returns
        -------
        DataFrame with columns: ds, prophet_yhat, arima_yhat, ensemble_yhat
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        prophet_fc = self._prophet.predict(periods, freq=freq)
        arima_fc = self._arima.predict(periods)

        # Align on date index
        result = prophet_fc[["ds", "yhat"]].copy()
        result = result.rename(columns={"yhat": "prophet_yhat"})
        result["arima_yhat"] = arima_fc.values

        result["ensemble_yhat"] = (
            self.prophet_weight * result["prophet_yhat"]
            + self.arima_weight * result["arima_yhat"]
        )

        logger.info("Ensemble forecast generated: %d steps", periods)
        return result[["ds", "prophet_yhat", "arima_yhat", "ensemble_yhat"]].reset_index(drop=True)
