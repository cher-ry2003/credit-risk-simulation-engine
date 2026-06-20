"""
ARIMA forecasting wrapper using statsmodels SARIMAX.

Interface:
  ARIMAForecaster.auto_order(series)  — AIC-based (p,d,q) grid search
  ARIMAForecaster.fit(series)         — fits model; calls auto_order if order not set
  ARIMAForecaster.predict(steps)      — returns pd.Series of point forecasts

Grid search space: p,q ∈ {0,1,2}, d ∈ {0,1}  (lightweight; avoids pmdarima dependency)
"""
from __future__ import annotations

import itertools
import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

_P_VALUES = (0, 1, 2)
_D_VALUES = (0, 1)
_Q_VALUES = (0, 1, 2)


class ARIMAForecaster:
    def __init__(self, order: Optional[Tuple[int, int, int]] = None) -> None:
        self.order = order
        self._model_fit = None
        self._last_index: Optional[pd.DatetimeIndex] = None

    def auto_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Grid search over (p,d,q) parameter space; selects order minimising AIC.
        Suppresses convergence warnings during the sweep.
        """
        best_aic = np.inf
        best_order = (1, 1, 1)

        for p, d, q in itertools.product(_P_VALUES, _D_VALUES, _Q_VALUES):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = SARIMAX(
                        series, order=(p, d, q), trend="n", enforce_stationarity=False
                    ).fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except Exception:
                continue

        logger.info("ARIMA auto_order → %s  (AIC=%.2f)", best_order, best_aic)
        return best_order

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Fit the ARIMA model on a time-indexed pd.Series.
        If order is not set, runs auto_order() first.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series must have a DatetimeIndex.")

        series = series.dropna().sort_index()

        if self.order is None:
            self.order = self.auto_order(series)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_fit = SARIMAX(
                series, order=self.order, trend="n", enforce_stationarity=False
            ).fit(disp=False)

        self._last_index = series.index
        logger.info("ARIMA%s fitted on %d observations.", self.order, len(series))
        return self

    def predict(self, steps: int) -> pd.Series:
        if self._model_fit is None:
            raise RuntimeError("Call fit() before predict().")

        forecast = self._model_fit.forecast(steps=steps)
        freq = pd.infer_freq(self._last_index) or "D"
        future_index = pd.date_range(
            start=self._last_index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=steps,
            freq=freq,
        )
        return pd.Series(forecast.values, index=future_index, name="arima_forecast")
