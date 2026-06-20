"""
Facebook Prophet wrapper for demand forecasting.

Interface:
  ProphetForecaster.fit(df)       — df with 'ds' (datetime) and 'y' (float) columns
  ProphetForecaster.predict(n)    — returns DataFrame with ds, yhat, yhat_lower, yhat_upper
  ProphetForecaster.from_series() — adapter to build the required ds/y DataFrame from a Series
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


class ProphetForecaster:
    def __init__(
        self,
        yearly_seasonality: bool | str = "auto",
        weekly_seasonality: bool | str = "auto",
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.95,
    ) -> None:
        self._model: Optional[Prophet] = None
        self._config = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "interval_width": interval_width,
        }
        self._last_ds: Optional[pd.Timestamp] = None

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """
        Fit Prophet on a DataFrame with columns 'ds' and 'y'.
        """
        required = {"ds", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")

        df = df[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)

        self._model = Prophet(**self._config)
        self._model.fit(df)
        self._last_ds = df["ds"].max()
        logger.info("Prophet fitted on %d observations (last: %s)", len(df), self._last_ds)
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        future = self._model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)

    @staticmethod
    def from_series(series: pd.Series, date_col_name: str = "ds") -> pd.DataFrame:
        """
        Convert a time-indexed Series or a Series with a DatetimeIndex to
        the ds/y DataFrame Prophet expects.
        """
        if isinstance(series.index, pd.DatetimeIndex):
            return pd.DataFrame({"ds": series.index, "y": series.values})
        raise ValueError(
            "series must have a DatetimeIndex. "
            "Set the date column as the index before calling from_series()."
        )
