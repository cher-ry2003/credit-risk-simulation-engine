"""
Tests for the forecasting engine:
  - ProphetForecaster
  - ARIMAForecaster
  - EnsembleForecast
  - RollingWindowBacktester (incl. MAPE gate behaviour)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def clean_time_series():
    """Smooth, low-noise series — ensemble should easily pass MAPE gate."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    y = (
        200
        + np.arange(200) * 0.5
        + 15 * np.sin(2 * np.pi * np.arange(200) / 7)
        + rng.normal(0, 1.5, 200)
    )
    prophet_df = pd.DataFrame({"ds": dates, "y": y})
    arima_series = pd.Series(y, index=dates)
    return prophet_df, arima_series


@pytest.fixture(scope="module")
def noisy_time_series():
    """High-noise series — built to exceed MAPE gate threshold."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    # Uniform noise: model cannot track this, MAPE will be very high
    y = rng.uniform(1, 1000, 200)
    prophet_df = pd.DataFrame({"ds": dates, "y": y})
    arima_series = pd.Series(y, index=dates)
    return prophet_df, arima_series


class TestProphetForecaster:
    def test_fit_and_predict_shape(self, clean_time_series):
        from src.models.prophet_model import ProphetForecaster

        prophet_df, _ = clean_time_series
        model = ProphetForecaster()
        model.fit(prophet_df)
        forecast = model.predict(30)

        assert len(forecast) == 30
        assert set(forecast.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}

    def test_predict_before_fit_raises(self):
        from src.models.prophet_model import ProphetForecaster

        with pytest.raises(RuntimeError, match="fit"):
            ProphetForecaster().predict(10)

    def test_missing_columns_raises(self):
        from src.models.prophet_model import ProphetForecaster

        bad_df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10), "value": range(10)})
        with pytest.raises(ValueError, match="missing columns"):
            ProphetForecaster().fit(bad_df)

    def test_from_series_adapter(self, clean_time_series):
        from src.models.prophet_model import ProphetForecaster

        _, arima_series = clean_time_series
        result = ProphetForecaster.from_series(arima_series)
        assert "ds" in result.columns and "y" in result.columns
        assert len(result) == len(arima_series)


class TestARIMAForecaster:
    def test_fit_and_predict_shape(self, clean_time_series):
        from src.models.arima_model import ARIMAForecaster

        _, arima_series = clean_time_series
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(arima_series)
        forecast = model.predict(14)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 14

    def test_predict_before_fit_raises(self):
        from src.models.arima_model import ARIMAForecaster

        with pytest.raises(RuntimeError, match="fit"):
            ARIMAForecaster().predict(5)

    def test_non_datetime_index_raises(self):
        from src.models.arima_model import ARIMAForecaster

        bad_series = pd.Series(range(50))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ARIMAForecaster().fit(bad_series)

    def test_auto_order_selects_valid_tuple(self, clean_time_series):
        from src.models.arima_model import ARIMAForecaster

        _, arima_series = clean_time_series
        model = ARIMAForecaster()
        order = model.auto_order(arima_series.iloc[:60])

        assert isinstance(order, tuple)
        assert len(order) == 3
        assert all(isinstance(v, int) for v in order)


class TestEnsembleForecast:
    def test_invalid_weights_raise(self):
        from src.models.ensemble import EnsembleForecast

        with pytest.raises(ValueError, match="sum to 1.0"):
            EnsembleForecast(prophet_weight=0.5, arima_weight=0.3)

    def test_fit_and_predict_columns(self, clean_time_series):
        from src.models.ensemble import EnsembleForecast

        prophet_df, arima_series = clean_time_series
        ens = EnsembleForecast()
        ens.fit(prophet_df, arima_series)
        result = ens.predict(14)

        assert set(result.columns) == {"ds", "prophet_yhat", "arima_yhat", "ensemble_yhat"}
        assert len(result) == 14

    def test_predict_before_fit_raises(self):
        from src.models.ensemble import EnsembleForecast

        with pytest.raises(RuntimeError, match="fit"):
            EnsembleForecast().predict(7)

    def test_ensemble_yhat_between_sub_models(self, clean_time_series):
        from src.models.ensemble import EnsembleForecast

        prophet_df, arima_series = clean_time_series
        ens = EnsembleForecast(prophet_weight=0.6, arima_weight=0.4)
        ens.fit(prophet_df, arima_series)
        result = ens.predict(14)

        # Weighted average must be a convex combination of the two sub-forecasts
        expected = 0.6 * result["prophet_yhat"] + 0.4 * result["arima_yhat"]
        pd.testing.assert_series_equal(
            result["ensemble_yhat"].round(6), expected.round(6), check_names=False
        )


class TestRollingWindowBacktester:
    def test_mape_history_length(self, clean_time_series):
        from src.models.backtesting import RollingWindowBacktester

        prophet_df, arima_series = clean_time_series
        bt = RollingWindowBacktester(n_splits=3, horizon=14)
        bt.run(prophet_df, arima_series)

        assert len(bt.mape_history) == 3

    def test_fold_results_populated(self, clean_time_series):
        from src.models.backtesting import RollingWindowBacktester

        prophet_df, arima_series = clean_time_series
        bt = RollingWindowBacktester(n_splits=3, horizon=14)
        bt.run(prophet_df, arima_series)

        assert len(bt.fold_results) == 3
        for fold in bt.fold_results:
            assert 0.0 <= fold.mape

    def test_clean_series_passes_mape_gate(self, clean_time_series):
        from src.models.backtesting import RollingWindowBacktester

        prophet_df, arima_series = clean_time_series
        bt = RollingWindowBacktester(n_splits=3, horizon=14)
        bt.run(prophet_df, arima_series)
        # Should not raise
        bt.validate_mape_gate(threshold=0.08)

    def test_noisy_series_fails_mape_gate(self, noisy_time_series):
        from src.models.backtesting import RollingWindowBacktester

        prophet_df, arima_series = noisy_time_series
        bt = RollingWindowBacktester(n_splits=3, horizon=14)
        bt.run(prophet_df, arima_series)

        with pytest.raises(RuntimeError, match="MAPE GATE FAILED"):
            bt.validate_mape_gate(threshold=0.08)

    def test_validate_before_run_raises(self):
        from src.models.backtesting import RollingWindowBacktester

        with pytest.raises(RuntimeError, match="run()"):
            RollingWindowBacktester().validate_mape_gate()

    def test_insufficient_data_raises(self):
        from src.models.backtesting import RollingWindowBacktester

        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        tiny_df = pd.DataFrame({"ds": dates, "y": range(10)})
        tiny_series = pd.Series(range(10), index=dates, dtype=float)

        with pytest.raises(ValueError, match="Insufficient data"):
            RollingWindowBacktester(n_splits=5, horizon=30).run(tiny_df, tiny_series)
