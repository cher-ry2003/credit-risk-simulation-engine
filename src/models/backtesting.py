"""
Rolling-window cross-validation and MAPE validation gate.

Design:
  - Expanding-window splits: training window grows by one fold each iteration,
    test window is always `horizon` days.
  - MAPE degradation trend computed via numpy polyfit slope over fold MAPEs.
  - Hard gate: raises RuntimeError if ANY fold MAPE exceeds `threshold` (default 8%).

Interface:
  RollingWindowBacktester(model, n_splits=5, horizon=30)
  .run(prophet_df, arima_series)    — executes all CV folds
  .validate_mape_gate(threshold)    — raises if any fold MAPE > threshold
  .summary_report()                 — prints fold-by-fold table
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAPE_GATE_DEFAULT = 0.08


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    mape: float
    prophet_mape: float
    arima_mape: float


@dataclass
class RollingWindowBacktester:
    """
    Rolling-window cross-validator for the EnsembleForecast model.

    Parameters
    ----------
    n_splits  : number of expanding-window CV folds
    horizon   : forecast horizon in periods (days) per fold
    """
    n_splits: int = 5
    horizon: int = 30
    mape_history: List[float] = field(default_factory=list, init=False)
    fold_results: List[FoldResult] = field(default_factory=list, init=False)

    def run(self, prophet_df: pd.DataFrame, arima_series: pd.Series) -> "RollingWindowBacktester":
        """
        Execute all CV folds on the provided time series data.

        Parameters
        ----------
        prophet_df    : Full ds/y DataFrame (used to derive folds for both models)
        arima_series  : DatetimeIndex-indexed Series (same time range as prophet_df)
        """
        from src.models.ensemble import EnsembleForecast

        prophet_df = prophet_df.copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
        arima_series = arima_series.sort_index()

        n = len(prophet_df)
        min_train = max(2 * self.horizon, n // (self.n_splits + 1))

        if n < min_train + self.horizon:
            raise ValueError(
                f"Insufficient data: need at least {min_train + self.horizon} rows, got {n}."
            )

        fold_step = max(1, (n - min_train - self.horizon) // max(1, self.n_splits - 1))
        self.mape_history = []
        self.fold_results = []

        for fold in range(self.n_splits):
            train_end_idx = min_train + fold * fold_step
            test_end_idx = train_end_idx + self.horizon

            if test_end_idx > n:
                logger.warning("Fold %d exceeds data length; stopping early.", fold + 1)
                break

            train_df = prophet_df.iloc[:train_end_idx]
            test_df = prophet_df.iloc[train_end_idx:test_end_idx]

            train_series = arima_series.iloc[:train_end_idx]

            model = EnsembleForecast()
            model.fit(train_df, train_series)
            forecast = model.predict(self.horizon)

            actual = test_df["y"].values
            ensemble_pred = forecast["ensemble_yhat"].values
            prophet_pred = forecast["prophet_yhat"].values
            arima_pred = forecast["arima_yhat"].values

            fold_mape = _mape(actual, ensemble_pred)
            self.mape_history.append(fold_mape)

            result = FoldResult(
                fold=fold + 1,
                train_size=train_end_idx,
                test_size=self.horizon,
                mape=fold_mape,
                prophet_mape=_mape(actual, prophet_pred),
                arima_mape=_mape(actual, arima_pred),
            )
            self.fold_results.append(result)
            logger.info(
                "Fold %d/%d — train=%d  MAPE=%.4f (Prophet=%.4f, ARIMA=%.4f)",
                fold + 1, self.n_splits, train_end_idx,
                fold_mape, result.prophet_mape, result.arima_mape,
            )

        return self

    def validate_mape_gate(self, threshold: float = MAPE_GATE_DEFAULT) -> None:
        """
        Raises RuntimeError if any fold MAPE exceeds `threshold`.
        Also warns if the MAPE trend slope is positive (degrading over time).
        """
        if not self.mape_history:
            raise RuntimeError("No backtest results found. Call run() first.")

        failing_folds = [
            (i + 1, m) for i, m in enumerate(self.mape_history) if m > threshold
        ]

        # Degradation trend (positive slope = worsening)
        if len(self.mape_history) > 1:
            x = np.arange(len(self.mape_history), dtype=float)
            slope = float(np.polyfit(x, self.mape_history, 1)[0])
            if slope > 0:
                logger.warning(
                    "MAPE DEGRADATION DETECTED: trend slope = +%.5f over %d folds. "
                    "Model accuracy is worsening across CV folds.",
                    slope, len(self.mape_history),
                )
        else:
            slope = 0.0

        if failing_folds:
            fold_detail = "  ".join(
                f"Fold {f}: MAPE={m:.4f} ({m*100:.2f}%)" for f, m in failing_folds
            )
            raise RuntimeError(
                f"\n{'!' * 60}\n"
                f"MAPE GATE FAILED — threshold = {threshold*100:.1f}%\n"
                f"Failing folds:\n  {fold_detail}\n"
                f"All MAPEs: {[round(m, 4) for m in self.mape_history]}\n"
                f"Trend slope: {slope:+.5f}\n"
                f"{'!' * 60}"
            )

        logger.info(
            "MAPE gate PASSED — all %d folds under %.1f%% threshold. "
            "Max MAPE = %.4f  Trend slope = %+.5f",
            len(self.mape_history), threshold * 100,
            max(self.mape_history), slope,
        )

    def summary_report(self) -> None:
        if not self.fold_results:
            print("No results — run() has not been called.")
            return

        header = f"{'Fold':>5}  {'Train rows':>11}  {'Test rows':>10}  {'MAPE':>8}  {'Prophet':>8}  {'ARIMA':>8}  Status"
        print(f"\n{'=' * 70}")
        print("Rolling-Window Cross-Validation — MAPE Report")
        print(f"{'=' * 70}")
        print(header)
        print("-" * 70)

        for r in self.fold_results:
            status = "PASS" if r.mape <= MAPE_GATE_DEFAULT else "FAIL ✗"
            print(
                f"{r.fold:>5}  {r.train_size:>11,}  {r.test_size:>10,}  "
                f"{r.mape:>8.4f}  {r.prophet_mape:>8.4f}  {r.arima_mape:>8.4f}  {status}"
            )

        all_mapes = [r.mape for r in self.fold_results]
        print("-" * 70)
        print(
            f"{'AVG':>5}  {'':>11}  {'':>10}  "
            f"{np.mean(all_mapes):>8.4f}  {'':>8}  {'':>8}"
        )
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    # Smooth trend + seasonality + small noise → should pass MAPE gate
    y = (
        100
        + np.arange(365) * 0.3
        + 10 * np.sin(2 * np.pi * np.arange(365) / 7)
        + rng.normal(0, 2, 365)
    )
    prophet_df = pd.DataFrame({"ds": dates, "y": y})
    arima_series = pd.Series(y, index=dates)

    backtester = RollingWindowBacktester(n_splits=5, horizon=14)
    backtester.run(prophet_df, arima_series)
    backtester.summary_report()
    backtester.validate_mape_gate(threshold=MAPE_GATE_DEFAULT)
    print("All folds passed the MAPE gate.")
