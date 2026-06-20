"""
Statistical evaluation module for consumer risk segment analysis.

Tests implemented:
  - Chi-Square test of independence (categorical segment splits)
    Effect size: Cramér's V
  - Mann-Whitney U test (non-parametric continuous distributions)
    Effect size: rank-biserial correlation r

Both functions return structured result dicts and print formatted reports.
`run_full_analysis(df)` orchestrates a predefined battery of tests.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA = 0.05


# ── Chi-Square ────────────────────────────────────────────────────────────────

def run_chi_square(
    df: pd.DataFrame,
    col_categorical: str,
    col_binary: str,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """
    Chi-Square test of independence between a categorical column and a binary outcome.

    Parameters
    ----------
    df              : DataFrame containing both columns
    col_categorical : Nominal feature column (e.g. 'merchant_category')
    col_binary      : Binary outcome column (e.g. 'default_flag')
    alpha           : Significance level (default 0.05)

    Returns
    -------
    dict with keys: chi2_stat, p_value, dof, expected_freq, cramers_v, is_significant
    """
    contingency_table = pd.crosstab(df[col_categorical], df[col_binary])

    chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)

    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    cramers_v = float(np.sqrt(chi2_stat / (n * (min(r, k) - 1))))

    is_significant = p_value < alpha

    result = {
        "test": "chi_square",
        "col_categorical": col_categorical,
        "col_binary": col_binary,
        "chi2_stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 6),
        "dof": int(dof),
        "expected_freq_shape": list(expected_freq.shape),
        "cramers_v": round(cramers_v, 4),
        "is_significant": is_significant,
        "alpha": alpha,
    }

    _print_chi_square_report(result, contingency_table)
    return result


def _print_chi_square_report(result: dict, table: pd.DataFrame) -> None:
    sig_label = "SIGNIFICANT" if result["is_significant"] else "NOT significant"
    print(
        f"\n{'=' * 60}\n"
        f"Chi-Square Test: {result['col_categorical']} × {result['col_binary']}\n"
        f"{'=' * 60}\n"
        f"  χ²  = {result['chi2_stat']:.4f}   (df = {result['dof']})\n"
        f"  p   = {result['p_value']:.6f}   (α = {result['alpha']})\n"
        f"  V   = {result['cramers_v']:.4f}  (Cramér's V effect size)\n"
        f"  → {sig_label} at α = {result['alpha']}\n"
        f"\nContingency table (top 5 rows):\n{table.head()}\n"
    )


# ── Mann-Whitney U ────────────────────────────────────────────────────────────

def run_mann_whitney(
    df: pd.DataFrame,
    col_continuous: str,
    col_group: str,
    group_values: tuple = (0, 1),
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """
    Mann-Whitney U test comparing distributions of a continuous variable
    across two groups defined by a binary column.

    Parameters
    ----------
    df              : DataFrame containing both columns
    col_continuous  : Numeric column to compare (e.g. 'amount')
    col_group       : Binary group column (e.g. 'churn_flag')
    group_values    : Tuple of the two group labels (default (0, 1))
    alpha           : Significance level (default 0.05)

    Returns
    -------
    dict with keys: U_statistic, p_value, rank_biserial_r, is_significant, group_stats
    """
    g0 = df.loc[df[col_group] == group_values[0], col_continuous].dropna().values
    g1 = df.loc[df[col_group] == group_values[1], col_continuous].dropna().values

    if len(g0) == 0 or len(g1) == 0:
        raise ValueError(
            f"One or both groups are empty for {col_group} in {group_values}. "
            "Check that the column contains the expected values."
        )

    u_stat, p_value = stats.mannwhitneyu(g0, g1, alternative="two-sided")

    n0, n1 = len(g0), len(g1)
    rank_biserial_r = float(1 - (2 * u_stat) / (n0 * n1))

    def _group_stats(arr: np.ndarray, label) -> dict:
        q25, q75 = np.percentile(arr, [25, 75])
        return {
            "group": label,
            "n": len(arr),
            "median": round(float(np.median(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "iqr": round(float(q75 - q25), 4),
            "std": round(float(np.std(arr)), 4),
        }

    result = {
        "test": "mann_whitney_u",
        "col_continuous": col_continuous,
        "col_group": col_group,
        "group_values": list(group_values),
        "U_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_value), 6),
        "rank_biserial_r": round(rank_biserial_r, 4),
        "is_significant": p_value < alpha,
        "alpha": alpha,
        "group_stats": [
            _group_stats(g0, group_values[0]),
            _group_stats(g1, group_values[1]),
        ],
    }

    _print_mann_whitney_report(result)
    return result


def _print_mann_whitney_report(result: dict) -> None:
    sig_label = "SIGNIFICANT" if result["is_significant"] else "NOT significant"
    gs = result["group_stats"]
    print(
        f"\n{'=' * 60}\n"
        f"Mann-Whitney U Test: {result['col_continuous']} by {result['col_group']}\n"
        f"{'=' * 60}\n"
        f"  U   = {result['U_statistic']:.4f}\n"
        f"  p   = {result['p_value']:.6f}   (α = {result['alpha']})\n"
        f"  r   = {result['rank_biserial_r']:.4f}  (rank-biserial correlation)\n"
        f"  → {sig_label} at α = {result['alpha']}\n"
        f"\nGroup descriptives:\n"
        f"  [{result['col_group']}={gs[0]['group']}]  "
        f"n={gs[0]['n']:,}  median={gs[0]['median']:.2f}  IQR={gs[0]['iqr']:.2f}\n"
        f"  [{result['col_group']}={gs[1]['group']}]  "
        f"n={gs[1]['n']:,}  median={gs[1]['median']:.2f}  IQR={gs[1]['iqr']:.2f}\n"
    )


# ── Full analysis orchestrator ────────────────────────────────────────────────

ANALYSIS_PLAN = [
    # (test_type, positional_args)
    ("chi_square",    ("merchant_category", "default_flag")),
    ("chi_square",    ("channel",           "churn_flag")),
    ("chi_square",    ("ltv_segment",       "default_flag")),
    ("mann_whitney",  ("amount",            "default_flag")),
    ("mann_whitney",  ("amount",            "churn_flag")),
]


def run_full_analysis(df: pd.DataFrame, alpha: float = ALPHA) -> dict[str, list]:
    """
    Run the full statistical test battery against df.

    Returns a report dict with 'chi_square_results' and 'mann_whitney_results'
    suitable for JSON serialization.
    """
    chi_square_results = []
    mann_whitney_results = []

    for test_type, cols in ANALYSIS_PLAN:
        try:
            if test_type == "chi_square":
                result = run_chi_square(df, cols[0], cols[1], alpha=alpha)
                chi_square_results.append(result)
            else:
                result = run_mann_whitney(df, cols[0], cols[1], alpha=alpha)
                mann_whitney_results.append(result)
        except Exception as exc:
            logger.warning("Skipping %s(%s): %s", test_type, cols, exc)

    report = {
        "chi_square_results": chi_square_results,
        "mann_whitney_results": mann_whitney_results,
    }
    return report


if __name__ == "__main__":
    from src.ingestion.generator import generate_transactions

    print("Generating sample data for statistical analysis...")
    df = generate_transactions(n_records=50_000, seed=42)
    report = run_full_analysis(df)
    print("\n\nFull analysis complete. JSON report:\n")
    print(json.dumps({k: len(v) for k, v in report.items()}, indent=2))
