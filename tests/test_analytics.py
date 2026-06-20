"""Tests for the statistical analysis module."""
import numpy as np
import pandas as pd
import pytest

from src.analytics.statistical_tests import run_chi_square, run_full_analysis, run_mann_whitney


@pytest.fixture(scope="module")
def sample_df():
    rng = np.random.default_rng(42)
    n = 2_000
    return pd.DataFrame(
        {
            "merchant_category": rng.choice(
                ["grocery", "fuel", "dining", "retail", "travel"], size=n
            ),
            "channel": rng.choice(["mobile", "web", "in-store"], size=n),
            "ltv_segment": rng.choice(
                ["STARTER", "GROWTH", "LOYAL", "CHAMPION", "CHURNED"], size=n
            ),
            "default_flag": rng.choice([0, 1], size=n, p=[0.94, 0.06]),
            "churn_flag": rng.choice([0, 1], size=n, p=[0.82, 0.18]),
            "amount": rng.lognormal(mean=3.8, sigma=1.2, size=n),
        }
    )


class TestChiSquare:
    def test_returns_required_keys(self, sample_df):
        result = run_chi_square(sample_df, "merchant_category", "default_flag")
        required_keys = {
            "test", "chi2_stat", "p_value", "dof",
            "cramers_v", "is_significant", "alpha",
        }
        assert required_keys.issubset(result.keys())

    def test_p_value_in_range(self, sample_df):
        result = run_chi_square(sample_df, "merchant_category", "default_flag")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_cramers_v_in_range(self, sample_df):
        result = run_chi_square(sample_df, "channel", "churn_flag")
        assert 0.0 <= result["cramers_v"] <= 1.0

    def test_chi2_non_negative(self, sample_df):
        result = run_chi_square(sample_df, "ltv_segment", "default_flag")
        assert result["chi2_stat"] >= 0.0

    def test_is_significant_type(self, sample_df):
        result = run_chi_square(sample_df, "merchant_category", "churn_flag")
        assert isinstance(result["is_significant"], (bool, np.bool_))

    def test_dof_positive(self, sample_df):
        result = run_chi_square(sample_df, "merchant_category", "default_flag")
        assert result["dof"] > 0

    def test_test_field_correct(self, sample_df):
        result = run_chi_square(sample_df, "channel", "default_flag")
        assert result["test"] == "chi_square"


class TestMannWhitneyU:
    def test_returns_required_keys(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "default_flag")
        required_keys = {
            "test", "U_statistic", "p_value", "rank_biserial_r",
            "is_significant", "alpha", "group_stats",
        }
        assert required_keys.issubset(result.keys())

    def test_p_value_in_range(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "churn_flag")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_rank_biserial_in_range(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "default_flag")
        assert -1.0 <= result["rank_biserial_r"] <= 1.0

    def test_u_statistic_non_negative(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "churn_flag")
        assert result["U_statistic"] >= 0.0

    def test_group_stats_length(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "default_flag")
        assert len(result["group_stats"]) == 2

    def test_group_stats_n_sums_to_total(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "churn_flag")
        gs = result["group_stats"]
        total_in_groups = gs[0]["n"] + gs[1]["n"]
        assert total_in_groups == len(sample_df)

    def test_empty_group_raises(self, sample_df):
        df_one_group = sample_df[sample_df["default_flag"] == 0].copy()
        with pytest.raises(ValueError, match="empty"):
            run_mann_whitney(df_one_group, "amount", "default_flag")

    def test_test_field_correct(self, sample_df):
        result = run_mann_whitney(sample_df, "amount", "default_flag")
        assert result["test"] == "mann_whitney_u"


class TestFullAnalysis:
    def test_returns_both_result_lists(self, sample_df):
        report = run_full_analysis(sample_df)
        assert "chi_square_results" in report
        assert "mann_whitney_results" in report

    def test_chi_square_count(self, sample_df):
        report = run_full_analysis(sample_df)
        # 3 chi-square tests defined in ANALYSIS_PLAN
        assert len(report["chi_square_results"]) == 3

    def test_mann_whitney_count(self, sample_df):
        report = run_full_analysis(sample_df)
        # 2 Mann-Whitney tests defined in ANALYSIS_PLAN
        assert len(report["mann_whitney_results"]) == 2
