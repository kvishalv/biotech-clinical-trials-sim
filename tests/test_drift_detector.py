"""tests/test_drift_detector.py — Tests for utils/drift_detector.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.drift_detector import DriftDetector, DriftReport, VariableDriftResult


@pytest.fixture
def identical_dfs(patient_df):
    """Two identical DataFrames — should produce no drift."""
    return patient_df.copy(), patient_df.copy()


@pytest.fixture
def shifted_df(patient_df):
    """A DataFrame with age shifted by 20 years — should produce drift."""
    shifted = patient_df.copy()
    shifted["age"] = shifted["age"] + 20.0
    return shifted


class TestKSTest:
    def test_identical_distributions_no_drift(self, patient_df):
        det = DriftDetector(alpha=0.05)
        ref = patient_df["age"].values
        result = det.ks_test(ref, ref.copy(), "age")
        assert not result.drifted

    def test_shifted_distribution_detected(self, patient_df):
        det = DriftDetector(alpha=0.05)
        ref = patient_df["age"].values
        shifted = ref + 20.0
        result = det.ks_test(ref, shifted, "age")
        assert result.drifted
        assert result.p_value < 0.05

    def test_result_has_correct_fields(self, patient_df):
        det = DriftDetector()
        ref = patient_df["age"].values
        result = det.ks_test(ref, ref, "age")
        assert isinstance(result, VariableDriftResult)
        assert result.test == "ks"
        assert result.variable == "age"
        assert 0 <= result.statistic <= 1.0


class TestPSI:
    def test_identical_distributions_low_psi(self, patient_df):
        det = DriftDetector()
        ref = patient_df["age"].values
        result = det.psi(ref, ref.copy(), "age")
        assert result.statistic < 0.1  # negligible PSI

    def test_shifted_distribution_high_psi(self, patient_df):
        det = DriftDetector()
        ref = patient_df["age"].values
        shifted = ref + 20.0
        result = det.psi(ref, shifted, "age")
        assert result.statistic > 0.1  # some PSI detected


class TestChi2Test:
    def test_identical_categoricals_no_drift(self, patient_df):
        det = DriftDetector(alpha=0.05)
        result = det.chi2_test(patient_df["arm"], patient_df["arm"], "arm")
        assert not result.drifted or result.p_value is None  # no drift or N/A

    def test_result_fields(self, patient_df):
        det = DriftDetector()
        result = det.chi2_test(patient_df["sex"], patient_df["sex"], "sex")
        assert result.test == "chi2"
        assert result.variable == "sex"


class TestCheckPatientDrift:
    def test_identical_df_no_drift(self, patient_df):
        det = DriftDetector()
        report = det.check_patient_drift(patient_df, patient_df.copy())
        assert isinstance(report, DriftReport)
        assert report.n_tested > 0
        # Identical data should have at most very few false positives
        assert report.n_drifted <= 1  # allow 1 for floating point edge cases

    def test_shifted_df_detects_age_drift(self, patient_df, shifted_df):
        det = DriftDetector(alpha=0.05)
        report = det.check_patient_drift(patient_df, shifted_df)
        drifted_vars = [r.variable for r in report.results if r.drifted]
        # At least one age-related variable should show drift
        assert any("age" in v for v in drifted_vars)

    def test_report_summary_is_string(self, patient_df):
        det = DriftDetector()
        report = det.check_patient_drift(patient_df, patient_df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "DriftReport" in summary

    def test_report_to_dict(self, patient_df):
        det = DriftDetector()
        report = det.check_patient_drift(patient_df, patient_df)
        d = report.to_dict()
        assert "n_drifted" in d
        assert "n_tested" in d
        assert "results" in d
        assert isinstance(d["results"], list)


class TestCheckBiomarkerDrift:
    def test_identical_biomarker_no_drift(self, biomarker_df):
        det = DriftDetector(alpha=0.05)
        report = det.check_biomarker_drift(
            biomarker_df,
            biomarker_df.copy(),
            biomarkers=["inflammation_index"],
            arms=["placebo"],
            check_weeks=[0, 12],
        )
        assert report.n_tested > 0
        # Identical data — should not drift
        assert report.n_drifted == 0
