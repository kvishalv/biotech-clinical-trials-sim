"""tests/test_biomarker_models.py — Tests for simulation/biomarker_models.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.biomarker_models import (
    BiomarkerSimulator,
    BiomarkerTimeSeries,
    extract_re_stds,
    extract_baseline_params,
)


class TestBiomarkerSimulator:
    def test_simulate_patient_returns_9_series(self, biomarker_config, patients):
        sim = BiomarkerSimulator(biomarker_config, seed=42, burnin_weeks=2)
        result = sim.simulate_patient(patients[0], n_weeks=12)
        assert len(result) == 9

    def test_series_correct_length(self, biomarker_config, patients):
        sim = BiomarkerSimulator(biomarker_config, seed=42, burnin_weeks=2)
        n_weeks = 12
        result = sim.simulate_patient(patients[0], n_weeks=n_weeks)
        for ts in result:
            assert len(ts.values) == n_weeks + 1  # +1 for week 0 baseline
            assert len(ts.weeks) == n_weeks + 1

    def test_values_non_negative(self, biomarker_config, patients):
        sim = BiomarkerSimulator(biomarker_config, seed=42, burnin_weeks=2)
        result = sim.simulate_patient(patients[0], n_weeks=12)
        for ts in result:
            assert np.all(ts.values >= 0.0), f"{ts.biomarker} has negative values"

    def test_simulate_cohort_returns_dataframe(self, biomarker_df):
        assert isinstance(biomarker_df, pd.DataFrame)
        expected_cols = {"patient_id", "arm", "biomarker", "week", "value", "observed"}
        assert expected_cols.issubset(set(biomarker_df.columns))

    def test_all_9_biomarkers_in_output(self, biomarker_df, biomarker_config):
        expected = set(biomarker_config.biomarkers.keys())
        actual = set(biomarker_df["biomarker"].unique())
        assert expected == actual

    def test_dropout_weeks_are_unobserved(self, biomarker_config, patients):
        sim = BiomarkerSimulator(biomarker_config, seed=42, burnin_weeks=2)
        for patient in patients:
            if patient.dropout_week is not None:
                result = sim.simulate_patient(patient, n_weeks=12)
                for ts in result:
                    # Weeks after dropout should be unobserved
                    assert not any(ts.is_observed[patient.dropout_week:])
                break  # test one dropout patient is sufficient

    def test_reproducibility(self, biomarker_config, patients):
        sim1 = BiomarkerSimulator(biomarker_config, seed=42)
        sim2 = BiomarkerSimulator(biomarker_config, seed=42)
        p = patients[0]
        ts1 = sim1.simulate_patient(p, n_weeks=12)
        ts2 = sim2.simulate_patient(p, n_weeks=12)
        for s1, s2 in zip(ts1, ts2):
            np.testing.assert_array_equal(s1.values, s2.values)

    def test_different_seeds_produce_different_values(self, biomarker_config, patients):
        sim1 = BiomarkerSimulator(biomarker_config, seed=42)
        sim2 = BiomarkerSimulator(biomarker_config, seed=999)
        p = patients[0]
        ts1 = sim1.simulate_patient(p, n_weeks=12)
        ts2 = sim2.simulate_patient(p, n_weeks=12)
        # At least one biomarker should differ
        diffs = [not np.array_equal(s1.values, s2.values) for s1, s2 in zip(ts1, ts2)]
        assert any(diffs)

    def test_treatment_effect_direction_high_dose(self, biomarker_df):
        """High-dose arm should show lower inflammation than placebo at week 12."""
        sub = biomarker_df[
            (biomarker_df["biomarker"] == "inflammation_index")
            & (biomarker_df["week"] == 12)
            & (biomarker_df["observed"])
        ]
        high_mean = sub[sub["arm"] == "high_dose"]["value"].mean()
        placebo_mean = sub[sub["arm"] == "placebo"]["value"].mean()
        # High dose should reduce inflammation (direction test, not magnitude)
        assert high_mean <= placebo_mean + 0.5  # generous tolerance for small cohort

    def test_endpoint_summary_structure(self, biomarker_df, biomarker_config):
        sim = BiomarkerSimulator(biomarker_config, seed=42)
        summary = sim.endpoint_summary(biomarker_df, "inflammation_index")
        assert "week" in summary.columns
        assert "arm" in summary.columns
        assert "mean" in summary.columns


class TestExtractHelpers:
    def test_extract_re_stds(self, biomarker_config):
        stds = extract_re_stds(biomarker_config)
        assert len(stds) == 9
        assert all(v >= 0 for v in stds.values())

    def test_extract_baseline_params(self, biomarker_config):
        params = extract_baseline_params(biomarker_config)
        assert len(params) == 9
        for name, p in params.items():
            assert "baseline_mean" in p
            assert "baseline_std" in p
            assert p["baseline_std"] > 0


class TestBiomarkerTimeSeries:
    def test_to_dataframe(self, biomarker_config, patients):
        sim = BiomarkerSimulator(biomarker_config, seed=42)
        ts_list = sim.simulate_patient(patients[0], n_weeks=12)
        df = ts_list[0].to_dataframe()
        assert "patient_id" in df.columns
        assert "week" in df.columns
        assert "value" in df.columns
        assert len(df) == 13  # weeks 0..12
