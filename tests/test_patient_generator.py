"""tests/test_patient_generator.py — Tests for simulation/patient_generator.py"""

from __future__ import annotations

import pytest
import numpy as np
from src.simulation.patient_generator import (
    PatientGenerator,
    Patient,
    _top_comorbidities,
    COMORBIDITY_PREVALENCE,
)


class TestPatientGenerator:
    def test_generates_correct_n_patients(self, patients):
        assert len(patients) == 50

    def test_all_patients_enrolled(self, patients):
        assert all(p.enrolled for p in patients)

    def test_age_within_inclusion_bounds(self, patients, small_trial_config):
        ic = small_trial_config.cohort.inclusion_criteria
        for p in patients:
            assert ic.min_age <= p.age <= ic.max_age, (
                f"Patient age {p.age} outside [{ic.min_age}, {ic.max_age}]"
            )

    def test_bmi_within_inclusion_bounds(self, patients, small_trial_config):
        ic = small_trial_config.cohort.inclusion_criteria
        for p in patients:
            assert ic.min_bmi <= p.bmi <= ic.max_bmi

    def test_no_excluded_comorbidities(self, patients, small_trial_config):
        excluded = set(small_trial_config.cohort.inclusion_criteria.excluded_conditions)
        for p in patients:
            overlap = p.comorbidities & excluded
            assert not overlap, f"Patient has excluded comorbidity: {overlap}"

    def test_arms_are_valid(self, patients, small_trial_config):
        valid_arms = {arm.name for arm in small_trial_config.cohort.treatment_arms}
        for p in patients:
            assert p.arm in valid_arms

    def test_arm_allocation_roughly_balanced(self, patients):
        from collections import Counter
        counts = Counter(p.arm for p in patients)
        total = len(patients)
        for arm, count in counts.items():
            # Allow ±15% deviation from expected allocation
            assert count / total > 0.15, f"Arm '{arm}' has fewer patients than expected"

    def test_patient_ids_are_unique(self, patients):
        ids = [p.patient_id for p in patients]
        assert len(ids) == len(set(ids))

    def test_site_ids_are_valid(self, patients, small_trial_config):
        n_sites = small_trial_config.cohort.n_sites
        valid_sites = {f"site_{i:02d}" for i in range(n_sites)}
        for p in patients:
            assert p.site_id in valid_sites

    def test_patient_re_keys_match_biomarkers(self, patients, biomarker_config):
        expected_keys = set(biomarker_config.biomarkers.keys())
        for p in patients:
            assert set(p.patient_re.keys()) == expected_keys

    def test_baseline_biomarkers_non_negative(self, patients):
        for p in patients:
            for bm, val in p.baseline_biomarkers.items():
                assert val >= 0.0, f"Patient {p.patient_id} has negative {bm}: {val}"

    def test_reproducibility_same_seed(self, small_trial_config, biomarker_config):
        """Same seed must produce identical cohorts."""
        from src.simulation.biomarker_models import extract_re_stds, extract_baseline_params

        pg1 = PatientGenerator(small_trial_config.cohort, list(biomarker_config.biomarkers.keys()), seed=42)
        pg2 = PatientGenerator(small_trial_config.cohort, list(biomarker_config.biomarkers.keys()), seed=42)
        p1 = pg1.generate(extract_re_stds(biomarker_config), extract_baseline_params(biomarker_config), n_weeks=12)
        p2 = pg2.generate(extract_re_stds(biomarker_config), extract_baseline_params(biomarker_config), n_weeks=12)
        assert len(p1) == len(p2)
        assert all(a.age == b.age for a, b in zip(p1, p2))

    def test_different_seeds_produce_different_cohorts(self, small_trial_config, biomarker_config):
        from src.simulation.biomarker_models import extract_re_stds, extract_baseline_params

        pg1 = PatientGenerator(small_trial_config.cohort, list(biomarker_config.biomarkers.keys()), seed=42)
        pg2 = PatientGenerator(small_trial_config.cohort, list(biomarker_config.biomarkers.keys()), seed=999)
        p1 = pg1.generate(extract_re_stds(biomarker_config), extract_baseline_params(biomarker_config), n_weeks=12)
        p2 = pg2.generate(extract_re_stds(biomarker_config), extract_baseline_params(biomarker_config), n_weeks=12)
        ages1 = [p.age for p in p1]
        ages2 = [p.age for p in p2]
        assert ages1 != ages2

    def test_to_dataframe_has_expected_columns(self, patient_df):
        expected_cols = {"patient_id", "age", "sex", "bmi", "arm", "site_id", "enrolled"}
        assert expected_cols.issubset(set(patient_df.columns))

    def test_cohort_summary_structure(self, patient_generator, patients):
        summary = patient_generator.cohort_summary(patients)
        assert "n_enrolled" in summary
        assert "age_mean" in summary
        assert "arm_distribution" in summary
        assert summary["n_enrolled"] == len(patients)


class TestTopComorbidities:
    def test_returns_top_n(self, patient_df):
        result = _top_comorbidities(patient_df, top_n=3)
        assert len(result) <= 3

    def test_rates_are_between_0_and_1(self, patient_df):
        result = _top_comorbidities(patient_df)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}: {v} out of range"
