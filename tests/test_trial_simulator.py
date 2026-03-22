"""tests/test_trial_simulator.py — Tests for simulation/trial_simulator.py"""

from __future__ import annotations

import pytest
from src.simulation.trial_simulator import TrialResult


class TestTrialSimulator:
    def test_run_returns_trial_result(self, trial_result):
        assert isinstance(trial_result, TrialResult)

    def test_run_id_format(self, trial_result):
        assert "-s" in trial_result.run_id

    def test_n_patients_matches_config(self, trial_result, small_trial_config):
        # Allow slight shortfall from I/E screening
        assert trial_result.n_patients >= small_trial_config.cohort.n_patients * 0.70

    def test_biomarker_df_has_9_markers(self, trial_result, biomarker_config):
        n_markers = trial_result.biomarker_df["biomarker"].nunique()
        assert n_markers == 9

    def test_survival_df_has_correct_columns(self, trial_result):
        expected = {"patient_id", "arm", "duration", "event"}
        assert expected.issubset(set(trial_result.survival_df.columns))

    def test_ate_results_has_primary_endpoint(self, trial_result, small_trial_config):
        primary = small_trial_config.outcome.primary_endpoint
        assert primary in trial_result.ate_results

    def test_ate_high_dose_vs_placebo_present(self, trial_result, small_trial_config):
        primary = small_trial_config.outcome.primary_endpoint
        assert "high_dose_vs_placebo" in trial_result.ate_results[primary]

    def test_logistic_summary_has_responder_rate(self, trial_result):
        assert "overall_responder_rate" in trial_result.logistic_summary
        rate = trial_result.logistic_summary["overall_responder_rate"]
        assert 0.0 <= rate <= 1.0

    def test_uplift_summary_present(self, trial_result):
        assert "uplift_mean" in trial_result.uplift_summary

    def test_to_summary_dict_is_serialisable(self, trial_result):
        import json
        summary = trial_result.to_summary_dict()
        # Should not raise
        json.dumps(summary, default=str)

    def test_elapsed_seconds_is_positive(self, trial_result):
        assert trial_result.elapsed_seconds > 0

    def test_reproducibility(self, small_trial_config, biomarker_config):
        """Two runs with same config + seed must produce identical patient counts."""
        from src.simulation.trial_simulator import TrialSimulator

        r1 = TrialSimulator(small_trial_config, biomarker_config).run()
        r2 = TrialSimulator(small_trial_config, biomarker_config).run()
        assert r1.n_patients == r2.n_patients
        assert r1.run_id == r2.run_id

    def test_sensitivity_runs_multiple_seeds(self, small_trial_config, biomarker_config):
        from src.simulation.trial_simulator import TrialSimulator

        sim = TrialSimulator(small_trial_config, biomarker_config)
        results = sim.run_sensitivity(seeds=[42, 43, 44])
        assert len(results) == 3
        run_ids = [r.run_id for r in results]
        assert len(set(run_ids)) == 3  # all distinct


class TestTrialResultCohortSummary:
    def test_cohort_summary_n_enrolled(self, trial_result):
        assert trial_result.cohort_summary["n_enrolled"] > 0

    def test_cohort_summary_arm_distribution(self, trial_result):
        arms = trial_result.cohort_summary["arm_distribution"]
        assert len(arms) >= 2  # at least 2 arms

    def test_cohort_summary_age_mean_in_range(self, trial_result, small_trial_config):
        ic = small_trial_config.cohort.inclusion_criteria
        age_mean = trial_result.cohort_summary["age_mean"]
        assert ic.min_age <= age_mean <= ic.max_age
