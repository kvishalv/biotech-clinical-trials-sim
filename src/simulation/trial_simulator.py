"""
simulation/trial_simulator.py
──────────────────────────────
Main simulation orchestrator for the biotech clinical trial platform.

This is the single entry point that wires together:
  PatientGenerator → BiomarkerSimulator → CausalEstimator → OutcomeModels

Key design principles:
  1. Every run is fully versioned: config hash + seed → deterministic run_id
  2. The simulator is a pure function of (config, seed) → results
  3. No global mutable state — all state lives in TrialResult
  4. The LLM agent layer is called AFTER outcomes are computed — never before
  5. Configs are frozen Pydantic models — no accidental mutation mid-run
  6. Experiment tracking is optional (injected, not hardwired)

Usage:
    from src.simulation.trial_simulator import TrialSimulator
    from src.utils.config import load_trial_config, load_biomarker_config

    trial_cfg = load_trial_config("configs/trial_config.yaml")
    biomarker_cfg = load_biomarker_config("configs/biomarker_config.yaml")

    sim = TrialSimulator(trial_cfg, biomarker_cfg)
    result = sim.run()
    print(result.ate_results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.config import (
    BiomarkerConfig,
    TrialConfig,
    build_run_id,
    hash_config,
    register_seed,
)
from src.simulation.patient_generator import Patient, PatientGenerator
from src.simulation.biomarker_models import (
    BiomarkerSimulator,
    extract_baseline_params,
    extract_re_stds,
)
from src.simulation.causal_model import CausalDAG, CausalEstimator
from src.simulation.outcome_models import (
    WeibullSurvivalModel,
    LogisticOutcomeModel,
    ContinuousOutcomeModel,
    HTEUpliftModel,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TrialResult:
    """
    Complete output of a single simulation run.

    Attributes:
        run_id:            Versioned run identifier (<config_hash>-s<seed>)
        trial_name:        Human-readable trial name from config
        config_hash:       SHA-256 hash of the trial config (first 8 chars)
        seed:              RNG seed used
        n_patients:        Number of enrolled patients
        n_weeks:           Duration of simulation
        patients:          List of Patient objects
        patient_df:        Patient-level DataFrame
        biomarker_df:      Long-format longitudinal biomarker measurements
        survival_df:       Time-to-event DataFrame
        cohort_summary:    Dict of cohort-level summary statistics
        ate_results:       Dict of ATE estimates per endpoint
        cate_results:      Dict of CATE DataFrames per endpoint
        survival_summary:  Survival model summary dict
        logistic_summary:  Logistic outcome model summary dict
        continuous_summary: Continuous endpoint OLS summary dict
        uplift_summary:    HTE uplift model summary dict
        elapsed_seconds:   Wall-clock simulation runtime
        metadata:          Arbitrary extension dict
    """

    run_id: str
    trial_name: str
    config_hash: str
    seed: int
    n_patients: int
    n_weeks: int
    patients: list[Patient]
    patient_df: pd.DataFrame
    biomarker_df: pd.DataFrame
    survival_df: pd.DataFrame
    cohort_summary: dict[str, Any]
    ate_results: dict[str, Any]
    cate_results: dict[str, pd.DataFrame]
    survival_summary: dict[str, Any]
    logistic_summary: dict[str, Any]
    continuous_summary: dict[str, Any]
    uplift_summary: dict[str, Any]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable summary of the trial result.

        Excludes large DataFrames — use .biomarker_df for raw data.
        Intended for LLM agent consumption and experiment logging.
        """
        return {
            "run_id": self.run_id,
            "trial_name": self.trial_name,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "n_patients": self.n_patients,
            "n_weeks": self.n_weeks,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "cohort_summary": self.cohort_summary,
            "ate_results": self.ate_results,
            "survival_summary": self.survival_summary,
            "logistic_summary": self.logistic_summary,
            "continuous_summary": self.continuous_summary,
            "uplift_summary": self.uplift_summary,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────


class TrialSimulator:
    """
    Top-level orchestrator for a single clinical trial simulation run.

    Args:
        trial_cfg:     Validated TrialConfig (frozen Pydantic model)
        biomarker_cfg: Validated BiomarkerConfig (frozen Pydantic model)
        dag:           Optional CausalDAG — defaults to the longevity DAG
        tracker:       Optional ExperimentTracker for MLflow logging

    Usage:
        sim = TrialSimulator(trial_cfg, biomarker_cfg)
        result = sim.run()
    """

    def __init__(
        self,
        trial_cfg: TrialConfig,
        biomarker_cfg: BiomarkerConfig,
        dag: CausalDAG | None = None,
        tracker: Any | None = None,  # ExperimentTracker (avoids circular import)
    ) -> None:
        self.trial_cfg = trial_cfg
        self.biomarker_cfg = biomarker_cfg
        self.dag = dag or CausalDAG.default_longevity_dag()
        self.tracker = tracker

        # Derived convenience attributes
        self.seed = trial_cfg.simulation.seed
        self.n_weeks = trial_cfg.simulation.n_weeks
        self.n_patients = trial_cfg.cohort.n_patients
        self.burnin = trial_cfg.simulation.burnin_weeks
        self.biomarker_names = list(biomarker_cfg.biomarkers.keys())
        self.run_id = build_run_id(trial_cfg)

        # Register seed in the global registry for reproducibility tracking
        register_seed(self.run_id, self.seed)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_patient_generator(self) -> PatientGenerator:
        """Instantiate a seeded PatientGenerator."""
        return PatientGenerator(
            cohort_cfg=self.trial_cfg.cohort,
            biomarker_names=self.biomarker_names,
            seed=self.seed,
        )

    def _build_biomarker_simulator(self) -> BiomarkerSimulator:
        """Instantiate a seeded BiomarkerSimulator."""
        return BiomarkerSimulator(
            biomarker_cfg=self.biomarker_cfg,
            seed=self.seed,
            burnin_weeks=self.burnin,
        )

    def _compute_ate_for_all_endpoints(
        self,
        biomarker_df: pd.DataFrame,
        estimator: CausalEstimator,
    ) -> dict[str, Any]:
        """
        Estimate ATE for the primary endpoint and all secondary endpoints.

        Args:
            biomarker_df: Long-format simulation output.
            estimator:    CausalEstimator instance.

        Returns:
            Dict {endpoint_name: {ate, se, ci_lower, ci_upper, p_value, ...}}
        """
        endpoints = [self.trial_cfg.outcome.primary_endpoint] + list(
            self.trial_cfg.outcome.secondary_endpoints
        )
        results: dict[str, Any] = {}

        for endpoint in endpoints:
            # High dose vs placebo
            ate = estimator.estimate_ate(
                df=biomarker_df,
                outcome_col=endpoint,
                treatment_value="high_dose",
                control_value="placebo",
            )
            # Low dose vs placebo
            ate_low = estimator.estimate_ate(
                df=biomarker_df,
                outcome_col=endpoint,
                treatment_value="low_dose",
                control_value="placebo",
            )
            results[endpoint] = {
                "high_dose_vs_placebo": ate,
                "low_dose_vs_placebo": ate_low,
                "is_primary": endpoint == self.trial_cfg.outcome.primary_endpoint,
            }

        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> TrialResult:
        """
        Execute a complete simulation run.

        Steps:
          1. Generate patient cohort
          2. Simulate longitudinal biomarkers (all 9 composite markers)
          3. Fit survival, logistic, continuous, and HTE outcome models
          4. Compute ATE and CATE estimates via causal estimator
          5. Optionally log everything to MLflow
          6. Return a TrialResult

        Returns:
            TrialResult with all outputs.

        Raises:
            RuntimeError: If cohort generation produces fewer patients than required.
        """
        t_start = time.perf_counter()

        # ── 1. Generate cohort ────────────────────────────────────────────────
        pg = self._build_patient_generator()
        re_stds = extract_re_stds(self.biomarker_cfg)
        baseline_params = extract_baseline_params(self.biomarker_cfg)

        patients: list[Patient] = pg.generate(
            biomarker_re_stds=re_stds,
            biomarker_params=baseline_params,
            n_weeks=self.n_weeks,
        )

        if len(patients) == 0:
            raise RuntimeError(
                f"No patients enrolled after I/E screening. "
                f"Check inclusion criteria in trial config (run_id={self.run_id})."
            )

        patient_df = pg.to_dataframe(patients)
        cohort_summary = pg.cohort_summary(patients)

        # ── 2. Simulate biomarker trajectories ────────────────────────────────
        bio_sim = self._build_biomarker_simulator()
        biomarker_df = bio_sim.simulate_cohort(patients, n_weeks=self.n_weeks)

        # ── 3. Survival model ─────────────────────────────────────────────────
        surv_model = WeibullSurvivalModel(seed=self.seed)
        survival_df = surv_model.generate_survival_data(patients, n_weeks=self.n_weeks)
        surv_model.fit(survival_df)
        survival_summary = surv_model.summary(survival_df)

        # ── 4. Logistic outcome model ─────────────────────────────────────────
        primary = self.trial_cfg.outcome.primary_endpoint
        logit_model = LogisticOutcomeModel(seed=self.seed)
        logit_data = logit_model.prepare_data(
            patient_df=patient_df,
            biomarker_df=biomarker_df,
            endpoint=primary,
            threshold_delta=self.trial_cfg.outcome.success_threshold_delta,
        )
        logit_model.fit(logit_data)
        logistic_summary = logit_model.summary(logit_data)

        # ── 5. Continuous outcome model ───────────────────────────────────────
        cont_model = ContinuousOutcomeModel(seed=self.seed)
        cont_model.fit(patient_df, biomarker_df, endpoint=primary)
        continuous_summary = cont_model.summary()

        # ── 6. HTE / Uplift model ─────────────────────────────────────────────
        uplift_data = logit_data.copy()  # reuse same feature matrix
        uplift_model = HTEUpliftModel(seed=self.seed)
        uplift_model.fit(uplift_data)
        uplift_summary = uplift_model.summary(uplift_data)

        # ── 7. Causal ATE + CATE ──────────────────────────────────────────────
        estimator = CausalEstimator(self.dag)
        ate_results = self._compute_ate_for_all_endpoints(biomarker_df, estimator)

        # CATE on primary endpoint only (expensive; add more as needed)
        cate_df = estimator.estimate_cate(
            patient_df=patient_df,
            biomarker_df=biomarker_df,
            outcome_col=primary,
        )
        cate_results = {primary: cate_df}

        elapsed = time.perf_counter() - t_start

        result = TrialResult(
            run_id=self.run_id,
            trial_name=self.trial_cfg.trial.get("name", "unnamed"),
            config_hash=hash_config(self.trial_cfg),
            seed=self.seed,
            n_patients=len(patients),
            n_weeks=self.n_weeks,
            patients=patients,
            patient_df=patient_df,
            biomarker_df=biomarker_df,
            survival_df=survival_df,
            cohort_summary=cohort_summary,
            ate_results=ate_results,
            cate_results=cate_results,
            survival_summary=survival_summary,
            logistic_summary=logistic_summary,
            continuous_summary=continuous_summary,
            uplift_summary=uplift_summary,
            elapsed_seconds=elapsed,
        )

        # ── 8. Optional experiment tracking ───────────────────────────────────
        if self.tracker is not None:
            self.tracker.log_trial_result(result)

        return result

    def run_sensitivity(self, seeds: list[int]) -> list[TrialResult]:
        """
        Run multiple simulations with different seeds for sensitivity analysis.

        Args:
            seeds: List of integer seeds to use.

        Returns:
            List of TrialResult instances, one per seed.
        """
        import copy

        results = []
        for s in seeds:
            # Clone config with new seed (Pydantic frozen → use model_copy)
            new_sim_cfg = self.trial_cfg.simulation.model_copy(update={"seed": s})
            new_trial_cfg = self.trial_cfg.model_copy(update={"simulation": new_sim_cfg})
            sim = TrialSimulator(new_trial_cfg, self.biomarker_cfg, self.dag, self.tracker)
            results.append(sim.run())

        return results
