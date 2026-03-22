"""
simulation/biomarker_models.py
───────────────────────────────
Longitudinal biomarker simulation engine for 9 composite longevity biomarkers.

Generative model per biomarker per patient per week:
─────────────────────────────────────────────────────
    y[t] = μ_arm(t)
           + ρ · (y[t-1] - μ_baseline)     ← AR(1) mean-reversion
           + u_i                             ← patient random effect (fixed at enrolment)
           + ε_site                          ← site-specific assay noise ~ N(0, σ_site²)
           + ε_residual                      ← residual noise ~ N(0, σ_resid²)

Treatment response heterogeneity:
  Each patient is independently designated a "responder" (Bernoulli draw with
  probability = responder_fraction).  Non-responders receive 50% of the arm effect.

Autocorrelation (AR(1)):
  The ρ·(y[t-1] - μ_baseline) term means biomarker values are correlated across
  weeks — biologically realistic. ρ ∈ [0, 1); set per biomarker in YAML.

All 9 biomarkers modelled:
  1. inflammation_index
  2. metabolic_risk_index
  3. epigenetic_age_acceleration
  4. frailty_progression
  5. organ_reserve_score
  6. latent_mitochondrial_dysfunction
  7. immune_resilience
  8. sleep_circadian_disruption
  9. recovery_velocity
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import BiomarkerConfig, SingleBiomarkerConfig
from src.simulation.patient_generator import Patient

# Default residual noise std (fraction of baseline_std not captured by AR(1))
_RESIDUAL_NOISE_FRACTION = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BiomarkerTimeSeries:
    """
    Longitudinal biomarker measurements for one patient.

    Attributes:
        patient_id:  UUID of the patient
        arm:         Treatment arm name
        biomarker:   Biomarker name
        values:      Array of shape (n_weeks+1,) — index 0 is baseline (t=0)
        weeks:       Array [0, 1, 2, ..., n_weeks]
        is_observed: Boolean mask — False when patient has dropped out
    """

    patient_id: str
    arm: str
    biomarker: str
    values: np.ndarray
    weeks: np.ndarray
    is_observed: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Melt to long-format DataFrame."""
        return pd.DataFrame(
            {
                "patient_id": self.patient_id,
                "arm": self.arm,
                "biomarker": self.biomarker,
                "week": self.weeks,
                "value": self.values,
                "observed": self.is_observed,
            }
        )


# ──────────────────────────────────────────────────────────────────────────────
# Biomarker simulator
# ──────────────────────────────────────────────────────────────────────────────


class BiomarkerSimulator:
    """
    Simulates longitudinal trajectories for all 9 composite biomarkers.

    Args:
        biomarker_cfg:  BiomarkerConfig loaded from biomarker_config.yaml
        seed:           RNG seed (combined with patient_id hash for per-patient reproducibility)
        burnin_weeks:   Weeks before treatment starts (pre-treatment period)

    Usage:
        sim = BiomarkerSimulator(biomarker_cfg, seed=42, burnin_weeks=4)
        results = sim.simulate_patient(patient, n_weeks=52)
        df = pd.concat([ts.to_dataframe() for ts in results])
    """

    def __init__(
        self,
        biomarker_cfg: BiomarkerConfig,
        seed: int = 42,
        burnin_weeks: int = 4,
    ) -> None:
        self.cfg = biomarker_cfg
        self.base_seed = seed
        self.burnin_weeks = burnin_weeks

    def _patient_rng(self, patient_id: str) -> np.random.Generator:
        """
        Create a deterministic per-patient RNG by XOR-ing the base seed with
        a hash of the patient UUID.  This ensures different patients get
        independent noise streams without requiring a central state machine.

        Args:
            patient_id: Patient UUID string.

        Returns:
            numpy Generator seeded deterministically per patient.
        """
        # Use first 8 hex chars of md5 digest as patient-specific offset
        import hashlib

        pid_hash = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(self.base_seed ^ pid_hash)

    def _is_responder(self, rng: np.random.Generator, responder_fraction: float) -> bool:
        """
        Draw whether this patient is a treatment responder.

        Args:
            rng:                Per-patient RNG
            responder_fraction: Fraction of patients who respond to treatment

        Returns:
            True if responder.
        """
        return bool(rng.random() < responder_fraction)

    def _arm_effect(
        self,
        arm: str,
        week: int,
        cfg: SingleBiomarkerConfig,
        is_responder: bool,
        burnin_weeks: int,
    ) -> float:
        """
        Compute the weekly treatment effect for a given arm at a given week.

        Treatment effect ramps in linearly over the first 4 post-burnin weeks
        to avoid unrealistic step-change responses.

        Args:
            arm:           Treatment arm name
            week:          Current simulation week (0-indexed)
            cfg:           SingleBiomarkerConfig for this biomarker
            is_responder:  Whether this patient responds to treatment
            burnin_weeks:  Pre-treatment weeks (effect is 0 during this period)

        Returns:
            Additive weekly effect on the biomarker value.
        """
        if week <= burnin_weeks:
            return 0.0  # no treatment during burnin

        # Base effect from config
        effect_map = cfg.treatment_effect.model_dump()
        base_effect = effect_map.get(arm, 0.0)

        # Non-responders get attenuated effect
        if not is_responder:
            base_effect *= 0.50

        # Ramp-in over 4 weeks after treatment start
        weeks_on_treatment = week - burnin_weeks
        ramp = min(weeks_on_treatment / 4.0, 1.0)

        return base_effect * ramp

    def simulate_patient(
        self, patient: Patient, n_weeks: int = 52
    ) -> list[BiomarkerTimeSeries]:
        """
        Simulate all 9 biomarker trajectories for a single patient.

        Args:
            patient:  Patient instance with arm, baseline_biomarkers, patient_re, etc.
            n_weeks:  Number of weeks to simulate.

        Returns:
            List of BiomarkerTimeSeries, one per biomarker.
        """
        rng = self._patient_rng(patient.patient_id)
        results: list[BiomarkerTimeSeries] = []

        for biomarker_name, bio_cfg in self.cfg.biomarkers.items():
            ts = self._simulate_single_biomarker(
                rng=rng,
                patient=patient,
                biomarker_name=biomarker_name,
                bio_cfg=bio_cfg,
                n_weeks=n_weeks,
            )
            results.append(ts)

        return results

    def _simulate_single_biomarker(
        self,
        rng: np.random.Generator,
        patient: Patient,
        biomarker_name: str,
        bio_cfg: SingleBiomarkerConfig,
        n_weeks: int,
    ) -> BiomarkerTimeSeries:
        """
        Core AR(1) simulation loop for one patient × one biomarker.

        Model:
            y[0] = baseline (from patient_generator)
            y[t] = arm_effect(t)
                   + ρ·(y[t-1] - μ_baseline)   [mean-reverting AR(1)]
                   + u_i                          [patient RE, fixed]
                   + ε_site[t]                   [assay noise]
                   + ε_resid[t]                  [residual noise]

        Args:
            rng:            Per-patient numpy Generator
            patient:        Patient instance
            biomarker_name: Which biomarker to simulate
            bio_cfg:        Config for this biomarker
            n_weeks:        Total simulation weeks

        Returns:
            BiomarkerTimeSeries for this patient × biomarker.
        """
        rho = bio_cfg.ar1_coefficient
        mu = bio_cfg.baseline_mean
        sigma_site = bio_cfg.site_assay_noise_std
        sigma_resid = bio_cfg.baseline_std * _RESIDUAL_NOISE_FRACTION

        # Patient random effect (drawn at enrolment, fixed here)
        u_i = patient.patient_re.get(biomarker_name, 0.0)

        # Site-specific noise: same site = correlated noise across patients
        # We use a deterministic site seed so site noise is consistent
        site_seed = int(patient.site_id.replace("site_", "")) + self.base_seed
        site_rng = np.random.default_rng(site_seed)

        # Responder status for treatment-response heterogeneity
        is_responder = self._is_responder(rng, bio_cfg.responder_fraction)

        # ── Initialise arrays ─────────────────────────────────────────────────
        values = np.zeros(n_weeks + 1)  # index 0 = baseline week (t=0)
        observed = np.ones(n_weeks + 1, dtype=bool)

        # t=0: use the pre-computed baseline from patient generator
        y0 = patient.baseline_biomarkers.get(biomarker_name, mu)
        values[0] = y0

        # Mark dropout weeks as unobserved
        if patient.dropout_week is not None:
            observed[patient.dropout_week :] = False

        # ── AR(1) forward simulation ──────────────────────────────────────────
        for t in range(1, n_weeks + 1):
            effect = self._arm_effect(
                arm=patient.arm,
                week=t,
                cfg=bio_cfg,
                is_responder=is_responder,
                burnin_weeks=self.burnin_weeks,
            )

            # AR(1) mean-reversion component
            ar_component = rho * (values[t - 1] - mu)

            # Stochastic noise components
            eps_site = float(site_rng.normal(0.0, sigma_site))
            eps_resid = float(rng.normal(0.0, sigma_resid))

            # Compose the next observation
            y_t = mu + ar_component + effect + u_i + eps_site + eps_resid

            # Non-negativity floor (ratio/index biomarkers can't go below 0)
            values[t] = max(y_t, 0.0)

        weeks = np.arange(n_weeks + 1)
        return BiomarkerTimeSeries(
            patient_id=patient.patient_id,
            arm=patient.arm,
            biomarker=biomarker_name,
            values=values,
            weeks=weeks,
            is_observed=observed,
        )

    def simulate_cohort(
        self, patients: list[Patient], n_weeks: int = 52
    ) -> pd.DataFrame:
        """
        Simulate all biomarkers for an entire patient cohort.

        Args:
            patients: List of Patient instances.
            n_weeks:  Trial duration in weeks.

        Returns:
            Long-format DataFrame with columns:
                patient_id, arm, biomarker, week, value, observed
        """
        frames: list[pd.DataFrame] = []
        for patient in patients:
            ts_list = self.simulate_patient(patient, n_weeks)
            for ts in ts_list:
                frames.append(ts.to_dataframe())
        return pd.concat(frames, ignore_index=True)

    def endpoint_summary(self, df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
        """
        Compute per-arm mean ± SD for a specific biomarker endpoint at each week.

        Args:
            df:       Long-format simulation output from simulate_cohort()
            endpoint: Biomarker name (must match keys in biomarker config)

        Returns:
            DataFrame with columns: week, arm, mean, std, n
        """
        subset = df[(df["biomarker"] == endpoint) & (df["observed"])]
        summary = (
            subset.groupby(["week", "arm"])["value"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        return summary


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build biomarker_re_stds dict from BiomarkerConfig
# ──────────────────────────────────────────────────────────────────────────────


def extract_re_stds(biomarker_cfg: BiomarkerConfig) -> dict[str, float]:
    """
    Extract the patient random-effect standard deviations from BiomarkerConfig.

    These are passed to PatientGenerator so it can draw patient REs at enrolment.

    Args:
        biomarker_cfg: Loaded BiomarkerConfig.

    Returns:
        Dict {biomarker_name: patient_re_std}.
    """
    return {
        name: cfg.patient_re_std for name, cfg in biomarker_cfg.biomarkers.items()
    }


def extract_baseline_params(biomarker_cfg: BiomarkerConfig) -> dict[str, dict[str, float]]:
    """
    Extract baseline mean/std for each biomarker from BiomarkerConfig.

    Args:
        biomarker_cfg: Loaded BiomarkerConfig.

    Returns:
        Dict {biomarker_name: {baseline_mean, baseline_std}}.
    """
    return {
        name: {"baseline_mean": cfg.baseline_mean, "baseline_std": cfg.baseline_std}
        for name, cfg in biomarker_cfg.biomarkers.items()
    }
