"""
simulation/patient_generator.py
────────────────────────────────
Generates synthetic patient cohorts for clinical trial simulation.

Design:
  - All randomness flows through a single seeded numpy Generator — fully reproducible
  - Patient-level random effects are drawn once at generation time and carried
    forward through the entire longitudinal simulation
  - Site assignment is stratified so each site gets a roughly equal slice
  - Comorbidity prevalence follows published epidemiological rates (approximated)
  - No real patient data is ever used — this module is purely generative

Generative model for baseline demographics:
    age      ~ Uniform(min_age, max_age) clipped to [18, 100]
    bmi      ~ Normal(27, 4) clipped to [min_bmi, max_bmi]
    sex      ~ Bernoulli(0.5)
    site     ~ Categorical(1/n_sites)  (stratified)
    comorbidities: each drawn independently from prevalence table
    patient_re: drawn per biomarker from N(0, patient_re_std²)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CohortConfig, InclusionCriteria

# ──────────────────────────────────────────────────────────────────────────────
# Comorbidity prevalence table (population-level approximations)
# ──────────────────────────────────────────────────────────────────────────────

COMORBIDITY_PREVALENCE: dict[str, float] = {
    "hypertension": 0.45,
    "type2_diabetes": 0.12,
    "dyslipidemia": 0.38,
    "obesity": 0.32,
    "metabolic_syndrome": 0.25,
    "osteoarthritis": 0.22,
    "depression": 0.18,
    "sleep_apnea": 0.15,
    "hypothyroidism": 0.10,
    "atrial_fibrillation": 0.05,
    "active_cancer": 0.02,
    "end_stage_renal_disease": 0.01,
}

ETHNICITY_LABELS = ["white", "black", "hispanic", "asian", "other"]
ETHNICITY_PROBS = [0.60, 0.13, 0.18, 0.06, 0.03]


# ──────────────────────────────────────────────────────────────────────────────
# Patient dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Patient:
    """
    Immutable snapshot of a synthetic patient at enrolment.

    Attributes:
        patient_id:       UUID4 string, unique per patient
        age:              Age in years at enrolment
        sex:              "M" or "F"
        bmi:              Body mass index (kg/m²)
        ethnicity:        Ethnic group label
        site_id:          Clinical site identifier (e.g., "site_03")
        arm:              Treatment arm name ("placebo", "low_dose", "high_dose")
        comorbidities:    Set of comorbidity labels present at baseline
        patient_re:       Per-biomarker random effect dict {biomarker_name: float}
        baseline_biomarkers: Initial biomarker values at t=0 {biomarker_name: float}
        enrolled:         Whether patient passed I/E and was enrolled
        dropout_week:     Week at which patient dropped out (None if completed)
        metadata:         Arbitrary key-value store for extension
    """

    patient_id: str
    age: float
    sex: str
    bmi: float
    ethnicity: str
    site_id: str
    arm: str
    comorbidities: set[str]
    patient_re: dict[str, float]
    baseline_biomarkers: dict[str, float]
    enrolled: bool = True
    dropout_week: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict (suitable for DataFrame construction)."""
        base = {
            "patient_id": self.patient_id,
            "age": self.age,
            "sex": self.sex,
            "bmi": self.bmi,
            "ethnicity": self.ethnicity,
            "site_id": self.site_id,
            "arm": self.arm,
            "enrolled": self.enrolled,
            "dropout_week": self.dropout_week,
            "comorbidities": ";".join(sorted(self.comorbidities)),
        }
        # Flatten baseline biomarkers with prefix
        for k, v in self.baseline_biomarkers.items():
            base[f"baseline_{k}"] = v
        # Flatten random effects with prefix
        for k, v in self.patient_re.items():
            base[f"re_{k}"] = v
        return base


# ──────────────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────────────


class PatientGenerator:
    """
    Generates a reproducible synthetic patient cohort.

    Args:
        cohort_cfg:       CohortConfig from trial_config.yaml
        biomarker_names:  Ordered list of biomarker names — drives random-effect dims
        seed:             Integer seed for full reproducibility

    Usage:
        gen = PatientGenerator(cfg.cohort, biomarker_names, seed=42)
        patients = gen.generate()
        df = gen.to_dataframe(patients)
    """

    def __init__(
        self,
        cohort_cfg: CohortConfig,
        biomarker_names: list[str],
        seed: int = 42,
    ) -> None:
        self.cfg = cohort_cfg
        self.biomarker_names = biomarker_names
        self.rng = np.random.default_rng(seed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _draw_demographics(self, n: int) -> dict[str, np.ndarray]:
        """
        Draw raw demographic attributes for n candidates before I/E filtering.

        Returns:
            Dict of arrays: age, bmi, sex, ethnicity
        """
        ic: InclusionCriteria = self.cfg.inclusion_criteria

        ages = self.rng.uniform(ic.min_age, ic.max_age, size=n)
        bmis = np.clip(
            self.rng.normal(27.0, 4.0, size=n),
            ic.min_bmi,
            ic.max_bmi,
        )
        sexes = self.rng.choice(["M", "F"], size=n)
        ethnicities = self.rng.choice(ETHNICITY_LABELS, size=n, p=ETHNICITY_PROBS)
        return {"age": ages, "bmi": bmis, "sex": sexes, "ethnicity": ethnicities}

    def _draw_comorbidities(self) -> set[str]:
        """
        Draw a set of comorbidities for one patient using prevalence priors.

        Returns:
            Set of comorbidity label strings.
        """
        return {
            label
            for label, prev in COMORBIDITY_PREVALENCE.items()
            if self.rng.random() < prev
        }

    def _passes_inclusion(self, demo: dict[str, Any], comorbidities: set[str]) -> bool:
        """
        Check whether a patient satisfies the inclusion/exclusion criteria.

        Args:
            demo:          Dict with 'age' and 'bmi' values (scalars)
            comorbidities: Set of comorbidity labels for this patient

        Returns:
            True if the patient should be enrolled.
        """
        ic = self.cfg.inclusion_criteria

        # Age and BMI bounds
        if not (ic.min_age <= demo["age"] <= ic.max_age):
            return False
        if not (ic.min_bmi <= demo["bmi"] <= ic.max_bmi):
            return False

        # Required comorbidities must all be present
        for req in ic.required_conditions:
            if req not in comorbidities:
                return False

        # Excluded comorbidities must all be absent
        for excl in ic.excluded_conditions:
            if excl in comorbidities:
                return False

        return True

    def _assign_sites(self, n: int) -> list[str]:
        """
        Stratified site assignment: round-robin then shuffle.

        Args:
            n: Number of patients to assign.

        Returns:
            List of site_id strings.
        """
        n_sites = self.cfg.n_sites
        # Round-robin ensures balanced assignment
        site_ids = [f"site_{(i % n_sites):02d}" for i in range(n)]
        self.rng.shuffle(site_ids)
        return site_ids

    def _assign_arms(self, n: int) -> list[str]:
        """
        Randomise patients to treatment arms according to allocation weights.

        Args:
            n: Number of enrolled patients.

        Returns:
            List of arm name strings.
        """
        arms = [arm.name for arm in self.cfg.treatment_arms]
        probs = np.array([arm.allocation for arm in self.cfg.treatment_arms])
        probs = probs / probs.sum()  # normalise in case of floating-point drift
        return self.rng.choice(arms, size=n, p=probs).tolist()

    def _draw_patient_random_effects(
        self, biomarker_re_stds: dict[str, float]
    ) -> dict[str, float]:
        """
        Draw patient-level random effects for each biomarker.

        These are drawn once at enrolment and held fixed for the patient
        throughout the entire simulation — they represent stable inter-individual
        variation in biomarker baseline and trajectory.

        Args:
            biomarker_re_stds: {biomarker_name: re_std} from biomarker config

        Returns:
            Dict {biomarker_name: random_effect_value}
        """
        return {
            name: float(self.rng.normal(0.0, std))
            for name, std in biomarker_re_stds.items()
        }

    def _draw_baseline_biomarkers(
        self, biomarker_params: dict[str, dict[str, float]], patient_re: dict[str, float]
    ) -> dict[str, float]:
        """
        Sample baseline biomarker values at t=0 incorporating patient random effects.

        Args:
            biomarker_params: {name: {baseline_mean, baseline_std}} from biomarker config
            patient_re:       Patient-level random effects for each biomarker

        Returns:
            Dict {biomarker_name: baseline_value}
        """
        baselines: dict[str, float] = {}
        for name, params in biomarker_params.items():
            mean = params["baseline_mean"]
            std = params["baseline_std"]
            re = patient_re.get(name, 0.0)
            # Baseline = population mean + individual deviation + RE
            raw = float(self.rng.normal(mean, std)) + re
            # Clip to non-negative for ratio-type biomarkers
            baselines[name] = max(raw, 0.0)
        return baselines

    def _assign_dropout_week(self, n_weeks: int) -> int | None:
        """
        Stochastically assign a dropout week or None (completed trial).

        Uses a geometric distribution parameterised by the weekly dropout rate.

        Args:
            n_weeks: Total simulation duration in weeks.

        Returns:
            Integer dropout week, or None if patient completes the trial.
        """
        rate = self.cfg.dropout_rate_per_week
        if rate <= 0:
            return None
        # Geometric: probability of dropping out at exactly week k
        # P(dropout at week k) = (1-rate)^(k-1) * rate
        trial_week = int(self.rng.geometric(rate))
        return trial_week if trial_week <= n_weeks else None

    # ── public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        biomarker_re_stds: dict[str, float] | None = None,
        biomarker_params: dict[str, dict[str, float]] | None = None,
        n_weeks: int = 52,
    ) -> list[Patient]:
        """
        Generate a full synthetic cohort of enrolled patients.

        Strategy: over-sample by 30% to account for I/E screening failures,
        then filter down to exactly n_patients enrolled.

        Args:
            biomarker_re_stds:  {biomarker_name: re_std} — if None, all REs = 0
            biomarker_params:   {biomarker_name: {baseline_mean, baseline_std}}
            n_weeks:            Trial duration, for dropout assignment

        Returns:
            List of Patient objects (enrolled == True for all, with possible dropouts).
        """
        target_n = self.cfg.n_patients
        oversample = int(target_n * 1.30) + 20  # buffer for I/E rejections

        # Default stubs if biomarker params not provided
        if biomarker_re_stds is None:
            biomarker_re_stds = {name: 0.0 for name in self.biomarker_names}
        if biomarker_params is None:
            biomarker_params = {
                name: {"baseline_mean": 1.0, "baseline_std": 0.1}
                for name in self.biomarker_names
            }

        demo = self._draw_demographics(oversample)
        site_ids = self._assign_sites(oversample)

        enrolled: list[Patient] = []

        for i in range(oversample):
            if len(enrolled) >= target_n:
                break

            d = {k: demo[k][i] for k in demo}
            comorbidities = self._draw_comorbidities()

            if not self._passes_inclusion(d, comorbidities):
                continue  # screened out — do not enrol

            patient_re = self._draw_patient_random_effects(biomarker_re_stds)
            baselines = self._draw_baseline_biomarkers(biomarker_params, patient_re)
            dropout = self._assign_dropout_week(n_weeks)

            enrolled.append(
                Patient(
                    patient_id=str(uuid.uuid4()),
                    age=float(d["age"]),
                    sex=str(d["sex"]),
                    bmi=float(d["bmi"]),
                    ethnicity=str(d["ethnicity"]),
                    site_id=site_ids[i],
                    arm="",  # assigned below after full enrolment list is built
                    comorbidities=comorbidities,
                    patient_re=patient_re,
                    baseline_biomarkers=baselines,
                    enrolled=True,
                    dropout_week=dropout,
                )
            )

        # Warn (don't crash) if we couldn't hit the target
        if len(enrolled) < target_n:
            import warnings

            warnings.warn(
                f"Only enrolled {len(enrolled)}/{target_n} patients after I/E screening. "
                "Consider relaxing inclusion criteria or increasing oversample factor.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Assign treatment arms to the enrolled cohort
        arms = self._assign_arms(len(enrolled))
        for patient, arm in zip(enrolled, arms):
            object.__setattr__(patient, "arm", arm) if hasattr(
                patient, "__dataclass_fields__"
            ) else None
            patient.arm = arm  # dataclass is not frozen, so direct assignment is fine

        return enrolled

    def to_dataframe(self, patients: list[Patient]) -> pd.DataFrame:
        """
        Convert a list of Patient objects to a tidy pandas DataFrame.

        Args:
            patients: List of Patient instances.

        Returns:
            DataFrame with one row per patient and all attributes as columns.
        """
        return pd.DataFrame([p.to_dict() for p in patients])

    def cohort_summary(self, patients: list[Patient]) -> dict[str, Any]:
        """
        Compute basic summary statistics for the generated cohort.

        Used by the LLM cohort narrator agent as its data source.

        Args:
            patients: List of enrolled Patient instances.

        Returns:
            Dict of summary stats (counts, means, proportions, arm distribution).
        """
        df = self.to_dataframe(patients)
        arm_counts = df["arm"].value_counts().to_dict()
        site_counts = df["site_id"].value_counts().to_dict()
        dropout_count = df["dropout_week"].notna().sum()

        return {
            "n_enrolled": len(patients),
            "age_mean": float(df["age"].mean()),
            "age_std": float(df["age"].std()),
            "bmi_mean": float(df["bmi"].mean()),
            "bmi_std": float(df["bmi"].std()),
            "sex_f_pct": float((df["sex"] == "F").mean() * 100),
            "arm_distribution": arm_counts,
            "site_distribution": site_counts,
            "dropout_n": int(dropout_count),
            "dropout_pct": float(dropout_count / len(patients) * 100),
            "top_comorbidities": _top_comorbidities(df, top_n=5),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────


def _top_comorbidities(df: pd.DataFrame, top_n: int = 5) -> dict[str, float]:
    """
    Return the most prevalent comorbidities and their cohort-level rates.

    Args:
        df:    Patient DataFrame with 'comorbidities' column (semicolon-delimited).
        top_n: Number of top conditions to return.

    Returns:
        Dict {comorbidity: prevalence_rate} sorted by prevalence descending.
    """
    from collections import Counter

    counter: Counter = Counter()
    total = len(df)
    for row in df["comorbidities"]:
        if row:
            for cond in row.split(";"):
                counter[cond] += 1

    return {k: round(v / total, 3) for k, v in counter.most_common(top_n)}
