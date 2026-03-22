"""
utils/config.py
───────────────
Centralised configuration management for the biotech simulation platform.

Responsibilities:
  - Load and validate trial + biomarker YAML configs via Pydantic v2 models
  - Expose a typed TrialConfig and BiomarkerConfig to all other modules
  - Maintain a seed registry so every simulation run is fully reproducible:
      seed → deterministic RNG state → identical cohort + outcomes
  - Hash every config dict so experiment runs are self-describing

Design decisions:
  - Pydantic v2 for strict typing; extra fields are forbidden to catch typos early
  - Seed registry is a simple dict keyed by run_id — no external dependency
  - Config is immutable after construction (model_config frozen=True)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models for trial_config.yaml
# ──────────────────────────────────────────────────────────────────────────────


class TreatmentArm(BaseModel):
    """A single treatment arm with name and randomisation allocation weight."""

    name: str
    allocation: float = Field(gt=0.0, le=1.0)


class InclusionCriteria(BaseModel):
    """Inclusion / exclusion filters applied during cohort generation."""

    min_age: int = Field(ge=18, le=120)
    max_age: int = Field(ge=18, le=120)
    min_bmi: float = Field(ge=10.0)
    max_bmi: float = Field(le=60.0)
    required_conditions: list[str] = Field(default_factory=list)
    excluded_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def age_range_valid(self) -> "InclusionCriteria":
        """Ensure min_age < max_age."""
        if self.min_age >= self.max_age:
            raise ValueError(f"min_age ({self.min_age}) must be < max_age ({self.max_age})")
        return self


class CohortConfig(BaseModel):
    """Cohort sizing, arms, and eligibility parameters."""

    n_patients: int = Field(ge=10)
    n_sites: int = Field(ge=1)
    treatment_arms: list[TreatmentArm]
    inclusion_criteria: InclusionCriteria
    dropout_rate_per_week: float = Field(ge=0.0, le=0.1)

    @model_validator(mode="after")
    def allocations_sum_to_one(self) -> "CohortConfig":
        """Allocation weights must sum to ~1.0 (±0.01 tolerance)."""
        total = sum(arm.allocation for arm in self.treatment_arms)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Treatment arm allocations must sum to 1.0, got {total:.4f}")
        return self


class SimulationConfig(BaseModel):
    """Simulation run parameters."""

    n_weeks: int = Field(ge=1, le=520)
    seed: int
    n_parallel_runs: int = Field(ge=1)
    burnin_weeks: int = Field(ge=0)


class OutcomeConfig(BaseModel):
    """Primary and secondary endpoints."""

    primary_endpoint: str
    secondary_endpoints: list[str] = Field(default_factory=list)
    success_threshold_delta: float


class CausalConfig(BaseModel):
    """Causal inference configuration."""

    dag_definition: str
    estimand: str = Field(pattern="^(ATE|ATT|CATE)$")
    confounder_adjustment: bool


class TrackingConfig(BaseModel):
    """MLflow experiment tracking settings."""

    mlflow_uri: str
    experiment_name: str
    log_artifacts: bool


class TrialConfig(BaseModel):
    """
    Top-level trial configuration — parsed from trial_config.yaml.
    Immutable after construction.
    """

    model_config = {"frozen": True}  # immutable once loaded

    trial: dict[str, Any]
    cohort: CohortConfig
    simulation: SimulationConfig
    outcome: OutcomeConfig
    causal: CausalConfig
    tracking: TrackingConfig


# ──────────────────────────────────────────────────────────────────────────────
# Biomarker config sub-models
# ──────────────────────────────────────────────────────────────────────────────


class BiomarkerTreatmentEffects(BaseModel):
    """Per-arm mean treatment effect on a biomarker (weekly delta)."""

    placebo: float
    low_dose: float
    high_dose: float


class SingleBiomarkerConfig(BaseModel):
    """
    Parameters governing one composite biomarker's stochastic process.

    The generative model for each biomarker follows:
        y_t = μ + ρ·y_{t-1} + u_i + ε_site + ε_noise
    where:
        ρ  = ar1_coefficient   (autocorrelation)
        u_i ~ N(0, patient_re_std²)   (patient random effect)
        ε_site ~ N(0, site_assay_noise_std²)
        ε_noise ~ N(0, residual_std²)
    """

    description: str
    baseline_mean: float
    baseline_std: float = Field(gt=0.0)
    ar1_coefficient: float = Field(ge=0.0, le=1.0)
    patient_re_std: float = Field(ge=0.0)
    site_assay_noise_std: float = Field(ge=0.0)
    treatment_effect: BiomarkerTreatmentEffects
    responder_fraction: float = Field(ge=0.0, le=1.0)


class BiomarkerConfig(BaseModel):
    """Full biomarker config — parsed from biomarker_config.yaml."""

    model_config = {"frozen": True}

    biomarkers: dict[str, SingleBiomarkerConfig]


# ──────────────────────────────────────────────────────────────────────────────
# Environment settings (injected via .env or shell)
# ──────────────────────────────────────────────────────────────────────────────


class EnvSettings(BaseSettings):
    """
    Runtime environment settings loaded from environment variables / .env file.
    These override nothing in the YAML configs — they are infrastructure concerns.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    ray_address: str = "auto"           # "auto" = use existing cluster; "local" = start new
    log_level: str = "INFO"


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a plain dict.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML as a Python dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r") as fh:
        return yaml.safe_load(fh)


def load_trial_config(path: str | Path = "configs/trial_config.yaml") -> TrialConfig:
    """
    Load and validate the trial configuration from YAML.

    Args:
        path: Path to trial_config.yaml (default: configs/trial_config.yaml).

    Returns:
        Validated, frozen TrialConfig instance.
    """
    raw = load_yaml(path)
    return TrialConfig(**raw)


def load_biomarker_config(path: str | Path = "configs/biomarker_config.yaml") -> BiomarkerConfig:
    """
    Load and validate the biomarker configuration from YAML.

    Args:
        path: Path to biomarker_config.yaml (default: configs/biomarker_config.yaml).

    Returns:
        Validated, frozen BiomarkerConfig instance.
    """
    raw = load_yaml(path)
    return BiomarkerConfig(**raw)


# ──────────────────────────────────────────────────────────────────────────────
# Config hashing — every run is self-describing
# ──────────────────────────────────────────────────────────────────────────────


def hash_config(config: BaseModel) -> str:
    """
    Produce a deterministic SHA-256 hash of a Pydantic config model.

    Used to version simulation runs: identical configs → identical hash →
    same entry in the experiment registry.

    Args:
        config: Any Pydantic BaseModel instance.

    Returns:
        8-character hex digest (first 8 chars of SHA-256).
    """
    # model_dump_json is deterministic in Pydantic v2 (sorted keys)
    serialised = config.model_dump_json()
    return hashlib.sha256(serialised.encode()).hexdigest()[:8]


# ──────────────────────────────────────────────────────────────────────────────
# Seed registry — in-memory, keyed by run_id
# ──────────────────────────────────────────────────────────────────────────────


_SEED_REGISTRY: dict[str, int] = {}


def register_seed(run_id: str, seed: int) -> None:
    """
    Register the RNG seed for a simulation run.

    Args:
        run_id: Unique identifier for this simulation run (e.g., "run-abc123").
        seed: Integer seed used to initialise numpy's default_rng.
    """
    _SEED_REGISTRY[run_id] = seed


def get_seed(run_id: str) -> int:
    """
    Retrieve the registered seed for a run.

    Args:
        run_id: Run identifier.

    Returns:
        The seed integer.

    Raises:
        KeyError: If run_id has not been registered.
    """
    if run_id not in _SEED_REGISTRY:
        raise KeyError(f"No seed registered for run_id='{run_id}'")
    return _SEED_REGISTRY[run_id]


def list_seeds() -> dict[str, int]:
    """Return a copy of the entire seed registry."""
    return dict(_SEED_REGISTRY)


def build_run_id(trial_cfg: TrialConfig) -> str:
    """
    Construct a unique, human-readable run ID from config hash + seed.

    Format: "<config_hash>-s<seed>"
    Example: "a3f7c1d2-s42"

    Args:
        trial_cfg: Validated TrialConfig.

    Returns:
        Run ID string.
    """
    cfg_hash = hash_config(trial_cfg)
    seed = trial_cfg.simulation.seed
    return f"{cfg_hash}-s{seed}"
