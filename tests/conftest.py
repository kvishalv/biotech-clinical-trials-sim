"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the biotech simulation test suite.

All fixtures use small cohorts and short simulation durations to keep
the test suite fast. All RNG seeds are fixed for determinism.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable without installing the package
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def trial_config():
    """Load the default trial config from configs/trial_config.yaml."""
    from src.utils.config import load_trial_config

    cfg_path = ROOT / "configs" / "trial_config.yaml"
    return load_trial_config(cfg_path)


@pytest.fixture(scope="session")
def biomarker_config():
    """Load the default biomarker config from configs/biomarker_config.yaml."""
    from src.utils.config import load_biomarker_config

    cfg_path = ROOT / "configs" / "biomarker_config.yaml"
    return load_biomarker_config(cfg_path)


@pytest.fixture(scope="session")
def small_trial_config(trial_config):
    """
    A small cohort config suitable for fast unit tests.
    50 patients, 12 weeks, seed=42.
    """
    new_sim = trial_config.simulation.model_copy(update={"seed": 42, "n_weeks": 12})
    new_cohort = trial_config.cohort.model_copy(update={"n_patients": 50})
    return trial_config.model_copy(update={"simulation": new_sim, "cohort": new_cohort})


# ── Patient / cohort fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="session")
def patient_generator(small_trial_config, biomarker_config):
    """PatientGenerator with small cohort config."""
    from src.simulation.patient_generator import PatientGenerator
    from src.simulation.biomarker_models import extract_re_stds, extract_baseline_params

    return PatientGenerator(
        cohort_cfg=small_trial_config.cohort,
        biomarker_names=list(biomarker_config.biomarkers.keys()),
        seed=42,
    )


@pytest.fixture(scope="session")
def patients(patient_generator, biomarker_config):
    """50 generated patients with all biomarker fixtures."""
    from src.simulation.biomarker_models import extract_re_stds, extract_baseline_params

    return patient_generator.generate(
        biomarker_re_stds=extract_re_stds(biomarker_config),
        biomarker_params=extract_baseline_params(biomarker_config),
        n_weeks=12,
    )


@pytest.fixture(scope="session")
def patient_df(patient_generator, patients):
    """Patient-level DataFrame."""
    return patient_generator.to_dataframe(patients)


# ── Biomarker simulation fixtures ────────────────────────────────────────────


@pytest.fixture(scope="session")
def biomarker_df(biomarker_config, patients):
    """Short longitudinal biomarker DataFrame (12 weeks)."""
    from src.simulation.biomarker_models import BiomarkerSimulator

    sim = BiomarkerSimulator(biomarker_config, seed=42, burnin_weeks=2)
    return sim.simulate_cohort(patients, n_weeks=12)


# ── Trial result fixture ─────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def trial_result(small_trial_config, biomarker_config):
    """Full TrialResult for a small simulation run."""
    from src.simulation.trial_simulator import TrialSimulator

    sim = TrialSimulator(small_trial_config, biomarker_config)
    return sim.run()


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def api_client():
    """HTTPX test client for the FastAPI app."""
    from httpx import AsyncClient
    from src.api.main import app

    # Return the app itself — individual tests create their own async client
    return app
