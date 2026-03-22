"""tests/test_config.py — Tests for utils/config.py"""

from __future__ import annotations

import hashlib
import pytest
from pathlib import Path
from src.utils.config import (
    load_trial_config,
    load_biomarker_config,
    hash_config,
    register_seed,
    get_seed,
    list_seeds,
    build_run_id,
    InclusionCriteria,
    CohortConfig,
)

ROOT = Path(__file__).parent.parent


class TestLoadConfigs:
    def test_trial_config_loads(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        assert cfg.cohort.n_patients > 0
        assert cfg.simulation.seed >= 0

    def test_biomarker_config_loads(self):
        cfg = load_biomarker_config(ROOT / "configs" / "biomarker_config.yaml")
        assert len(cfg.biomarkers) == 9, f"Expected 9 biomarkers, got {len(cfg.biomarkers)}"

    def test_all_9_biomarkers_present(self):
        cfg = load_biomarker_config(ROOT / "configs" / "biomarker_config.yaml")
        expected = {
            "inflammation_index", "metabolic_risk_index", "epigenetic_age_acceleration",
            "frailty_progression", "organ_reserve_score", "latent_mitochondrial_dysfunction",
            "immune_resilience", "sleep_circadian_disruption", "recovery_velocity",
        }
        assert set(cfg.biomarkers.keys()) == expected

    def test_arm_allocations_sum_to_one(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        total = sum(arm.allocation for arm in cfg.cohort.treatment_arms)
        assert abs(total - 1.0) < 0.01

    def test_trial_config_is_frozen(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        with pytest.raises(Exception):
            cfg.simulation = None  # type: ignore

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_trial_config("nonexistent/path.yaml")


class TestHashConfig:
    def test_hash_is_deterministic(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2

    def test_hash_length_is_8(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        assert len(hash_config(cfg)) == 8

    def test_different_seeds_produce_different_hashes(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        cfg2 = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update={"seed": 9999})})
        assert hash_config(cfg) != hash_config(cfg2)


class TestSeedRegistry:
    def test_register_and_get_seed(self):
        register_seed("test-run-1", 42)
        assert get_seed("test-run-1") == 42

    def test_get_unregistered_raises(self):
        with pytest.raises(KeyError):
            get_seed("nonexistent-run-xyz")

    def test_list_seeds_returns_copy(self):
        register_seed("test-run-2", 99)
        seeds = list_seeds()
        seeds["mutated"] = 0  # mutating the copy should not affect registry
        assert "mutated" not in list_seeds()


class TestInclusionCriteria:
    def test_age_range_invalid_raises(self):
        with pytest.raises(Exception):
            InclusionCriteria(min_age=80, max_age=45, min_bmi=18.5, max_bmi=35.0)

    def test_valid_criteria_passes(self):
        ic = InclusionCriteria(min_age=45, max_age=80, min_bmi=18.5, max_bmi=35.0)
        assert ic.min_age == 45


class TestBuildRunId:
    def test_run_id_format(self):
        cfg = load_trial_config(ROOT / "configs" / "trial_config.yaml")
        run_id = build_run_id(cfg)
        assert "-s" in run_id
        parts = run_id.split("-s")
        assert len(parts[0]) == 8  # config hash portion
        assert parts[1].isdigit()   # seed portion
