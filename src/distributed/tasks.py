"""
distributed/tasks.py
──────────────────────
Ray remote task definitions for parallelised clinical trial simulations.

Design:
  - Each @ray.remote function is a pure function of (config, seed) → result dict
  - No shared mutable state between tasks — Ray actors are not used here because
    the simulation kernel is stateless; actors would be overkill and add failure surface
  - Results are returned as JSON-serialisable dicts (not DataFrames) to avoid
    Ray's object store serialisation overhead for large pandas objects
  - Large DataFrames (biomarker_df) are returned as Parquet bytes via io.BytesIO
    so they can be reassembled efficiently on the driver

Sweep strategy:
  - Parameter sweeps are expressed as a list of (trial_cfg, seed) pairs
  - Ray distributes these across available workers automatically
  - Each worker reconstructs its own TrialSimulator from the config — no object sharing
"""

from __future__ import annotations

import io
from typing import Any

import ray

from src.utils.config import TrialConfig, BiomarkerConfig


# ──────────────────────────────────────────────────────────────────────────────
# Remote simulation task
# ──────────────────────────────────────────────────────────────────────────────


@ray.remote
def run_simulation_task(
    trial_cfg_dict: dict[str, Any],
    biomarker_cfg_dict: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """
    Run a single clinical trial simulation as a Ray remote task.

    Args are passed as plain dicts (not Pydantic models) because Ray's serialiser
    handles dicts more efficiently than arbitrary Python objects across workers.

    Args:
        trial_cfg_dict:    Trial config as a plain dict (model_dump() output).
        biomarker_cfg_dict: Biomarker config as a plain dict.
        seed:              RNG seed for this run.

    Returns:
        Dict containing:
            - run_id:        Versioned run identifier
            - summary:       JSON-serialisable TrialResult.to_summary_dict()
            - biomarker_parquet: Parquet bytes of longitudinal biomarker DataFrame
            - patient_parquet:   Parquet bytes of patient-level DataFrame
            - error:         None on success, error string on failure
    """
    try:
        # Reconstruct Pydantic configs inside the worker process
        trial_cfg = TrialConfig(**trial_cfg_dict)

        # Override seed for this specific run
        new_sim = trial_cfg.simulation.model_copy(update={"seed": seed})
        trial_cfg = trial_cfg.model_copy(update={"simulation": new_sim})

        bio_cfg = BiomarkerConfig(**biomarker_cfg_dict)

        # Local import avoids serialising heavy modules into the Ray object store
        from src.simulation.trial_simulator import TrialSimulator

        sim = TrialSimulator(trial_cfg, bio_cfg)
        result = sim.run()

        # Serialise DataFrames to Parquet bytes for efficient transport
        bio_buf = io.BytesIO()
        result.biomarker_df.to_parquet(bio_buf, index=False)

        pat_buf = io.BytesIO()
        result.patient_df.to_parquet(pat_buf, index=False)

        return {
            "run_id": result.run_id,
            "summary": result.to_summary_dict(),
            "biomarker_parquet": bio_buf.getvalue(),
            "patient_parquet": pat_buf.getvalue(),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        # Never let a worker crash silently — return structured error
        import traceback

        return {
            "run_id": f"FAILED-s{seed}",
            "summary": {},
            "biomarker_parquet": b"",
            "patient_parquet": b"",
            "error": traceback.format_exc(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Remote biomarker-only task (lighter weight, for sensitivity sweeps)
# ──────────────────────────────────────────────────────────────────────────────


@ray.remote
def run_biomarker_task(
    trial_cfg_dict: dict[str, Any],
    biomarker_cfg_dict: dict[str, Any],
    seed: int,
    biomarker_name: str,
) -> dict[str, Any]:
    """
    Simulate a single biomarker trajectory for a cohort.

    Lighter than run_simulation_task — skips outcome modelling.
    Useful for rapid parameter sweeps over biomarker configs.

    Args:
        trial_cfg_dict:    Trial config dict.
        biomarker_cfg_dict: Biomarker config dict.
        seed:              RNG seed.
        biomarker_name:    Which biomarker to return (filters output).

    Returns:
        Dict with 'run_id', 'biomarker_parquet', 'endpoint_summary', 'error'.
    """
    try:
        from src.utils.config import TrialConfig, BiomarkerConfig
        from src.simulation.patient_generator import PatientGenerator
        from src.simulation.biomarker_models import (
            BiomarkerSimulator,
            extract_re_stds,
            extract_baseline_params,
        )

        trial_cfg = TrialConfig(**trial_cfg_dict)
        new_sim = trial_cfg.simulation.model_copy(update={"seed": seed})
        trial_cfg = trial_cfg.model_copy(update={"simulation": new_sim})
        bio_cfg = BiomarkerConfig(**biomarker_cfg_dict)

        biomarker_names = list(bio_cfg.biomarkers.keys())

        pg = PatientGenerator(trial_cfg.cohort, biomarker_names, seed=seed)
        patients = pg.generate(
            biomarker_re_stds=extract_re_stds(bio_cfg),
            biomarker_params=extract_baseline_params(bio_cfg),
            n_weeks=trial_cfg.simulation.n_weeks,
        )

        bio_sim = BiomarkerSimulator(bio_cfg, seed=seed, burnin_weeks=trial_cfg.simulation.burnin_weeks)
        biomarker_df = bio_sim.simulate_cohort(patients, n_weeks=trial_cfg.simulation.n_weeks)

        # Filter to requested biomarker
        filtered = biomarker_df[biomarker_df["biomarker"] == biomarker_name]
        endpoint_summary = bio_sim.endpoint_summary(biomarker_df, biomarker_name).to_dict(orient="records")

        buf = io.BytesIO()
        filtered.to_parquet(buf, index=False)

        return {
            "run_id": f"bio-{seed}-{biomarker_name}",
            "biomarker_parquet": buf.getvalue(),
            "endpoint_summary": endpoint_summary,
            "error": None,
        }

    except Exception:  # noqa: BLE001
        import traceback

        return {
            "run_id": f"FAILED-bio-s{seed}",
            "biomarker_parquet": b"",
            "endpoint_summary": [],
            "error": traceback.format_exc(),
        }
