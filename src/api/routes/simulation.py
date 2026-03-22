"""
api/routes/simulation.py
─────────────────────────
FastAPI routes for simulation execution.

Endpoints:
  POST /simulate          — Run a single simulation, return full result summary
  POST /sweep             — Run a Ray-distributed parameter sweep
  GET  /trials/{run_id}   — Retrieve a cached trial result summary
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    RunSimulationRequest,
    SimulationResponse,
    SweepRequest,
    SweepResponse,
)
from src.utils.config import load_biomarker_config, load_trial_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulate", tags=["simulation"])

# In-memory result cache — replace with Redis or Postgres in production
_result_cache: dict[str, Any] = {}


@router.post("", response_model=SimulationResponse)
async def run_simulation(req: RunSimulationRequest) -> SimulationResponse:
    """
    Run a single clinical trial simulation.

    Returns a full result summary including ATE estimates, cohort stats,
    survival analysis, and optionally LLM-generated narratives.
    """
    try:
        trial_cfg = load_trial_config(req.trial_config_path)
        bio_cfg = load_biomarker_config(req.biomarker_config_path)

        # Override seed and cohort size from request
        new_sim = trial_cfg.simulation.model_copy(update={"seed": req.seed, "n_weeks": req.n_weeks})
        new_cohort = trial_cfg.cohort.model_copy(update={"n_patients": req.n_patients})
        trial_cfg = trial_cfg.model_copy(update={"simulation": new_sim, "cohort": new_cohort})

        from src.simulation.trial_simulator import TrialSimulator

        sim = TrialSimulator(trial_cfg, bio_cfg)
        result = sim.run()

        # Cache result for agent routes
        _result_cache[result.run_id] = result

        # Optionally run LLM agents
        narrative = None
        interpretation = None
        if req.run_agents:
            try:
                from src.agents.cohort_narrator import CohortNarratorAgent
                from src.agents.result_interpreter import ResultInterpreterAgent

                narrator = CohortNarratorAgent()
                interpreter = ResultInterpreterAgent()
                narrative = narrator.narrate(result.cohort_summary)
                interpretation = interpreter.interpret(result.to_summary_dict())
            except Exception as agent_exc:
                logger.warning(f"Agent calls failed: {agent_exc}")

        summary = result.to_summary_dict()
        return SimulationResponse(
            run_id=result.run_id,
            trial_name=result.trial_name,
            seed=result.seed,
            n_patients=result.n_patients,
            n_weeks=result.n_weeks,
            elapsed_seconds=result.elapsed_seconds,
            cohort_summary=summary["cohort_summary"],
            ate_results=summary["ate_results"],
            survival_summary=summary["survival_summary"],
            logistic_summary=summary["logistic_summary"],
            continuous_summary=summary["continuous_summary"],
            uplift_summary=summary["uplift_summary"],
            agent_narrative=narrative,
            agent_interpretation=interpretation,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{run_id}", response_model=SimulationResponse)
async def get_trial(run_id: str) -> SimulationResponse:
    """
    Retrieve a cached trial result by run_id.
    """
    if run_id not in _result_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for run_id='{run_id}'. Run POST /simulate first.",
        )
    result = _result_cache[run_id]
    summary = result.to_summary_dict()
    return SimulationResponse(
        run_id=result.run_id,
        trial_name=result.trial_name,
        seed=result.seed,
        n_patients=result.n_patients,
        n_weeks=result.n_weeks,
        elapsed_seconds=result.elapsed_seconds,
        cohort_summary=summary["cohort_summary"],
        ate_results=summary["ate_results"],
        survival_summary=summary["survival_summary"],
        logistic_summary=summary["logistic_summary"],
        continuous_summary=summary["continuous_summary"],
        uplift_summary=summary["uplift_summary"],
    )


@router.post("/sweep", response_model=SweepResponse)
async def run_sweep(req: SweepRequest) -> SweepResponse:
    """
    Run a distributed parameter sweep using Ray.

    Each seed produces one independent simulation run.
    Returns aggregated ATE results across all runs.
    """
    try:
        import ray

        trial_cfg = load_trial_config(req.trial_config_path)
        bio_cfg = load_biomarker_config(req.biomarker_config_path)

        # Apply request overrides
        new_sim = trial_cfg.simulation.model_copy(update={"n_weeks": req.n_weeks})
        new_cohort = trial_cfg.cohort.model_copy(update={"n_patients": req.n_patients})
        trial_cfg = trial_cfg.model_copy(update={"simulation": new_sim, "cohort": new_cohort})

        from src.distributed.ray_runner import RayClusterManager

        manager = RayClusterManager(address="local")
        if not ray.is_initialized():
            manager.init()

        sweep_result = manager.run_sweep(trial_cfg, bio_cfg, seeds=req.seeds)

        # Cache individual results
        for run_id, pat_df in sweep_result.patient_dfs.items():
            # Store lightweight cached entry (no full TrialResult object)
            _result_cache[run_id] = {"patient_df": pat_df}

        return SweepResponse(
            n_successful=sweep_result.n_successful,
            n_failed=sweep_result.n_failed,
            run_ids=list(sweep_result.patient_dfs.keys()),
            errors=sweep_result.errors,
            ate_summary=sweep_result.ate_summary_df().to_dict(orient="records"),
        )

    except Exception as exc:
        logger.exception("Sweep failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
