"""
api/routes/agents.py
──────────────────────
FastAPI routes for the LLM agent layer.

Endpoints:
  POST /agents/lint        — Lint a trial protocol config
  POST /agents/narrate     — Generate cohort narrative for a run
  POST /agents/interpret   — Interpret and critique simulation results
  POST /agents/plan        — Generate a next-experiment plan
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    InterpretRequest,
    InterpretResponse,
    LintRequest,
    LintResponse,
    NarrateRequest,
    NarrateResponse,
)
from src.api.routes.simulation import _result_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/lint", response_model=LintResponse)
async def lint_protocol(req: LintRequest) -> LintResponse:
    """
    Lint a trial protocol configuration file.

    Runs both deterministic rule-checks and an LLM-powered review.
    Returns ERRORS, WARNINGS, and SUGGESTIONS.
    """
    try:
        from src.utils.config import load_trial_config
        from src.agents.protocol_linter import ProtocolLinterAgent

        trial_cfg = load_trial_config(req.trial_config_path)
        agent = ProtocolLinterAgent()
        report = agent.lint(trial_cfg.model_dump())
        return LintResponse(**report.to_dict())

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Lint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/narrate", response_model=NarrateResponse)
async def narrate_cohort(req: NarrateRequest) -> NarrateResponse:
    """
    Generate a clinical narrative for a previously simulated cohort.

    Requires a run_id from POST /simulate.
    """
    if req.run_id not in _result_cache:
        raise HTTPException(
            status_code=404,
            detail=f"run_id='{req.run_id}' not found. Run POST /simulate first.",
        )

    result = _result_cache[req.run_id]

    # Cached result might be a full TrialResult or a lightweight dict
    cohort_summary = (
        result.cohort_summary
        if hasattr(result, "cohort_summary")
        else result.get("cohort_summary", {})
    )

    if not cohort_summary:
        raise HTTPException(status_code=422, detail="No cohort summary available for this run_id.")

    try:
        from src.agents.cohort_narrator import CohortNarratorAgent

        agent = CohortNarratorAgent()
        narrative = agent.narrate(cohort_summary)
        return NarrateResponse(run_id=req.run_id, narrative=narrative)

    except Exception as exc:
        logger.exception("Narrate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/interpret", response_model=InterpretResponse)
async def interpret_results(req: InterpretRequest) -> InterpretResponse:
    """
    Interpret and critique simulation results for a completed run.

    Requires a run_id from POST /simulate.
    """
    if req.run_id not in _result_cache:
        raise HTTPException(
            status_code=404,
            detail=f"run_id='{req.run_id}' not found. Run POST /simulate first.",
        )

    result = _result_cache[req.run_id]

    summary = (
        result.to_summary_dict()
        if hasattr(result, "to_summary_dict")
        else result
    )

    try:
        from src.agents.result_interpreter import ResultInterpreterAgent

        agent = ResultInterpreterAgent()
        interpretation = agent.interpret(summary)
        return InterpretResponse(run_id=req.run_id, interpretation=interpretation)

    except Exception as exc:
        logger.exception("Interpret failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/plan")
async def plan_experiments(body: dict) -> dict:
    """
    Generate a next-experiment plan based on a research goal and current results.

    Body: {"goal": "...", "run_id": "..."}
    """
    run_id = body.get("run_id", "")
    goal = body.get("goal", "Maximise treatment effect on epigenetic age acceleration.")

    current_results = {}
    if run_id and run_id in _result_cache:
        result = _result_cache[run_id]
        current_results = (
            result.to_summary_dict()
            if hasattr(result, "to_summary_dict")
            else result
        )

    try:
        from src.agents.result_interpreter import ExperimentPlannerAgent

        planner = ExperimentPlannerAgent()
        plan = planner.plan(goal=goal, current_results=current_results)
        return {"run_id": run_id, "goal": goal, "plan": plan}

    except Exception as exc:
        logger.exception("Plan generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
