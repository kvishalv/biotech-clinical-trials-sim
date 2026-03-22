"""
api/schemas.py
───────────────
Pydantic v2 request and response schemas for the FastAPI serving layer.

All schemas validate inputs strictly — extra fields are forbidden.
Response schemas mirror TrialResult and agent output structures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────────────────


class RunSimulationRequest(BaseModel):
    """Request body for POST /simulate"""

    seed: int = Field(default=42, description="RNG seed for reproducibility")
    n_patients: int = Field(default=200, ge=10, le=5000)
    n_weeks: int = Field(default=52, ge=4, le=260)
    trial_config_path: str = Field(default="configs/trial_config.yaml")
    biomarker_config_path: str = Field(default="configs/biomarker_config.yaml")
    run_agents: bool = Field(
        default=False,
        description="Whether to run LLM agents on the result (requires ANTHROPIC_API_KEY)",
    )


class SweepRequest(BaseModel):
    """Request body for POST /sweep"""

    seeds: list[int] = Field(default=[42, 43, 44, 45], min_length=1, max_length=50)
    n_patients: int = Field(default=200, ge=10, le=5000)
    n_weeks: int = Field(default=52, ge=4, le=260)
    trial_config_path: str = Field(default="configs/trial_config.yaml")
    biomarker_config_path: str = Field(default="configs/biomarker_config.yaml")


class LintRequest(BaseModel):
    """Request body for POST /agents/lint"""

    trial_config_path: str = Field(default="configs/trial_config.yaml")


class NarrateRequest(BaseModel):
    """Request body for POST /agents/narrate"""

    run_id: str = Field(description="run_id from a completed simulation")


class InterpretRequest(BaseModel):
    """Request body for POST /agents/interpret"""

    run_id: str = Field(description="run_id from a completed simulation")


class DriftRequest(BaseModel):
    """Request body for POST /drift"""

    reference_run_id: str
    new_run_id: str
    check_biomarkers: bool = Field(default=True)


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────


class SimulationResponse(BaseModel):
    """Response from POST /simulate"""

    run_id: str
    trial_name: str
    seed: int
    n_patients: int
    n_weeks: int
    elapsed_seconds: float
    cohort_summary: dict[str, Any]
    ate_results: dict[str, Any]
    survival_summary: dict[str, Any]
    logistic_summary: dict[str, Any]
    continuous_summary: dict[str, Any]
    uplift_summary: dict[str, Any]
    agent_narrative: str | None = None
    agent_interpretation: str | None = None


class SweepResponse(BaseModel):
    """Response from POST /sweep"""

    n_successful: int
    n_failed: int
    run_ids: list[str]
    errors: list[dict[str, str]]
    ate_summary: list[dict[str, Any]]


class LintResponse(BaseModel):
    """Response from POST /agents/lint"""

    rule_errors: list[str]
    llm_report: str
    has_errors: bool


class NarrateResponse(BaseModel):
    """Response from POST /agents/narrate"""

    run_id: str
    narrative: str


class InterpretResponse(BaseModel):
    """Response from POST /agents/interpret"""

    run_id: str
    interpretation: str


class DriftResponse(BaseModel):
    """Response from POST /drift"""

    reference_run_id: str
    new_run_id: str
    n_drifted: int
    n_tested: int
    overall_drift: bool
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Response from GET /health"""

    status: str
    version: str
    ray_initialized: bool
