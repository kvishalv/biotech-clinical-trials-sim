"""
api/routes/biomarkers.py
─────────────────────────
FastAPI routes for biomarker data access and drift detection.

Endpoints:
  GET  /biomarkers/{run_id}                 — Biomarker endpoint summary for a run
  GET  /biomarkers/{run_id}/{biomarker}     — Per-arm trajectory for one biomarker
  POST /drift                               — Run distribution drift tests
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import DriftRequest, DriftResponse
from src.api.routes.simulation import _result_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/biomarkers", tags=["biomarkers"])


@router.get("/{run_id}")
async def get_biomarker_summary(run_id: str) -> dict:
    """
    Return a summary of all biomarker endpoint trajectories for a simulation run.

    Returns per-arm mean ± SD at weeks 0, 13, 26, 39, 52 for every biomarker.
    """
    if run_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"run_id='{run_id}' not found.")

    result = _result_cache[run_id]
    if not hasattr(result, "biomarker_df"):
        raise HTTPException(status_code=422, detail="No biomarker data for this run_id.")

    bio_df = result.biomarker_df
    check_weeks = [w for w in [0, 13, 26, 39, 52] if w in bio_df["week"].values]

    summary = {}
    for biomarker in bio_df["biomarker"].unique():
        sub = bio_df[
            (bio_df["biomarker"] == biomarker)
            & (bio_df["week"].isin(check_weeks))
            & (bio_df["observed"])
        ]
        stats = (
            sub.groupby(["week", "arm"])["value"]
            .agg(mean="mean", std="std", n="count")
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        )
        summary[biomarker] = stats

    return {"run_id": run_id, "biomarkers": summary}


@router.get("/{run_id}/{biomarker}")
async def get_biomarker_trajectory(run_id: str, biomarker: str) -> dict:
    """
    Return the full per-arm weekly trajectory for a specific biomarker.
    """
    if run_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"run_id='{run_id}' not found.")

    result = _result_cache[run_id]
    if not hasattr(result, "biomarker_df"):
        raise HTTPException(status_code=422, detail="No biomarker data for this run_id.")

    bio_df = result.biomarker_df

    if biomarker not in bio_df["biomarker"].unique():
        available = sorted(bio_df["biomarker"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Biomarker '{biomarker}' not found. Available: {available}",
        )

    sub = bio_df[(bio_df["biomarker"] == biomarker) & (bio_df["observed"])]
    trajectory = (
        sub.groupby(["week", "arm"])["value"]
        .agg(mean="mean", std="std", n="count")
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "run_id": run_id,
        "biomarker": biomarker,
        "trajectory": trajectory,
    }


@router.post("/drift", response_model=DriftResponse)
async def check_drift(req: DriftRequest) -> DriftResponse:
    """
    Run distribution drift tests between two simulation runs.

    Compares patient demographics and (optionally) biomarker means.
    """
    for rid in [req.reference_run_id, req.new_run_id]:
        if rid not in _result_cache:
            raise HTTPException(
                status_code=404,
                detail=f"run_id='{rid}' not found. Run POST /simulate first.",
            )

    ref = _result_cache[req.reference_run_id]
    new = _result_cache[req.new_run_id]

    if not (hasattr(ref, "patient_df") and hasattr(new, "patient_df")):
        raise HTTPException(status_code=422, detail="Patient DataFrames not available for one or both runs.")

    try:
        from src.utils.drift_detector import DriftDetector

        detector = DriftDetector()
        report = detector.check_patient_drift(ref.patient_df, new.patient_df)

        if req.check_biomarkers and hasattr(ref, "biomarker_df") and hasattr(new, "biomarker_df"):
            bio_report = detector.check_biomarker_drift(
                ref.biomarker_df,
                new.biomarker_df,
                check_weeks=[0, 26],
            )
            report.results.extend(bio_report.results)
            report.n_tested += bio_report.n_tested
            report.n_drifted += bio_report.n_drifted

        return DriftResponse(
            reference_run_id=req.reference_run_id,
            new_run_id=req.new_run_id,
            **report.to_dict(),
        )

    except Exception as exc:
        logger.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
