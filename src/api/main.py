"""
api/main.py
────────────
FastAPI application entry point for the biotech clinical trials simulation platform.

Endpoints registered:
  GET  /health                              — Health check
  POST /simulate                            — Run a simulation
  GET  /simulate/{run_id}                   — Get cached result
  POST /simulate/sweep                      — Ray distributed sweep
  GET  /biomarkers/{run_id}                 — Biomarker summaries
  GET  /biomarkers/{run_id}/{biomarker}     — Single biomarker trajectory
  POST /biomarkers/drift                    — Drift detection
  POST /agents/lint                         — Protocol linting
  POST /agents/narrate                      — Cohort narration
  POST /agents/interpret                    — Result interpretation
  POST /agents/plan                         — Experiment planning

OpenAPI docs: http://localhost:8000/docs
ReDoc:        http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Ray is optional — the API degrades gracefully without it
# (distributed sweeps unavailable, single simulations still work)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    RAY_AVAILABLE = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import simulation, biomarkers, agents
from src.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Startup:  Initialise Ray local cluster (if not already running).
    Shutdown: Gracefully shut down Ray.
    """
    logger.info("Starting biotech-sim API server...")

    # Initialise Ray — gracefully skip if not installed or fails
    if not RAY_AVAILABLE:
        logger.warning("Ray not installed — distributed sweeps unavailable.")
    else:
        try:
            if not ray.is_initialized():
                ray.init(
                    address="local",
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
                logger.info("Ray local cluster initialised.")
            else:
                logger.info("Ray already initialised — reusing.")
        except Exception as exc:
            logger.warning(f"Ray init skipped: {exc}. Distributed sweeps will be unavailable.")

    yield  # Application runs here

    logger.info("Shutting down biotech-sim API server...")
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
        logger.info("Ray cluster shut down.")


# ──────────────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Biotech Clinical Trials Simulator",
        description=(
            "Open-source distributed simulation platform for biotech clinical trials "
            "with causal AI, longevity biomarker modelling, and agentic LLM interpretation. "
            "All patient data is synthetic."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route groups
    app.include_router(simulation.router)
    app.include_router(biomarkers.router)
    app.include_router(agents.router)

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Application health check."""
        return HealthResponse(
            status="ok",
            version=APP_VERSION,
            ray_initialized=RAY_AVAILABLE and ray.is_initialized(),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
