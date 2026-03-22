"""
distributed/ray_runner.py
──────────────────────────
Ray cluster orchestration and sweep launcher for clinical trial simulations.

Responsibilities:
  - Initialise / connect to a Ray cluster (local or remote)
  - Launch parameter sweeps as batches of @ray.remote tasks
  - Collect, reassemble, and validate results from workers
  - Provide a clean shutdown hook

Cluster modes:
  - "local":  ray.init() — spawns a local Ray cluster in the current process.
              Use this for development and single-machine runs.
  - "auto":   ray.init(address="auto") — connects to an existing cluster.
              Use this in production / Kubernetes environments.
  - Custom:   ray.init(address="ray://<host>:<port>") for Ray Serve / head node.

Extension points (noted in the spec):
  - Ray Serve can be layered on top for inference / simulation serving
  - GPU batch inference: pass num_gpus=1 to @ray.remote decorators
  - EHR event streams: add a streaming task that yields events via Ray actors
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import ray

from src.utils.config import BiomarkerConfig, TrialConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    """
    Aggregated results from a distributed parameter sweep.

    Attributes:
        summaries:       List of per-run summary dicts
        biomarker_dfs:   Dict {run_id: biomarker_df}
        patient_dfs:     Dict {run_id: patient_df}
        errors:          List of {run_id, error_msg} for failed tasks
        n_successful:    Count of completed runs
        n_failed:        Count of failed runs
    """

    summaries: list[dict[str, Any]] = field(default_factory=list)
    biomarker_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    patient_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    n_successful: int = 0
    n_failed: int = 0

    def combined_biomarker_df(self) -> pd.DataFrame:
        """Concatenate all biomarker DataFrames, adding run_id column."""
        if not self.biomarker_dfs:
            return pd.DataFrame()
        frames = []
        for run_id, df in self.biomarker_dfs.items():
            df = df.copy()
            df["run_id"] = run_id
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def ate_summary_df(self) -> pd.DataFrame:
        """Extract ATE results across all runs into a tidy DataFrame."""
        rows = []
        for summary in self.summaries:
            run_id = summary.get("run_id", "unknown")
            seed = summary.get("seed", -1)
            for endpoint, ate_data in summary.get("ate_results", {}).items():
                for contrast, stats in ate_data.items():
                    if isinstance(stats, dict) and "ate" in stats:
                        rows.append(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "endpoint": endpoint,
                                "contrast": contrast,
                                "ate": stats.get("ate"),
                                "p_value": stats.get("p_value"),
                                "ci_lower": stats.get("ci_lower"),
                                "ci_upper": stats.get("ci_upper"),
                            }
                        )
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Ray cluster manager
# ──────────────────────────────────────────────────────────────────────────────


class RayClusterManager:
    """
    Manages Ray cluster lifecycle and distributes simulation sweeps.

    Args:
        address:     Ray cluster address — "local", "auto", or "ray://<host>:<port>".
        num_cpus:    Override for local cluster CPU count (None = all available).
        num_gpus:    Override for local cluster GPU count (None = 0).
        log_to_driver: Whether Ray worker logs are forwarded to the driver.
    """

    def __init__(
        self,
        address: str = "local",
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        log_to_driver: bool = True,
    ) -> None:
        self.address = address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.log_to_driver = log_to_driver
        self._initialised = False

    def init(self) -> "RayClusterManager":
        """
        Initialise the Ray cluster.

        - "local": spawns an embedded cluster (development mode)
        - "auto" or URL: connects to an existing cluster (production mode)

        Returns:
            Self (for chaining).
        """
        if ray.is_initialized():
            logger.info("Ray already initialised — reusing existing cluster.")
            self._initialised = True
            return self

        init_kwargs: dict[str, Any] = {
            "log_to_driver": self.log_to_driver,
            "ignore_reinit_error": True,
        }

        if self.address == "local":
            if self.num_cpus is not None:
                init_kwargs["num_cpus"] = self.num_cpus
            if self.num_gpus is not None:
                init_kwargs["num_gpus"] = self.num_gpus
            ray.init(**init_kwargs)
            logger.info("Ray initialised in LOCAL mode.")
        else:
            ray.init(address=self.address, **init_kwargs)
            logger.info(f"Ray connected to cluster at {self.address}.")

        self._initialised = True
        return self

    def shutdown(self) -> None:
        """Shut down the Ray cluster (no-op if not initialised)."""
        if self._initialised and ray.is_initialized():
            ray.shutdown()
            self._initialised = False
            logger.info("Ray cluster shut down.")

    def cluster_resources(self) -> dict[str, Any]:
        """Return current cluster resource availability."""
        if not ray.is_initialized():
            return {}
        return dict(ray.cluster_resources())

    # ── Sweep launcher ─────────────────────────────────────────────────────────

    def run_sweep(
        self,
        trial_cfg: TrialConfig,
        biomarker_cfg: BiomarkerConfig,
        seeds: list[int],
        batch_size: int = 8,
    ) -> SweepResult:
        """
        Launch a parameter sweep: run the simulation once per seed in parallel.

        Tasks are batched so Ray's object store doesn't get overwhelmed on large
        sweeps. Each batch of `batch_size` tasks is submitted and collected before
        the next batch starts.

        Args:
            trial_cfg:     Base trial configuration (seed will be overridden per run).
            biomarker_cfg: Biomarker configuration (same for all runs in sweep).
            seeds:         List of RNG seeds — one simulation run per seed.
            batch_size:    Max concurrent tasks per batch.

        Returns:
            SweepResult with all run outputs and any errors.
        """
        if not self._initialised:
            raise RuntimeError("Call .init() before running a sweep.")

        from src.distributed.tasks import run_simulation_task

        # Serialise configs once — reused across all task submissions
        trial_dict = trial_cfg.model_dump()
        bio_dict = biomarker_cfg.model_dump()

        sweep = SweepResult()
        total = len(seeds)

        logger.info(f"Starting sweep: {total} runs across seeds {seeds}")

        for batch_start in range(0, total, batch_size):
            batch_seeds = seeds[batch_start : batch_start + batch_size]
            logger.info(
                f"  Batch {batch_start // batch_size + 1}: "
                f"seeds {batch_seeds}"
            )

            # Submit this batch of tasks
            refs = [
                run_simulation_task.remote(trial_dict, bio_dict, s)
                for s in batch_seeds
            ]

            # Block until all tasks in the batch complete
            raw_results = ray.get(refs)

            for raw in raw_results:
                if raw.get("error"):
                    logger.error(
                        f"Task {raw['run_id']} failed: {raw['error'][:200]}"
                    )
                    sweep.errors.append(
                        {"run_id": raw["run_id"], "error": raw["error"]}
                    )
                    sweep.n_failed += 1
                else:
                    sweep.summaries.append(raw["summary"])
                    sweep.n_successful += 1

                    # Deserialise DataFrames from Parquet bytes
                    if raw["biomarker_parquet"]:
                        bio_df = pd.read_parquet(io.BytesIO(raw["biomarker_parquet"]))
                        sweep.biomarker_dfs[raw["run_id"]] = bio_df

                    if raw["patient_parquet"]:
                        pat_df = pd.read_parquet(io.BytesIO(raw["patient_parquet"]))
                        sweep.patient_dfs[raw["run_id"]] = pat_df

        logger.info(
            f"Sweep complete: {sweep.n_successful} successful, {sweep.n_failed} failed."
        )
        return sweep

    def run_biomarker_sweep(
        self,
        trial_cfg: TrialConfig,
        biomarker_cfg: BiomarkerConfig,
        seeds: list[int],
        biomarker_name: str,
        batch_size: int = 16,
    ) -> SweepResult:
        """
        Lightweight sweep: simulate a single biomarker across many seeds.

        Useful for sensitivity analysis on biomarker parameters without
        running the full outcome modelling pipeline.

        Args:
            trial_cfg:      Base trial config.
            biomarker_cfg:  Biomarker config.
            seeds:          List of seeds.
            biomarker_name: Which biomarker to simulate.
            batch_size:     Concurrent task limit.

        Returns:
            SweepResult with biomarker_dfs populated.
        """
        if not self._initialised:
            raise RuntimeError("Call .init() before running a sweep.")

        from src.distributed.tasks import run_biomarker_task

        trial_dict = trial_cfg.model_dump()
        bio_dict = biomarker_cfg.model_dump()
        sweep = SweepResult()

        for batch_start in range(0, len(seeds), batch_size):
            batch = seeds[batch_start : batch_start + batch_size]
            refs = [
                run_biomarker_task.remote(trial_dict, bio_dict, s, biomarker_name)
                for s in batch
            ]
            for raw in ray.get(refs):
                if raw.get("error"):
                    sweep.errors.append({"run_id": raw["run_id"], "error": raw["error"]})
                    sweep.n_failed += 1
                else:
                    if raw["biomarker_parquet"]:
                        df = pd.read_parquet(io.BytesIO(raw["biomarker_parquet"]))
                        sweep.biomarker_dfs[raw["run_id"]] = df
                    sweep.summaries.append({"endpoint_summary": raw.get("endpoint_summary", [])})
                    sweep.n_successful += 1

        return sweep
