"""
tracking/experiment_tracker.py
────────────────────────────────
MLflow-backed experiment tracker for clinical trial simulation runs.

Responsibilities:
  - Log every simulation run as an MLflow experiment with full config versioning
  - Track: config hash, seed, ATE, p-values, cohort stats, elapsed time
  - Store config YAML and biomarker DataFrames as MLflow artifacts
  - Provide a query interface for past runs (by config hash, seed, endpoint)
  - Wrap MLflow so it can be swapped for another backend (W&B, DVC) later

Design:
  - The tracker is injected into TrialSimulator — it is not a singleton
  - If MLflow is unreachable (URI wrong, server down), logging silently degrades
    and a warning is emitted — the simulation still runs
  - Config hash is logged as an MLflow tag so runs with the same config are
    grouped automatically in the MLflow UI
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — mlflow is optional; tracker degrades gracefully without it
try:
    import mlflow
    import mlflow.data
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — experiment tracking disabled.")


class ExperimentTracker:
    """
    MLflow-backed experiment tracker.

    Args:
        tracking_uri:    MLflow server URI (e.g., "http://localhost:5000" or
                         "sqlite:///mlruns.db" for local file tracking).
        experiment_name: MLflow experiment name. Created if it doesn't exist.
        log_artifacts:   Whether to log DataFrames as Parquet artifacts.
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "biotech-sim",
        log_artifacts: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_artifacts = log_artifacts
        self._available = MLFLOW_AVAILABLE
        self._experiment_id: str | None = None

        if self._available:
            self._setup()

    def _setup(self) -> None:
        """Configure MLflow tracking URI and create experiment if needed."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self._experiment_id = experiment.experiment_id
            logger.info(
                f"MLflow tracker ready: uri={self.tracking_uri}, "
                f"experiment='{self.experiment_name}', id={self._experiment_id}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MLflow setup failed: {exc}. Tracking disabled.")
            self._available = False

    def log_trial_result(self, result: Any) -> str | None:
        """
        Log a TrialResult to MLflow.

        Logs:
          - Params: seed, n_patients, n_weeks, config_hash, trial_name
          - Metrics: ATE, p_value, responder_rate, elapsed_seconds per endpoint
          - Tags: run_id, config_hash
          - Artifacts: biomarker_df.parquet, patient_df.parquet, summary.json

        Args:
            result: TrialResult instance (avoids circular import — typed as Any).

        Returns:
            MLflow run_id string, or None if logging is unavailable.
        """
        if not self._available:
            logger.debug("MLflow unavailable — skipping log_trial_result.")
            return None

        try:
            with mlflow.start_run(experiment_id=self._experiment_id) as run:
                # ── Tags ──────────────────────────────────────────────────────
                mlflow.set_tags({
                    "run_id": result.run_id,
                    "config_hash": result.config_hash,
                    "trial_name": result.trial_name,
                })

                # ── Params ────────────────────────────────────────────────────
                mlflow.log_params({
                    "seed": result.seed,
                    "n_patients": result.n_patients,
                    "n_weeks": result.n_weeks,
                    "config_hash": result.config_hash,
                    "trial_name": result.trial_name,
                })

                # ── Metrics ───────────────────────────────────────────────────
                mlflow.log_metric("elapsed_seconds", result.elapsed_seconds)

                # ATE per endpoint
                for endpoint, ate_data in result.ate_results.items():
                    safe_ep = endpoint.replace(" ", "_")[:40]
                    for contrast, stats in ate_data.items():
                        if isinstance(stats, dict) and "ate" in stats:
                            safe_c = contrast.replace(" ", "_")
                            prefix = f"{safe_ep}.{safe_c}"
                            if stats.get("ate") is not None:
                                mlflow.log_metric(f"{prefix}.ate", stats["ate"])
                            if stats.get("p_value") is not None:
                                mlflow.log_metric(f"{prefix}.p_value", stats["p_value"])

                # Cohort summary
                cs = result.cohort_summary
                mlflow.log_metrics({
                    "cohort.n_enrolled": cs.get("n_enrolled", 0),
                    "cohort.age_mean": cs.get("age_mean", 0.0),
                    "cohort.dropout_pct": cs.get("dropout_pct", 0.0),
                })

                # Logistic summary
                ls = result.logistic_summary
                mlflow.log_metric(
                    "logistic.overall_responder_rate",
                    ls.get("overall_responder_rate", 0.0),
                )

                # ── Artifacts ─────────────────────────────────────────────────
                if self.log_artifacts:
                    self._log_artifacts(result)

                mlflow_run_id = run.info.run_id
                logger.info(f"MLflow run logged: {mlflow_run_id}")
                return mlflow_run_id

        except Exception as exc:  # noqa: BLE001
            logger.error(f"MLflow logging failed: {exc}")
            return None

    def _log_artifacts(self, result: Any) -> None:
        """
        Persist DataFrames and summary JSON as MLflow artifacts.

        Args:
            result: TrialResult instance.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Summary JSON
            summary_path = tmp / "summary.json"
            with summary_path.open("w") as f:
                json.dump(result.to_summary_dict(), f, indent=2, default=str)
            mlflow.log_artifact(str(summary_path), artifact_path="outputs")

            # Biomarker DataFrame as Parquet
            if hasattr(result, "biomarker_df") and result.biomarker_df is not None:
                bio_path = tmp / "biomarker_df.parquet"
                result.biomarker_df.to_parquet(bio_path, index=False)
                mlflow.log_artifact(str(bio_path), artifact_path="outputs")

            # Patient DataFrame as Parquet
            if hasattr(result, "patient_df") and result.patient_df is not None:
                pat_path = tmp / "patient_df.parquet"
                result.patient_df.to_parquet(pat_path, index=False)
                mlflow.log_artifact(str(pat_path), artifact_path="outputs")

    def search_runs(
        self,
        config_hash: str | None = None,
        min_responder_rate: float | None = None,
        max_p_value: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query past simulation runs from MLflow.

        Args:
            config_hash:         Filter to runs with this config hash.
            min_responder_rate:  Filter to runs with responder_rate >= this.
            max_p_value:         Filter to runs with ATE p_value <= this.

        Returns:
            List of run dicts with run_id, params, metrics, and tags.
        """
        if not self._available:
            return []

        try:
            filter_parts = []
            if config_hash:
                filter_parts.append(f"tags.config_hash = '{config_hash}'")
            if min_responder_rate is not None:
                filter_parts.append(
                    f"metrics.`logistic.overall_responder_rate` >= {min_responder_rate}"
                )

            filter_str = " and ".join(filter_parts) if filter_parts else ""

            runs_df = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=filter_str,
                max_results=100,
            )

            if runs_df.empty:
                return []

            return runs_df.to_dict(orient="records")

        except Exception as exc:  # noqa: BLE001
            logger.error(f"MLflow search failed: {exc}")
            return []
