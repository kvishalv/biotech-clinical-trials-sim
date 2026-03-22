"""
utils/drift_detector.py
────────────────────────
Distribution drift detector for simulated cohort and biomarker data.

Drift tests catch when a new simulation run produces a cohort or biomarker
distribution that has drifted unexpectedly from a reference run — indicating
a bug in the stochastic model, a config change, or a seed collision.

Tests implemented:
  1. KS-test (Kolmogorov-Smirnov) for continuous variables
  2. PSI (Population Stability Index) for categorical binned distributions
  3. Chi-squared test for categorical variables (arm/site distribution)
  4. Biomarker mean drift: per-arm per-endpoint Welch's t-test

Usage:
    ref_df = patient_df_run_1
    new_df = patient_df_run_2

    detector = DriftDetector(alpha=0.05, psi_threshold=0.2)
    report = detector.check_patient_drift(ref_df, new_df)
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# PSI thresholds (industry standard):
#   PSI < 0.10  → negligible drift
#   PSI 0.10-0.25 → some drift, worth investigating
#   PSI > 0.25  → significant drift, investigate before proceeding
PSI_NEGLIGIBLE = 0.10
PSI_WARNING = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VariableDriftResult:
    """Drift test result for a single variable."""

    variable: str
    test: str             # "ks", "psi", "chi2", "t-test"
    statistic: float
    p_value: float | None
    drifted: bool
    detail: str


@dataclass
class DriftReport:
    """
    Aggregated drift report comparing two simulation cohorts.

    Attributes:
        results:       Per-variable drift results
        n_drifted:     Count of variables with detected drift
        n_tested:      Total variables tested
        overall_drift: True if any variable shows drift
    """

    results: list[VariableDriftResult] = field(default_factory=list)
    n_drifted: int = 0
    n_tested: int = 0

    @property
    def overall_drift(self) -> bool:
        """True if any variable drifted."""
        return self.n_drifted > 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"DriftReport: {self.n_drifted}/{self.n_tested} variables drifted",
            f"Overall drift detected: {self.overall_drift}",
            "",
        ]
        for r in self.results:
            flag = "⚠️ DRIFT" if r.drifted else "✓  ok  "
            lines.append(f"  {flag}  {r.variable:35s} [{r.test}] {r.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API response."""
        return {
            "n_drifted": self.n_drifted,
            "n_tested": self.n_tested,
            "overall_drift": self.overall_drift,
            "results": [
                {
                    "variable": r.variable,
                    "test": r.test,
                    "statistic": round(r.statistic, 4),
                    "p_value": round(r.p_value, 4) if r.p_value is not None else None,
                    "drifted": r.drifted,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Drift Detector
# ──────────────────────────────────────────────────────────────────────────────


class DriftDetector:
    """
    Detects distribution drift between two simulation cohort DataFrames.

    Args:
        alpha:          Significance level for KS and chi-squared tests.
        psi_threshold:  PSI value above which drift is flagged.
        n_bins:         Number of bins for PSI computation.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        psi_threshold: float = PSI_WARNING,
        n_bins: int = 10,
    ) -> None:
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins

    # ── KS Test (continuous variables) ────────────────────────────────────────

    def ks_test(self, ref: np.ndarray, new: np.ndarray, variable: str) -> VariableDriftResult:
        """
        Two-sample Kolmogorov-Smirnov test for continuous variable drift.

        Args:
            ref:      Reference distribution sample.
            new:      New distribution sample.
            variable: Variable name (for reporting).

        Returns:
            VariableDriftResult.
        """
        stat, p_val = stats.ks_2samp(ref, new)
        drifted = bool(p_val < self.alpha)
        return VariableDriftResult(
            variable=variable,
            test="ks",
            statistic=float(stat),
            p_value=float(p_val),
            drifted=drifted,
            detail=f"KS={stat:.4f}, p={p_val:.4f}",
        )

    # ── PSI (Population Stability Index) ──────────────────────────────────────

    def psi(self, ref: np.ndarray, new: np.ndarray, variable: str) -> VariableDriftResult:
        """
        Population Stability Index for a continuous variable.

        PSI = Σ (Observed% - Expected%) * ln(Observed% / Expected%)

        Bins are defined on the reference distribution; new distribution is
        mapped into the same bins.

        Args:
            ref:      Reference distribution array.
            new:      New distribution array.
            variable: Variable name.

        Returns:
            VariableDriftResult.
        """
        # Define bins from reference distribution
        bin_edges = np.percentile(ref, np.linspace(0, 100, self.n_bins + 1))
        bin_edges = np.unique(bin_edges)  # remove duplicates for degenerate dists

        if len(bin_edges) < 2:
            return VariableDriftResult(
                variable=variable,
                test="psi",
                statistic=0.0,
                p_value=None,
                drifted=False,
                detail="PSI=N/A (degenerate distribution)",
            )

        ref_counts = np.histogram(ref, bins=bin_edges)[0]
        new_counts = np.histogram(new, bins=bin_edges)[0]

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
        new_pct = (new_counts + eps) / (len(new) + eps * len(new_counts))

        psi_val = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
        drifted = psi_val > self.psi_threshold

        severity = (
            "negligible" if psi_val < PSI_NEGLIGIBLE
            else "moderate" if psi_val < PSI_WARNING
            else "significant"
        )

        return VariableDriftResult(
            variable=variable,
            test="psi",
            statistic=psi_val,
            p_value=None,
            drifted=drifted,
            detail=f"PSI={psi_val:.4f} ({severity})",
        )

    # ── Chi-squared test (categorical) ────────────────────────────────────────

    def chi2_test(
        self, ref: pd.Series, new: pd.Series, variable: str
    ) -> VariableDriftResult:
        """
        Chi-squared goodness-of-fit test for categorical distribution drift.

        Args:
            ref:      Reference categorical Series.
            new:      New categorical Series.
            variable: Variable name.

        Returns:
            VariableDriftResult.
        """
        all_cats = set(ref.unique()) | set(new.unique())
        ref_counts = ref.value_counts().reindex(all_cats, fill_value=0)
        new_counts = new.value_counts().reindex(all_cats, fill_value=0)

        # Need at least 5 per cell for chi2 to be valid
        if ref_counts.min() < 5 or new_counts.min() < 5:
            return VariableDriftResult(
                variable=variable,
                test="chi2",
                statistic=0.0,
                p_value=None,
                drifted=False,
                detail="chi2=N/A (insufficient cell counts)",
            )

        stat, p_val = stats.chisquare(f_obs=new_counts.values, f_exp=ref_counts.values)
        drifted = bool(p_val < self.alpha)
        return VariableDriftResult(
            variable=variable,
            test="chi2",
            statistic=float(stat),
            p_value=float(p_val),
            drifted=drifted,
            detail=f"chi2={stat:.4f}, p={p_val:.4f}",
        )

    # ── Biomarker mean drift ───────────────────────────────────────────────────

    def biomarker_mean_drift(
        self,
        ref_bio_df: pd.DataFrame,
        new_bio_df: pd.DataFrame,
        biomarker: str,
        arm: str,
        week: int,
    ) -> VariableDriftResult:
        """
        Test whether a biomarker's mean at a specific week has drifted.

        Uses Welch's t-test (unequal variance assumption).

        Args:
            ref_bio_df: Reference longitudinal biomarker DataFrame.
            new_bio_df: New longitudinal biomarker DataFrame.
            biomarker:  Biomarker name.
            arm:        Treatment arm to test.
            week:       Week to compare.

        Returns:
            VariableDriftResult.
        """
        def _extract(df: pd.DataFrame) -> np.ndarray:
            return df[
                (df["biomarker"] == biomarker)
                & (df["arm"] == arm)
                & (df["week"] == week)
                & (df["observed"])
            ]["value"].values

        ref_vals = _extract(ref_bio_df)
        new_vals = _extract(new_bio_df)
        variable = f"{biomarker}@{arm}@w{week}"

        if len(ref_vals) < 5 or len(new_vals) < 5:
            return VariableDriftResult(
                variable=variable,
                test="t-test",
                statistic=0.0,
                p_value=None,
                drifted=False,
                detail="t-test=N/A (insufficient data)",
            )

        stat, p_val = stats.ttest_ind(ref_vals, new_vals, equal_var=False)
        drifted = bool(p_val < self.alpha)
        ref_mean = float(ref_vals.mean())
        new_mean = float(new_vals.mean())
        pct_change = ((new_mean - ref_mean) / abs(ref_mean) * 100) if ref_mean != 0 else 0.0

        return VariableDriftResult(
            variable=variable,
            test="t-test",
            statistic=float(stat),
            p_value=float(p_val),
            drifted=drifted,
            detail=f"t={stat:.4f}, p={p_val:.4f}, Δmean={pct_change:+.1f}%",
        )

    # ── High-level API ────────────────────────────────────────────────────────

    def check_patient_drift(
        self,
        ref_df: pd.DataFrame,
        new_df: pd.DataFrame,
        continuous_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> DriftReport:
        """
        Run all drift tests on patient-level DataFrames.

        Args:
            ref_df:           Reference patient DataFrame.
            new_df:           New patient DataFrame.
            continuous_cols:  Columns to test with KS + PSI (defaults: age, bmi).
            categorical_cols: Columns to test with chi2 (defaults: arm, sex, ethnicity).

        Returns:
            DriftReport with per-variable results.
        """
        if continuous_cols is None:
            continuous_cols = ["age", "bmi"]
        if categorical_cols is None:
            categorical_cols = ["arm", "sex", "ethnicity", "site_id"]

        report = DriftReport()

        for col in continuous_cols:
            if col not in ref_df.columns or col not in new_df.columns:
                continue
            ref_arr = ref_df[col].dropna().values
            new_arr = new_df[col].dropna().values

            ks_result = self.ks_test(ref_arr, new_arr, f"{col}__ks")
            psi_result = self.psi(ref_arr, new_arr, f"{col}__psi")
            report.results.extend([ks_result, psi_result])
            report.n_tested += 2
            if ks_result.drifted:
                report.n_drifted += 1
            if psi_result.drifted:
                report.n_drifted += 1

        for col in categorical_cols:
            if col not in ref_df.columns or col not in new_df.columns:
                continue
            chi2_result = self.chi2_test(ref_df[col], new_df[col], col)
            report.results.append(chi2_result)
            report.n_tested += 1
            if chi2_result.drifted:
                report.n_drifted += 1

        return report

    def check_biomarker_drift(
        self,
        ref_bio_df: pd.DataFrame,
        new_bio_df: pd.DataFrame,
        biomarkers: list[str] | None = None,
        arms: list[str] | None = None,
        check_weeks: list[int] | None = None,
    ) -> DriftReport:
        """
        Run biomarker mean drift tests across endpoints, arms, and weeks.

        Args:
            ref_bio_df:   Reference biomarker DataFrame.
            new_bio_df:   New biomarker DataFrame.
            biomarkers:   Biomarker names to test (defaults to all in ref).
            arms:         Arms to test (defaults to all in ref).
            check_weeks:  Weeks to test at (defaults to [0, 26, 52]).

        Returns:
            DriftReport.
        """
        if biomarkers is None:
            biomarkers = list(ref_bio_df["biomarker"].unique())
        if arms is None:
            arms = list(ref_bio_df["arm"].unique())
        if check_weeks is None:
            check_weeks = [0, 26, int(ref_bio_df["week"].max())]
        check_weeks = [w for w in check_weeks if w in ref_bio_df["week"].values]

        report = DriftReport()
        for bm in biomarkers:
            for arm in arms:
                for week in check_weeks:
                    result = self.biomarker_mean_drift(
                        ref_bio_df, new_bio_df, bm, arm, week
                    )
                    report.results.append(result)
                    report.n_tested += 1
                    if result.drifted:
                        report.n_drifted += 1

        return report
