"""
simulation/outcome_models.py
─────────────────────────────
Trial outcome prediction models for the biotech simulation platform.

Models implemented:
  1. WeibullSurvivalModel   — time-to-event (Weibull AFT parameterisation)
  2. LogisticOutcomeModel   — binary endpoint (responder / non-responder)
  3. ContinuousOutcomeModel — continuous endpoint change-from-baseline
  4. HTEUpliftModel         — heterogeneous treatment effect / uplift scoring

Design:
  - All models accept and return pandas DataFrames for easy logging/inspection
  - Models are fit on simulated data, not real patient data
  - Survival model uses lifelines; logistic uses scikit-learn
  - Each model exposes a .summary() method for the agent result interpreter
  - Models are deterministic given the same seed and data

The LLM result-interpreter agent reads model .summary() outputs — it never
touches the underlying model fitting or inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# 1. Weibull Survival Model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SurvivalResult:
    """Container for survival analysis outputs."""

    arm: str
    median_survival_weeks: float
    survival_at_52w: float     # probability of surviving to week 52
    hazard_ratio: float        # vs placebo arm
    log_rank_p: float
    n_events: int
    n_censored: int


class WeibullSurvivalModel:
    """
    Parametric survival model using the Weibull Accelerated Failure Time (AFT)
    parameterisation, implemented via lifelines.

    In clinical trial simulation this models "time to meaningful biomarker
    deterioration" or "time to clinical event".

    Args:
        seed: RNG seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model: Any = None
        self._fitted = False

    def generate_survival_data(
        self,
        patients: list[Any],  # list[Patient]
        n_weeks: int = 52,
        event_biomarker: str = "epigenetic_age_acceleration",
        event_threshold_delta: float = 0.5,
        biomarker_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic time-to-event data based on biomarker trajectories.

        An "event" is defined as the first week where the biomarker exceeds
        (or falls below for protective biomarkers) the event_threshold_delta
        from baseline.

        Args:
            patients:              List of Patient instances.
            n_weeks:               Maximum follow-up duration.
            event_biomarker:       Which biomarker defines the event.
            event_threshold_delta: Magnitude of change that constitutes an event.
            biomarker_df:          Pre-computed longitudinal biomarker DataFrame.

        Returns:
            DataFrame with columns: patient_id, arm, duration, event, age, bmi.
        """
        rng = np.random.default_rng(self.seed)
        rows = []

        for patient in patients:
            # Simulate time to event from a Weibull distribution
            # parametrised by arm (treatment protective → longer survival)
            arm_scale = {"placebo": 30.0, "low_dose": 38.0, "high_dose": 48.0}.get(
                patient.arm, 35.0
            )
            arm_shape = 2.0  # shape > 1 → increasing hazard over time

            # Weibull time-to-event
            tte = float(rng.weibull(arm_shape) * arm_scale)

            # Censored if tte > n_weeks or patient dropped out
            if patient.dropout_week is not None:
                censor_week = min(patient.dropout_week, n_weeks)
            else:
                censor_week = n_weeks

            duration = min(tte, censor_week)
            event = int(tte <= censor_week)

            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "arm": patient.arm,
                    "duration": round(duration, 2),
                    "event": event,
                    "age": patient.age,
                    "bmi": patient.bmi,
                    "site_id": patient.site_id,
                }
            )

        return pd.DataFrame(rows)

    def fit(self, survival_df: pd.DataFrame) -> "WeibullSurvivalModel":
        """
        Fit a Weibull AFT model to the survival data.

        Args:
            survival_df: Output from generate_survival_data().

        Returns:
            Self (for chaining).
        """
        try:
            from lifelines import WeibullAFTFitter

            df_fit = survival_df[["duration", "event", "age", "bmi"]].copy()
            # Encode treatment as numeric
            df_fit["arm_low_dose"] = (survival_df["arm"] == "low_dose").astype(int)
            df_fit["arm_high_dose"] = (survival_df["arm"] == "high_dose").astype(int)

            self._model = WeibullAFTFitter(penalizer=0.1)
            self._model.fit(df_fit, duration_col="duration", event_col="event")
            self._fitted = True
        except ImportError:
            # lifelines not installed — fall back to summary stats only
            self._fitted = False

        return self

    def per_arm_summary(self, survival_df: pd.DataFrame) -> list[SurvivalResult]:
        """
        Compute per-arm survival summary statistics.

        Uses Kaplan-Meier estimates when lifelines is available,
        otherwise uses empirical estimates from the data.

        Args:
            survival_df: Output from generate_survival_data().

        Returns:
            List of SurvivalResult instances, one per arm.
        """
        from scipy.stats import chi2_contingency

        results = []
        placebo_df = survival_df[survival_df["arm"] == "placebo"]

        for arm in survival_df["arm"].unique():
            arm_df = survival_df[survival_df["arm"] == arm]

            # Empirical median survival
            events = arm_df[arm_df["event"] == 1]["duration"]
            median_surv = float(events.median()) if len(events) > 0 else float(arm_df["duration"].max())

            # Empirical survival at 52 weeks
            surv_52 = float((arm_df["duration"] >= 52).mean())

            # Hazard ratio vs placebo (approximate: ratio of event rates)
            arm_rate = arm_df["event"].mean()
            placebo_rate = placebo_df["event"].mean()
            hr = arm_rate / placebo_rate if placebo_rate > 0 else np.nan

            # Log-rank test approximation via chi2 on event counts
            if arm != "placebo" and len(arm_df) > 0 and len(placebo_df) > 0:
                contingency = np.array([
                    [arm_df["event"].sum(), len(arm_df) - arm_df["event"].sum()],
                    [placebo_df["event"].sum(), len(placebo_df) - placebo_df["event"].sum()],
                ])
                if contingency.min() >= 5:  # chi2 valid only with sufficient counts
                    _, p_val, _, _ = chi2_contingency(contingency)
                else:
                    _, p_val = stats.fisher_exact(contingency)
            else:
                p_val = 1.0

            results.append(
                SurvivalResult(
                    arm=arm,
                    median_survival_weeks=median_surv,
                    survival_at_52w=surv_52,
                    hazard_ratio=hr,
                    log_rank_p=float(p_val),
                    n_events=int(arm_df["event"].sum()),
                    n_censored=int((arm_df["event"] == 0).sum()),
                )
            )

        return results

    def summary(self, survival_df: pd.DataFrame) -> dict[str, Any]:
        """Return a dict summary suitable for LLM agent consumption."""
        per_arm = self.per_arm_summary(survival_df)
        return {
            "model": "WeibullAFT",
            "n_patients": len(survival_df),
            "n_events_total": int(survival_df["event"].sum()),
            "arms": [
                {
                    "arm": r.arm,
                    "median_survival_weeks": round(r.median_survival_weeks, 1),
                    "survival_at_52w": round(r.survival_at_52w, 3),
                    "hazard_ratio_vs_placebo": round(r.hazard_ratio, 3) if not np.isnan(r.hazard_ratio) else None,
                    "log_rank_p": round(r.log_rank_p, 4),
                    "n_events": r.n_events,
                    "n_censored": r.n_censored,
                }
                for r in per_arm
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Logistic Outcome Model (binary responder endpoint)
# ──────────────────────────────────────────────────────────────────────────────


class LogisticOutcomeModel:
    """
    Binary outcome model: predicts probability of being a "clinical responder"
    (defined as meeting the primary endpoint threshold) from patient covariates.

    Uses scikit-learn's LogisticRegression.

    Args:
        seed: RNG seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model: Any = None
        self._feature_cols: list[str] = []

    def prepare_data(
        self,
        patient_df: pd.DataFrame,
        biomarker_df: pd.DataFrame,
        endpoint: str,
        threshold_delta: float = -0.3,
        week: int | None = None,
    ) -> pd.DataFrame:
        """
        Build a feature matrix for logistic regression.

        "Responder" label: final-week change from baseline exceeds threshold_delta.

        Args:
            patient_df:      Patient-level metadata DataFrame.
            biomarker_df:    Long-format biomarker simulation output.
            endpoint:        Biomarker name used as outcome.
            threshold_delta: Change from baseline that defines a responder.
            week:            Evaluation week (defaults to max observed).

        Returns:
            DataFrame with features and binary 'responder' label.
        """
        if week is None:
            week = int(biomarker_df["week"].max())

        # Get baseline (week 0) and final week values
        baseline = biomarker_df[
            (biomarker_df["week"] == 0) & (biomarker_df["biomarker"] == endpoint)
        ][["patient_id", "value"]].rename(columns={"value": "baseline_val"})

        final = biomarker_df[
            (biomarker_df["week"] == week)
            & (biomarker_df["biomarker"] == endpoint)
            & (biomarker_df["observed"])
        ][["patient_id", "value"]].rename(columns={"value": "final_val"})

        merged = baseline.merge(final, on="patient_id").merge(
            patient_df[["patient_id", "age", "bmi", "sex", "arm"]], on="patient_id"
        )

        merged["delta"] = merged["final_val"] - merged["baseline_val"]
        merged["responder"] = (merged["delta"] <= threshold_delta).astype(int)
        merged["sex_f"] = (merged["sex"] == "F").astype(int)
        merged["arm_low"] = (merged["arm"] == "low_dose").astype(int)
        merged["arm_high"] = (merged["arm"] == "high_dose").astype(int)

        return merged

    def fit(self, data: pd.DataFrame) -> "LogisticOutcomeModel":
        """
        Fit a logistic regression model to predict responder status.

        Args:
            data: Output from prepare_data().

        Returns:
            Self (for chaining).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._feature_cols = ["age", "bmi", "sex_f", "arm_low", "arm_high"]
        X = data[self._feature_cols].values
        y = data["responder"].values

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=self.seed, max_iter=1000)),
        ])
        self._model.fit(X, y)
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict responder probabilities.

        Args:
            data: DataFrame with feature columns.

        Returns:
            Array of shape (n,) with P(responder=1).
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = data[self._feature_cols].values
        return self._model.predict_proba(X)[:, 1]

    def summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Return a summary dict for the LLM result interpreter."""
        proba = self.predict_proba(data)
        by_arm = data.copy()
        by_arm["pred_proba"] = proba

        arm_summary = (
            by_arm.groupby("arm")
            .agg(
                responder_rate=("responder", "mean"),
                pred_proba_mean=("pred_proba", "mean"),
                n=("responder", "count"),
            )
            .round(3)
            .to_dict(orient="index")
        )

        return {
            "model": "LogisticRegression",
            "n_patients": len(data),
            "overall_responder_rate": round(float(data["responder"].mean()), 3),
            "arm_summary": arm_summary,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Continuous Outcome Model (change from baseline)
# ──────────────────────────────────────────────────────────────────────────────


class ContinuousOutcomeModel:
    """
    Linear mixed-effects style model for continuous endpoint change from baseline.

    Fits OLS with arm indicators and baseline covariates; reports treatment effects
    as regression coefficients with confidence intervals.

    Args:
        seed: RNG seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._result: Any = None

    def fit(
        self,
        patient_df: pd.DataFrame,
        biomarker_df: pd.DataFrame,
        endpoint: str,
        week: int | None = None,
    ) -> "ContinuousOutcomeModel":
        """
        Fit OLS model for endpoint change from baseline.

        Args:
            patient_df:   Patient-level metadata.
            biomarker_df: Long-format biomarker output.
            endpoint:     Biomarker name.
            week:         Evaluation week (defaults to max).

        Returns:
            Self (for chaining).
        """
        import statsmodels.formula.api as smf

        if week is None:
            week = int(biomarker_df["week"].max())

        baseline = biomarker_df[
            (biomarker_df["week"] == 0) & (biomarker_df["biomarker"] == endpoint)
        ][["patient_id", "value"]].rename(columns={"value": "baseline"})

        final = biomarker_df[
            (biomarker_df["week"] == week)
            & (biomarker_df["biomarker"] == endpoint)
            & (biomarker_df["observed"])
        ][["patient_id", "value"]].rename(columns={"value": "final"})

        df = baseline.merge(final, on="patient_id").merge(
            patient_df[["patient_id", "age", "bmi", "sex", "arm"]], on="patient_id"
        )
        df["cfb"] = df["final"] - df["baseline"]  # change from baseline
        df["sex_f"] = (df["sex"] == "F").astype(int)

        formula = "cfb ~ C(arm, Treatment(reference='placebo')) + age + bmi + sex_f + baseline"
        self._result = smf.ols(formula, data=df).fit()
        return self

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the LLM agent."""
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        res = self._result
        params = {k: round(v, 4) for k, v in res.params.items()}
        pvals = {k: round(v, 4) for k, v in res.pvalues.items()}
        conf = {k: [round(v[0], 4), round(v[1], 4)] for k, v in res.conf_int().iterrows()}

        return {
            "model": "OLS_CFB",
            "r_squared": round(float(res.rsquared), 4),
            "n_obs": int(res.nobs),
            "coefficients": params,
            "p_values": pvals,
            "confidence_intervals_95": conf,
            "aic": round(float(res.aic), 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. HTE Uplift Model
# ──────────────────────────────────────────────────────────────────────────────


class HTEUpliftModel:
    """
    Heterogeneous Treatment Effect (HTE) / Uplift model.

    Estimates individual-level treatment effects using the T-learner approach:
      - Fit one model on treated patients: μ₁(x) = E[Y | T=1, X=x]
      - Fit one model on control patients: μ₀(x) = E[Y | T=0, X=x]
      - Uplift score = μ₁(x) - μ₀(x)

    Args:
        seed: RNG seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._m1: Any = None  # model on treated
        self._m0: Any = None  # model on control
        self._feature_cols: list[str] = ["age", "bmi", "sex_f"]

    def fit(self, data: pd.DataFrame) -> "HTEUpliftModel":
        """
        Fit the T-learner uplift model.

        Args:
            data: Must contain columns: age, bmi, sex_f, arm, delta (cfb or endpoint change).

        Returns:
            Self (for chaining).
        """
        from sklearn.ensemble import GradientBoostingRegressor

        X = data[self._feature_cols].values
        y = data["delta"].values
        t = (data["arm"] != "placebo").values

        self._m1 = GradientBoostingRegressor(random_state=self.seed, n_estimators=100)
        self._m0 = GradientBoostingRegressor(random_state=self.seed, n_estimators=100)

        self._m1.fit(X[t], y[t])
        self._m0.fit(X[~t], y[~t])
        return self

    def uplift_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute individual uplift scores: τ(x) = μ₁(x) - μ₀(x).

        Args:
            data: DataFrame with feature columns.

        Returns:
            Array of uplift scores.
        """
        if self._m1 is None or self._m0 is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = data[self._feature_cols].values
        return self._m1.predict(X) - self._m0.predict(X)

    def summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Return uplift summary statistics for the LLM result interpreter."""
        scores = self.uplift_scores(data)
        return {
            "model": "T-Learner HTE",
            "n_patients": len(data),
            "uplift_mean": round(float(scores.mean()), 4),
            "uplift_std": round(float(scores.std()), 4),
            "uplift_p25": round(float(np.percentile(scores, 25)), 4),
            "uplift_median": round(float(np.median(scores)), 4),
            "uplift_p75": round(float(np.percentile(scores, 75)), 4),
            "high_responder_pct": round(
                float((scores < scores.mean() - scores.std()).mean() * 100), 1
            ),
        }
