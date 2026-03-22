"""
agents/protocol_linter.py
──────────────────────────
Protocol linting agent: detects contradictions and logical errors in
inclusion/exclusion criteria without ever generating synthetic data.

Use cases:
  - Catch age range contradictions (min_age > max_age)
  - Detect comorbidity conflicts (requiring a condition that is also excluded)
  - Flag unrealistic BMI ranges or missing mandatory fields
  - Warn when exclusion rate is likely to be > 80% (under-powered trial)
  - Identify ambiguous arm allocation weights

The agent receives the trial config dict and returns a structured lint report
as plain text. It does NOT modify the config.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent


class ProtocolLinterAgent(BaseAgent):
    """
    LLM-powered trial protocol linter.

    Analyses inclusion/exclusion criteria, arm allocation, and simulation
    parameters for logical contradictions and potential issues.

    Usage:
        linter = ProtocolLinterAgent()
        report = linter.run(trial_config.model_dump())
        print(report)
    """

    def _system_prompt(self) -> str:
        return (
            "You are a senior clinical trial statistician and regulatory expert. "
            "Your job is to review synthetic trial protocol configurations for "
            "logical contradictions, statistical design flaws, and inclusion/exclusion "
            "criteria errors. You are rigorous, precise, and concise. "
            "You never invent patient data — you only analyse configuration parameters. "
            "Format your response as a structured lint report with sections: "
            "ERRORS (must fix), WARNINGS (should review), and SUGGESTIONS (nice to have). "
            "If nothing is wrong, say so clearly."
        )

    def _build_prompt(self, data: dict[str, Any]) -> str:
        """
        Build a lint prompt from the trial configuration dict.

        Args:
            data: Trial config dict (from TrialConfig.model_dump()).

        Returns:
            Formatted prompt string.
        """
        config_json = json.dumps(data, indent=2, default=str)
        return (
            f"Please lint the following clinical trial protocol configuration.\n\n"
            f"Check for:\n"
            f"1. Inclusion/exclusion criteria contradictions "
            f"(e.g. a condition is both required AND excluded)\n"
            f"2. Age or BMI range errors (min > max, biologically impossible values)\n"
            f"3. Treatment arm allocation weights that do not sum to 1.0\n"
            f"4. Unrealistic simulation parameters (n_weeks, n_patients, dropout rates)\n"
            f"5. Missing or ambiguous primary endpoints\n"
            f"6. Likely statistical power issues (very small arms, very short trials)\n\n"
            f"Protocol config:\n```json\n{config_json}\n```\n\n"
            f"Provide a structured lint report."
        )

    def lint(self, trial_cfg_dict: dict[str, Any]) -> "LintReport":
        """
        Run the linter and return a structured LintReport.

        Also runs a fast rule-based pre-check before calling the LLM,
        so obvious errors are caught even without an API key.

        Args:
            trial_cfg_dict: Trial config as plain dict.

        Returns:
            LintReport instance with both rule-based and LLM findings.
        """
        rule_errors = _rule_based_lint(trial_cfg_dict)
        llm_text = self.run_safe(trial_cfg_dict, fallback="[LLM lint unavailable]")
        return LintReport(rule_errors=rule_errors, llm_report=llm_text)


class LintReport:
    """
    Combined output from rule-based + LLM protocol linting.

    Attributes:
        rule_errors: List of deterministic rule violations found without LLM.
        llm_report:  Free-text LLM lint report.
    """

    def __init__(self, rule_errors: list[str], llm_report: str) -> None:
        self.rule_errors = rule_errors
        self.llm_report = llm_report

    @property
    def has_errors(self) -> bool:
        """True if rule-based errors were found."""
        return len(self.rule_errors) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API response."""
        return {
            "rule_errors": self.rule_errors,
            "llm_report": self.llm_report,
            "has_errors": self.has_errors,
        }

    def __repr__(self) -> str:
        return (
            f"LintReport(errors={len(self.rule_errors)}, "
            f"llm_report_len={len(self.llm_report)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based lint checks (deterministic, no LLM required)
# ──────────────────────────────────────────────────────────────────────────────


def _rule_based_lint(cfg: dict[str, Any]) -> list[str]:
    """
    Fast deterministic protocol linter — runs before the LLM call.

    Args:
        cfg: Trial config dict.

    Returns:
        List of error/warning strings (empty = no issues found).
    """
    errors: list[str] = []
    cohort = cfg.get("cohort", {})
    ic = cohort.get("inclusion_criteria", {})
    arms = cohort.get("treatment_arms", [])
    simulation = cfg.get("simulation", {})

    # Age range
    min_age = ic.get("min_age", 0)
    max_age = ic.get("max_age", 0)
    if min_age >= max_age:
        errors.append(
            f"[ERROR] min_age ({min_age}) must be < max_age ({max_age})."
        )

    # BMI range
    min_bmi = ic.get("min_bmi", 0)
    max_bmi = ic.get("max_bmi", 0)
    if min_bmi >= max_bmi:
        errors.append(
            f"[ERROR] min_bmi ({min_bmi}) must be < max_bmi ({max_bmi})."
        )

    # Required vs excluded comorbidities overlap
    required = set(ic.get("required_conditions", []))
    excluded = set(ic.get("excluded_conditions", []))
    overlap = required & excluded
    if overlap:
        errors.append(
            f"[ERROR] Conditions appear in BOTH required and excluded lists: {sorted(overlap)}. "
            f"These patients can never exist."
        )

    # Arm allocations sum
    if arms:
        total_alloc = sum(a.get("allocation", 0) for a in arms)
        if abs(total_alloc - 1.0) > 0.01:
            errors.append(
                f"[ERROR] Treatment arm allocations sum to {total_alloc:.4f}, expected 1.0."
            )

    # Minimum arm count
    if len(arms) < 2:
        errors.append("[WARNING] Trial has fewer than 2 treatment arms — no comparison possible.")

    # Simulation length
    n_weeks = simulation.get("n_weeks", 0)
    burnin = simulation.get("burnin_weeks", 0)
    if burnin >= n_weeks:
        errors.append(
            f"[ERROR] burnin_weeks ({burnin}) >= n_weeks ({n_weeks}). "
            f"No post-treatment observation period."
        )

    # Patient count
    n_patients = cohort.get("n_patients", 0)
    if n_patients < 30:
        errors.append(
            f"[WARNING] n_patients={n_patients} is very small. "
            f"Statistical power will be insufficient for most endpoints."
        )

    return errors
