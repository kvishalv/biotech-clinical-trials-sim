"""
agents/cohort_narrator.py
──────────────────────────
Cohort narrator agent: generates a human-readable narrative summary of
the synthetic cohort statistics produced by PatientGenerator.cohort_summary().

The agent receives ONLY aggregate statistics (means, proportions, counts).
It NEVER receives individual patient records — privacy-preserving by design.

Output: a 150-300 word clinical narrative suitable for a trial status report
or regulatory submission appendix.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent


class CohortNarratorAgent(BaseAgent):
    """
    Generates a clinical narrative from cohort summary statistics.

    Usage:
        narrator = CohortNarratorAgent()
        summary = patient_generator.cohort_summary(patients)
        narrative = narrator.run(summary)
    """

    def _system_prompt(self) -> str:
        return (
            "You are a medical writer specialising in clinical trial documentation. "
            "You write clear, accurate, and concise cohort descriptions based solely "
            "on the aggregate statistics provided to you. "
            "You never invent individual patient stories or fabricate numbers. "
            "Use clinical language appropriate for a Phase II trial report. "
            "Write in third-person past tense. Target 150-250 words."
        )

    def _build_prompt(self, data: dict[str, Any]) -> str:
        stats_json = json.dumps(data, indent=2, default=str)
        return (
            "Write a clinical narrative describing the enrolled cohort based on "
            "these aggregate statistics from a synthetic clinical trial simulation.\n\n"
            "Statistics:\n"
            f"```json\n{stats_json}\n```\n\n"
            "Include: total enrolment, age and BMI distribution, sex split, "
            "treatment arm balance, site distribution, dropout rate, and top comorbidities. "
            "Note that all data is synthetic and generated for simulation purposes only."
        )

    def narrate(self, cohort_summary: dict[str, Any]) -> str:
        """
        Generate a cohort narrative from summary statistics.

        Args:
            cohort_summary: Output from PatientGenerator.cohort_summary().

        Returns:
            Clinical narrative string.
        """
        return self.run_safe(
            cohort_summary,
            fallback="[Cohort narrative unavailable — check ANTHROPIC_API_KEY]",
        )
