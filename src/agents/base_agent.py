"""
agents/base_agent.py
─────────────────────
Base LLM agent using the Anthropic SDK.

Architecture contract (strictly enforced):
  - The LLM NEVER generates numeric outcomes, patient data, or biomarker values
  - The LLM ONLY receives structured summaries (dicts) and returns text
  - All number generation happens in the simulation kernel
  - Agents are stateless — each call is independent (no conversation history)
  - API key is read from ANTHROPIC_API_KEY env var — never hardcoded

Agents implemented:
  1. ProtocolLinterAgent   — detects I/E contradictions in trial protocols
  2. CohortNarratorAgent   — narrates synthetic cohort summary statistics
  3. ResultInterpreterAgent — interprets and critiques simulation outcomes

Pattern: each agent wraps a structured prompt template + Anthropic SDK call.
The simulator calls agent.run(data_dict) and gets back a plain string.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Default model — use the latest Sonnet for cost/quality balance
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


class BaseAgent(ABC):
    """
    Abstract base for all LLM agents in the simulation platform.

    Args:
        model:      Anthropic model string (defaults to claude-sonnet-4-6).
        max_tokens: Maximum tokens in the response.
        api_key:    Optional override for ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. Agent calls will fail unless a key is provided."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key or "placeholder")

    @abstractmethod
    def _build_prompt(self, data: dict[str, Any]) -> str:
        """
        Build the user-turn prompt string from structured data.

        Args:
            data: Structured dict from the simulation layer.

        Returns:
            Prompt string to send to the LLM.
        """

    @abstractmethod
    def _system_prompt(self) -> str:
        """Return the system prompt for this agent."""

    def run(self, data: dict[str, Any]) -> str:
        """
        Execute the agent: build prompt → call API → return response text.

        Args:
            data: Structured simulation data to summarise/critique.

        Returns:
            LLM response as a plain string.

        Raises:
            anthropic.APIError: On API failure (rate limit, auth, etc.)
        """
        user_prompt = self._build_prompt(data)
        system = self._system_prompt()

        logger.debug(f"{self.__class__.__name__}: calling {self.model}")

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response = message.content[0].text
        logger.debug(f"{self.__class__.__name__}: response length={len(response)}")
        return response

    def run_safe(self, data: dict[str, Any], fallback: str = "[Agent unavailable]") -> str:
        """
        run() with exception handling — returns fallback string on any API error.

        Use this in production paths where agent failure should not crash the pipeline.

        Args:
            data:     Data dict.
            fallback: String to return on failure.

        Returns:
            LLM response or fallback string.
        """
        try:
            return self.run(data)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{self.__class__.__name__} failed: {exc}")
            return fallback
