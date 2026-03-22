"""tests/test_api.py — Tests for the FastAPI serving layer"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="module")
def app():
    """Import the FastAPI app without starting Ray."""
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
    from src.api.main import create_app
    return create_app()


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_ok(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "ray_initialized" in data

    async def test_health_version_is_string(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert isinstance(resp.json()["version"], str)


@pytest.mark.asyncio
class TestSimulateEndpoint:
    async def test_simulate_returns_200(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/simulate",
                json={"seed": 42, "n_patients": 30, "n_weeks": 8, "run_agents": False},
            )
        assert resp.status_code == 200, resp.text

    async def test_simulate_returns_run_id(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/simulate",
                json={"seed": 42, "n_patients": 30, "n_weeks": 8},
            )
        data = resp.json()
        assert "run_id" in data
        assert len(data["run_id"]) > 0

    async def test_simulate_cohort_summary_present(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/simulate",
                json={"seed": 42, "n_patients": 30, "n_weeks": 8},
            )
        data = resp.json()
        assert "cohort_summary" in data
        assert data["cohort_summary"]["n_enrolled"] > 0

    async def test_get_trial_after_simulate(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # First simulate
            post_resp = await client.post(
                "/simulate",
                json={"seed": 77, "n_patients": 30, "n_weeks": 8},
            )
            run_id = post_resp.json()["run_id"]

            # Then retrieve
            get_resp = await client.get(f"/simulate/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == run_id

    async def test_get_nonexistent_run_returns_404(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/simulate/nonexistent-run-id")
        assert resp.status_code == 404

    async def test_simulate_invalid_params_returns_422(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/simulate",
                json={"seed": 42, "n_patients": -1},  # invalid n_patients
            )
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestBiomarkersEndpoint:
    async def test_biomarkers_returns_all_9(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Run a simulation first
            post_resp = await client.post(
                "/simulate",
                json={"seed": 55, "n_patients": 30, "n_weeks": 8},
            )
            run_id = post_resp.json()["run_id"]

            bio_resp = await client.get(f"/biomarkers/{run_id}")
        assert bio_resp.status_code == 200
        data = bio_resp.json()
        assert len(data["biomarkers"]) == 9

    async def test_single_biomarker_trajectory(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            post_resp = await client.post(
                "/simulate",
                json={"seed": 66, "n_patients": 30, "n_weeks": 8},
            )
            run_id = post_resp.json()["run_id"]
            resp = await client.get(f"/biomarkers/{run_id}/inflammation_index")
        assert resp.status_code == 200
        data = resp.json()
        assert data["biomarker"] == "inflammation_index"
        assert "trajectory" in data

    async def test_invalid_biomarker_returns_404(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            post_resp = await client.post(
                "/simulate",
                json={"seed": 88, "n_patients": 30, "n_weeks": 8},
            )
            run_id = post_resp.json()["run_id"]
            resp = await client.get(f"/biomarkers/{run_id}/nonexistent_biomarker")
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestAgentLintEndpoint:
    async def test_lint_returns_200(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/agents/lint", json={})
        # 200 even if LLM unavailable (rule-based checks still run)
        assert resp.status_code == 200

    async def test_lint_response_has_rule_errors(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/agents/lint", json={})
        data = resp.json()
        assert "rule_errors" in data
        assert "has_errors" in data
        assert isinstance(data["rule_errors"], list)
