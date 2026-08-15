"""Endpoint tests for debug.py and tasks.py's HTTP status route (LIT-187).

Neither had any route-level test coverage before this issue.
"""

from __future__ import annotations

import pytest


class TestDebugSessionRoute:
    @pytest.mark.asyncio
    async def test_reflects_session_id_and_cookies(self, client):
        session_resp = await client.get("/session")
        cookies = session_resp.cookies
        sid = session_resp.json()["sid"]

        r = await client.get("/debug/session", cookies=cookies)
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert body["cookies"].get("sid") == sid

    @pytest.mark.asyncio
    async def test_returns_200_even_without_prior_session(self, client):
        # SessionMiddleware auto-provisions a session on every request, so
        # this still 200s (and reflects the freshly-created id) with no
        # cookie sent up front.
        r = await client.get("/debug/session")
        assert r.status_code == 200
        assert r.json()["session_id"] is not None


class TestTaskStatusRoute:
    @pytest.mark.asyncio
    async def test_unknown_task_returns_unknown_state(self, client):
        # No RQ/Redis broker is reachable in this test environment, so
        # fetch_job() swallows the connection failure and returns None --
        # verifying the route degrades to a clean 200/UNKNOWN rather than a
        # 500 is exactly the "server-side responsiveness" this issue's DoD
        # cares about.
        r = await client.get("/api/tasks/definitely-not-a-real-job-id/status")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "definitely-not-a-real-job-id"
        assert body["state"] == "UNKNOWN"
