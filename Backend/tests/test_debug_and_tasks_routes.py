"""Endpoint tests for debug.py and tasks.py's HTTP status route (LIT-187).

Neither had any route-level test coverage before this issue. LIT-223
tightened the debug endpoint: it is disabled unless ``DEBUG_ENABLED`` is set,
and when enabled it reveals the session id only (never cookies or headers).
"""

from __future__ import annotations

import pytest

from app.infrastructure.settings import settings


class TestDebugSessionRoute:
    @pytest.mark.asyncio
    async def test_disabled_by_default_returns_404(self, client, monkeypatch):
        monkeypatch.setattr(settings, "DEBUG_ENABLED", False)
        r = await client.get("/debug/session")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_enabled_returns_session_id_without_cookies_or_headers(self, client, monkeypatch):
        monkeypatch.setattr(settings, "DEBUG_ENABLED", True)
        session_resp = await client.get("/session")
        cookies = session_resp.cookies
        sid = session_resp.json()["sid"]

        r = await client.get("/debug/session", cookies=cookies)
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        # LIT-223: the endpoint used to echo cookies and full request headers;
        # it must not anymore.
        assert "cookies" not in body
        assert "headers" not in body


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
