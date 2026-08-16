"""Endpoint tests for datasets.py's new GET /datasets/list (LIT-235).

No route-level coverage existed for datasets.py before this issue.
"""
from __future__ import annotations

import pytest

from app.api.routes import datasets as datasets_routes


class TestListDatasets:
    @pytest.mark.asyncio
    async def test_returns_registered_corpora(self, client):
        r = await client.get("/datasets/list")
        assert r.status_code == 200
        body = r.json()
        assert "common-voice" in body["datasets"]
        assert "ravdess" in body["datasets"]
        assert "l2-arctic" in body["datasets"]
        assert "asvspoof-2021" in body["datasets"]
        assert "crema-d" in body["datasets"]

    @pytest.mark.asyncio
    async def test_excludes_corpora_with_no_loader(self, client):
        # "esd" is registered (LIT-106 inventory) but LIT-208 deliberately
        # left it unwired - listing it would offer a dataset that 404s.
        r = await client.get("/datasets/list")
        assert "esd" not in r.json()["datasets"]

    @pytest.mark.asyncio
    async def test_does_not_shadow_the_per_dataset_metadata_route(self, client, monkeypatch):
        # Guards against a routing regression: /datasets/list must not be
        # swallowed by the /{dataset}/metadata catch-all, and vice versa.
        called = {}
        monkeypatch.setattr(
            datasets_routes, "load_metadata", lambda dataset, session_id: called.setdefault("dataset", dataset) or []
        )
        r = await client.get("/some-corpus/metadata")
        assert r.status_code == 200
        assert called["dataset"] == "some-corpus"
