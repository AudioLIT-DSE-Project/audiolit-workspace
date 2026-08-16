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
    async def test_includes_esd_now_that_it_has_a_loader(self, client):
        # Was excluded pre-LIT-236 (registered but unwired, would have 404d
        # on every request); ESDLoader landing means it belongs in the list now.
        r = await client.get("/datasets/list")
        assert "esd" in r.json()["datasets"]

    @pytest.mark.asyncio
    async def test_exclusion_filter_still_works_for_a_future_unwired_corpus(self, client, monkeypatch):
        # Exercises the filter mechanism itself, independent of any one
        # corpus's current wiring state (unlike the ESD case above, which is
        # now permanently wired and can't cover this path anymore).
        monkeypatch.setattr(
            datasets_routes.dataset_ingestion, "list_supported_corpora", lambda: ["loadable-a", "unloadable-b"]
        )
        monkeypatch.setattr(datasets_routes, "_UNLOADABLE_CORPORA", {"unloadable-b"})
        r = await client.get("/datasets/list")
        assert r.json()["datasets"] == ["loadable-a"]

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
