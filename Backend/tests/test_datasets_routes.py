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
            datasets_routes,
            "load_metadata",
            lambda dataset, session_id, **kwargs: called.setdefault("dataset", dataset) or [],
        )
        r = await client.get("/some-corpus/metadata")
        assert r.status_code == 200
        assert called["dataset"] == "some-corpus"

    @pytest.mark.asyncio
    async def test_includes_licenses_for_non_commercial_corpora(self, client):
        # LIT-237, FR2.3: non-commercial corpora must be flagged so the
        # frontend can render a licence notice.
        r = await client.get("/datasets/list")
        licenses = r.json()["licenses"]
        assert licenses["ravdess"]["non_commercial"] is True
        assert licenses["ravdess"]["license"]
        assert licenses["common-voice"]["non_commercial"] is False

    @pytest.mark.asyncio
    async def test_includes_availability_per_corpus(self, client):
        r = await client.get("/datasets/list")
        available = r.json()["available"]
        assert set(available) == set(r.json()["datasets"])
        assert all(isinstance(v, bool) for v in available.values())

    @pytest.mark.asyncio
    async def test_metadata_query_params_pass_limit_and_offset_through(self, client, monkeypatch):
        seen = {}

        def _fake_load_metadata(dataset, session_id, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(datasets_routes, "load_metadata", _fake_load_metadata)
        r = await client.get("/some-corpus/metadata?limit=5&offset=10")
        assert r.status_code == 200
        assert seen == {"limit": 5, "offset": 10}


class TestDatasetsFootprint:
    @pytest.mark.asyncio
    async def test_returns_usage_and_limit(self, client):
        r = await client.get("/datasets/footprint")
        assert r.status_code == 200
        body = r.json()
        assert "per_dataset_bytes" in body
        assert "total_bytes" in body
        assert "limit_bytes" in body
        assert isinstance(body["over_limit"], bool)

    @pytest.mark.asyncio
    async def test_limit_bytes_reflects_settings(self, client, monkeypatch):
        monkeypatch.setattr(datasets_routes.settings, "DATASET_FOOTPRINT_LIMIT_GB", 1.0)
        r = await client.get("/datasets/footprint")
        assert r.json()["limit_bytes"] == 1024 ** 3

    @pytest.mark.asyncio
    async def test_over_limit_true_when_usage_exceeds_configured_limit(self, client, monkeypatch):
        monkeypatch.setattr(
            datasets_routes.dataset_ingestion, "measure_footprint", lambda: {"big-corpus": 500}
        )
        monkeypatch.setattr(datasets_routes.settings, "DATASET_FOOTPRINT_LIMIT_GB", 1e-9)
        r = await client.get("/datasets/footprint")
        body = r.json()
        assert body["total_bytes"] == 500
        assert body["over_limit"] is True
