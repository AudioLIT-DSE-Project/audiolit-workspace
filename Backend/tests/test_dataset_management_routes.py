"""
Endpoint tests for the custom-dataset file-manipulation routes (LIT-187).

Backend Test Plan Section 3.1.x — server-side responsiveness, data streaming
validation, and file manipulation status codes for
app/api/routes/dataset_management.py (mounted under /upload). No route-level
coverage existed for this file before this issue -- the manager class
(custom_dataset_service.py) has no unit tests of its own either, so these
exercise both layers together through the real HTTP surface.

SESSIONS_BASE_DIR (custom_dataset_service.py) is a relative path that would
otherwise write real files under Backend/uploads/sessions/ -- isolated_sessions_dir
redirects it into pytest's tmp_path so these tests don't touch or depend on
real on-disk state.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure import custom_dataset_service


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")


def _wav_bytes(seconds: float = 0.2, sr: int = 16_000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


async def _session_cookies(client):
    r = await client.get("/session")
    assert r.status_code == 200
    return r.cookies


class TestCreateDataset:
    @pytest.mark.asyncio
    async def test_create_returns_201_with_formatted_name(self, client):
        cookies = await _session_cookies(client)
        r = await client.post(
            "/upload/dataset/create", data={"dataset_name": "my-set"}, cookies=cookies
        )
        assert r.status_code == 201
        body = r.json()
        assert body["original_name"] == "my-set"
        assert body["dataset_name"].startswith("custom:")
        assert body["dataset_name"].endswith(":my-set")
        assert body["metadata"]["total_files"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_create_returns_400(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "dup"}, cookies=cookies)
        r = await client.post("/upload/dataset/create", data={"dataset_name": "dup"}, cookies=cookies)
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_no_cookie_still_succeeds_via_auto_provisioned_session(self, client):
        # SessionMiddleware.dispatch calls ensure_session() unconditionally on
        # every request, so require_session_id's 400 branch is unreachable
        # through the real app -- a session is silently auto-created, and the
        # response carries a Set-Cookie for the client to persist it.
        r = await client.post("/upload/dataset/create", data={"dataset_name": "no-session"})
        assert r.status_code == 201
        assert "set-cookie" in r.headers


class TestListDatasets:
    @pytest.mark.asyncio
    async def test_list_empty_session_returns_empty_list(self, client):
        cookies = await _session_cookies(client)
        r = await client.get("/upload/dataset/list", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["datasets"] == []
        assert r.json()["total_datasets"] == 0

    @pytest.mark.asyncio
    async def test_list_reflects_created_datasets(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "a"}, cookies=cookies)
        await client.post("/upload/dataset/create", data={"dataset_name": "b"}, cookies=cookies)
        r = await client.get("/upload/dataset/list", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["total_datasets"] == 2
        names = {d["dataset_name"] for d in r.json()["datasets"]}
        assert names == {"a", "b"}


class TestUploadFilesToDataset:
    @pytest.mark.asyncio
    async def test_upload_valid_wav_returns_200(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "audio-set"}, cookies=cookies)

        files = [("files", ("clip.wav", _wav_bytes(), "audio/wav"))]
        r = await client.post("/upload/dataset/audio-set/files", files=files, cookies=cookies)

        assert r.status_code == 200
        body = r.json()
        assert body["total_files"] == 1
        assert body["uploaded_files"][0]["filename"] == "clip.wav"
        assert body["dataset_metadata"]["total_files"] == 1

    @pytest.mark.asyncio
    async def test_upload_duplicate_filename_gets_suffixed(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "dupes"}, cookies=cookies)

        files = [("files", ("clip.wav", _wav_bytes(), "audio/wav"))]
        r1 = await client.post("/upload/dataset/dupes/files", files=files, cookies=cookies)
        r2 = await client.post("/upload/dataset/dupes/files", files=files, cookies=cookies)

        assert r1.json()["uploaded_files"][0]["filename"] == "clip.wav"
        assert r2.json()["uploaded_files"][0]["filename"] == "clip_1.wav"

    @pytest.mark.asyncio
    async def test_upload_wrong_extension_returns_400(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "bad-ext"}, cookies=cookies)

        files = [("files", ("clip.ogg", _wav_bytes(), "audio/ogg"))]
        r = await client.post("/upload/dataset/bad-ext/files", files=files, cookies=cookies)
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_non_audio_content_type_returns_400(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "not-audio"}, cookies=cookies)

        files = [("files", ("clip.wav", b"not really audio", "text/plain"))]
        r = await client.post("/upload/dataset/not-audio/files", files=files, cookies=cookies)
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_to_nonexistent_dataset_returns_404(self, client):
        cookies = await _session_cookies(client)
        files = [("files", ("clip.wav", _wav_bytes(), "audio/wav"))]
        r = await client.post("/upload/dataset/ghost/files", files=files, cookies=cookies)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_no_files_returns_400(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "empty-upload"}, cookies=cookies)
        r = await client.post("/upload/dataset/empty-upload/files", files=[], cookies=cookies)
        assert r.status_code in (400, 422)  # FastAPI 422s an empty required List[UploadFile]


class TestDatasetMetadataAndFiles:
    @pytest.mark.asyncio
    async def test_metadata_for_existing_dataset(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "meta-set"}, cookies=cookies)
        r = await client.get("/upload/dataset/meta-set/metadata", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["dataset_name"] == "meta-set"

    @pytest.mark.asyncio
    async def test_metadata_for_missing_dataset_returns_404(self, client):
        cookies = await _session_cookies(client)
        r = await client.get("/upload/dataset/ghost/metadata", cookies=cookies)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_list_files_for_existing_dataset(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "files-set"}, cookies=cookies)
        files = [("files", ("clip.wav", _wav_bytes(), "audio/wav"))]
        await client.post("/upload/dataset/files-set/files", files=files, cookies=cookies)

        r = await client.get("/upload/dataset/files-set/files", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["total_files"] == 1

    @pytest.mark.asyncio
    async def test_list_files_for_missing_dataset_returns_404(self, client):
        cookies = await _session_cookies(client)
        r = await client.get("/upload/dataset/ghost/files", cookies=cookies)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_serve_existing_file_returns_200(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "serve-set"}, cookies=cookies)
        files = [("files", ("clip.wav", _wav_bytes(), "audio/wav"))]
        await client.post("/upload/dataset/serve-set/files", files=files, cookies=cookies)

        r = await client.get("/upload/dataset/serve-set/files/clip.wav", cookies=cookies)
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"

    @pytest.mark.asyncio
    async def test_serve_missing_file_returns_404(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "serve-set2"}, cookies=cookies)
        r = await client.get("/upload/dataset/serve-set2/files/ghost.wav", cookies=cookies)
        assert r.status_code == 404


class TestDeleteAndCleanup:
    @pytest.mark.asyncio
    async def test_delete_existing_dataset_returns_200(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "to-delete"}, cookies=cookies)
        r = await client.delete("/upload/dataset/to-delete", cookies=cookies)
        assert r.status_code == 200

        # Actually gone, not just reported gone.
        follow_up = await client.get("/upload/dataset/to-delete/metadata", cookies=cookies)
        assert follow_up.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_missing_dataset_returns_404(self, client):
        cookies = await _session_cookies(client)
        r = await client.delete("/upload/dataset/ghost", cookies=cookies)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_cleanup_returns_200(self, client):
        cookies = await _session_cookies(client)
        await client.post("/upload/dataset/create", data={"dataset_name": "cleanup-me"}, cookies=cookies)
        r = await client.post("/upload/dataset/cleanup", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["success"] is True

        # Session's datasets are actually gone after cleanup.
        listing = await client.get("/upload/dataset/list", cookies=cookies)
        assert listing.json()["total_datasets"] == 0
