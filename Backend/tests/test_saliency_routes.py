"""Endpoint-level tests for /saliency/generate.

Every previous failure of this endpoint was invisible to the unit tests because
they all called ``generate_saliency`` directly and never went through the route:

* the response model was rewritten to require ``success``/``max_val``/
  ``target_class``/``duration_s``/``sample_rate`` - none of which the service
  produces - so pydantic rejected correct results with a 422;
* ``provenance`` was typed as a dict while the LIT-238 contract makes it a
  string enum value;
* an ``UNAVAILABLE`` handler referenced an undefined name and turned a graceful
  degradation into a 500.

The service was right in all three cases. The wiring was wrong, and only a
request exercises the wiring.

Models are patched out deliberately: these assert the *contract*, not the maths.
Attribution correctness lives in test_saliency_service.py and test_grad_cam.py.
"""

from unittest.mock import patch

import pytest

from app.api.routes.saliency import SaliencyResponse


# The exact shape `generate_saliency` returns. Verified against all six return
# paths in app/domain/saliency_service.py - if you change one, change this.
SERVICE_PAYLOAD = {
    "model": "openai/whisper-base",
    "method": "gradcam",
    "segments": [{"start_time": 0.0, "end_time": 1.2, "word": "It", "score": 0.4}],
    "total_duration": 5.04,
    "series": [0.1, 0.5, 0.9],
    "base_spectrogram": [[0.1, 0.2], [0.3, 0.4]],
    "saliency_matrix": [[0.5, 0.6], [0.7, 0.8]],
    "provenance": "measured",
    "provenance_reason": None,
}

# The wav2vec2 paths add one key on top of the above.
SER_PAYLOAD = dict(SERVICE_PAYLOAD, model="wav2vec2", emotion="happy")


class TestResponseContract:
    """The regression: a response model that outran its service."""

    def test_model_requires_nothing_the_service_omits(self):
        required = {
            name
            for name, field in SaliencyResponse.model_fields.items()
            if field.is_required()
        }
        missing = required - set(SERVICE_PAYLOAD)
        assert not missing, (
            f"SaliencyResponse requires {sorted(missing)}, which generate_saliency "
            "never returns. Every such field is a 422 on a correct result."
        )

    def test_accepts_the_real_service_payload(self):
        assert SaliencyResponse(**SERVICE_PAYLOAD).method == "gradcam"

    def test_accepts_the_ser_payload_with_emotion(self):
        assert SaliencyResponse(**SER_PAYLOAD).emotion == "happy"

    def test_provenance_is_a_string_not_a_dict(self):
        """LIT-238 defines provenance as a string enum value."""
        assert SaliencyResponse(**SERVICE_PAYLOAD).provenance == "measured"

    @pytest.mark.parametrize("provenance", ["measured", "fallback", "unavailable"])
    def test_every_provenance_state_serialises(self, provenance):
        payload = dict(SERVICE_PAYLOAD, provenance=provenance, provenance_reason="why")
        assert SaliencyResponse(**payload).provenance == provenance


@pytest.mark.asyncio
class TestSaliencyRoute:
    """Requests, not function calls."""

    async def _post(self, client, **overrides):
        body = {
            "model": "whisper-base",
            "method": "gradcam",
            "dataset": "common-voice",
            "dataset_file": "sample-000775.mp3",
            "no_cache": True,
        }
        body.update(overrides)
        return await client.post("/saliency/generate", json=body)

    async def test_measured_result_returns_200_with_matrices(self, client, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        with patch("app.api.routes.saliency.resolve_audio_reference", return_value=audio), \
             patch("app.api.routes.saliency.generate_saliency", return_value=SERVICE_PAYLOAD):
            r = await self._post(client)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["saliency_matrix"], "empty saliency_matrix reaches the canvas as a blank overlay"
        assert body["base_spectrogram"], "empty base_spectrogram reaches the canvas as a blank base"
        assert body["provenance"] == "measured"

    async def test_fallback_provenance_survives_serialisation(self, client, tmp_path):
        """A flagged fallback must reach the UI still flagged (FR17 / A2)."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        payload = dict(SERVICE_PAYLOAD, provenance="fallback",
                       provenance_reason="attribution empty; showing encoder energy")
        with patch("app.api.routes.saliency.resolve_audio_reference", return_value=audio), \
             patch("app.api.routes.saliency.generate_saliency", return_value=payload):
            r = await self._post(client)
        assert r.status_code == 200, r.text
        assert r.json()["provenance"] == "fallback"
        assert r.json()["provenance_reason"]

    async def test_unknown_method_is_400_not_500(self, client, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        with patch("app.api.routes.saliency.resolve_audio_reference", return_value=audio), \
             patch("app.api.routes.saliency.generate_saliency",
                   side_effect=ValueError("Unsupported saliency method 'nope'")):
            r = await self._post(client, method="nope")
        assert r.status_code == 400, r.text

    async def test_missing_audio_reference_is_400(self, client):
        r = await client.post("/saliency/generate", json={"model": "whisper-base"})
        assert r.status_code == 400, r.text

    async def test_missing_model_is_400(self, client):
        r = await self._post(client, model="")
        assert r.status_code == 400, r.text

    async def test_service_error_becomes_500_with_a_reason(self, client, tmp_path):
        """A NameError in an error path used to surface as an opaque failure."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        with patch("app.api.routes.saliency.resolve_audio_reference", return_value=audio), \
             patch("app.api.routes.saliency.generate_saliency",
                   side_effect=NameError("name 'segment_duration' is not defined")):
            r = await self._post(client)
        assert r.status_code == 500
        assert "segment_duration" in r.text, "the cause must reach the client, not be swallowed"
