"""Load tests for the AudioLIT API, measured against SRS section 3.4.1.

Why this exists alongside tests/test_performance_load.py
-------------------------------------------------------
That pytest module mocks the models and drives concurrency in-process, so it
measures the code path and nothing else. It cannot see what actually degrades
under load on this system: RQ queue depth, Redis round-trips, uvicorn's worker
pool, and the per-model saliency lock that serialises attribution against plain
inference. Those only appear when real HTTP requests contend for a real stack.

This file drives the running server the way a browser does, and scores the
result against the engineering targets the SRS commits to, rather than against
numbers invented here.

Running it
----------
Bring the stack up first (Redis, API, workers - see README section "Step-by-Step
Execution Guide"), then from ``Backend/``::

    locust -f loadtests/locustfile.py --host http://localhost:8000 \
           --headless -u 8 -r 2 -t 3m

``-u`` is concurrent users and ``-r`` the spawn rate. Keep ``-u`` modest on CPU:
attribution is seconds per request, so a high user count measures the queue
backing up rather than the system's response, which is not what the SRS targets
describe.

A hardware caveat, stated in the SRS itself
-------------------------------------------
Section 3.4.1 opens with "GPU figures assume an NVIDIA T4 or equivalent free
cloud tier; CPU fallback is proportionally slower but functional." So the
model-bound targets (cold inference, attribution) are *expected* to be missed on
a CPU box, and failing the run for that would make this suite permanently red
and therefore ignored. Those targets are reported but not enforced unless
``LOADTEST_ENFORCE_MODEL_TARGETS=1``. The infrastructure targets - cached
response, enqueue acknowledgement - are not hardware-bound in the same way and
are always enforced.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

# --------------------------------------------------------------------------- #
# SRS section 3.4.1 - Performance Requirements
# --------------------------------------------------------------------------- #
# `model_bound` marks a target dominated by model inference time, which the SRS
# itself qualifies as GPU-assuming. See the module docstring.

SRS_TARGETS: dict[str, dict] = {
    "cached prediction": {
        "budget_ms": 200,
        "model_bound": False,
        "srs": "API response for a cached request < 200 ms",
    },
    "enqueue multitask": {
        "budget_ms": 50,
        "model_bound": False,
        "srs": "Cache miss to task enqueue < 50 ms",
    },
    "cold prediction": {
        "budget_ms": 3_000,
        "model_bound": True,
        "srs": "Cold ASR inference (Whisper-base, 15 s audio) < 3 s",
    },
    "attribution": {
        "budget_ms": 8_000,
        "model_bound": True,
        "srs": "Interpretability attribution (IG / saliency) < 8 s",
    },
    "acoustic profile": {
        "budget_ms": 2_000,
        "model_bound": True,
        "srs": "Canvas mutation - backend result < 2 s (nearest committed target)",
    },
}

ENFORCE_MODEL_TARGETS = os.getenv("LOADTEST_ENFORCE_MODEL_TARGETS") == "1"

# Below this many observations a p95 is not a percentile, so a target is
# reported but not judged. Raise the run duration, not this number.
MIN_SAMPLES_TO_ENFORCE = int(os.getenv("LOADTEST_MIN_SAMPLES", "20"))

DATASET = "common-voice"
MODEL = "whisper-base"
SALIENCY_METHODS = ("gradcam", "integrated_gradients", "lime", "shap")

# The corpus is read from disk rather than hardcoded, so this keeps working when
# the dataset is re-provisioned. Split so "cached" and "cold" never collide: a
# cold task touching a warmed file would silently measure a cache hit.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "common_voice_valid_dev"
_ALL_CLIPS = sorted(p.name for p in _DATA_DIR.glob("*.mp3"))

WARM_CLIPS = _ALL_CLIPS[:3]
COLD_CLIPS = _ALL_CLIPS[3:]


def _audio_path(filename: str) -> str:
    return str(_DATA_DIR / filename)


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #

@events.test_start.add_listener
def warm_the_cache(environment, **_kwargs):
    """Prime WARM_CLIPS so "cached prediction" measures a hit, not a cold run.

    Without this the first request per clip pays full inference cost and lands
    in the same statistic as the hits, turning a 200 ms target into an
    unmeetable one for reasons that have nothing to do with the cache.
    """
    if not _ALL_CLIPS:
        print(f"[warmup] no clips found under {_DATA_DIR} - is the dataset provisioned?")
        return

    import requests

    host = environment.host or "http://localhost:8000"
    print(f"[warmup] priming {len(WARM_CLIPS)} clip(s) against {host} ...", flush=True)
    for clip in WARM_CLIPS:
        try:
            requests.post(
                f"{host}/inferences/run",
                json={"model": MODEL, "dataset": DATASET, "dataset_file": clip},
                timeout=300,
            )
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            print(f"[warmup] {clip} failed: {type(exc).__name__}: {exc}")
    print("[warmup] done", flush=True)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

@events.test_stop.add_listener
def score_against_srs(environment, **_kwargs):
    """Print a p95-vs-SRS table and fail the run on an enforced breach.

    p95 rather than the mean: the SRS targets describe what a user should
    experience, and a mean hides the tail that people actually complain about.
    """
    stats = environment.stats
    rows, breaches = [], []

    for name, spec in SRS_TARGETS.items():
        entry = stats.get(name, "POST") or stats.get(name, "GET")
        if not entry or entry.num_requests == 0:
            rows.append((name, "-", spec["budget_ms"], "not exercised", "", 0))
            continue

        n = entry.num_requests
        p95 = entry.get_response_time_percentile(0.95)
        within = p95 <= spec["budget_ms"]
        enforced = not spec["model_bound"] or ENFORCE_MODEL_TARGETS

        # A p95 over a handful of samples is the second-worst observation, not a
        # percentile. Failing a build on that trains people to ignore the suite,
        # so below the floor the number is shown and left unjudged.
        if n < MIN_SAMPLES_TO_ENFORCE:
            verdict = f"n={n}, need {MIN_SAMPLES_TO_ENFORCE} to judge"
        elif within:
            verdict = "PASS"
        elif enforced:
            verdict = "FAIL"
            breaches.append(name)
        else:
            verdict = "over (model-bound, not enforced)"

        rows.append((name, round(p95), spec["budget_ms"], verdict, spec["srs"], n))

    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * 92)
    print("SRS 3.4.1 performance targets" + ("" if ENFORCE_MODEL_TARGETS
                                             else "   (model-bound targets reported only)"))
    print("=" * 92)
    print(f"{'operation':<{width}}{'n':>6}{'p95 ms':>9}{'budget':>9}   verdict")
    print("-" * 92)
    for name, p95, budget, verdict, srs, n in rows:
        print(f"{name:<{width}}{n:>6}{str(p95):>9}{budget:>9}   {verdict}")
        if srs:
            print(f"{'':<{width}}{'':>24}   {srs}")
    print("=" * 92)

    failures = stats.total.num_failures
    if failures:
        print(f"{failures} request failure(s) - see the table above for which endpoints.")

    if breaches or failures:
        environment.process_exit_code = 1
        if breaches:
            print(f"FAILED enforced targets: {', '.join(breaches)}")
    else:
        print("All enforced targets met.")


# --------------------------------------------------------------------------- #
# Load profile
# --------------------------------------------------------------------------- #

# Separate user classes rather than one class with weighted tasks. With a single
# class, a user that draws the 13 s acoustic profile is blocked for 13 s and
# stops sampling the fast endpoints - a 90 s run produced 12 cached-prediction
# samples, so its "p95" was really the second-worst of twelve and reported a
# failure that was pure noise. Splitting the roles keeps a population of users
# continuously exercising the fast paths while a smaller population generates
# the heavy load they have to contend with.

class CachedReadUser(HttpUser):
    """Clicking between rows whose results are already cached.

    The dominant interaction in a real session, and the one SRS 3.4.1's 200 ms
    cached-response target describes.
    """

    weight = 10
    wait_time = between(0.5, 1.5)

    @task(9)
    def cached_prediction(self):
        self.client.post(
            "/inferences/run",
            json={
                "model": MODEL,
                "dataset": DATASET,
                "dataset_file": random.choice(WARM_CLIPS),
            },
            name="cached prediction",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="health")


class AsyncEnqueueUser(HttpUser):
    """Fires multi-task jobs and does not wait for them.

    Measures acknowledgement only (SRS 50 ms), not completion. Per SAD 5.1 the
    gateway "never loads AI models directly", so this should stay flat however
    busy the workers are; if it does not, the request path is doing work it
    should have handed to a queue.
    """

    weight = 6
    wait_time = between(1, 2)

    @task
    def enqueue_multitask(self):
        self.client.post(
            "/api/inference/multitask",
            json={
                "audio_ref": _audio_path(random.choice(_ALL_CLIPS)),
                "tasks": ["asr", "ser", "add"],
            },
            name="enqueue multitask",
        )


class HeavyAnalysisUser(HttpUser):
    """Attribution, cold inference and acoustic profiling.

    Deliberately a small slice of the population - this is what a couple of
    analysts running real analyses do to everyone else's latency.
    """

    weight = 2
    wait_time = between(2, 5)

    @task(3)
    def attribution(self):
        """Saliency across all four methods, on warmed clips.

        Warmed so this measures attribution cost rather than attribution plus
        the cold transcription the Grad-CAM path would otherwise run first.
        """
        self.client.post(
            "/saliency/generate",
            json={
                "model": MODEL,
                "method": random.choice(SALIENCY_METHODS),
                "dataset": DATASET,
                "dataset_file": random.choice(WARM_CLIPS),
            },
            name="attribution",
        )

    @task(2)
    def cold_prediction(self):
        if not COLD_CLIPS:
            return
        self.client.post(
            "/inferences/run",
            json={
                "model": MODEL,
                "dataset": DATASET,
                "dataset_file": random.choice(COLD_CLIPS),
            },
            name="cold prediction",
        )

    @task(1)
    def acoustic_profile(self):
        self.client.post(
            "/acoustic/profile",
            json={"dataset": DATASET, "dataset_file": random.choice(WARM_CLIPS)},
            name="acoustic profile",
        )
