# RQ fan-out/fan-in pattern (LIT-225)

Prototype: `Backend/app/services/fanout_orchestrator_service.py`
Tests: `Backend/tests/test_fanout_orchestrator.py`

## Why this exists

RQ (unlike Celery) has no built-in chord/fan-in primitive. "Dispatch ASR +
SER + ADD, wait for all, aggregate" (the real shape of LIT-150's orchestrator)
has to be composed from RQ's own primitives. This prototype proves the
composition works with three trivial stub jobs before LIT-127/LIT-150 build
the real thing on top of it.

## The pattern

1. **Fan out**: enqueue one job per child task, each on its own queue
   (`CHILD_QUEUE_NAMES`).
2. **Fan in**: enqueue one aggregator job whose `depends_on` is an RQ
   `Dependency(jobs=child_jobs, allow_failure=True)` — **not** a bare list of
   jobs. The `allow_failure=True` is the load-bearing detail: with a bare
   list, RQ never enqueues the aggregator at all once *any* child fails,
   silently discarding the other children's results. With it, the aggregator
   always runs once every child has either finished or exhausted its
   retries, and reports each child as succeeded or failed individually
   (`aggregate_children` in the service module).
3. **Progress**: each child publishes its own progress fraction over a
   per-job pub/sub channel (`fanout:progress:<job_id>`) as it runs. A real
   caller (e.g. a WebSocket relay) subscribes to the channel for the jobs it
   started.
4. **Recovery from a killed worker**: enqueue child jobs with
   `retry=Retry(max=1)`. This needs **no custom recovery code** — RQ's own
   `Worker` forks a work-horse child process per job and monitors it via
   `waitpid()`. If that work-horse is killed (crash, OOM, `kill -9`), the
   parent worker notices the moment it waits on the dead pid, marks the job
   failed, and — because a retry budget remains — immediately requeues it
   instead of losing it. This is proven directly in
   `TestFanoutRecovery.test_recovers_a_killed_job`: it forks a real `Worker`,
   waits for the in-flight job to report its work-horse pid (via
   `job.meta["pid"]`), `SIGKILL`s that pid, and asserts the same job finishes
   successfully afterward with its retry budget consumed.

## What's *not* proven here

- `StartedJobRegistry.add()`/`.cleanup()` — the registry-based recovery
  pattern from older RQ versions — is unimplemented in RQ 2.10 (superseded
  by the heartbeat/fork-monitor mechanism above). Don't reach for it; it
  raises `NotImplementedError`.
- Recovery from an **entire worker process** dying (not just its work-horse)
  is a different, harder failure mode — the process watching the job is
  itself gone, so nothing left alive detects the crash. That needs a
  separate reaper/monitor (stale-worker detection across the fleet) and is
  out of scope for this prototype; flag it if LIT-127/LIT-150 need it.
- The real aggregator will need real result payloads (attribution tensors,
  transcripts, etc.) instead of the trivial stub dict — this prototype only
  proves the control flow, not the data contract.

## Reusing this for LIT-150

Swap `run_stub_child` for the real ASR/SER/ADD task functions and
`CHILD_QUEUE_NAMES`/payload shape for the real per-model queues; the
`enqueue_fanout` / `aggregate_children` / `Dependency(allow_failure=True)` /
`Retry` shape carries over unchanged.

## Running the killed-worker test locally

`TestFanoutRecovery::test_recovers_a_killed_job` needs a real Redis (it
forks an actual OS process and kills it, which fakeredis's in-process store
can't observe across processes). It skips automatically if Redis isn't
reachable — this is the expected outcome in CI, since no Redis service is
provisioned there (see below). To run it locally:

```
cd Backend
docker-compose up -d redis
pytest tests/test_fanout_orchestrator.py -v
```

**Why CI doesn't run it**: adding a `redis:7-alpine` service container to
the `backend-test` CI job was tried and reverted — it made 7 unrelated,
pre-existing tests (rate-limiting/security/performance suites, all of which
call `GET /health`) fail deterministically, every run, even with this
prototype's own test file excluded.

Root cause, confirmed by tracing it: `Backend/app/api/routes/health.py` does
`from ...core.redis import redis` (a direct name import) instead of
importing the module and reading `redis_module.redis`. That binds `health`'s
local `redis` name to the real async client at **import time**, so:

- it bypasses `conftest.py`'s `fake_redis` fixture entirely, which only
  monkeypatches the `redis` attribute on the `app.core.redis` module object
  - `health.py`'s own already-bound name is untouched;
- with no real Redis reachable (the situation in every test run to date),
  `redis.ping()` fails fast as a connection error and `/health` returns a
  harmless 503, which the affected tests explicitly treat as an accepted
  status code;
- the moment a real Redis *is* reachable, `redis.ping()` succeeds against a
  long-lived async connection pool that was opened under whatever asyncio
  event loop was active the first time `/health` was hit - and because this
  repo's pytest-asyncio config hands each async test its own event loop, a
  later test reusing that same pooled connection hits
  `RuntimeError: Event loop is closed` deep in the pool's read path. That
  exception then propagates up through Starlette's middleware stack as an
  `ExceptionGroup`, taking down every concurrent/rate-limit/perf test that
  happens to call `/health`.

This is a pre-existing test-isolation gap, not something LIT-225 introduced
- fixing it is out of scope for a fan-out/fan-in prototype and belongs in
its own issue (fix: import the module and reference `redis_module.redis`,
matching every other route, so the fixture's monkeypatch actually takes).
Flagged to the team rather than fixed here; don't add a real Redis service
to CI until it's resolved.
