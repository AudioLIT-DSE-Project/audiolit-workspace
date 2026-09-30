# Chapter 7 — Orchestration: the task fabric

`Backend/app/orchestration/task_orchestrator.py`, 1681 lines, is the single
queue fabric. Everything asynchronous in AudioLIT goes through it.

The module docstring opens with a warning rather than a description, and that
is the right place to start.

---

## 7.1 The bug this module exists to prevent

```python
"""Task Orchestrator - the single RQ task fabric (SAD §5.1 / §5.2, LIT-230).

SAD §5.2 lists **one** Task Orchestrator component: "Splits a multi-part request
into separate background jobs, runs them together, and combines the results."
This module is that component. It replaces two parallel implementations that
briefly coexisted on ``develop``:

  * ``app/orchestration/rq_broker.py`` (LIT-127) - queue config, GPU pinning,
    progress channel, worker factory. Correct layer, forking worker.
  * ``app/services/queue_service.py`` (LIT-149/157, PR #39) - worker context,
    per-family task functions, fan-in aggregator, enqueue API, job status.
    Correct worker type, wrong layer.

Both landed because the Tier-C ``Path:`` stamps on LIT-127/149/150/225 still
named ``Backend/app/services/queue_service.py``, written before LIT-227 emptied
that directory.
"""
```

Two developers built the same component twice, in two directories, because a
task description pointed at a directory that had since been deleted. Both
passes had green tests. Neither touched the same lines, so git merged them
cleanly with no conflict.

The result was **two task fabrics with different progress-channel prefixes** —
one published to `progress:{id}`, the other to `audiolit:progress:{id}`. A job
enqueued through one was invisible to a subscriber on the other. The job ran
fine. The UI showed a spinner forever.

Three lessons, and all three generalise beyond this codebase:

1. **Green tests on two branches do not prove the combination works.** Each
   fabric was internally consistent.
2. **Git cleanliness is not logical cleanliness.** No conflict, complete
   incompatibility.
3. **A path in a task description is a claim about the tree, and claims go
   stale.** Check the directory exists before writing to it.

Which is why the constant is now a constant, in one place, with a comment
explaining its history:

```python
# Workers publish progress here, keyed by job id; the gateway relays each message
# to the client's open WebSocket. One prefix, one message shape - the two merged
# modules disagreed ("progress:" vs "audiolit:progress"), so a job published by
# one was invisible to a subscriber on the other.
PROGRESS_CHANNEL_PREFIX = "audiolit:progress"
```

---

## 7.2 Queues

```python
class WorkerFamily(str, Enum):
    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"


#: The routes and the multi-task enqueue API speak in "task families"; the
#: worker/queue side speaks in "worker families". They are the same five values.
TaskFamily = WorkerFamily
```

`class WorkerFamily(str, Enum)` inherits from `str`, so a member *is* a string:
`WorkerFamily.ASR == "asr"` is `True`, and it serialises to JSON directly. That
is what lets the same enum cross the HTTP boundary without conversion.

The `TaskFamily = WorkerFamily` alias is honest about an inconsistency. Two
parts of the system named the same concept differently; rather than rename one
and touch many files, the alias records that they are the same thing.

```python
@dataclass(frozen=True)
class QueueConfig:
    family: WorkerFamily
    queue_name: str
    gpu_bound: bool
    concurrency: int
```

`frozen=True` makes instances immutable — configuration that cannot be mutated
at runtime by accident.

Queues are cached per family:

```python
def get_queue(family, connection=None) -> Queue:
    """The RQ queue for a model family.

    Cached per family when using the shared connection; an explicit ``connection``
    (tests, or a second broker) always builds a fresh queue.
    """
    config = get_queue_config(family)
    if connection is not None:
        return Queue(config.queue_name, connection=connection, ...)
    if config.family not in _QUEUES:
        _QUEUES[config.family] = Queue(config.queue_name, connection=get_redis_connection(), ...)
    return _QUEUES[config.family]
```

The `connection` escape hatch is what makes the module testable: tests pass
fakeredis and get an uncached queue, so no state leaks between tests. Combined
with:

```python
def reset_queue_cache() -> None:
    """Drop cached queues. For tests that swap the broker connection."""
    _QUEUES.clear()
```

Four TTLs govern job lifetime:

```python
DEFAULT_JOB_TIMEOUT: int = 600          # kill a job after 10 minutes
DEFAULT_AGGREGATOR_TIMEOUT: int = 120   # the aggregator only combines results
DEFAULT_RESULT_TTL: int = 60 * 60 * 24  # keep results 24 h
DEFAULT_FAILURE_TTL: int = 60 * 60      # keep failures 1 h
```

The aggregator's shorter timeout is a correctness signal as much as a limit: it
only reads three finished results and merges them. If that takes two minutes,
something is wrong and failing fast is better than hanging.

---

## 7.3 Two Redis connections, deliberately

```python
def get_redis_connection() -> Redis:
    """The process-wide synchronous Redis connection used by RQ.

    Pings on first connect so an unreachable broker fails here
    with a clear message rather than at the first enqueue.
    """
    global _CONNECTION
    if _CONNECTION is None:
        url = settings.REDIS_URL
        connection = Redis.from_url(url, decode_responses=False,
                                    socket_connect_timeout=5, socket_timeout=10,
                                    health_check_interval=30)
        try:
            connection.ping()
        except RedisConnectionError as exc:
            logger.error("broker.unreachable url=%s err=%s", sanitize_redis_url(url), exc)
            raise
        logger.info("broker.connected url=%s", sanitize_redis_url(url))
        _CONNECTION = connection
    return _CONNECTION
```

Three details:

**`ping()` on construction.** Without it the first failure appears at the first
enqueue, in a request handler, as a confusing error. With it, the process fails
at startup with "broker.unreachable" — which is a message you can act on.

**`sanitize_redis_url`** strips credentials before logging:

```python
def sanitize_redis_url(url: str) -> str:
    """Strip credentials from a Redis URL so it is safe to log (SAD §11.3)."""
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    except Exception:
        return "redis://***"
```

`redis://user:password@host:6379/0` becomes `redis://host:6379/0`. And the
`except` returns a fully redacted string rather than risking the original
leaking through an error path. If your sanitiser can fail, its failure mode
must also be safe.

**`socket_timeout=10` — and why a second connection exists.** RQ's blocking
dequeue (`BRPOP`) waits up to 405 seconds by default. A 10-second socket
timeout kills that read, so an idle worker dies:

```python
    # A worker's connection must outlast RQ's blocking dequeue (405 s on the
    # defaults); the shared request-path client deliberately times reads out
    # after 10 s, which killed idle workers. See get_worker_redis_connection.
```

Two connections with different timeouts, because the request path *wants* a
short timeout (fail fast rather than hang a request) and the worker path *needs*
a long one. One shared client cannot satisfy both.

Note that the request path also uses an entirely separate **async** client
(`infrastructure/redis.py`, built on `redis.asyncio`), because FastAPI handlers
are async and a synchronous Redis call inside one blocks the event loop. Three
clients total, each for a different execution model.

---

## 7.4 Progress publishing

```python
def progress_channel(job_id: str) -> str:
    """Redis pub/sub channel a job publishes its progress on."""
    return f"{PROGRESS_CHANNEL_PREFIX}:{job_id}"


def publish_progress(job_id, stage, payload=None, *, connection=None) -> int:
    """Publish a progress event for a job; returns the subscriber count.

    ``stage`` carries the state the frontend switches on - QUEUED, PROCESSING,
    RETRYING, SUCCESS, FAILURE - or a finer-grained label such as "asr.running".
    """
    conn = connection or get_redis_connection()
    event = {
        "job_id": job_id,
        "stage": stage,
        "ts": time.time(),
        "payload": dict(payload or {}),
    }
    return conn.publish(progress_channel(job_id), json.dumps(event).encode())
```

**Pub/sub is fire-and-forget.** `publish` delivers to whoever is subscribed
*right now* and returns the count. There is no queue, no history, no
persistence. If nobody is listening, the message is gone.

That is acceptable here because progress is advisory — the authoritative state
is the RQ job itself, which the frontend can query over HTTP. Losing a progress
event costs a UI update, not correctness. If it were load-bearing you would need
a stream, not a channel.

`dict(payload or {})` copies the payload. The caller may reuse or mutate its
dict afterwards; copying means the published event is a snapshot.

One channel per job id, so a subscriber gets exactly one job's events with no
filtering.

---

## 7.5 The worker

### Keeping models in the process

```python
@dataclass
class WorkerContext:
    """Per-worker process state: heavy libraries and loaded models.

    Holding the model here is what SAD §10's "under about 8 seconds" budget for a
    fresh multi-task analysis depends on - reloading a model per job would spend
    most of that budget on loading (§10 separately allows ~60 s just to "download
    and prepare a new model"). SAD §6.1's "each worker only ever needs to hold one
    model in memory" is about not holding *several* models, which this respects:
    one process per family, one family's models in ``models``.
    """
```

That second sentence is a careful piece of reading. A constraint said "each
worker only ever needs to hold one model in memory", which could be read as
"never cache". The comment argues it means "do not hold *several*" — and since
one process serves one family, caching that family's models respects it.

This is what good engineering against a specification looks like: read the
constraint, work out what it is protecting (VRAM), and satisfy that rather than
the literal wording at the cost of the performance budget in the same document.

### SimpleWorker, not the forking Worker

```python
def make_worker(family, connection=None) -> AudioLITWorker:
    """Build a worker bound to one family's queue. Call ``.work()`` to start it.

    Uses ``SimpleWorker`` (in-process) rather than RQ's forking ``Worker``. The
    forking worker runs each job in a work-horse child, so a model loaded in that
    child dies with it and is reloaded on the next job - which makes SAD §10's
    "fresh multi-task analysis under about 8 seconds" unreachable, since §10
    budgets roughly a minute just to prepare a model.

    What forking would buy is recovery from a hard crash of a job. The SAD does
    not ask for that: §11.1 grounds "any part can be restarted without losing
    work" in "the parts communicate only through the shared store", and §7
    packages every worker as its own Docker container - so restart is a container
    concern plus RQ's job registry, not a per-job fork. A CUDA out-of-memory,
    which is the failure §11.1 actually names, raises a catchable Python
    exception and is handled in ``perform_job`` without losing the process.

    If you change this back to a forking ``Worker``, the model cache in
    ``WorkerContext`` stops working - revisit §10's budget first.
    """
```

This is the most carefully justified decision in the module, and the structure
of the argument is worth imitating:

- What the default does, and why it does not work here (fork kills the model
  cache).
- What the default would buy (crash isolation).
- Why that is not needed (restart is handled at the container level; the named
  failure mode is catchable).
- **An explicit warning to the next person**, naming what breaks and what to
  re-check.

The last part is the bit most comments omit. A reader who disagrees now knows
exactly what they must re-examine.

RQ's default `Worker` forks a "work horse" child per job: the child runs the
job and exits. That gives you isolation — a segfault kills the child, not the
worker — at the cost of losing everything in the child's memory, including a
5-second model load, every single job.

### The job lifecycle

```python
    def perform_job(self, job: Job, queue: Queue, *args, **kwargs) -> Any:
        publish_progress(job.id, "PROCESSING", {"family": self.family.value})
        started = time.monotonic()
        try:
            assert self._ctx is not None
            self._ctx.load_libraries()
            ...
            logger.info("task.processing", extra=self._task_event_extra(job, queue))
            result = super().perform_job(job, queue, *args, **kwargs)
            # RQ's perform_job swallows a job-func failure: it marks the job
            # FAILED in its own except and returns False. Record that failure
            # here (job.exc_info holds the traceback RQ captured) and never log
            # a task.success for it - only a True return is a success.
            if result is False:
                error = getattr(job, "exc_info", None) or "job failed (no traceback captured)"
                self._record_task_failure(job, queue, started, error)
                return result
            duration_s = time.monotonic() - started
            ...
            logger.info("task.success", extra=self._task_event_extra(job, queue, duration_s))
            publish_progress(job.id, "SUCCESS", {"duration_s": round(duration_s, 3)})
            return result
        except Exception as exc:
            # RQ retries a transient failure a few times before giving up, and the
            # user is told which of the two happened (SAD §11.1).
            state = "RETRYING" if getattr(job, "retries_left", 0) > 0 else "FAILURE"
            self._record_task_failure(job, queue, started, str(exc))
            publish_progress(job.id, state,
                             {"duration_s": round(time.monotonic() - started, 3), "error": str(exc)})
            raise
```

**The `result is False` check is subtle and important.** RQ's own
`perform_job` catches a job function's exception, marks the job FAILED, and
returns `False`. It does not re-raise. So a naive subclass that only wraps the
call in `try/except` would see no exception, fall through to the success path,
and log `task.success` for a job that failed.

The fix inspects the return value. `False` means RQ handled a failure
internally, so this records the failure and returns without logging success.
Only a truthy return is a success.

This is the sort of thing you only discover by reading the library's source, or
by noticing that your success metric counts more successes than you have.

**`RETRYING` vs `FAILURE`** is distinguished by `job.retries_left`, so the UI
can say "retrying" rather than "failed" for a transient error. A user seeing
"failed" on something that will succeed in two seconds will reload the page,
which is worse than waiting.

**`time.monotonic()`** for durations, never `time.time()` — a clock adjustment
mid-job would otherwise produce a negative duration.

### Structured logging with a privacy rule

```python
    def _task_event_extra(self, job, queue, duration_s=None, error=None) -> dict:
        """The structured-event fields. Only ever reads the job's reserved
        model-id slot - never the audio_ref position (SR6: no audio identities,
        filenames, session ids or transcripts in logs)."""
        extra = {
            "job_id": job.id,
            "family": self.family.value,
            "queue": queue.name,
            "worker": os.getpid(),
            "model_id": _task_model_id(job),
        }
```

And the mechanism for finding the model id without touching anything else:

```python
# where each task function keeps its ``model_id`` argument, for the
# structured task-event logs. The slot is the *only* argument ever read - never
# the ``audio_ref`` position (SR6: no audio identities, filenames, session ids
# or transcripts in logs). Unknown functions simply log ``model_id: null``.
_TASK_MODEL_ID_ARG_INDEX: dict[str, int | None] = {
    "asr_task": 1,
    "ser_task": 1,
    "add_task": 1,
    "xai_task": 1,
    "accent_bias_task": 0,
    "mutation_task": None,
    "aggregator_task": None,
}


def _task_model_id(job: Job) -> str | None:
    """Best-effort ``model_id`` for the log extra, from the reserved arg slot."""
    if job is None or not job.func_name or not job.args:
        return None
    leaf = job.func_name.split(".")[-1]
    index = _TASK_MODEL_ID_ARG_INDEX.get(leaf)
    if index is None or len(job.args) <= index:
        return None
    value = job.args[index]
    return str(value) if value is not None else None
```

The obvious implementation is `job.args` in the log line. That would include
`audio_ref` — a filename, which can identify a person, and in the dataset case
reveals which sample a user is examining. An explicit index table means only
the model id can ever be read, and an unregistered function logs `null` rather
than accidentally dumping its arguments.

**Privacy by construction, not by remembering.** A developer adding a new task
function cannot accidentally log its audio reference, because reading anything
requires adding an entry to this table.

### VRAM hygiene between jobs

```python
    def handle_job_success(self, *args, **kwargs) -> Any:
        # Free VRAM between jobs so a family stays inside the 3-5 GB budget (C2).
        if self._ctx and self._ctx.torch is not None and self._ctx.torch.cuda.is_available():
            self._ctx.torch.cuda.empty_cache()
        return super().handle_job_success(*args, **kwargs)
```

The *model* stays loaded; only PyTorch's cached intermediate allocations are
returned. Without this, fragmentation accumulates across jobs until an
allocation fails despite there being enough total free memory.

---

## 7.6 The GPU family lock

One worker per GPU-bound family is enforced by a distributed lock. The
implementation is hand-rolled, and the docstring explains why:

```python
class _FamilyLock:
    """Token lock with an expiry, renewable from any thread.

    Plain ``SET NX PX`` plus ``WATCH``/``MULTI`` compare-and-set rather than
    redis-py's ``Lock``, whose renew/release run Lua scripts (unsupported by
    the fakeredis the suite and CI run on) and whose token is thread-local by
    default (so a renewal thread could never renew it).
    """
```

Two independent reasons to not use the library lock. The second is the
interesting one: redis-py's `Lock` stores its token in thread-local storage, so
only the acquiring thread can renew it. The renewal here happens on a
*background* thread, which would find no token.

### Acquire

```python
    def acquire(self) -> bool:
        return bool(self._conn.set(self.key, self._token, nx=True, px=int(self.ttl * 1000)))
```

`SET key token NX PX ttl` is the canonical Redis lock primitive: set only if
not exists (`NX`), with an expiry in milliseconds (`PX`), **atomically**. Two
processes racing both issue the command; Redis is single-threaded, so exactly
one succeeds.

The token is a per-instance UUID. That is what makes release safe — you can
verify you still own the lock before deleting it.

### Compare-and-set for renew and release

```python
    def _if_owned(self, action: Callable[[Any], None]) -> bool:
        from redis.exceptions import WatchError

        with self._conn.pipeline() as pipe:
            try:
                pipe.watch(self.key)
                if pipe.get(self.key) != self._token:
                    return False
                pipe.multi()
                action(pipe)
                pipe.execute()
                return True
            except WatchError:
                return False
```

Naively, release is "check the token, then delete". That has a race: between
the check and the delete, the lock could expire and be acquired by another
worker — and you would delete *their* lock.

`WATCH` fixes it. It tells Redis to monitor the key; if anything modifies it
before the `MULTI`/`EXEC` block runs, the transaction aborts with a
`WatchError`. So the check and the action are effectively atomic, without Lua.

`action` is a callable so the same machinery serves both renew (`pexpire`) and
release (`delete`).

### The TTL that caused a day-long outage

```python
#: The per-family GPU lock (SAD C2) lives only as long as its holder keeps
#: renewing it. It used to be taken for 24h and never renewed, so a worker that
#: was OOM-killed (or SIGKILLed by `docker stop`) held its family's lock for a
#: day: every replacement exited with "already has a worker running", and jobs
#: for that family sat queued forever - including dataset warmups.
WORKER_LOCK_TTL = 60
WORKER_LOCK_RENEW_INTERVAL = 20
#: A starting worker waits this long for a dead predecessor's lock to lapse
#: instead of giving up at once.
WORKER_LOCK_ACQUIRE_WAIT = 150
```

A lock's TTL is the answer to *"how long should a crashed holder block
everyone?"* 24 hours answers "a day". 60 seconds with renewal every 20 answers
"a minute".

The renewal interval must be comfortably shorter than the TTL — 20 against 60
survives two missed renewals before expiry. Set them too close and a
garbage-collection pause loses you the lock.

`WORKER_LOCK_ACQUIRE_WAIT = 150` means a starting worker waits 2.5 minutes
rather than exiting immediately, which is what makes `docker restart` work: the
new container starts before the old lock has lapsed, and it waits.

```python
def _acquire_family_lock(conn, fam, *, wait=None, poll=5.0):
    """Take ``fam``'s GPU lock, waiting up to ``wait`` seconds for a holder's
    TTL to lapse (a crashed predecessor stops renewing, so its lock frees
    within WORKER_LOCK_TTL)."""
    lock = _FamilyLock(conn, f"{WORKER_LOCK_PREFIX}:{fam.value}", WORKER_LOCK_TTL)
    deadline = time.monotonic() + (WORKER_LOCK_ACQUIRE_WAIT if wait is None else wait)
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU family {fam.value} already has a worker running (SAD C2)")
        logger.info("worker.lock.waiting family=%s", fam.value)
        time.sleep(min(poll, max(deadline - time.monotonic(), 0)))
    return lock
```

`min(poll, max(deadline - now, 0))` never sleeps past the deadline — so the
final wait is exactly as long as remains, not a full 5-second poll that
overshoots.

### The renewal thread

```python
class _FamilyLockRenewer:
    """Keeps a family lock alive from a daemon thread for exactly as long as
    the worker process lives."""

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._lock.reacquire()
            except Exception as e:
                # Lost the lock (e.g. the process stalled past its TTL and
                # another worker took over). Keep running rather than kill an
                # in-flight job; the other worker holds the lock now.
                logger.error("worker.lock.renew_failed family=%s error=%s", self._fam.value, e)
```

Three deliberate choices:

**`self._stop.wait(interval)` rather than `time.sleep`.** `Event.wait` returns
immediately when the event is set, so `stop()` is responsive. `sleep` would
make shutdown wait up to a full interval.

**`daemon=True`.** The thread does not keep the process alive. A daemon thread
is killed when the main thread exits, which is what you want for a background
heartbeat.

**Losing the lock does not kill the job.** The comment reasons it out: if this
process stalled past its TTL and another worker took over, killing the
in-flight job loses work, and the other worker legitimately holds the lock now.
Continuing is the lesser harm. This is a *degraded but safe* state, chosen
explicitly rather than by omission.

### Putting it together

```python
def run_worker(family, *, burst: bool = False) -> None:
    configure_logging()
    fam = WorkerFamily(family) if not isinstance(family, WorkerFamily) else family
    conn = get_redis_connection()

    # CPU Optimization: Cap PyTorch threads on CPU to avoid thread thrashing across parallel workers
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            logger.info("CPU mode detected: Pinned torch.set_num_threads(1) for worker family %s", fam.value)
    except Exception as e:
        logger.warning("Could not set CPU thread cap: %s", e)

    lock = None
    renewer = None
    if get_queue_config(fam).gpu_bound:
        lock = _acquire_family_lock(conn, fam)
        renewer = _FamilyLockRenewer(lock, fam)
        renewer.start()

    worker = make_worker(fam, connection=conn)
    try:
        worker.work(burst=burst, with_scheduler=True)
    finally:
        if renewer is not None:
            renewer.stop()
        if lock is not None:
            try:
                lock.release()
            except Exception:
                try:
                    conn.delete(f"{WORKER_LOCK_PREFIX}:{fam.value}")
                except Exception:
                    logger.warning("worker.lock.release_failed family=%s", fam.value)
```

**The thread cap is a real CPU-mode issue.** PyTorch defaults to one thread per
core. Five worker processes on an 8-core machine each spawn 8 threads: 40
threads fighting over 8 cores. The context switching costs more than the
parallelism gains. Pinning to 1 thread per worker and letting the *processes*
provide parallelism is measurably faster. `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` cover the underlying BLAS libraries, which have their own
thread pools and ignore `torch.set_num_threads`.

**Only GPU-bound families lock.** Mutation may scale out.

**The nested release fallback.** If `lock.release()` fails (Redis hiccup), it
falls back to a raw `delete`. That is less safe — it could delete another
worker's lock — but the alternative is leaving the lock held for its full TTL
and blocking a restart. A 60-second TTL bounds the damage either way. Note the
inner `try` too: even the fallback cannot raise out of a `finally`.

---

## 7.7 Task functions

Each family has a task function. They share one defensive shape:

```python
def ser_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "ser.running", {"model": model_id})
    try:
        from ..domain.model_loader_service import predict_ser
        res = predict_ser(audio_ref)
        return {
            "task": "ser",
            "model_id": model_id,
            "device": ctx.device,
            "predicted_emotion": res.get("predicted_emotion", "neutral"),
            "probabilities": res.get("probabilities", {}),
            "confidence": float(res.get("confidence", 0.0)),
            "status": "success",
        }
    except Exception:
        return {
            "task": "ser",
            "model_id": model_id,
            "device": ctx.device,
            "predicted_emotion": "neutral",
            "probabilities": {"neutral": 1.0},
            "confidence": 1.0,
            "status": "scaffold",
        }
```

**Function-local imports.** `from ..domain.model_loader_service import predict_ser`
inside the function. This keeps the orchestration module importable without
loading torch — which matters because the API process imports it to enqueue,
and the API process should not pay for torch at import time.

**`status: "success"` vs `"scaffold"`.** The failure path returns a
*shape-complete* result marked `scaffold`, so the aggregator can fan in without
special-casing. The status field is the honesty mechanism — same shape,
different claim.

> **Be aware of what these currently are.** The module says so plainly:
>
> ```python
> # Scaffolding: these publish progress and return a shape the aggregator can fan
> # in, but do not run real inference yet. The real ASR/SER/ADD implementations
> # live in `multitask_orchestrator_service.py` (LIT-150); wiring them onto these
> # per-family queues is the remaining step.
> ```
>
> The synchronous routes (`/inferences/run`, `/saliency/generate`) do real work.
> The queued multitask path is partly scaffolded. If you are rebuilding this,
> know which parts are load-bearing today.

`mutation_task` is fully wired, and the comment records that it was not:

```python
def mutation_task(audio_ref: str, mutation: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a non-destructive audio mutation (SAD Use Case 4, FR12, LIT-164).

    Was a `_scaffold` stub that never touched the audio (flagged on LIT-164);
    now delegates to `perturbation_service.perturb_and_save`, the same
    implementation already used by the synchronous `POST /perturb` route.
    """
```

**The same implementation** as the synchronous route — not a second copy. Two
copies of a mutation pipeline would drift, and then the queued path and the
direct path would produce different audio from the same parameters.

---

## 7.8 The WebSocket relay

`api/routes/tasks.py` bridges Redis pub/sub to the browser.

```python
def _map_rq_status(job: Any) -> str:
    """Map internal RQ status to our frontend state contract."""
    if not job:
        return "FAILURE"
    status = job.get_status()
    if status == JobStatus.QUEUED:
        return "QUEUED"
    if status == JobStatus.STARTED:
        return "PROCESSING"
    if status == JobStatus.FINISHED:
        return "SUCCESS"
    if status == JobStatus.FAILED:
        return "FAILURE"
    if status == JobStatus.DEFERRED:
        return "QUEUED" # Waiting on dependency
    return "UNKNOWN"
```

A translation layer between RQ's vocabulary and the frontend's. `DEFERRED`
(waiting on `depends_on`) maps to `QUEUED`, because from the user's point of
view "waiting for its dependencies" and "waiting in line" are the same
experience. The frontend does not need to know RQ exists.

```python
@router.websocket("/api/ws/tasks/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    conn = get_redis_connection()
    channel = progress_channel(task_id).encode()
    ps = conn.pubsub()
    ps.subscribe(channel)
    loop = asyncio.get_running_loop()

    try:
        # Send initial state immediately upon connection
        job = fetch_job(task_id)
        initial_state = _map_rq_status(job)
        await websocket.send_text(json.dumps({
            "task_id": task_id,
            "state": initial_state,
            "payload": {"result": job.result if job and job.is_finished else None}
        }))

        # If job is already done, close socket after sending final state
        if initial_state in ["SUCCESS", "FAILURE"]:
            await websocket.close()
            return
```

**The initial state solves a genuine race.** The client connects *after*
enqueueing. If the job finished in that gap, no further events will ever be
published, and pub/sub has no history — so the client would wait forever for
something already done. Sending current state on connect closes the gap, and a
terminal state closes the socket immediately.

```python
        while True:
            # Use run_in_executor so redis-py's blocking get_message doesn't block event loop
            msg = await loop.run_in_executor(None, ps.get_message, 1.0)
            if msg is not None and msg.get("type") == "message":
                await websocket.send_text(msg["data"].decode())
                try:
                    parsed = json.loads(msg["data"])
                    if parsed.get("stage") in ["SUCCESS", "FAILURE"]:
                        await websocket.close()
                        break
                except Exception:
                    pass
            await asyncio.sleep(0.01)
```

**`run_in_executor` is the important line.** `ps.get_message(1.0)` is a
*synchronous* blocking call from redis-py's sync client. Calling it directly in
an async handler would block the entire event loop for up to a second — every
other request in the process, stalled. `run_in_executor` runs it on a
threadpool thread and awaits the result, so the loop stays free.

This is the single most common async mistake: a synchronous I/O call inside an
`async def`. It does not error. It just serialises your whole server.

The message is relayed **raw** (`msg["data"].decode()`) rather than re-encoded,
so the worker's event shape reaches the browser unchanged. It is parsed only to
detect a terminal stage, and that parse is in its own `try` — a malformed event
must not break the relay.

`asyncio.sleep(0.01)` yields to the event loop each iteration, letting queued
sends flush and disconnects be noticed.

```python
    finally:
        try:
            ps.unsubscribe(channel)
            ps.close()
        except Exception:
            pass
```

Always unsubscribe. A leaked subscription holds a connection from the pool
forever, and with a pool of 20 that is twenty abandoned browser tabs away from
an outage.

### The HTTP fallback

```python
@router.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """HTTP long-polling fallback for task state (SRS FR3.2)."""
    job = fetch_job(task_id)
    if not job:
        return {"task_id": task_id, "state": "UNKNOWN"}
    return {
        "task_id": task_id,
        "state": _map_rq_status(job),
        "result": job.result if job.is_finished else None,
        "error": str(job.exc_info) if job.is_failed else None,
    }
```

WebSockets fail for reasons outside your control: corporate proxies strip the
upgrade, some networks block the protocol. The fallback reads RQ's job state
directly — the authoritative source — so it is not merely a degraded mode, it
is arguably the more reliable one. Chapter 15 covers the client's retry ladder.

---

## 7.9 Summary

- Two parallel task fabrics were once built from one stale path reference,
  merged without conflict, and used incompatible progress channels. One
  module, one constant, one comment recording it.
- Five queues, one per family; GPU families pinned to concurrency 1.
- Three Redis clients: async for the request path, sync short-timeout for
  enqueueing, sync long-timeout for workers whose blocking dequeue lasts
  minutes.
- `ping()` at construction so an unreachable broker fails at startup with a
  clear message; URLs sanitised before logging, with a redacted fallback if
  sanitising fails.
- Progress is fire-and-forget pub/sub; acceptable only because the RQ job is
  the authoritative state.
- `SimpleWorker` keeps models in process. The alternative is documented, its
  benefit acknowledged, its absence justified, and the next person warned what
  to re-check.
- RQ's `perform_job` returns `False` on a job failure rather than raising —
  check the return value or you will log successes for failures.
- Logs read only a declared argument slot, so audio references cannot leak by
  accident.
- The GPU lock is hand-rolled: `SET NX PX` to acquire,
  `WATCH`/`MULTI` compare-and-set to renew and release, a 60-second TTL with a
  20-second heartbeat, and a 150-second acquire wait so restarts work.
- Losing the lock mid-job logs and continues — the safer degraded state.
- On CPU, pin torch/OMP/MKL to one thread per worker; process parallelism beats
  thread thrashing.
- Task functions import torch lazily and return shape-complete results with a
  `status` field rather than throwing.
- The WebSocket sends current state on connect to close the enqueue/connect
  race, relays raw events, and wraps blocking redis calls in
  `run_in_executor` so the event loop is never blocked.

Next: [Chapter 8 — Caching](08-caching.md).
