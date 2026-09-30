# Chapter 16 — Testing and verification

63 test files on the backend, 9 Jest suites on the frontend, a Playwright suite,
a Postman collection and a Locust load test. This chapter is about **what kinds
of bug this system is prone to, and which tests catch which** — because the
answer is not obvious, and the default kinds of test miss most of them.

---

## 16.1 The bug class this project actually has

Go back through the handbook and list the worst defects:

| Defect | Did it crash? | Was the output wrong? |
|---|---|---|
| Random SER classifier head | no | yes — noise |
| Every Grad-CAM was an energy map | no | yes — wrong explanation |
| Whisper LIME collapsed to zeros | no | yes — wrong explanation |
| Fabricated attention | no | yes — fiction |
| Fabricated faithfulness metric | no | yes — certifies noise |
| Deepfake model anti-correlated | no | yes — worse than guessing |
| WER without normalisation | no | yes — bias index off 56% |
| Language misdetection | no | yes — measured the wrong thing |
| Model cache with zero callers | no | **no** — just 73% slower |
| Embedding routed to wrong extractor | no | yes — wrong latent space |
| Projection row reorder (hypothetical) | no | yes — every label wrong |

**Not one of them crashes.** Every single one returns a well-formed,
plausible-looking, wrong answer.

This is the defining property of a machine-learning interpretability system:
**its output is not verifiable by inspection.** If a web form returns the wrong
total, you notice. If a heatmap highlights the wrong region, it looks like a
heatmap.

Which means the default testing instinct — assert the function returns
something of the right type — catches almost nothing here. From the defect log:

> *The defects that mattered most were not crashes. D04, D05, D06 and D07 all
> returned a confident, well-formed, wrong answer, and D04 actively fabricated
> evidence — it showed one model's attention as another's. In an
> interpretability tool that is the worst available failure mode, because the
> output's whole purpose is to be trusted as measurement. Unit tests calling
> functions directly could not see any of them; each needed either a second
> call to compare against or a real end-to-end path.*

**"Either a second call to compare against or a real end-to-end path."** That
is the actionable conclusion, and the rest of this chapter is about how to get
one or the other.

---

## 16.2 Six techniques that actually catch these

### 1. Differential testing — compare two calls

A single call has nothing to be wrong *relative to*. Two calls do.

```python
def test_pca_and_tsne_do_not_produce_the_same_projection(self):
    embs = _embeddings(n=20, seed=3)
    pca = reduce_dimensions(embs, method="pca", n_components=2)
    tsne = reduce_dimensions(embs, method="tsne", n_components=2)

    assert pca.shape == tsne.shape
    # Two different algorithms on the same input must differ. Equality here
    # would mean one label is lying about which ran.
    assert not np.allclose(pca, tsne)
```

This does not check either algorithm is *correct*. It checks the **method
parameter reaches the reducer**. If dispatch broke and everything fell through
to PCA, shapes would match, determinism would hold, clusters would separate —
and only this test would fail.

The same technique applies to the Grad-CAM/IG confusion: assert the two
produce *different* outputs for the same input. They are different algorithms;
equality means one label is wrong.

```python
def test_a_second_model_gets_its_own_entry(self, monkeypatch):
    a = ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)
    b = ml._get_whisper_pipeline("openai/whisper-small", -1, torch.float32)
    assert _FakePipe.builds == 2
    # Two checkpoints must not collide on one cache entry, or selecting a
    # different model would silently return the first model's pipeline.
    assert a is not b
```

Same shape: two models must not be the same object.

### 2. Counting stubs — make invisible behaviour observable

The dead pipeline cache (§6.2) produced correct output. No correctness
assertion could find it.

```python
class _FakePipe:
    """Stands in for an HF pipeline; records how many were constructed."""
    builds = 0

    def __init__(self, model_id: str):
        self.model_id = model_id
        type(self).builds += 1

    def __call__(self, audio, **kwargs):
        return {"text": f"transcript from {self.model_id}"}


def _install_counting_pipeline(monkeypatch):
    """Replace transformers.pipeline with a counter, and clear the cache."""
    _FakePipe.builds = 0
    monkeypatch.setattr(ml, "_pipeline_cache", {})

    def fake_pipeline(task, model=None, **kwargs):
        return _FakePipe(model)

    monkeypatch.setattr(ml, "pipeline", fake_pipeline)
```

```python
def test_repeated_calls_build_the_model_once(self, monkeypatch):
    _install_counting_pipeline(monkeypatch)
    for _ in range(4):
        ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)
    assert _FakePipe.builds == 1, (
        f"built the pipeline {_FakePipe.builds} times for one model; the "
        "cache is not being consulted"
    )
```

`type(self).builds += 1` increments the *class* attribute, so the count is
shared across instances. `_FakePipe.builds = 0` resets per test.

Note `monkeypatch.setattr(ml, "_pipeline_cache", {})` — it clears module state
so tests do not leak into each other. Module-level caches are the most common
source of order-dependent test failures.

**General technique: when the behaviour you care about is a side effect
(caching, reuse, call counts), instrument the dependency and assert on the
instrument.**

### 3. Design the test around what a wrong answer would look like

The row-order test (§12.6) is the exemplar:

```python
def test_row_order_is_preserved(self):
    """Row i of the output must be the projection of row i of the input.

    The panel maps output rows back to filenames by position, so a reorder
    would attach every point to the wrong file with no visible symptom.
    """
    far = rng.normal(loc=20.0, scale=0.1, size=dim).astype("float32")
    near = [rng.normal(loc=0.0, scale=0.1, size=dim).astype("float32") for _ in range(6)]

    # Put the outlier last; it must still be the last output row.
    out = reduce_dimensions(near + [far], method="pca", n_components=2)
    distances = np.linalg.norm(out - out[:-1].mean(axis=0), axis=1)
    assert int(np.argmax(distances)) == len(out) - 1
```

The author asked: *if the rows were reordered, what would I see?* Answer:
nothing — a perfect-looking plot with wrong labels. So the test constructs an
input where a reorder becomes detectable, by planting an identifiable outlier
in a known position.

### 4. Verify the test by breaking the code

Every important test in this project was checked by reverting the fix:

```
Verified by reverting: 2 fail with the right diagnostics.
```

```
Verified by disabling the check: the rejection test fails.
```

```
Verified by removing the call.
```

**An untested test is not a test.** A test can pass because the code works, or
because it asserts nothing meaningful. Breaking the code deliberately is the
only way to tell the two apart, and it takes thirty seconds.

This is mutation testing applied by hand, and for a small number of critical
tests it is entirely practical.

### 5. Contract tests over cache shapes

The cache's failure mode is a right key with a wrong value (§8.1).
`test_warmup_cache_contract.py` — 26 tests — asserts that what a writer stores
is what a reader expects. That is not a property of either side alone, so
neither side's unit tests can check it.

### 6. Failure-path tests

The `logger` bug (§4.3) survived because nothing ever executed the
decode-failure branch.

```python
class TestDecodeRejection:
    """An undecodable body must be reported as such, not as a server error.

    The handler for this path logged before raising, and `logger` was never
    defined in the module, so the handler itself raised NameError. The outer
    `except Exception` caught that and returned
    `500 name 'logger' is not defined` - an internal error for what is
    actually a bad request, with the real reason hidden from the user and the
    422 branch unreachable. Found by sending a non-audio body, which no test
    had done.
    """

    async def test_a_non_audio_body_is_rejected_as_unprocessable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/upload",
                files={"file": ("broken.wav", b"not audio at all" * 50, "audio/wav")},
                data={"model": "whisper-base"},
            )
        assert response.status_code == 422
        assert "decoded" in response.json()["detail"]
        assert not list(tmp_path.iterdir()), "the undecodable file was left on disk"
```

**Error paths are code.** Code that never runs is code that has never been
tested. The happy path here ran thousands of times; this branch had run zero.

The third assertion is worth noting too — it checks the *cleanup* happened, not
just the status code. A rejection that leaves a file behind is a slow leak.

---

## 16.3 Test infrastructure

### pytest configuration

```ini
[pytest]
pythonpath = .
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
timeout = 300
timeout_method = thread
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: ...
    security: ...
    performance: ...
    critical: ...
    important: ...
```

**`asyncio_mode = auto`** means `async def` tests run without an explicit
`@pytest.mark.asyncio` on each.

**`--strict-markers`** makes an unregistered marker an error. Without it,
`@pytest.mark.slwo` (typo) silently does nothing, and you believe you have
marked a test that you have not.

**The timeout, and why it exists:**

```ini
# A per-test ceiling, not a performance target. The orchestrator tests have
# hung outright three times in one session: a burst-mode SimpleWorker draining a
# dependency-gated aggregator against fakeredis stops making progress and never
# returns. Without a timeout that stalls the whole run, so CI burns its entire
# job allowance and reports nothing. 300 s is far above the slowest legitimate
# test here (the worker-supervision respawn tests take ~70 s on Windows), so
# this only ever fires on a genuine hang.
```

A hung test is worse than a failing one: CI burns its whole allowance and
reports *nothing*, so you learn less than if the suite had failed immediately.
The timeout converts a hang into a failure.

300 seconds is chosen with the slowest legitimate test measured (~70 s), so the
ceiling never fires spuriously. A timeout tuned too tightly produces flaky
failures, which erodes trust in the suite faster than slow tests do.

`timeout_method = thread` because the signal-based method does not work on
Windows.

### The deadlock that took three sessions to find

```python
@pytest.fixture(autouse=True, scope="session")
def _no_pipeline_finaliser():
    """`redis.client.Pipeline.__del__` calls `reset()`, which sends UNWATCH on its
    connection. When the garbage collector fires that finaliser while fakeredis is
    mid-command, the UNWATCH re-enters a socket that is not re-entrant, and the
    process deadlocks.
    """
    original = Pipeline.__del__
    Pipeline.__del__ = lambda self: None
    try:
        yield
    finally:
        Pipeline.__del__ = original
```

Worth understanding in full, because it is the hardest bug in the project:

1. redis-py's `Pipeline` has a `__del__` finaliser that calls `reset()`.
2. `reset()` sends `UNWATCH` over the connection.
3. Python's garbage collector runs `__del__` at **unpredictable times** —
   including in the middle of another Redis command.
4. fakeredis's socket emulation is not re-entrant.
5. Re-entering it deadlocks the process.

Why it was so hard: it depends on **GC timing**, which depends on memory
pressure, which depends on what else is running. So it never reproduced when
the tests ran alone, only during a full suite. Three sessions attributed it to
coverage instrumentation.

The fix neutralises the finaliser for the test session only, restoring it in
`finally`. Plus `gc.disable()` around the specific worker drains that provoked
it.

The general lesson: **an intermittent hang that only appears under load is
often a garbage-collection-timing interaction.** If a bug correlates with "how
much else is running" rather than with any input, suspect finalisers.

### The Redis fake

```python
@pytest.fixture(autouse=True, scope="function")
async def fake_redis(monkeypatch):
```

`autouse=True` means **every** test gets fakeredis whether it asks or not. That
is deliberate: a test that accidentally hits real Redis pollutes a developer's
local state and behaves differently on CI.

`scope="function"` gives each test a clean database, so no ordering
dependencies.

> **Know this gap:** the autouse fixture patches
> `app.infrastructure.redis.redis` (the async client). Code that calls a
> task-orchestrator function touching `publish_progress` or
> `get_redis_connection` needs a `broker` fixture too, which patches
> `rq_connection._CONNECTION`. And `broker` lives in individual test files, not
> in `conftest.py`, so a new test module has to bring its own.
>
> The trap: "I mocked the domain call, so no Redis is involved" is wrong —
> check the orchestrator wrapper, not just the function you mocked.

### Running against no Redis at all

```bash
REDIS_URL="redis://127.0.0.1:1/0" pytest -q
```

Port 1 is guaranteed closed. This is mandatory before pushing, and the reason
is specific:

> *CI has no Redis service, and a locally-reachable Redis masks failures that
> only bite on CI.*

If your machine has Redis running, a test that accidentally uses it passes
locally and fails on CI. Pointing at a closed port makes your local environment
match CI's.

**Make your local environment match CI in the ways that matter, or CI becomes a
lottery.**

---

## 16.4 The five test layers

### Backend unit tests — 63 files

Domain logic, called directly. Fast, no network, no models.

```python
def _embeddings(n: int = 12, dim: int = 64, seed: int = 0) -> list:
    """n distinct high-dimensional vectors, as the extractor would return."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=dim).astype("float32") for _ in range(n)]
```

Synthetic data with a fixed seed. `np.random.default_rng(seed)` is a local
generator — it does not disturb global random state, so tests cannot affect each
other through it.

Assertion messages carry the *reason*, not just the fact:

```python
assert out.shape == (12, n_components), (
    f"projected {len(embs)} embeddings to {out.shape}; the plot binds "
    "each row to a file, so a row count mismatch mislabels every point"
)
```

A failure tells you why it matters, which is what you need at 2am when you did
not write the test.

### Backend route tests — httpx

```python
async with AsyncClient(app=app, base_url="http://test") as client:
    response = await client.post("/upload", files={...}, data={...})
```

The whole ASGI stack — middleware, validation, serialisation — with no network.

> **A real trap here.** `TestClient` (Starlette's synchronous client) drives the
> app on its own portal event loop. That binds the session middleware's async
> redis pool to *that* loop, and then the conftest fixture's teardown —
> running on pytest-asyncio's loop — fails with
> `<Queue ...> is bound to a different event loop`.
>
> The symptom is a teardown error on every test that made a request, with the
> tests themselves passing. `httpx.AsyncClient` runs on the same loop as the
> fixtures and avoids it entirely. Match your client to your fixture's event
> loop.

### Frontend unit tests — Jest

9 suites, 75 tests. ts-jest plus jsdom. The path alias must be mirrored in
`jest.config.cjs` or imports resolve in the app and fail in tests.

The heatmap module is separate from the component specifically so its colour
maths — including the luminance monotonicity the requirement demands — can be
tested without rendering anything (§15.5).

### End-to-end — Playwright

```ts
projects: [
  { name: "chromium", use: { ...devices["Desktop Chrome"] },
    testIgnore: /(dataflow|accessibility)\.spec\.ts/ },
  { name: "firefox", ... },
  { name: "webkit", ... },
  { name: "accessibility", use: { ...devices["Desktop Chrome"] },
    testMatch: /accessibility\.spec\.ts/ },
  { name: "dataflow", use: { ...devices["Desktop Chrome"] },
    testMatch: /dataflow\.spec\.ts/, timeout: 180_000 },
]
```

Three projects for layout across three engines; two single-engine projects with
documented reasoning:

```ts
// Opt-in: `npm run test:e2e:dataflow`. Requires the backend, Redis and the
// RQ workers to be running. Chromium only - what these assert is that data
// survives the round trip from model to panel, which is not a per-browser
// property; running it three times would triple a suite whose individual
// requests already take tens of seconds on CPU.

// Axe-core WCAG 2.1 AA scan. Backend-free like the layout suite, but run
// once on Chromium rather than per-browser: axe evaluates the same DOM and
// the same computed styles in each engine, so tripling it would treble the
// runtime for near-identical findings.
```

**Decide per suite whether cross-browser matters.** Layout does — engines differ
in flexbox and font metrics. Data round-tripping does not. Saying which and why
is what makes the choice reviewable rather than arbitrary.

### The `networkidle` trap

Both the accessibility spec and an existing quickstart spec waited for
`networkidle`. It **never fires** in this app, because the WebSocket keeps a
connection open indefinitely. The tests hung until timeout.

The fix waits for a specific element that proves the workbench rendered:

```ts
// wait for the panel group, not networkidle: the WebSocket keeps a
// connection open so networkidle never fires.
await page.waitForSelector('[data-panel-group]');
```

**Wait for a condition that means what you need**, not for a proxy that happens
to work in simpler apps.

### API tests — Postman/newman

13 requests, 39 assertions, run headless:

```bash
npx newman run Backend/apitests/AudioLIT.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:8000
```

An **independent harness**. The pytest route tests and the application share a
Python process, a serialisation layer and a set of assumptions. A separate tool
speaking raw HTTP catches contract problems that in-process tests cannot see —
a wrong content type, a missing header, a field that serialises differently
than you expect.

`newman` is invoked via `npx` rather than being a declared dependency, because
as a devDependency it pulled in 14 security advisories including the only
critical one in the tree.

### Load tests — Locust

`Backend/loadtests/locustfile.py`, on the `testing` branch. Concurrent-user
simulation against the enqueue path, which is where the tight latency budget
applies.

---

## 16.5 Measurement is harder than testing

The defect log has a section that belongs in every engineering handbook:

```
Three findings during this work were **wrong on first measurement** and were
corrected before they reached a fix:

- An enqueue latency reported at 2100 ms (42x budget) was an artefact of
  `urllib` splitting headers and body across packets; `requests` measured
  23.6 ms. Caught because the load test said 41 ms and the contradiction was
  chased rather than averaged away.
- Acoustic profiling "taking 13 s" was a cold-start run; warm it is 0.1 s.
- A claimed ~1.8x saliency slowdown disappeared once the untouched control path
  was measured and had moved by the same factor - it was machine load.
```

Three false findings. Each would have caused work on a problem that did not
exist.

**Every one was caught by a contradiction**, and the first is the model case:
two tools reported 2100 ms and 41 ms for the same thing. Averaging them, or
picking the alarming one, would have sent someone optimising a code path that
was already fast. Chasing the disagreement found a measurement artefact.

> *This is recorded because it bears on how the rest of the table should be
> read: no single timing observation was treated as evidence anywhere above.*

The third case shows the discipline: **measure a control.** The saliency path
looked 1.8× slower — until the untouched path was measured and had also moved
1.8×. The machine was loaded. Without a control you cannot distinguish "my
change is slow" from "this machine is busy".

### A synthetic benchmark that proved the opposite of the truth

```
| shape                | p50   | p95      |
| `async def`          | 44 ms | 70-75 ms |
| `def` plus threadpool| 70 ms | 120 ms   |

The synthetic test had used a 200 ms stub. A real loopback round trip is 1.67 ms,
and per-request thread dispatch costs more than that. The change was reverted and
the test deleted, because a test asserting the slower shape is worse than no
test.

The lesson generalises: a synthetic benchmark whose parameters do not match
production can prove the opposite of the truth, confidently.
```

A synthetic benchmark with a 200 ms stubbed operation showed threadpooling was
faster. In production the operation takes 1.67 ms, and thread dispatch costs
more than that — so threadpooling was **1.7× slower**. The benchmark was
internally valid and answered a question about a system that did not exist.

And note the disposition: the test was **deleted**, not left passing. *"A test
asserting the slower shape is worse than no test"* — it would actively prevent
the correct implementation.

---

## 16.6 CI

```yaml
jobs:
  frontend-lint-and-build:   # npm ci, npm audit, eslint, jest, vite build
  backend-test:              # pip install, verify CPU torch, pytest, pip-audit
  frontend-e2e:              # playwright install, 3 engines, accessibility
  docker-image-scan:         # build both images, Trivy at CRITICAL
```

Details worth copying:

**Dependency scanning on every build.** `pip-audit` against the installed
environment (so what is scanned is what the tests ran against) and `npm audit
--omit=dev --audit-level=high` on production dependencies. The dev side is
deliberately not gated, and the reason is stated in the workflow rather than
left implicit.

**`pip-audit` runs after pytest**, so a new advisory never masks a test failure.

**CPU-only torch, verified:**

```yaml
- name: Verify torch is the CPU build
  run: python -c "import torch; print('torch', torch.__version__, '| cuda_available', torch.cuda.is_available())"
```

The default Linux torch wheel is the multi-gigabyte CUDA build; runners are
CPU-only. Installing the CPU wheel from PyTorch's index shrinks the download
and the cache — and the verification step means a silent revert to the CUDA
wheel is visible rather than just slow.

**Trivy with `ignore-unfixed`:**

```yaml
# ignore-unfixed keeps the gate opinionated: only a fixable CRITICAL CVE fails
# the run (an unfixable one in a pinned base image is a bump-the-tag decision,
# not a red CI).
```

A gate that fires on things you cannot fix trains people to ignore it.

---

## 16.7 The incidents that testing alone cannot prevent

### Two green PRs, one broken combination

> *PR #10 added `app/core/rq_connection.py` importing `app.core.settings`;
> PR #13 separately moved `settings.py` out of `app/core/`. Neither touched the
> same lines, so they merged with no conflict at all — and broke `pytest`
> collection for everyone until a third PR fixed it.*

Both branches were green. Neither test suite could have caught it, because
neither branch contained the problem. Only the *combination* did, and git saw no
conflict because different files changed.

The mitigation is procedural: after someone else's merge lands, merge it into
your branch and run the full suite again before pushing. A clean *git* merge can
be a broken *logical* merge.

> *A develop-merge on PR #27 kept both a new and an old assertion and shipped a
> failing test.*

### A stale path built a module twice

Covered in §7.1. A task description named a directory that had been deleted, two
people implemented the same component in two places, both green, no conflict,
incompatible at runtime.

The mitigation is one command: `ls` the directory before writing to it.

### Status is not evidence

> *LIT-7 ("Setup the Echo 1.0") was marked Done, but the real ECHO 1.0 codebase
> had never actually been merged into this repo — `develop` carried a parallel
> from-scratch scaffold instead. Caught by comparing git history against the
> real fork, not by trusting the status.*

A tracker says what someone *believed*. The repository says what *is*. When it
matters, spend the one command.

### Tooling lies too

> *`gh pr view --json additions,deletions,changedFiles` can return stale counts
> captured when the PR was opened — on two PRs it reported ~180 files/+38k when
> the real diffs were 5 files/+577 and 2 files/+188, and a review round
> confidently repeated both numbers before anyone measured.*

Use `git diff --shortstat origin/develop...<head>`. And do not reason about what
a merge will do — run it in a throwaway worktree and look.

---

## 16.8 Current state

Verified on 2026-09-30:

| Check | Result |
|---|---|
| Backend suite, Redis unreachable | **1 failed**, 782 passed, 7 skipped (measured 2026-09-30, 9m29s) |
| Coverage | 70 % |
| Frontend `npm ci && lint && test && build` | pass, 75 Jest tests |
| `pip-audit` | clean |
| `npm audit --omit=dev` | 0 vulnerabilities |
| Playwright (chromium + accessibility) | pass |
| Postman/newman | 13 requests, 39 assertions, 0 failed |

The failure is `test_security.py::TestFileUploadSecurity::test_file_type_validation`:
uploading `malicious.exe` returns **500** where the test expects 400. It is under
active investigation at the time of writing and is recorded here rather than
omitted — a handbook that reports a suite as green when it is not teaches the
wrong lesson twice over.

> An earlier draft of this table said "783 passed, 0 failed". That number was
> **predicted** from a previous run plus one added test, not measured. The
> measurement disagreed. This is the §16.5 lesson applied to the handbook
> itself: a single observation is not evidence, and an arithmetic extrapolation
> from one is not a measurement.

Known gaps, stated rather than hidden:

- `inferences.py` — 763 statements at 8% coverage, the largest untested
  surface.
- Three fixed defects have no guarding test (a worker lock prefix, a frontend
  XAI method list, and a Redis memory cap that is deployment config rather than
  code). Each is listed with what a test would need.
- GPU-bound performance targets are unmeasured — there is no GPU host.
- The Firefox and WebKit Playwright projects have not run locally (they fail on
  Windows with `spawn UNKNOWN`); CI is their first real execution.

---

## 16.9 Summary

- This system's bugs do not crash. Every serious defect returned a well-formed,
  plausible, wrong answer. Type-and-shape assertions catch almost none of them.
- What works: **a second call to compare against, or a real end-to-end path.**
- Differential tests catch "the parameter is being ignored" — assert two
  algorithms *differ*.
- Counting stubs make invisible behaviour (caching, reuse) assertable.
- Design tests around what a wrong answer would look like; plant something that
  makes the wrong answer detectable.
- Verify every important test by breaking the code. An untested test is not a
  test.
- Contract tests over cache shapes, because the property belongs to neither side
  alone.
- Error paths are code and are usually untested. The `logger` bug lived in a
  branch that had never executed.
- A timeout converts a hang (CI reports nothing) into a failure (CI reports
  something). Set it well above the slowest real test.
- `--strict-markers`, or a typo'd marker silently does nothing.
- The GC-timing deadlock: an intermittent hang correlated with load rather than
  input suggests a finaliser.
- `autouse` fakeredis for every test; know that the orchestrator needs a
  separate `broker` fixture that lives in test files, not conftest.
- Run with Redis unreachable before pushing — a local Redis masks CI failures.
- Match your test client to your fixture's event loop, or every teardown errors.
- Decide per suite whether cross-browser matters, and write down why.
- Wait for conditions that mean what you need; `networkidle` never fires with an
  open WebSocket.
- An independent harness (Postman) catches contract bugs in-process tests
  cannot.
- **Measurement is harder than testing.** Three findings here were wrong on
  first measurement; each was caught by chasing a contradiction, and one needed
  a control. Never treat a single timing observation as evidence.
- A synthetic benchmark whose parameters do not match production can
  confidently prove the opposite of the truth. Delete a test that asserts the
  wrong shape.
- Two green branches do not prove the combination works; re-run the full suite
  after merging someone else's work.
- Tracker status and tooling output are claims. The repository is evidence.

Next: [Chapter 17 — Building it from scratch](17-build-from-scratch.md).

For the full record — every tool, every test file, the measured numbers, all 23 defects
and the open items — see [Chapter 18](18-testing-and-evaluation-in-practice.md).
