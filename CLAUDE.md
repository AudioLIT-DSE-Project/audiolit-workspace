# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AudioLIT is an interpretability workbench for ASR, Speech Emotion Recognition,
and Audio Deepfake Detection, extending the open-source **ECHO 1.0** baseline
(`AudioLIT-DSE-Project/ECHO`, forked from `AnasSAV/ECHO`). FastAPI + Redis +
RQ backend, React 18 + TypeScript + Vite frontend. Built by 3 developers
(Tharusha Perera, Rahim Iqbal, Ravindu Pathirana) across an academic-project
timeline (Phase 2 MVP → Phase 3 refinement/testing → Phase 4 submission).

**Read in this order before touching architecture or scope:**

1. `docs/README.md` — conventions, branch model, errata (**read this first**)
2. `docs/SAD.md` — architecture of record
3. `docs/SRS.md` — committed requirements (FRs, NFRs, scope)
4. Linear issue **LIT-228** — Tier-C stamping convention doc, FR→issue map
5. `docs/ISSUE_PLAN.md` — dependency-ordered local index of every committed
   issue, with status/assignee, so you can see what's unblocked without
   opening 50 Linear tickets

If any of these disagree, the order above is the authority — SAD/SRS win
over Linear issue bodies, which win over your own assumptions.

---

## Commands

### Backend (`cd Backend`, Python 3.11, venv active)

```bash
pip install -r requirements.txt          # CI also installs: pytest httpx
pytest                                    # full suite (pytest.ini: testpaths=tests, asyncio_mode=auto)
pytest tests/test_task_orchestrator.py    # one file
pytest tests/test_task_orchestrator.py::TestEnqueue::test_x   # one test
pytest -m "not slow"                      # markers: slow, integration, security, performance, critical, important
uvicorn app.main:app --reload --port 8000 # API + OpenAPI docs at /docs
docker compose up -d                      # Redis 7 on :6379 (container lit-redis)
python -m app.orchestration.worker all    # all five RQ worker families in one process
python -m app.orchestration.worker asr    # or one family: asr | ser | add | xai | mutation
```

**Run the backend suite with Redis unreachable before pushing.** CI has no
Redis service, and a locally-reachable Redis masks failures that only bite on
CI. The portable way, no container knowledge needed:

```bash
REDIS_URL="redis://127.0.0.1:1/0" pytest -q
```

For orchestrator/RQ-touching tests also stress-run them (~15× in a loop with a
background-and-kill timeout — `perl alarm` does not work, Python resets
SIGALRM): `SimpleWorker(burst=True)` draining a dependency-gated aggregator on
fakeredis has hung pytest intermittently, and twice masked a real CI hang.

### Frontend (`cd Frontend`, Node 20)

```bash
npm ci
npm run lint      # eslint .
npm test          # jest (ts-jest + jsdom, tests in src/**/*.test.ts(x))
npm run build     # vite build
npm run dev       # Vite dev server on :8080 (NOT 5173 — README.md is wrong on this)
npm test -- src/tests/WaveformViewer.test.tsx   # one Jest file
npx playwright test                              # e2e/ — boots its own dev server, no backend needed
npx playwright test --project=chromium
```

**The full local CI equivalent is `npm ci && npm run lint && npm test && npm run
build` — `npm test` is in `.github/workflows/ci.yml` and is easy to forget.**
CI on this project has genuinely failed from dependency version drift (PR #7);
don't assume green.

---

## Architecture — the parts that need several files to see

**Request path.** `app/main.py` mounts ~15 routers from `app/api/routes/`. Heavy
work does not run on the request path: a route enqueues onto the Task
Orchestrator, returns a job id, and the frontend follows the job over a
WebSocket.

```
route → task_orchestrator.enqueue → RQ queue (one per family) → worker process
      → Redis (result cache + progress pubsub) → WS /api/ws/tasks/{id} → useTaskStatus
```

**`app/orchestration/task_orchestrator.py` is the single queue fabric** (SAD
§5.2). Five worker families — `asr`, `ser`, `add`, `xai`, `mutation` — one
queue and one worker process each, so a process only ever holds its own model
in memory (SAD §6.1). GPU-bound families are pinned to concurrency 1 via a
per-family Redis lock (SAD C2). It uses `SimpleWorker`, not the forking
`Worker`. Progress is published on `audiolit:progress:{job_id}`;
`app/api/routes/tasks.py` subscribes and relays to the browser, with
`useTaskStatus.ts` falling back to polling. **Extend this module; never add a
parallel queue fabric** — that exact bug is documented below.

**Two cache-key schemes coexist, deliberately.** Know which one you're in:

- `app/infrastructure/cache_keys.py` — the inherited ECHO scheme, MD5 over the
  *resolved path*, in two spellings: `path_hash` (`md5(path)`, what almost
  every route computes, sometimes under the misleading local name
  `file_content_hash`) and `content_hash` (`md5(path_size_mtime)`, used by the
  saliency route). Every hot path uses this. Warm both spellings, and respect
  the documented *value shape* per key family — a right key with a wrong shape
  is worse than a miss, because consumers won't fall back to recomputation.
- `app/core/redis.py` — the FR4 content-addressed manager (`RedisCacheManager`,
  SHA-256 over audio bytes + model + task + params, msgpack/lz4, dedup lock).
  Its only production consumer is `app/api/routes/results.py`.

Whether `app/core/redis.py` moves to `app/infrastructure/` or `app/core/` is
formally retired is an **open decision — raise it, don't resolve it by deleting
a module a route depends on.**

**Layers** (SAD §5.1, landed via PR #16 / LIT-227):

```
Backend/app/{api/routes, domain, orchestration, infrastructure}
Frontend/src/{pages, contexts, components/{layout,panels,audio,visualization,ui,analysis,dataset,predictions}, hooks, lib}
```

- `app/domain/` — model registry/loader, hook manager, saliency, acoustic
  profiler, perturbation, accent-bias, evaluation, provenance.
- `app/infrastructure/` — settings, redis client + helpers, rq_connection,
  session middleware, dataset loaders/ingestion, cache_keys.
- `app/orchestration/` — task_orchestrator, worker, inference/fanout/multitask
  orchestrator services, session queue.
- **`app/services/` is gone** (LIT-230 removed the last of it; only a stale
  `__pycache__` remains). If you find yourself adding a file there, **stop** —
  its return is the exact bug LIT-230 fixed.

**Don't trust this block over the repo** — `ls Backend/app/` settles the current
shape in one command, and this doc drifts the moment someone forgets to update
it (which is how the previous version of this section went wrong).

**Frontend.** Effectively a single page: `pages/Index.tsx` composes the
workbench panels. `lib/api.ts` exports `API_BASE`
(`VITE_API_BASE_URL` || `http://localhost:8000`) — derive WebSocket origins
from it, never from `window.location` (that pointed the socket at the Vite dev
server and forced `ws://` on an `https://` page). Note the **two context
directories**: `src/context/` holds `ModelRegistryContext.tsx`, `src/contexts/`
holds `EmbeddingContext`/`PlaybackContext`. Check which one you mean. Path
alias `@/` → `src/`, mirrored in `jest.config.cjs`.

**Test fixtures.** `Backend/tests/conftest.py` has an autouse `fake_redis`
fixture that monkeypatches `app.infrastructure.redis.redis` with fakeredis.
Any test that calls a task-orchestrator function — not just ones that enqueue —
needs the `broker` fixture (patches `rq_connection._CONNECTION`) if that
function or anything it calls touches `publish_progress` /
`get_redis_connection`. Don't reason "I mocked the domain call, so no Redis is
involved"; check the orchestrator wrapper itself.

---

## Rules that must never be silently violated

- **RQ + Redis only.** Celery is removed project-wide. If you see a Celery
  mention anywhere (old issue bodies, comments), it's stale — never
  reintroduce it.
- **soundfile only** for audio I/O. torchaudio has been removed (LIT-226,
  merged) — never reintroduce it.
- **Single monorepo** (`audiolit-workspace`). Some old issue bodies and an
  earlier SAD draft describe a two-repo split (workspace + ds-engine) — that
  topology is superseded, documented in `docs/README.md` errata E1.
- **Branch model**: `main` is production, never receives a PR directly.
  `develop` is the integration branch — all feature work branches off it,
  PRs merge into it. `testing` is a dedicated test-harness/evaluation branch
  (a superset of `develop`, carrying Playwright `dataflow` E2E, Locust load
  tests, and diagnostic scripts). One feature branch per Linear issue
  (`feature/lit-xxx-...`, use the issue's own `gitBranchName` field), one PR
  per issue referencing its LIT-id.
- **Do not invent FRs.** There is no FR5, FR13, or FR14 in the reconciled
  SRS (FR5 — multi-model comparison — was demoted to non-committed stretch).
- **Never promote a §4.4 stretch item to committed scope.** Stretch issues
  are listed in `docs/ISSUE_PLAN.md`'s "Non-committed / stretch" table —
  check there before starting anything that sounds like it might be one.

---

## A trap already hit once — verify claims against the actual source

LIT-228 (the Tier-C bootstrapping doc) originally cited SAD section numbers
(`§5.2.1`–`§5.2.5`, `§8.3`, `§11.4`) and class names (`HookManager`,
`CacheGateway`, `TensorCodec`) that **do not exist** in the actual
`docs/SAD.md` — they were fabricated by an earlier pass and never checked
against the real document. This has been corrected (LIT-228, all 49
Tier-C-stamped issues, and `docs/README.md`'s repo-layout table now cite the
real structure: SAD `§5.1`/`§5.2` only, `§6.1`, `§8.1`/`§8.2`, `§11.1`–
`§11.3`, plain component names — Model Registry, Explanation Strategies,
Acoustic Profiler, Mutation Engine, Bias Profiler and Faithfulness Auditor,
Cache Manager, Task Orchestrator, Workspace).

The lesson, not just the fix: **if you're about to cite a SAD/SRS section
number, constraint ID, or class name you didn't just read yourself in this
session, `grep`/read the actual file first.** Convention docs and prior Tier-C
stamps can be wrong; the source documents are ground truth.

The same applies to tooling output. **`gh pr view --json
additions,deletions,changedFiles` can return stale counts** captured when the PR
was opened — on PRs #34/#36 it reported ~180 files/+38k when the real diffs were
5 files/+577 and 2 files/+188, and a review round confidently repeated both
numbers before anyone measured. Use `git diff --shortstat
origin/develop...<head>`. Likewise, **don't reason about what a merge will do —
run it**: `git worktree add --detach /tmp/x <base> && git merge --no-commit
<head>` answers "will this conflict / will this revert X" in seconds, and in
this repo it disproved a confident "merging this reverts LIT-128".

Similarly: LIT-7 ("Setup the Echo 1.0") was marked Done in Linear, but the
real ECHO 1.0 codebase had never actually been merged into this repo —
`develop` carried a parallel from-scratch scaffold instead. Caught by comparing
git history against the real fork, not by trusting the Linear status. LIT-227
and LIT-207 were likewise briefly marked Done with their DoD unmet. **Linear
status and repo evidence are not the same thing — when something matters,
spend the one command it takes to check the tree.**

---

## Picking up an implementation issue — step by step

1. **Check `docs/ISSUE_PLAN.md`** for the lowest unstarted tier. Everything
   in a tier is parallelizable; an issue is only really unblocked if
   everything in its own **Blocked by** column is done (tier position is a
   guide, not a guarantee — some issues have cross-tier dependencies).
2. **Fetch the issue from Linear** (MCP tools prefixed
   `mcp__claude_ai_Linear__`, team is `LIT`). Read its Tier-C stamp — the
   `SRS:`/`SAD:`/`Milestone:`/`Path:`/`Acceptance:`/`Out of scope:` header —
   for the actual scope and acceptance criteria. `docs/ISSUE_PLAN.md` only
   has a one-line summary; Linear is the source of truth for scope.
3. **Create the branch** using the issue's own `gitBranchName` field off
   `develop` (`git checkout -b <gitBranchName> develop`).
4. **Implement**, respecting the "Path:" field (**verified against the actual
   repo tree** — see the stale-stamp incident below) and the "Out of scope"
   line (don't build stretch functionality bundled in the same issue body).
5. **Verify locally before pushing** — run the actual CI steps yourself
   (both command blocks above, backend suite with Redis unreachable).
6. **Do NOT run `git commit` — the developer makes the commits.** Stage the
   work (`git add` the intended files, and check nothing stray got swept in),
   then hand over a ready-to-paste `git commit` with the message already
   written. Say plainly that nothing is committed yet. Commits carry the
   developer's name and are the permanent record of who wrote what — they
   want that authorship, and a last look at the diff before it is sealed into
   history. If you have already committed by reflex, offer `git reset --soft
   <branch-point>` so they can make the commit themselves; the work stays
   staged and nothing is lost.
7. **Once they have committed, pushing and opening the PR is yours to do** —
   `git push -u origin <branch>` and `gh pr create --base develop` with the
   LIT-id in the title and body. A PR is a wrapper around commits that already
   exist, so it does not carry the authorship weight a commit does. Then wait
   for `gh pr checks <n>` to report an actual terminal pass/fail for every
   check before treating CI as verified — "opened" is not "green." If a check
   fails, diagnose and fix the root cause (don't disable the check, don't skip
   hooks, don't force-merge).
8. **Stop there. Do not merge the PR yourself.** Every PR needs at least one
   approving review from a different team member before merging into
   `develop` — mandatory regardless of CI status. Opening the PR already moved
   the Linear issue to **In Review** (LIT-134's automation) — that's the
   correct, expected state. Only merge if a human explicitly asks you to merge
   that specific PR. **A PR with zero recorded reviews still isn't mergeable
   just because the user asked** — say so and let them approve or merge it.
9. **Update `docs/ISSUE_PLAN.md`'s status column** for the issue (and any
   issue it unblocks) so the next session/developer sees accurate state.
10. If you find a conflict along the way (issue body contradicts SAD/SRS, a
    mapping in LIT-228 points at the wrong thing, an issue marked Done with
    no evidence in the repo) — **flag it in a Linear comment and to the
    user, don't silently resolve it** by guessing which side is right.

---

## Incidents worth not repeating

**A stale Tier-C `Path:` stamp caused a whole module to be built twice
(LIT-230).** LIT-149's stamp said `Path: Backend/app/services/queue_service.py`
— written before LIT-227 emptied that directory. The developer followed it
exactly as step 4 says to, and `app/services/queue_service.py` (383 lines)
landed duplicating `app/orchestration/rq_broker.py`, already merged as LIT-127.
Two individually-green PRs, no git conflict, one silently duplicated task
fabric with **two different progress-channel prefixes**, so a job published by
one was invisible to a subscriber on the other. Both were consolidated into
today's `task_orchestrator.py`. **A `Path:` field is a claim about the tree, and
stale stamps are a known failure mode here — `ls` the directory before you write
to it, and if the stamp points somewhere that no longer exists, fix the stamp
and flag it rather than recreating the directory.**

**Two green PRs are not proof the combination works.** PR #10 (LIT-225) added
`app/core/rq_connection.py` importing `app.core.settings`; PR #13 (LIT-227)
separately moved `settings.py` out of `app/core/`. Neither touched the same
lines, so they merged with no conflict at all — and broke `pytest` collection
for everyone until a third PR fixed it. LIT-150 hit the identical trap later
(imports written against the pre-migration layout).

**Don't bind a name directly across a module boundary.** `import the module,
not the name`, so a later reassignment upstream is still visible to test
monkeypatching — see `app/domain/saliency_service.py` and the now-fixed
`app/api/routes/health.py` for the pattern. (A function-local
`from ... import redis` is benign, since it binds at call time.)

---

## Multiple concurrent sessions

More than one developer may be running a Claude Code session on this repo at
the same time (see `docs/TEAM_AGENT_WORKFLOW.md`). This is a real collision
risk, not a hypothetical one — see the PR #10/#13 incident above.

Before starting real implementation work in a session:

1. `git fetch origin` and skim `gh pr list --state open` — if someone else
   has an open PR touching the area you're about to work in, coordinate
   before you also touch it, even if your change looks unrelated on paper.
2. Re-check `docs/ISSUE_PLAN.md` **and** Linear for the issue's current
   status immediately before starting, not from memory of an earlier
   session — status changes fast when several sessions land work the same day.
3. If your change and another open PR both touch a file that only one of
   you renamed/moved, that's exactly the shape of bug above — flag it rather
   than assuming CI passing on your branch alone means the combination is safe.
4. After a merge you didn't make lands on `develop`, and before you push your
   own PR, merge (or rebase onto) latest `develop` and run the full test suite
   once more. A clean *git* merge can still be a broken *logical* merge — a
   develop-merge on PR #27 kept both a new and an old assertion and shipped a
   failing test.

---

## Known open items (check before assuming these need fresh triage)

- **`app/core/redis.py`'s eventual home** — open decision, see Architecture
  above. Don't resolve it by deleting the module.
- LIT-124/143/144 — flagged as superseded duplicates of LIT-207/210/211,
  not stamped, recommended for closing. Don't resume work on them without
  checking whether that closure decision has been made.
- LIT-180 — self-labeled stretch (sub-task of stretch LIT-166), excluded
  from Tier-C stamping.
- LIT-184 — reassigned from FR15 to FR16 (content/parent both say FR16;
  LIT-228's own mapping table had it wrong).
- LIT-154 — inherits stretch status from its LIT-129→LIT-153 chain but has
  no banner of its own yet.
- `docs/RAVINDU_TESTING_ISSUES_PLAN.md` — working plan for the 2026-09-12
  issue batch, not yet indexed in `docs/ISSUE_PLAN.md`. Linear wins over it.

---

_Keep this file and `docs/ISSUE_PLAN.md` current as work lands — a stale
CLAUDE.md is worse than no CLAUDE.md, because it reads as authoritative._
