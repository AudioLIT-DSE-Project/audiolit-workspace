# AudioLIT — instructions for Claude Code sessions

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

## Rules that must never be silently violated

- **RQ + Redis only.** Celery is removed project-wide. If you see a Celery
  mention anywhere (old issue bodies, comments), it's stale — never
  reintroduce it.
- **soundfile only** for audio I/O. torchaudio has been removed (LIT-226,
  merged) — never reintroduce it.
- **Single monorepo** (`audiolit-workspace`). Some old issue bodies and an
  earlier SAD draft describe a two-repo split (workspace + ds-engine) — that
  topology is superseded, documented in `docs/README.md` errata E1.
- **Branch model**: `main` is production, never receive a PR directly.
  `develop` is the integration branch — all feature work branches off it,
  PRs merge into it. One feature branch per Linear issue
  (`feature/lit-xxx-...`, use the issue's own `gitBranchName` field), one PR
  per issue referencing its LIT-id.
- **Do not invent FRs.** There is no FR5, FR13, or FR14 in the reconciled
  SRS (FR5 — multi-model comparison — was demoted to non-committed stretch).
- **Never promote a §4.4 stretch item to committed scope.** Stretch issues
  are listed in `docs/ISSUE_PLAN.md`'s "Non-committed / stretch" table —
  check there before starting anything that sounds like it might be one.

## A trap already hit once — verify claims against the actual source, don't trust convention docs blindly

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
`develop` carried a parallel from-scratch scaffold instead. This was caught
by comparing git history against the real `AudioLIT-DSE-Project/ECHO` fork,
not by trusting the Linear status. **Linear status and PR/attachment
evidence are not the same thing — when something matters, check the repo.**

---

## Repo structure — five-layer migration is merged

The real ECHO 1.0 baseline is merged (PR #7), and the SAD §5.1/§5.2 five-layer
layout landed via PR #16 (LIT-227). Structure on `develop` as of 2026-08-05:

```
Backend/app/{api/routes, domain, orchestration, infrastructure}
Frontend/src/{pages, contexts, components/{layout,panels,audio,visualization,ui,analysis,dataset,predictions}, hooks, lib}
```

`app/core/` **and** `app/services/` are both **gone** (LIT-230 removed the last
of `services/`). The tree now matches SAD §5.1's five layers exactly:
`settings`/`redis`/`session`/`rq_connection` in `app/infrastructure/`, model +
explanation logic in `app/domain/`, the RQ fabric in `app/orchestration/`,
routes in `app/api/routes/`. **If you find yourself adding a file to
`app/services/`, stop** — that directory is ECHO 1.0 legacy and its return is
the exact bug LIT-230 fixed. **Don't trust this block over the repo** — `ls
Backend/app/` settles the current shape in one command, and this doc drifts the
moment someone forgets to update it (the whole reason the previous version of
this section was wrong).

Inside `app/orchestration/`, `task_orchestrator.py` is the SAD §5.2 Task
Orchestrator — **one** queue fabric, worker and enqueue API. Extend it; do not
add a parallel one (see the duplicate-module incident below).

Per SAD §8.2 the migration is incremental ("infra services set up first, then
model-loading reorganised, then explanation code tidied, then background-
processing replaces the old queue, then new features built on top," system kept
working at each step) — not a big-bang rewrite. What #16 finished: the layered
packages + hook registration wired into the registry. **Still genuinely open
after #16**: the queue → real RQ / no-synchronous-inference-on-the-request-path
step, which changes `/upload`'s HTTP contract and needs LIT-157 (not started —
verify in Linear/`docs/ISSUE_PLAN.md`). See LIT-227's Linear comments for what's
left and why.

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
4. **Implement**, respecting the "Path:" field (verified against the actual
   repo tree, not folder-level guidance) and the "Out of scope" line (don't
   build stretch functionality bundled in the same issue body).
5. **Verify locally before pushing** — run the actual CI steps yourself
   first (`cd Frontend && npm ci && npm run lint && npm run build`;
   `cd Backend && pip install -r requirements.txt && pytest`). Don't assume
   green — CI on this project has genuinely failed before due to dependency
   version drift (see PR #7 history).
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
   `develop` — this is mandatory on this project regardless of CI status.
   Opening the PR already moved the Linear issue to **In Review**
   automatically (LIT-134's automation) — that's the correct, expected
   state; don't try to advance it further. Only merge if a human explicitly
   asks you to merge that specific PR. **A PR with zero recorded reviews
   still isn't mergeable just because the user asked** — say so and let them
   approve or merge it themselves.
9. **Update `docs/ISSUE_PLAN.md`'s status column** for the issue (and any
   issue it unblocks) so the next session/developer sees accurate state.
10. If you find a conflict along the way (issue body contradicts SAD/SRS, a
    mapping in LIT-228 points at the wrong thing, an issue marked Done with
    no evidence in the repo) — **flag it in a Linear comment and to the
    user, don't silently resolve it** by guessing which side is right.

## Known open items (check before assuming these need fresh triage)

- **A stale Tier-C `Path:` stamp caused a whole module to be built twice
  (LIT-230).** LIT-149's stamp said `Path: Backend/app/services/queue_service.py`
  — written before LIT-227 emptied that directory. The developer followed it
  exactly as step 4 above says to, and `app/services/queue_service.py` (383
  lines) landed duplicating `app/orchestration/rq_broker.py`, already merged as
  LIT-127. Two individually-green PRs, no git conflict, one silently duplicated
  task fabric with **two different progress-channel prefixes**, so a job
  published by one was invisible to a subscriber on the other. LIT-127, LIT-149,
  LIT-150 and LIT-225 all carried the same stale path.
  **The lesson: a `Path:` field is a claim about the tree, and stale stamps are
  a known failure mode here — `ls` the directory before you write to it, and if
  the stamp points somewhere that no longer exists, fix the stamp and flag it
  rather than recreating the directory.** This is the same class of bug as the
  fabricated SAD section numbers above: convention metadata drifted from the
  source, and nobody checked.

- LIT-124/143/144 — flagged as superseded duplicates of LIT-207/210/211,
  not stamped, recommended for closing. Don't resume work on them without
  checking whether that closure decision has been made.
- LIT-180 — self-labeled stretch (sub-task of stretch LIT-166), excluded
  from Tier-C stamping.
- LIT-184 — reassigned from FR15 to FR16 (content/parent both say FR16;
  LIT-228's own mapping table had it wrong).
- LIT-154 — inherits stretch status from its LIT-129→LIT-153 chain but has
  no banner of its own yet; was wrongly blocking the urgent LIT-132 (fixed).
- `PredictionPanel.tsx`'s second `useEffect` (whisper prediction fetch) is
  missing its unmount-cleanup function — pre-existing ECHO bug, not yet
  filed as its own issue.
- **LIT-229** — `Backend/app/api/routes/health.py` imports the redis client
  by direct name (`from ...infrastructure.redis import redis`, verified still
  present on `develop` 2026-08-05), bypassing the `fake_redis` test fixture.
  Harmless today (no test run has a real Redis reachable), but adding a real
  Redis to CI/local test runs before this is fixed will break 7 unrelated
  tests with `RuntimeError: Event loop is closed`. Don't add a Redis service
  container to CI until this is resolved. **This is also the reference example
  for "don't bind a name directly across a module boundary"** — see
  `app/domain/saliency_service.py` for the fix pattern (import the module, not
  the name, so a later reassignment upstream is still visible).
- **LIT-227 and LIT-207 were briefly marked Done in Linear without their own
  DoD actually being met** (`app/domain`/`app/orchestration` were empty
  placeholders; hook registration wasn't wired into the registry). Caught by
  a repo audit, not by anyone reading the status — flagged in Linear
  comments on both issues rather than silently re-toggled, and PR #16 (now
  merged) closed the real gap. **Second occurrence of the LIT-7 lesson above**
  (Linear status ≠ repo evidence) — if you're relying on a Done status for
  something that matters, spend the one command it takes to verify it against
  the actual tree/tests instead of trusting the label.
- **LIT-150 — removed by #21, being RE-ADDED FIXED via PR #22 (2026-08-05).**
  History: orchestrator merged (#17) → reverted (#19) → re-applied (#20) →
  **PR #21 merged (by Ravindu, 2026-08-05) which DELETED it** — both
  `app/services/multitask_orchestrator_service.py` and its test are gone from
  `develop` as of b01ccd0. The reason #21 gave was a red CI pipeline; the actual
  cause was never a logic bug but a **post-migration import break**:
  `multitask_orchestrator_service.py` was written against the pre-migration
  layout and still imported `..core.rq_connection`,
  `.fanout_orchestrator_service`, `.model_loader_service` — all of which PR #16
  relocated, so pytest collection died with `ModuleNotFoundError: app.core` (the
  same "two green PRs break in combination" trap as the earlier rq_connection
  incident). **PR #22** (`fix/lit-150-post-migration-imports-ci`) re-introduces
  the orchestrator **with the imports repointed** to `..infrastructure` /
  `..orchestration` / `..domain` (one file, three lines) plus its restored test.
  Verified on Python 3.11: full backend suite **149 passed, 2 skipped**, frontend
  `npm ci && lint && build` green. So the current plan is: **LIT-150 comes back,
  fixed, through PR #22's review** — don't re-revert it, and don't build on
  `multitask_orchestrator_service.py` until #22 merges. The infra tier
  (LIT-207/211/225/226/227) is Done, so a large batch of Tier 2–5 work is
  unblocked regardless — see `docs/ISSUE_PLAN.md`. Re-run `gh pr list --state
  open` to confirm current PR state.
- **AGREED SEQUENCING PLAN (2026-08-05) — read before claiming an issue so
  concurrent sessions don't collide:** (1) land the LIT-150 re-add/fix PR #22 above first;
  (2) **then complete LIT-123 (Ravindu, dataset ingestion core) and LIT-127
  (Rahim, RQ broker, Urgent) FIRST, before anyone starts a downstream critical
  path** — these two are the shared base that de-risks everything else, so they
  go through review + merge before the parallel build-out; (3) **then work the
  LIT-127 critical path step by step** (LIT-127 → LIT-149 workers → real
  orchestrator wiring → LIT-131/157 frontend async), in parallel with the
  LIT-123 → LIT-142 → LIT-128 → LIT-148 dataset/ADD path. If you're a fresh
  session: LIT-150 fix, LIT-123, LIT-127 are already claimed/in-flight — pick
  genuinely independent unblocked work (e.g. LIT-206/224 SER, LIT-126/130 XAI,
  LIT-222) rather than touching those, and coordinate per "Multiple concurrent
  sessions" below.

## Multiple concurrent sessions

More than one developer may now be running a Claude Code session on this
repo at the same time (see `docs/TEAM_AGENT_WORKFLOW.md`). This is a real
collision risk, not a hypothetical one — it already happened once: PR #10
(LIT-225) added `app/core/rq_connection.py` importing `app.core.settings`;
PR #13 (LIT-227) separately moved `settings.py` out of `app/core/`. Neither
PR touched the same lines, so they merged into `develop` with no git
conflict at all — and the combination broke `pytest` collection entirely
(`ModuleNotFoundError`) for everyone, on every branch, until a third PR
fixed it. Two individually-green PRs are not proof the combination works.

Before starting real implementation work in a session:

1. `git fetch origin` and skim `gh pr list --state open` — if someone else
   has an open PR touching the area you're about to work in, coordinate
   before you also touch it, even if your change looks unrelated on paper.
2. Re-check `docs/ISSUE_PLAN.md` **and** Linear for the issue's current
   status immediately before starting, not from memory of an earlier
   session — status changes fast when several sessions are landing work the
   same day.
3. If your change and another open PR both touch a file that only one of
   you renamed/moved, that's exactly the shape of bug above — flag it
   rather than assuming CI passing on your branch alone means the
   combination is safe.
4. After a merge you didn't make lands on `develop`, and before you push
   your own PR, merge (or rebase onto) latest `develop` and run the full
   test suite once more — don't assume your branch is still consistent with
   `develop` just because it was when you started.

---

_Keep this file and `docs/ISSUE_PLAN.md` current as work lands — a stale
CLAUDE.md is worse than no CLAUDE.md, because it reads as authoritative._
