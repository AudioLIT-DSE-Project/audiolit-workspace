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
- **soundfile only** for audio I/O. torchaudio is on a discontinuation path
  and is being removed (LIT-226). Never reintroduce it.
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

Similarly: LIT-7 ("Setup the Echo 1.0") was marked Done in Linear, but the
real ECHO 1.0 codebase had never actually been merged into this repo —
`develop` carried a parallel from-scratch scaffold instead. This was caught
by comparing git history against the real `AudioLIT-DSE-Project/ECHO` fork,
not by trusting the Linear status. **Linear status and PR/attachment
evidence are not the same thing — when something matters, check the repo.**

---

## Repo structure — current state, not yet the SAD's target

The real ECHO 1.0 baseline is merged (PR #7). Structure on `develop` as of
this writing:

```
Backend/app/{core, api/routes, services}    # ECHO's own layout
Frontend/src/{pages, contexts, components/{layout,panels,audio,visualization,ui,analysis,dataset,predictions}, hooks, lib}
```

The SAD's target five-layer layout (`Backend/app/{api,domain,infrastructure,
orchestration}`) is **mid-migration, not merged into `develop` yet** — PR #16
(LIT-227 slice 2) does the real move (every service into its actual
domain/infrastructure/orchestration home, grounded in the real SAD §5.1/§5.2
text, not guessed) but is sitting in review. **Check whether #16 has merged
before assuming either tree shape** — `git log --oneline -5` or `ls
Backend/app/` settles it in one command; don't guess from this doc, which
will drift the moment that PR lands and someone forgets to update it.

Per SAD §8.2 the migration is incremental ("infra services set up first,
then model-loading reorganised, then explanation code tidied, then
background-processing replaces the old queue, then new features built on
top," system kept working at each step) — not a big-bang rewrite blocking
everything else. `app/core/` retiring entirely and everything moving out of
flat `app/services/` is what #16 finishes; the queue → real RQ /
no-synchronous-inference-on-the-request-path step is still genuinely open
after #16 (see LIT-227's Linear comments for exactly what's left and why —
it changes `/upload`'s HTTP contract and needs LIT-157, not started).

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
6. **Push, open the PR into `develop`**, title/body referencing the LIT-id.
7. **Wait for `gh pr checks <n>` to report an actual terminal pass/fail for
   every check** before treating CI as verified — "opened" is not "green."
   If a check fails, diagnose and fix the root cause (don't disable the
   check, don't skip hooks, don't force-merge).
8. **Stop there. Do not merge the PR yourself.** Every PR needs at least one
   approving review from a different team member before merging into
   `develop` — this is mandatory on this project regardless of CI status.
   Opening the PR already moved the Linear issue to **In Review**
   automatically (LIT-134's automation) — that's the correct, expected
   state; don't try to advance it further. Only merge if a human explicitly
   asks you to merge that specific PR.
9. **Update `docs/ISSUE_PLAN.md`'s status column** for the issue (and any
   issue it unblocks) so the next session/developer sees accurate state.
10. If you find a conflict along the way (issue body contradicts SAD/SRS, a
   mapping in LIT-228 points at the wrong thing, an issue marked Done with
   no evidence in the repo) — **flag it in a Linear comment and to the
   user, don't silently resolve it** by guessing which side is right.

## Known open items (check before assuming these need fresh triage)

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
  by direct name (`from ...core.redis import redis`, or
  `from ...infrastructure.redis import redis` once PR #16 merges), bypassing
  the `fake_redis` test fixture. Harmless today (no test run has a real
  Redis reachable), but adding a real Redis to CI/local test runs before
  this is fixed will break 7 unrelated tests with `RuntimeError: Event loop
  is closed`. Don't add a Redis service container to CI until this is
  resolved. **This is also the reference example for "don't bind a name
  directly across a module boundary"** — see `app/domain/saliency_service.py`
  for the fix pattern (import the module, not the name, so a later
  reassignment upstream is still visible).
- **LIT-227 and LIT-207 were briefly marked Done in Linear without their own
  DoD actually being met** (`app/domain`/`app/orchestration` were empty
  placeholders; hook registration wasn't wired into the registry). Caught by
  a repo audit, not by anyone reading the status — flagged in Linear
  comments on both issues rather than silently re-toggled, and PR #16 closes
  the real gap. **Second occurrence of the LIT-7 lesson above** (Linear
  status ≠ repo evidence) — if you're relying on a Done status for something
  that matters, spend the one command it takes to verify it against the
  actual tree/tests instead of trusting the label.
- **As of this writing, three PRs are open awaiting review**: #12 (LIT-211,
  HookManager), #16 (LIT-227 slice 2 + LIT-207 hook-wiring fix), #17 (draft,
  LIT-150 real ASR+SER orchestrator, ADD stubbed — LIT-128 doesn't exist
  yet). Check `gh pr list` before starting related work so you don't
  duplicate what's already sitting in review.
- **Current critical-path bottlenecks, unstarted**: LIT-123 (Ravindu, Multi-
  task dataset ingestion core — blocks 4 other dataset issues plus ADD
  integration) and LIT-127 (Rahim, Deploy RQ broker, Urgent — blocks the
  entire async fabric: worker scaffolding, the real orchestrator, frontend
  wiring). If you're picking up work for either of them and neither reason
  applies, these are the highest-leverage places to start.

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

*Keep this file and `docs/ISSUE_PLAN.md` current as work lands — a stale
CLAUDE.md is worse than no CLAUDE.md, because it reads as authoritative.*
