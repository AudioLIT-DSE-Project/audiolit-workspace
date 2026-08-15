# Project Rules — AudioLIT

> Loaded in Phase 4 alongside `core/rules.md`. Twelve maximum.
> Sourced from `docs/README.md`, `.antigravity/project/TEAM_AGENT_WORKFLOW.md`,
> `.antigravity/project/PR_REVIEW_CHECKLIST.md` and `docs/SAD.md`.

**A1** RQ + Redis only, `soundfile` only for audio I/O. Celery and torchaudio were
removed project-wide — a mention anywhere is stale, and neither is ever
reintroduced. Grep the diff if unsure.

**A2** **Never fabricate fallback data.** When something cannot be computed — an
unsupported architecture, a missing classifier head, a model not yet integrated —
raise a typed error or an explicit not-implemented marker. Never return
plausible-looking synthetic numbers. This project has shipped exactly this bug
once (silent synthetic-attention fallback, FR17 / LIT-222).

**A3** `main` is production and never receives a PR directly. All work branches
off `develop` and merges back into it.

**A4** One feature branch per Linear issue, named from the issue's own
`gitBranchName`, **created before any planning or code**. One PR per issue, with
the PR title set to the exact `gitBranchName`. **A unit is not complete until its PR is open and gh pr checks passes green.**

**A5** Never self-merge. Every PR needs a human review before merging, regardless
of CI. Never bypass a safety rail — no skipped hook, no `--no-verify`, no
disabled check, no force-push.

**A6** Do not invent FRs. There is no FR5, FR13 or FR14 in the reconciled SRS.

**A7** Never promote a stretch or non-committed item into scope. Check the
non-committed table in `.antigravity/project/ISSUE_PLAN.md` first.

**A8** No synchronous inference on the request path. Long work goes to RQ.

**A9** Import the module, not the name, across a boundary where an upstream
reassignment must stay visible (the LIT-229 pattern; see
`app/domain/saliency_service.py`).

**A10** Extend `app/orchestration/task_orchestrator.py`. Never add a parallel task
fabric — this project has shipped a duplicated one already. Do not add to
`app/core/`, and never recreate `app/services/`.

**A11** A `Path:` stamp is a claim about the tree. `ls` before writing there; if
it is stale, fix the stamp and flag it rather than recreating what it describes.

**A12** Work only your own assigned issues. Building a teammate's issue as an
unassigned draft PR is allowed **only after asking the human** — never a default
move, and never touch their Linear assignment.
