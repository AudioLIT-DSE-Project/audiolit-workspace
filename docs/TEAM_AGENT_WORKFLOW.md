# Running a Claude Code session on AudioLIT — team quickstart

**Who this is for:** any of the three developers picking up a Linear issue
by starting a Claude Code session, rather than writing the code by hand.
`CLAUDE.md` is the full reference every session reads automatically — this
doc is the fast path through it: what to actually type, in what order, and
where the "stop and ask a human" lines are.

**What this is not:** a way to skip review. Every PR a session opens still
needs a human's approval before it merges into `develop` — that rule doesn't
change because an AI wrote the diff. Budget real time for review; see
`docs/PR_REVIEW_CHECKLIST.md`.

---

## Before you type anything

Claude Code reads `CLAUDE.md` automatically at the start of every session in
this repo — you don't need to paste it in. But you do need to tell the
session **who you are** and **what you want**, because it has no memory of
previous sessions (including ones run by you, or by a teammate, an hour
ago).

A good first message looks like:

> I'm Rahim. Pick up my next unblocked issue from `docs/ISSUE_PLAN.md` and
> implement it, following the workflow in `CLAUDE.md`.

or, if you already know which issue:

> I'm Ravindu. Implement LIT-123 following the workflow in `CLAUDE.md`.

Don't skip the name — `docs/ISSUE_PLAN.md` and Linear both track assignee,
and the session needs it to pick the right issue and avoid touching a
teammate's in-flight work.

## The loop the session should follow

This is `CLAUDE.md`'s "Picking up an implementation issue" section,
compressed:

1. **Check for collisions first.** `git fetch origin`, skim
   `gh pr list --state open`, re-check `docs/ISSUE_PLAN.md` and Linear for
   current status (not from an earlier session's memory). See `CLAUDE.md`'s
   "Multiple concurrent sessions" section — this step exists because of a
   real incident, not caution theater.
2. **Pick the issue.** Lowest unstarted tier in `docs/ISSUE_PLAN.md` where
   your name is the assignee and the **Blocked by** column is actually
   clear (tier position is a guide, not a guarantee).
3. **Fetch the issue from Linear** and read its Tier-C stamp (`SRS:`/
   `SAD:`/`Path:`/`Acceptance:`/`Out of scope:`) for the real scope.
   `docs/ISSUE_PLAN.md` is a one-line pointer, not the scope definition.
4. **Branch off `develop`** using the issue's own `gitBranchName`.
5. **Implement**, respecting the `Path:` and `Out of scope:` fields.
6. **Run the actual CI commands locally before pushing** —
   `cd Backend && pytest`; `cd Frontend && npm run lint && npm run build`.
   Don't push on the assumption it'll be fine; this project's CI has failed
   for real reasons before.
7. **Push, open the PR into `develop`**, title/body referencing the LIT-id.
8. **Wait for `gh pr checks <n>` to report an actual pass/fail**, not just
   "opened." Fix root causes, don't disable checks.
9. **Stop.** No self-merge, ever — a human reviews every PR before it
   merges, regardless of CI status.
10. **Update `docs/ISSUE_PLAN.md`'s status column** so the next session
    (yours or a teammate's) sees accurate state.

## When the session should stop and ask you, not guess

Tell it to raise these with you rather than silently picking a side:

- **An issue's Linear `blockedBy` relations disagree with
  `docs/ISSUE_PLAN.md`'s Blocked-by column.** (This happened this session —
  LIT-211 listed a stale duplicate issue as a blocker; the doc had it as
  loose. Worth five minutes of investigation before either coding or
  waiting.)
- **An issue is marked Done in Linear but the code that would prove it
  isn't findable in the repo.** Two real instances of this on this project
  already (LIT-7; LIT-227/LIT-207) — don't be the third silent one.
- **The scope feels bigger than the issue text**, especially if it would
  mean touching another teammate's assigned issue or an area another open
  PR already modified. Building someone else's issue as an unassigned
  "draft, head start" PR (not touching their Linear assignment) is fine
  *if you check with the user first* — it's not a default move.
- **A fix would require bypassing a safety rail** — skipping a hook,
  force-pushing, disabling a CI check, adding `--no-verify`. Always stop.

## What "ready to open a PR" means

Before the session pushes:

- [ ] The actual CI commands were run locally and passed (not assumed).
- [ ] No Celery, no torchaudio, anywhere in the diff.
- [ ] Any SAD/SRS section number or class name cited was read from the real
      file in *this* session, not recalled from a previous one or copied
      from an issue body.
- [ ] The diff matches the issue's `Path:`/`Out of scope:` fields — no
      scope creep into adjacent, unassigned work.
- [ ] Tests are real (exercise the actual behavior) — not written to make a
      number go green.
- [ ] `docs/ISSUE_PLAN.md` updated for this issue's row.

## After the PR is open

The session's job is done — it shouldn't keep working on the same branch
unless you ask it to (e.g. to respond to review comments, or to merge
`develop` back in if something else landed first — see `CLAUDE.md`'s
concurrent-sessions section for why that step matters). Tell the reviewer
(Tharusha) the PR number; review happens per
`docs/PR_REVIEW_CHECKLIST.md`.
