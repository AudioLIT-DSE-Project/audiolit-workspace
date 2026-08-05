# PR review checklist — AI-authored and human-authored alike

**Why this exists:** once more than one Claude Code session is producing PRs
on this repo (see `docs/TEAM_AGENT_WORKFLOW.md`), review volume goes up, not
down — delegating implementation doesn't delegate judgment. This is the same
checklist regardless of whether a human or a session wrote the diff; an
AI-authored PR earns exactly as much trust as its evidence supports, no
more.

This isn't a replacement for actually reading the diff — it's the specific
list of things that have gone wrong on *this* project before, so they're
fast to check rather than easy to forget.

## Before you open the diff

- [ ] `gh pr checks <n>` shows every check **passed**, not pending or
      "opened." A PR description claiming green CI is not the same as CI
      actually being green — verify it yourself.
- [ ] The PR title/body references the right LIT-id, and the Linear issue
      actually moved to **In Review** (confirms the automation fired and
      you're looking at the issue the PR claims to close).
- [ ] If it's a **draft** PR for someone else's assigned issue (a "head
      start" — see `CLAUDE.md`'s pattern for this), confirm the Linear
      assignee wasn't touched. It shouldn't move off the original owner.

## Reading the diff

- [ ] **Scope matches the issue.** Check the diff against the issue's
      `Path:` and `Out of scope:` fields from its Tier-C stamp. Flag
      anything that wandered into adjacent files without a stated reason.
- [ ] **No Celery, no torchaudio.** Grep if you're not sure —
      `git diff main --  | grep -i celery` — either reintroduces something
      this project explicitly removed.
- [ ] **Any cited SAD/SRS section number, constraint ID, or class name is
      real.** Open `docs/SAD.md`/`docs/SRS.md` and check — this project has
      shipped fabricated citations before (LIT-228's original pass invented
      `§5.2.1`–`§5.2.5`, `HookManager`, `CacheGateway`, `TensorCodec` that
      didn't exist). A citation that sounds plausible is not the same as
      one that's correct.
- [ ] **Tests exercise real behavior.** A test suite that's green because
      every assertion is trivially true, or because the thing under test is
      mocked into meaninglessness, isn't evidence of anything. Spot-check
      one or two tests actually fail if you comment out the implementation.
- [ ] **No fabricated fallback data.** If something couldn't be computed
      (an unsupported model architecture, a missing classifier head, a
      not-yet-integrated model), the code should say so with a typed error
      or an explicit "not implemented" marker — not silently return
      plausible-looking fake numbers. This project has a specific,
      previously-shipped bug in this exact shape (silent synthetic-attention
      fallback, FR17/LIT-222) — don't let a second instance through.
- [ ] **Claimed-Done means actually done.** If the PR or its issue asserts a
      Definition of Done is met, check the repo state, not just the prose —
      this project has shipped "Done" issues twice now (LIT-7; LIT-227/
      LIT-207) where the claim didn't match what was actually in the tree.
- [ ] **No self-merge already happened.** Shouldn't be possible given branch
      protection, but confirm the PR is still open and unmerged before you
      start reviewing it.
- [ ] **`docs/ISSUE_PLAN.md` updated** for the issue's row, if the PR
      changes its status.

## If the PR touches a file another open PR also touches

Don't assume "both are green individually" means the combination is safe —
it wasn't, once, on this project (see `CLAUDE.md`'s concurrent-sessions
section: two individually-passing PRs broke `pytest` collection for
everyone once merged together, with zero git conflict). If you're merging
one of two PRs that overlap, check out `develop` with the other one already
merged and run the full suite before approving the second.

## Merging

Once approved: merge, then tell whoever opened the PR (or start a fresh
session) to update `docs/ISSUE_PLAN.md` and pick up whatever the merge just
unblocked. If the merge lands after another PR you're about to merge, rebase
first — see the note above.
