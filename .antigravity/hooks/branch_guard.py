#!/usr/bin/env python3
"""PreToolUse - refuse to write or commit while on the wrong branch.

The failure this exists to prevent, observed on this repository: seven issues
were implemented correctly, with correct conventional commits carrying their
issue ids - and every one landed on an unrelated `chore/` branch that happened to
be checked out. No phase had instructed branch creation, so none happened. The
result cannot become one pull request per issue without surgery.

Reads `ag state get branch`. If state names a branch and HEAD is somewhere else,
the write is denied with the command needed to fix it.

Instruction twin: core/workflow/p2.md, p4.md, p6.md; project/rules.md A5
"""
import os
import subprocess

from _common import blob_of, emit, note, payload, tool_of

WRITE = ("write", "edit", "create", "replace", "apply", "patch")
SHELL = ("run_command", "bash", "shell", "execute", "terminal")
MUTATING = ("git commit", "git push")

p = payload()
tool, blob = tool_of(p), blob_of(p)

is_write = any(k in tool for k in WRITE)
is_mutating = any(k in tool for k in SHELL) and any(c in blob for c in MUTATING)
if not (is_write or is_mutating):
    emit("allow")

# Never block the system's own bookkeeping.
for safe in ("runtime/", ".antigravity/plans/", ".antigravity/logs/", ".antigravity/project/issue_plan", "docs/issue_plan"):
    if safe in blob.replace("\\", "/").lower():
        emit("allow")

try:
    from aglib import state
    from aglib.util import REPO_ROOT, cfg
    expected = (state.get("branch") or "").strip()
except Exception as exc:
    note("branch_guard: cannot read state (%s) - failing open" % exc)
    emit("allow")

if not expected:
    # Phase 2 has not recorded a branch. That is itself the bug this hook exists
    # for, so say so loudly rather than silently allowing.
    ticket = ""
    try:
        ticket = state.get("ticket") or ""
    except Exception:
        pass
    if ticket:
        emit("ask",
             "No branch recorded for %s. Phase 2 should have created one before any\n"
             "work started - one branch per issue, one pull request per issue.\n"
             "Create it, then record it:\n"
             "    git fetch origin\n"
             "    git checkout -b <branch-from-tracker> origin/%s\n"
             "    ag state set branch <branch>\n"
             "Allow only if you are deliberately working outside the issue workflow."
             % (ticket, cfg("default_branch", "develop")))
    emit("allow")

try:
    current = subprocess.run(["git", "branch", "--show-current"], capture_output=True,
                             text=True, timeout=10, cwd=REPO_ROOT).stdout.strip()
except Exception:
    emit("allow")

if not current or current == expected:
    emit("allow")

emit("deny",
     "On branch '%s' but this issue's branch is '%s'.\n"
     "Committing here would put the issue on the wrong branch and make one\n"
     "pull request per issue impossible without splitting it afterwards.\n"
     "    git checkout %s\n"
     "...or, if the branch does not exist yet:\n"
     "    git fetch origin && git checkout -b %s origin/%s"
     % (current, expected, expected, expected, cfg("default_branch", "develop")))
