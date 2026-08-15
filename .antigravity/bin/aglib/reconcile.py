"""ag reconcile - find where the tracker, the forge and the tree disagree.

A local issue index drifts from the tracker the moment someone forgets to update
it, and both drift from the repository. This project's own index carries the
warning in its header and was last reconciled nine days before this tool was
written.

Reconciliation is mechanical, so it should not cost a model anything. This walks
all three sources and prints only the disagreements.

Sources:
  index   docs/ISSUE_PLAN.md   (local, hand-maintained)
  forge   git log / gh pr list (what landed)
  tree    the working copy     (what exists)

Linear is reached through the agent's MCP tools, not from here - a script cannot
authenticate. Rows the tracker owns are marked `tracker?` for the agent to check.
"""
import os
import re
import subprocess

from .util import (REPO_ROOT, RUNTIME, dim, err, ok, read, warn, write,
                   write_json)

INDEX = os.path.join(REPO_ROOT, ".antigravity", "project", "ISSUE_PLAN.md")
if not os.path.exists(INDEX):
    INDEX = os.path.join(REPO_ROOT, "docs", "ISSUE_PLAN.md")
ROW = re.compile(r"^\|\s*([A-Z]+-\d+)\s*\|([^|]*)\|[^|]*\|[^|]*\|\s*([^|]*?)\s*\|")
DONE = ("✅", "done")
INPROG = ("🟡", "🔵", "progress", "review")


def sh(args, timeout=25):
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, cwd=REPO_ROOT)
        return r.stdout
    except Exception:
        return ""


def index_rows():
    if not os.path.exists(INDEX):
        return {}
    out = {}
    for line in read(INDEX).splitlines():
        m = ROW.match(line)
        if m:
            out[m.group(1)] = {
                "title": m.group(2).strip(),
                "status": m.group(3).strip(),
            }
    return out


def integration_branch():
    from .util import cfg
    return "origin/" + (cfg("default_branch") or "develop")


def landed_ids():
    """Ids referenced by commits ON THE INTEGRATION BRANCH.

    Searching --all counts unmerged feature branches as landed, which is a
    different and much weaker claim. Verified against this repository: LIT-151
    has commits on a branch but none on develop.
    """
    log = sh(["git", "log", "--oneline", "-600", integration_branch()])
    if not log:
        log = sh(["git", "log", "--oneline", "-600"])
    return set(re.findall(r"[A-Z]+-\d+", log.upper()))


def branch_only_ids(landed):
    """Ids on some branch but not on the integration branch - work in flight."""
    allrefs = set(re.findall(r"[A-Z]+-\d+",
                             sh(["git", "log", "--oneline", "--all", "-600"]).upper()))
    return allrefs - landed


def open_pr_ids():
    out = sh(["gh", "pr", "list", "--state", "open", "--limit", "50",
              "--json", "number,title,headRefName"])
    return set(re.findall(r"[A-Z]+-\d+", out.upper())), out


def classify(status):
    s = status.lower()
    if any(k in s for k in DONE):
        return "done"
    if any(k in s for k in INPROG):
        return "active"
    return "todo"


def fetch_linear_tracker():
    """Query Linear GraphQL API directly to get live status & assignees for all issues."""
    try:
        bin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import sys
        if bin_dir not in sys.path:
            sys.path.insert(0, bin_dir)
        import linear_query
        query = """
        query {
          issues(first: 150) {
            nodes {
              identifier
              title
              state { name type }
              assignee { name }
            }
          }
        }
        """
        data = linear_query.execute_graphql(query)
        nodes = data.get("issues", {}).get("nodes", [])
        out = {}
        for n in nodes:
            iid = n.get("identifier")
            if iid:
                out[iid] = {
                    "title": n.get("title", ""),
                    "state": (n.get("state") or {}).get("name", ""),
                    "state_type": (n.get("state") or {}).get("type", ""),
                    "assignee": (n.get("assignee") or {}).get("name", "Unassigned"),
                }
        return out
    except Exception:
        return {}


def main(argv):
    print("ag reconcile\n")
    rows = index_rows()
    if not rows:
        err("no issue index found at docs/ISSUE_PLAN.md")
        dim("this project may track everything in the tracker - use the agent's")
        dim("linear_query skill instead, and skip this command")
        return 1

    from .util import cfg
    developer = (cfg("developer") or "").strip()
    linear_data = fetch_linear_tracker()
    if linear_data:
        ok("retrieved live status for %d issues from Linear API" % len(linear_data))

    merged = landed_ids()
    in_flight = branch_only_ids(merged)
    open_ids, pr_raw = open_pr_ids()

    stale_done, ghost_done, silent_progress, ready = [], [], [], []

    for iid, meta in sorted(rows.items()):
        l_info = linear_data.get(iid, {})
        l_state = l_info.get("state", "").lower()
        l_state_type = l_info.get("state_type", "").lower()
        l_assignee = l_info.get("assignee", "Unassigned")

        is_linear_done = l_state_type == "completed" or l_state in ("done", "closed", "completed", "duplicate")
        state = "done" if is_linear_done else classify(meta["status"])
        in_log = iid in merged or is_linear_done
        in_pr = iid in open_ids

        is_in_flight = in_pr or (iid in in_flight) or (l_state in ("in progress", "in-progress", "started", "in review", "in-review"))

        if is_linear_done or (state == "done" and in_log):
            continue
        elif state == "done" and not in_log:
            ghost_done.append(iid)
        elif state != "done" and (iid in merged):
            stale_done.append(iid)
        elif is_in_flight:
            silent_progress.append(iid)
        else:
            if developer and l_assignee != "Unassigned" and developer.lower() not in l_assignee.lower():
                continue
            ready.append((iid, l_assignee))

    total = len(rows)
    dim("%d issues in index | %d landed on branch | %d in open PRs\n"
        % (total, len(merged & set(rows)), len(open_ids & set(rows))))

    if ghost_done:
        warn("marked done, but the id appears in no commit on %s (%d):"
             % (integration_branch(), len(ghost_done)))
        for i in ghost_done[:12]:
            print("      %-10s %s" % (i, rows[i]["title"][:58]))
        dim("      NOT proof the work is missing - a squashed merge loses the id.")
        dim("      verify each against the tree before acting (this is the LIT-7 lesson)")
    else:
        ok("no issues marked done without a trace in history")

    if stale_done:
        err("on %s, but the index does not say done (%d):"
            % (integration_branch(), len(stale_done)))
        for i in stale_done[:12]:
            print("      %-10s %-46s index: %s"
                  % (i, rows[i]["title"][:46], rows[i]["status"][:14]))
        dim("      the index is behind - update it, and check the tracker too")
    else:
        ok("index matches git history on completed work")

    if silent_progress:
        warn("work in flight (branch or open PR) but index says not started (%d):"
             % len(silent_progress))
        for i in silent_progress[:10]:
            print("      %-10s %s" % (i, rows[i]["title"][:58]))

    ready_ids = [item[0] if isinstance(item, tuple) else item for item in ready]
    ok("%d issues read as unstarted and unblocked-by-history for %s" % (len(ready), developer or "all"))
    if "--ready" in argv:
        for item in ready:
            if isinstance(item, tuple):
                iid, assignee = item
            else:
                iid, assignee = item, "Unassigned"
            print("      %-10s %-48s (%s)" % (iid, rows[iid]["title"][:48], assignee))

    print()
    dim("tracker? Linear API status and developer assignment applied.")

    if "--fix" in argv:
        n = apply_fix(rows, stale_done, ghost_done, silent_progress)
        print()
        if n:
            ok("updated %d status cell(s) in %s" % (n, os.path.relpath(INDEX, REPO_ROOT)))
            dim("      nothing is committed - review the diff, then commit it yourself")
        else:
            ok("no status cells needed changing")

    write_json(os.path.join(RUNTIME, "reconcile.json"), {
        "total": total, "ghost_done": ghost_done, "stale_done": stale_done,
        "silent_progress": silent_progress, "ready": ready_ids,
        "integration_branch": integration_branch(),
    })
    return 1 if (ghost_done or stale_done) else 0


# --- --fix ------------------------------------------------------------------

STAMP = "<!-- ag reconcile: status cells below are machine-updated -->"


def _replace_status(line, new):
    """Rewrite the 5th pipe-delimited cell (Status) of an index row."""
    parts = line.split("|")
    if len(parts) < 6:
        return line
    width = len(parts[5])
    parts[5] = (" " + new).ljust(max(width, len(new) + 2))
    return "|".join(parts)


def apply_fix(rows, stale_done, ghost_done, silent_progress):
    """Write corrected statuses back into the index.

    Only the Status cell is touched. Titles, blockers and notes are left alone -
    a reconciler that rewrites prose is a reconciler nobody trusts.
    """
    if not os.path.exists(INDEX):
        return 0
    text = read(INDEX)
    lines = text.splitlines()
    changed = 0

    new_status = {}
    for i in stale_done:
        new_status[i] = "\u2705 Done"
    for i in ghost_done:
        new_status[i] = "\u2705 Done (unverified)"
    for i in silent_progress:
        new_status[i] = "\U0001f7e1 In flight"

    for n, line in enumerate(lines):
        m = ROW.match(line)
        if not m:
            continue
        iid = m.group(1)
        if iid in new_status and new_status[iid].strip() != m.group(3).strip():
            lines[n] = _replace_status(line, new_status[iid])
            changed += 1

    if not changed:
        return 0

    banner = [
        "",
        STAMP,
        "> **Status cells reconciled against `%s` by `ag reconcile --fix`.**"
        % integration_branch(),
        "> `Done` means the issue id appears in a commit on that branch - which is"
        " evidence,",
        "> not proof its acceptance criteria were met. `Done (unverified)` means the"
        " index",
        "> claimed done and no commit references it; a squashed merge loses the id,"
        " so",
        "> check the tree before concluding anything. `In flight` means a branch or"
        " open PR",
        "> exists. **The tracker remains authoritative for status.**",
        "",
    ]
    # Insert the banner after the first heading block, once.
    if STAMP not in text:
        for n, line in enumerate(lines):
            if line.startswith("# "):
                lines[n + 1:n + 1] = banner
                break

    write(INDEX, "\n".join(lines) + "\n")
    return changed
