#!/usr/bin/env bash
#
# Build the production tree for `main` from `develop`.
#
# `main` holds the codebase that ships to a client. `develop` holds that plus
# everything used to build it: the handbook, the SAD/SRS, the defect log, agent
# instruction files, working plans, scratch output. This script produces the
# former from the latter, on a release branch, without touching `main` itself.
#
# It does NOT merge, commit to main, or push. It leaves you a release branch with
# one commit to review. The merge into main is a human decision.
#
# Usage:
#   scripts/prepare-release.sh v1.0.0            # from origin/develop
#   scripts/prepare-release.sh v1.0.0 develop    # from a named base
#
# Then:
#   git diff main..release/v1.0.0 --stat     # review what changes on main
#   git switch main && git merge --no-ff release/v1.0.0
#   git push origin main
#
set -euo pipefail

VERSION="${1:-}"
BASE="${2:-origin/develop}"

if [ -z "$VERSION" ]; then
  echo "usage: $0 <version> [base-ref]" >&2
  echo "   eg: $0 v1.0.0" >&2
  exit 64
fi

case "$VERSION" in
  v*) ;;
  *) echo "error: version should look like v1.0.0 (got '$VERSION')" >&2; exit 64 ;;
esac

BRANCH="release/${VERSION}"

# --------------------------------------------------------------------------- #
# Paths stripped from the release.
#
# This list is one half of a pair: the other is FORBIDDEN in
# .github/workflows/main-hygiene.yml, which fails the build if any of these
# reaches main. Add to both together. A path in one but not the other is exactly
# how this drifts, and the workflow is what catches it if it does.
#
# NOT stripped, deliberately: Backend/tests, Frontend/e2e, Frontend/src/**/*.test.tsx,
# Backend/apitests, Backend/loadtests, .github/workflows. A client who cannot run
# the suite cannot verify their own deployment, and CI on main cannot gate a
# release without it.
# --------------------------------------------------------------------------- #
STRIP=(
  "docs"
  "plans"
  "scratch"
  "CLAUDE.md"
  "GEMINI.md"
  ".DS_Store"
  "newman-report.html"
  # This script itself. It is a development tool, and on main it would reference
  # docs/ and CLAUDE.md - paths that do not exist there. Note this removes the
  # ROOT scripts/ only; Backend/scripts/evaluate_models.py is the evaluation
  # runner and is deliberately kept.
  "scripts"
)

# Files that must exist in the release, and are sourced from `release/` on the
# base branch. These are the client-facing guides; they live under release/ on
# develop so that develop's own README can stay a contributor guide.
declare -A PROMOTE=(
  ["release/README.md"]="README.md"
  ["release/DEPLOYMENT.md"]="DEPLOYMENT.md"
)

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
say "Preflight"

if [ -n "$(git status --porcelain)" ]; then
  die "working tree is not clean. Commit or stash first - this script switches branches."
fi

git rev-parse --verify --quiet "$BASE" >/dev/null || die "base ref '$BASE' does not exist. Run 'git fetch origin' first?"

if git rev-parse --verify --quiet "$BRANCH" >/dev/null; then
  die "branch '$BRANCH' already exists. Delete it or pick another version."
fi

# The release must be built from a base that passes its own gates. This script
# cannot run the suites for you (they need Redis, Node, a browser), so it
# refuses to guess: it states what must have passed and asks for confirmation.
cat <<'PREFLIGHT'

  This script assumes the base ref has already passed, on this commit:

    Backend    REDIS_URL="redis://127.0.0.1:1/0" pytest -q
    Frontend   npm ci && npm run typecheck && npm run lint && npm test && npm run build
    E2E        npm run test:e2e
    API        npx newman run Backend/apitests/AudioLIT.postman_collection.json --env-var baseUrl=http://127.0.0.1:8000

  A release built from a red base is a red release.

PREFLIGHT

if [ "${RELEASE_ASSUME_VERIFIED:-}" != "1" ]; then
  printf 'Have those passed on %s? [y/N] ' "$BASE"
  read -r reply
  case "$reply" in
    [yY]*) ;;
    *) die "aborted. Re-run with RELEASE_ASSUME_VERIFIED=1 to skip this prompt." ;;
  esac
fi

ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
restore() { git switch --quiet "$ORIGINAL_BRANCH" 2>/dev/null || true; }

# --------------------------------------------------------------------------- #
# Build the release branch
# --------------------------------------------------------------------------- #
say "Creating $BRANCH from $BASE"
git switch --quiet --create "$BRANCH" "$BASE"

say "Promoting client-facing guides"
for src in "${!PROMOTE[@]}"; do
  dst="${PROMOTE[$src]}"
  [ -f "$src" ] || { restore; git branch -D "$BRANCH" >/dev/null 2>&1 || true; die "'$src' not found on $BASE. The client-facing guides live under release/ on develop."; }
  git mv --force "$src" "$dst"
  say "  $src -> $dst"
done

say "Stripping development artefacts"
for path in "${STRIP[@]}"; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1 || [ -e "$path" ]; then
    git rm -r --quiet --ignore-unmatch "$path" || true
    rm -rf "$path"
    say "  removed $path"
  fi
done

# `release/` itself should not survive - its contents were promoted above.
if [ -d "release" ]; then
  git rm -r --quiet --ignore-unmatch "release" || true
  rm -rf "release"
  say "  removed release/ (contents promoted)"
fi

# --------------------------------------------------------------------------- #
# Self-check: the same assertions main-hygiene.yml makes, run before committing
# so a bad release never becomes a commit at all.
# --------------------------------------------------------------------------- #
say "Self-check"
FAILED=0
for path in "${STRIP[@]}"; do
  [ -e "$path" ] && { printf '  still present: %s\n' "$path"; FAILED=1; }
done
for required in README.md DEPLOYMENT.md docker-compose.yml Backend/Dockerfile Frontend/Dockerfile; do
  [ -f "$required" ] || { printf '  missing: %s\n' "$required"; FAILED=1; }
done
# Keeping the suites is the point of this release shape, so assert it.
for kept in Backend/tests Frontend/e2e Backend/apitests .github/workflows; do
  [ -e "$kept" ] || { printf '  should have been kept but is absent: %s\n' "$kept"; FAILED=1; }
done
if [ "$FAILED" -ne 0 ]; then
  restore
  git branch -D "$BRANCH" >/dev/null 2>&1 || true
  die "self-check failed; release branch discarded."
fi
say "  ok"

say "Committing"
git add -A
git commit --quiet -m "release: ${VERSION} production tree

Built from ${BASE} by scripts/prepare-release.sh.

Development artefacts removed for the production branch: $(printf '%s ' "${STRIP[@]}").
Client-facing README.md and DEPLOYMENT.md promoted from release/.

Test suites, API tests, load tests and CI workflows are deliberately retained:
a client who cannot run the suite cannot verify their own deployment, and CI on
main cannot gate a release without them."

say "Done"
cat <<EOF

  Branch ${BRANCH} is ready. Nothing has been pushed and main is untouched.

  Review, then merge:

    git diff main..${BRANCH} --stat
    git diff main..${BRANCH} -- README.md DEPLOYMENT.md
    git switch main && git merge --no-ff ${BRANCH}
    git push origin main

  Pushing will run .github/workflows/main-hygiene.yml, which re-checks
  everything the self-check above verified.

EOF
