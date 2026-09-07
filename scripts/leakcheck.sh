#!/usr/bin/env bash
# Fails if internal-lab details appear in the tree, skipping the paths that
# release/public-exclude.txt (internal-only) removes from the release. Run it on the
# exported tree (everything scanned, generic patterns) or on the internal repo
# (internal-only paths skipped, plus the concrete internal identifiers from
# release/leakcheck.internal-patterns, internal-only) -- both must pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Generic credential/identity shapes -- safe to ship. Case-insensitive.
PATTERNS=(
  'sk-proj-'                            # OpenAI project keys
  'AIza[0-9A-Za-z_-]{20}'               # Google API keys
  'ghp_[0-9A-Za-z]{20}'                 # GitHub PATs
  'hf_[0-9A-Za-z]{30}'                  # Hugging Face tokens
  'BEGIN (RSA|OPENSSH) PRIVATE KEY'
  '/(home|users|scratch|workspace1)/[a-z][a-z0-9_]{2,}/'  # personal absolute paths
)

# Concrete internal identifiers live in an export-excluded file so the shipped
# checker never carries them.
if [ -f release/leakcheck.internal-patterns ]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "${line}" | xargs || true)"
    [ -z "${line}" ] && continue
    PATTERNS+=("${line}")
  done < release/leakcheck.internal-patterns
fi

# Paths never scanned: the exclusion manifest (absent from the export), plus
# binary/vendored assets, the leak tooling itself, and git internals.
SKIP_ARGS=(
  ":!scripts/leakcheck.sh"
  ":!release/leakcheck.internal-patterns" ":!tests/test_public_release_configs.py"
  ":!assets" ":!*.png" ":!*.jpg" ":!*.stl" ":!*.usda" ":!*.pdf"
)
if [ -f release/public-exclude.txt ]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "${line}" | xargs || true)"
    [ -z "${line}" ] && continue
    SKIP_ARGS+=(":!${line}")
  done < release/public-exclude.txt
fi

# Personal-path shapes that are legitimate documentation placeholders.
ALLOW_RE='/(home|users)/(USER|user|<[a-z_-]+>)|Path.home|\$\{?HOME'

fail=0
for pat in "${PATTERNS[@]}"; do
  hits=$(git grep -I -n -E -i "${pat}" -- . "${SKIP_ARGS[@]}" 2>/dev/null || true)
  hits=$(printf '%s' "${hits}" | grep -Ev "${ALLOW_RE}" || true)
  if [ -n "${hits}" ]; then
    echo "LEAK [${pat}]:"
    echo "${hits}" | head -20
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "leakcheck: FAILED"
  exit 1
fi
echo "leakcheck: clean"
