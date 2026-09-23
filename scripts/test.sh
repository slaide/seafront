#!/usr/bin/env bash
# Run the test suite with the whole filesystem read-only, the repository included. The only
# writable place is a scratch /tmp, removed when the run ends, so a test that writes anywhere
# else (the real ~/seafront, the source tree) fails with "Read-only file system" and names
# the path, instead of leaving files behind.
#
# usage: scripts/test.sh [pytest args...]     e.g. scripts/test.sh tests/test_imaging_order.py -x
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$repo/.venv/bin/python"

if ! command -v bwrap >/dev/null; then
    echo "scripts/test.sh: bwrap (bubblewrap) is required to sandbox the run" >&2
    exit 2
fi
if [[ ! -x $python ]]; then
    echo "scripts/test.sh: no $python; run 'uv sync --group dev' first" >&2
    exit 2
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/seafront-test-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# /dev/bus is there read-only because the Galaxy camera SDK refuses to initialise without the
# USB bus to enumerate, even when a test only hands it fake devices.
#
# bytecode and coverage data are the only writes a run makes on purpose: bytecode is not
# written at all, and coverage data goes to the scratch /tmp
bwrap \
    --ro-bind / / \
    --dev /dev \
    --ro-bind-try /dev/bus /dev/bus \
    --proc /proc \
    --bind "$scratch" /tmp \
    --setenv TMPDIR /tmp \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    --setenv COVERAGE_FILE /tmp/.coverage \
    --setenv MOCK_NO_DELAYS "${MOCK_NO_DELAYS:-1}" \
    --chdir "$repo" \
    --die-with-parent \
    -- "$python" -m pytest -p no:cacheprovider "$@"
