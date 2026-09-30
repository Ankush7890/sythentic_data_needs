#!/usr/bin/env bash
# The whole dc_xfer run in the brief's order; every stage resumes, so re-running is safe.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
S=scripts/run_dc_xfer.sh
while pgrep -f "dc_xfer.py --stage warm" > /dev/null; do sleep 30; done
$S fit instructions && $S cells instructions
$S warm hu_harm && $S fit hu_harm && $S cells hu_harm
$S warm highstakes && $S fit highstakes && $S cells highstakes
