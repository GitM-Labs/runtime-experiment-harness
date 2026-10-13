#!/usr/bin/env bash
# Live dashboard for the rex parallelism sweep: job status + the InferenceX-
# format results table, refreshed on an interval. Reads the LOCAL mirror, so
# run scripts/sync_results.sh in another terminal to keep results flowing in.
#
#   scripts/poll_status.sh [interval-seconds]   (default 120)
#
# Ctrl-C to stop. Nothing here launches or changes cluster state — read-only.
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-120}"

while true; do
  clear 2>/dev/null || true
  echo "============================================================"
  echo " rex sweep status — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  echo
  echo "## GPU jobs"
  kubectl get jobs -l app=rex 2>/dev/null || echo "  (kubectl unavailable here)"
  echo
  echo "## results so far (InferenceX format; FAILURES show a concrete reason)"
  python3 scripts/summarize_metrics.py 2>/dev/null || echo "  (no results pulled to the mirror yet)"
  echo
  echo "-- refreshing every ${INTERVAL}s; Ctrl-C to stop --"
  sleep "${INTERVAL}"
done
