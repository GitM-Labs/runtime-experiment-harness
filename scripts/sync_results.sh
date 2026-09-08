#!/usr/bin/env bash
# Continuously pull completed experiment results off the cluster.
#
#   scripts/sync_results.sh [local-dir]      # loops until Ctrl-C
#
# Cluster credits can disappear without notice; nothing important may exist
# only on the cluster. This tars the shared results tree through the staging
# pod every SYNC_INTERVAL_S (default 300s) into a timestamp-free local mirror
# (later pulls overwrite with newer copies of the same paths; completed run
# dirs never change, so the mirror converges).
set -euo pipefail

REX_DIR="${REX_DIR:-/mnt/shared/rex}"
STAGE_POD="${STAGE_POD:-rex-stage}"
DEST="${1:-./mi355x-results}"
SYNC_INTERVAL_S="${SYNC_INTERVAL_S:-300}"

mkdir -p "${DEST}"
echo "==> syncing ${STAGE_POD}:${REX_DIR}/results -> ${DEST} every ${SYNC_INTERVAL_S}s (Ctrl-C to stop)"

while true; do
  started=$(date +%s)
  if kubectl exec "${STAGE_POD}" -- sh -c "cd '${REX_DIR}' && tar cf - results" \
      | tar xf - -C "${DEST}"; then
    echo "$(date '+%H:%M:%S') synced ($(du -sh "${DEST}" | cut -f1) local)"
  else
    echo "$(date '+%H:%M:%S') sync failed; retrying next round" >&2
  fi
  elapsed=$(( $(date +%s) - started ))
  sleep_for=$(( SYNC_INTERVAL_S > elapsed ? SYNC_INTERVAL_S - elapsed : 0 ))
  sleep "${sleep_for}"
done
