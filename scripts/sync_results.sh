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
#
# Each successful pull is then mirrored to Google Drive via rclone when
# GDRIVE_REMOTE (default gdrive:mi355x-results) points at a configured
# remote. One-time setup:  rclone config create gdrive drive
# Set GDRIVE_REMOTE="" to disable the upload leg.
set -euo pipefail

REX_DIR="${REX_DIR:-/mnt/shared/rex}"
STAGE_POD="${STAGE_POD:-rex-stage}"
DEST="${1:-./mi355x-results}"
SYNC_INTERVAL_S="${SYNC_INTERVAL_S:-300}"
GDRIVE_REMOTE="${GDRIVE_REMOTE-gdrive:mi355x-results}"

if [[ -n "${GDRIVE_REMOTE}" ]]; then
  remote_name="${GDRIVE_REMOTE%%:*}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "WARNING: rclone not installed; results will NOT reach Google Drive." >&2
    echo "         brew install rclone && rclone config create ${remote_name} drive" >&2
    GDRIVE_REMOTE=""
  elif ! rclone listremotes | grep -qx "${remote_name}:"; then
    echo "WARNING: rclone remote '${remote_name}:' not configured; results will NOT reach Google Drive." >&2
    echo "         Run once (opens browser for Google sign-in):  rclone config create ${remote_name} drive" >&2
    GDRIVE_REMOTE=""
  fi
fi

mkdir -p "${DEST}"
echo "==> syncing ${STAGE_POD}:${REX_DIR}/results -> ${DEST} every ${SYNC_INTERVAL_S}s (Ctrl-C to stop)"
if [[ -n "${GDRIVE_REMOTE}" ]]; then
  echo "==> each pull is mirrored to ${GDRIVE_REMOTE}"
fi

drive_link_printed=""

# Pull one top-level results dir via the chunked puller. kubectl exec streams
# to this cluster die after ~60-90s, so any single-stream tar of a multi-GB
# dir can never finish; the chunked script compresses on the pod and pulls
# the archive in short, individually-retried dd reads.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pull_dir() {
  DEST="${DEST}" REX_DIR="${REX_DIR}" STAGE_POD="${STAGE_POD}" \
    bash "${SCRIPT_DIR}/pull_results_chunked.sh" "$1"
}

mkdir -p "${DEST}/results"
SIZES="${DEST}/.remote-sizes"
touch "${SIZES}"

while true; do
  started=$(date +%s)
  round_ok=1
  # -sk, not -sb: the staging pod is BusyBox, whose du has no -b.
  listing=$(kubectl exec "${STAGE_POD}" -- sh -c "cd '${REX_DIR}/results' && du -sk -- */ 2>/dev/null" || true)
  if [[ -z "${listing}" ]]; then
    round_ok=""
  fi
  while read -r size dir; do
    [[ -z "${dir}" ]] && continue
    dir="${dir%/}"
    # Completed run dirs never change; skip any dir whose remote byte size
    # matches what we recorded after its last successful pull.
    if grep -qxF "${size} ${dir}" "${SIZES}"; then
      continue
    fi
    if pull_dir "${dir}"; then
      # exact-field match: "foo" must not evict the record for "foo-v2"
      awk -v d="${dir}" '$2 != d' "${SIZES}" > "${SIZES}.tmp" || true
      echo "${size} ${dir}" >> "${SIZES}.tmp"
      mv "${SIZES}.tmp" "${SIZES}"
      echo "$(date '+%H:%M:%S') pulled ${dir}"
    else
      round_ok=""
    fi
  done <<< "${listing}"
  if [[ -n "${round_ok}" ]]; then
    echo "$(date '+%H:%M:%S') synced ($(du -sh "${DEST}" | cut -f1) local)"
    if [[ -n "${GDRIVE_REMOTE}" ]]; then
      # copy, not sync: completed run dirs never change, so already-uploaded
      # files are skipped and nothing on Drive is ever deleted.
      if rclone copy "${DEST}" "${GDRIVE_REMOTE}" \
          --transfers 4 --drive-chunk-size 128M --quiet; then
        echo "$(date '+%H:%M:%S') uploaded to ${GDRIVE_REMOTE}"
        if [[ -z "${drive_link_printed}" ]]; then
          link=$(rclone link "${GDRIVE_REMOTE}" 2>/dev/null || true)
          if [[ -n "${link}" ]]; then
            echo "==> Google Drive folder: ${link}"
            drive_link_printed=1
          fi
        fi
      else
        echo "$(date '+%H:%M:%S') Drive upload failed; retrying next round" >&2
      fi
    fi
  else
    echo "$(date '+%H:%M:%S') sync failed; retrying next round" >&2
  fi
  elapsed=$(( $(date +%s) - started ))
  sleep_for=$(( SYNC_INTERVAL_S > elapsed ? SYNC_INTERVAL_S - elapsed : 0 ))
  sleep "${sleep_for}"
done
