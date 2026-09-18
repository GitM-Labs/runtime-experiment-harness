#!/usr/bin/env bash
# Pull result directories off the cluster through the staging pod, robustly.
#
#   scripts/pull_results_chunked.sh              # pull every top-level results dir
#   scripts/pull_results_chunked.sh qwen-36-35b  # pull just the named dir(s)
#
# Why this shape: kubectl exec streams to this cluster die after ~60-90s
# (API-server proxy timeout), so any single-stream tar of a multi-GB dir can
# never finish. Instead each dir is compressed ON the pod into a temp archive
# on /mnt/shared (no stream involved, detached so the exec returns instantly),
# then the archive is pulled in CHUNK_MB-sized dd reads — each chunk a fresh
# short exec, retried independently — reassembled locally, verified with
# `gzip -t`, and extracted. Written for BusyBox on the pod and bash 3.2 on macOS.
set -uo pipefail

REX_DIR="${REX_DIR:-/mnt/shared/rex}"
STAGE_POD="${STAGE_POD:-rex-stage}"
DEST="${DEST:-./mi355x-results}"
CHUNK_MB="${CHUNK_MB:-64}"          # keep each stream well under the ~60s cutoff
CHUNK_RETRIES="${CHUNK_RETRIES:-5}"
COMPRESS_TIMEOUT_S="${COMPRESS_TIMEOUT_S:-1800}"
TMP_REMOTE="${REX_DIR}/.pulltmp"

pod() { kubectl exec "${STAGE_POD}" -- sh -c "$1"; }

remote_size() {  # archive byte size, empty on failure
  pod "wc -c < '$1' 2>/dev/null" | tr -d '[:space:]'
}

pull_archive() {  # $1 remote tgz, $2 local tgz, $3 total bytes
  local remote="$1" local_f="$2" total="$3"
  local chunk_bytes=$((CHUNK_MB * 1024 * 1024))
  local offset=0 part="${local_f}.part" chunk="${local_f}.chunk"
  : > "${part}"
  while (( offset < total )); do
    local want=$((total - offset)); (( want > chunk_bytes )) && want=${chunk_bytes}
    local ok="" attempt
    for attempt in $(seq 1 "${CHUNK_RETRIES}"); do
      # bs=1M with skip in MB units: offset is always a CHUNK_MB multiple.
      if pod "dd if='${remote}' bs=1M skip=$((offset / 1024 / 1024)) count=${CHUNK_MB} 2>/dev/null" > "${chunk}"; then
        local got; got=$(wc -c < "${chunk}" | tr -d '[:space:]')
        if [[ "${got}" == "${want}" ]]; then ok=1; break; fi
        echo "    chunk @${offset}: got ${got}, want ${want} (attempt ${attempt})" >&2
      else
        echo "    chunk @${offset}: exec failed (attempt ${attempt})" >&2
      fi
      sleep 3
    done
    [[ -z "${ok}" ]] && { rm -f "${part}" "${chunk}"; return 1; }
    cat "${chunk}" >> "${part}"
    offset=$((offset + want))
    printf '  %s/%s MB\r' "$((offset / 1024 / 1024))" "$((total / 1024 / 1024))" >&2
  done
  echo >&2
  rm -f "${chunk}"
  mv "${part}" "${local_f}"
}

pull_dir() {  # $1 top-level dir name under ${REX_DIR}/results
  local d="$1" tgz="${TMP_REMOTE}/$1.tgz" marker="${TMP_REMOTE}/$1.done"
  echo "==> ${d}: compressing on pod"
  pod "mkdir -p '${TMP_REMOTE}' && rm -f '${tgz}' '${marker}' && cd '${REX_DIR}/results' && (nohup sh -c \"tar czf '${tgz}' '${d}' && touch '${marker}'\" >/dev/null 2>&1 &)" \
    || { echo "==> ${d}: failed to start compression" >&2; return 1; }
  local waited=0
  until pod "test -f '${marker}'" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    if (( waited >= COMPRESS_TIMEOUT_S )); then
      echo "==> ${d}: compression timed out after ${COMPRESS_TIMEOUT_S}s" >&2; return 1
    fi
  done
  local size; size=$(remote_size "${tgz}")
  [[ -z "${size}" || "${size}" == "0" ]] && { echo "==> ${d}: empty/unreadable archive" >&2; return 1; }
  echo "==> ${d}: pulling $((size / 1024 / 1024)) MB in ${CHUNK_MB}MB chunks"
  local local_tgz="${DEST}/.incoming-$1.tgz"
  pull_archive "${tgz}" "${local_tgz}" "${size}" || { echo "==> ${d}: chunk pull failed" >&2; return 1; }
  if ! gzip -t "${local_tgz}"; then
    echo "==> ${d}: archive failed integrity check" >&2; rm -f "${local_tgz}"; return 1
  fi
  tar xzf "${local_tgz}" -C "${DEST}/results" || { echo "==> ${d}: extract failed" >&2; return 1; }
  rm -f "${local_tgz}"
  pod "rm -f '${tgz}' '${marker}'" || true
  echo "==> ${d}: done"
}

mkdir -p "${DEST}/results"
if [[ $# -gt 0 ]]; then
  dirs="$*"
else
  dirs=$(pod "cd '${REX_DIR}/results' && ls -1") || { echo "cannot list results" >&2; exit 1; }
fi

overall=0
for d in ${dirs}; do
  pull_dir "${d}" || overall=1
done
echo "==> local total: $(du -sh "${DEST}" | cut -f1)"
exit "${overall}"
