#!/usr/bin/env bash
# Stage rex onto the cluster's shared filesystem and launch one Job per
# (shard, node-index) assignment.
#
#   scripts/launch_mi355x_shards.sh                # stage + launch everything
#   scripts/launch_mi355x_shards.sh kimi-k25       # just one shard's jobs
#   scripts/launch_mi355x_shards.sh qwen-36-35b 1  # one shard, override node count
#                                                  # (smoke-test a single Job first)
#
# Node budget (per the run plan): kimi-k25 gets 4 nodes, every other model 2.
# A "node" here is one 8-GPU Job; the scheduler places them. Multiple indexes
# of the same shard run the same manifest — rex stamps each run with its own
# timestamp, so parallel copies of a sweep don't collide in the results tree.
#
# Prereqs: kubectl context pointed at the cluster; hf-token-secret applied;
# a staging pod or any pod that mounts /mnt/shared (STAGE_POD below) to copy
# files through. Build artifacts are staged, not rebuilt on-cluster.
set -euo pipefail

cd "$(dirname "$0")/.."

REX_DIR="${REX_DIR:-/mnt/shared/rex}"
STAGE_POD="${STAGE_POD:-rex-stage}"
# Container image for the rex Job. Default keeps prior behavior; override to pin
# a specific vLLM ROCm build for a reproduction, e.g.
#   IMAGE=vllm/vllm-openai-rocm:v0.24.0 ./scripts/launch_mi355x_shards.sh kimi-k25-int4-repro
# (:latest drifts — always pin for a result you intend to compare or publish.)
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:latest}"
ONLY_SHARD="${1:-}"
# Optional second arg: node count for that shard (requires ONLY_SHARD).
COUNT_OVERRIDE="${2:-}"
if [[ -n "${COUNT_OVERRIDE}" && -z "${ONLY_SHARD}" ]]; then
  echo "a count override needs a shard name first" >&2; exit 1
fi

# shard -> parallel node count ("shard=count", plain array: macOS ships bash 3.2,
# which has no associative arrays)
BUDGET=(
  kimi-k25=4
  kimi-k25-inferencex=2
  kimi-k25-ix-pareto=1
  kimi-k25-int4-repro=1
  kimi-k25-tp4=1
  kimi-k25-tp2-dp4-qr=1
  kimi-k25-tp4-dp2-qr=1
  kimi-k25-tp8-qr=1
  kimi-k25-tp2-dp4-ep-qr=1
  kimi-k25-tp4-dp2-ep-qr=1
  beat-baseline=1
  beat-quickreduce=1
  beat-dp-ep=1
  beat-dp-ep-qr=1
  beat-trace=1
  kimi-k26-inferencex-repro=1
  kimi-k3=2
  deepseek-v4-flash=2
  deepseek-v4-pro=2
  glm-52=2
  qwen-36-35b=2
)

echo "==> building wheel"
# --no-isolation: the default builds a throwaway venv and pip-installs
# hatchling into it on every run — the slow part of launching. Install the
# build deps into the current env once and reuse them.
python3 -m pip install --quiet build hatchling
python3 -m build --wheel --no-isolation --outdir dist

echo "==> ensuring staging pod ${STAGE_POD} (mounts /mnt/shared)"
if ! kubectl get pod "${STAGE_POD}" >/dev/null 2>&1; then
  kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${STAGE_POD}
  labels: {app: rex-stage}
spec:
  containers:
  - name: stage
    image: busybox:1.36
    command: ["sleep", "infinity"]
    volumeMounts:
    - {name: shared, mountPath: /mnt/shared}
  volumes:
  - name: shared
    hostPath: {path: /mnt/shared}
EOF
  kubectl wait --for=condition=ready "pod/${STAGE_POD}" --timeout=120s
fi

echo "==> staging wheel + shard manifests to ${REX_DIR}"
kubectl exec "${STAGE_POD}" -- mkdir -p "${REX_DIR}/wheel" "${REX_DIR}/shards" "${REX_DIR}/results"
for wheel in dist/runtime_experiment_harness-*.whl; do
  kubectl cp "${wheel}" "${STAGE_POD}:${REX_DIR}/wheel/$(basename "${wheel}")"
done
for shard_file in experiments/mi355x/*.yaml; do
  kubectl cp "${shard_file}" "${STAGE_POD}:${REX_DIR}/shards/$(basename "${shard_file}")"
done

echo "==> launching jobs"
for entry in "${BUDGET[@]}"; do
  shard="${entry%=*}"
  count="${entry#*=}"
  if [[ -n "${ONLY_SHARD}" && "${shard}" != "${ONLY_SHARD}" ]]; then
    continue
  fi
  for index in $(seq 1 "${COUNT_OVERRIDE:-${count}}"); do
    job="rex-${shard}-${index}"
    # A Job's pod template is immutable, so `kubectl apply` over a prior run of
    # the same name fails ("field is immutable"). Delete any existing Job first
    # (a completed/failed one, or one being relaunched) so create always works.
    kubectl delete job "${job}" --ignore-not-found --wait=true
    sed -e "s|__SHARD__|${shard}|g" \
        -e "s|__INDEX__|${index}|g" \
        -e "s|__REX_DIR__|${REX_DIR}|g" \
        -e "s|__IMAGE__|${IMAGE}|g" \
        deploy/mi355x-rex-job.yaml | kubectl create -f -
  done
done

echo "==> done. watch: kubectl get jobs -l app=rex; logs: kubectl logs -l app=rex -f --max-log-requests 20"
