# InferenceX K2.5 improvement roadmap — what runs where

Goal: close the gap to ATOM (~5,370 tok/s/GPU vs ~2,700 on the vLLM lane,
same MI355X silicon) by porting ATOM's wins into vLLM, then adding levers no
K2.5 lane has yet. InferenceX requires a vLLM/SGLang submission before other
frameworks, so vLLM is the lane that counts.

## Bucket A — config-level, implemented in `kimi-k25-inferencex.yaml`

| Lever | Arm(s) | Confidence |
| --- | --- | --- |
| ~~MTP speculative decoding~~ | removed 2026-09-15 | **dead for this checkpoint** — see below |
| Scheduler: `--scheduler-delay-factor 1` + `--async-scheduling` | `ix-levers/sched-delay` | high |
| KV cache fp8 | `ix-levers/kv-fp8` | high |
| AITER MLA decode + FA prefill (K3-lane pattern) | `ix-levers/aiter-mla` | env names = VERIFY in server.log |
| Quick Reduce INT4 (proven on DSV4 8k/1k) | `ix-levers/quickreduce-int4` | high |
| Rust frontend (MiniMax-lane pattern) | `ix-levers/rust-frontend` | VERIFY build ships it |
| TP4 (x2-replica story; per-GPU tok/s vs TP8) | `ix-tp4` | high; CK FP4 MoE fault only if FP4 quant |
| DP attention + EP (+ DBO) | `ix-dp-attn` | DBO flag name = VERIFY |

A/B validity rule: any arm whose lever is an env var must be confirmed in the
run's `server.log` (vLLM echoes config at startup). An unrecognized env
silently measures the control — never publish such an arm as a technique row.

## MTP: dead for the released K2.5 checkpoint (established 2026-09-15)

The released checkpoint (snapshot 4d01dfe0) ships **no MTP head**:
`text_config.num_nextn_predict_layers = 0` and the safetensors index has zero
mtp/nextn/eh_proj tensors. vLLM's `Unsupported speculative method: 'mtp'` was
correct behavior — this build knows `kimi_k3_mtp` but there is no K2.5 head
to support. No config or vLLM patch conjures absent weights.

Speculative-decoding paths that remain, in ascending effort:

- **n-gram spec decode** (model-free, supported): near-useless on synthetic
  random benchmark text; skip for InferenceX rows.
- **Train an EAGLE3 draft head** for K2.5 (this build supports `eagle3`):
  a real but training-scale project; would be the first K2.5 spec-decode lane
  anywhere.
- **The ask upstream**: Moonshot shipping a K3-style MTP head for K2.5 —
  document as the gap in the submission notes.

## Bucket B — vLLM code work (the actual ATOM port; not a manifest change)

1. **A4W4 fused MoE + fused router.** ATOM's core win: activations and
   weights both 4-bit; one fused SwiGLU→quantize→scale-write kernel between
   the grouped GEMMs; sorting fused into GEMM metadata; per-shape tuning for
   K2's experts; single-kernel top-k specialized for 8-of-384 routing. The
   AITER kernels are public (AITER #3832, #3466); the work is dispatching to
   them from vLLM's MoE path (fused_moe layer → rocm_aiter backend).
   **Scope narrowed 2026-09-15**: the released K2.5 checkpoint already ships
   routed experts as INT4 weights (compressed-tensors, group-32; attention,
   shared expert, dense MLP, lm_head, vision excluded) — so the gap on the
   vLLM lane is specifically 4-bit *activations* plus the fused kernels and
   router, not weight quantization. This is a vLLM PR, not a flag.
2. **MLA small-kernel fusion.** K3's wrapper fuses q_a/kv_a RMSNorm into one
   AITER launch; port the same to K2.5's MLA layer if the env-flag arm shows
   the unfused launches on the critical path (the rocprof kernel plane from
   the kimi-k25 sweep answers this).
3. **Shared-expert fusion.** Added to DSV4 MI355X paths; port to K2.5's MoE
   if absent in the build (inspect vLLM's rocm MoE path first — may already
   be env-gated, in which case it moves to Bucket A).

## Bucket C — out of scope on one 8x node

- **Multi-node disagg + wide EP** (1P+2D, TP8/EP8 decode, MTP on decode
  workers): the largest remaining headline gap vs GB200 NVL72, but needs
  multiple nodes and a P/D split; revisit when the marketplace grants a
  multi-node window. The JobSet pattern in
  `~/Desktop/Workspaces/nationalcompute/kimi-deploy.yaml` is the starting
  point for the serving topology.

## Run it

```bash
python3 -m pytest tests/ -q          # manifest loads through the harness
./scripts/launch_mi355x_shards.sh kimi-k25-inferencex   # 2 nodes, ~3h each
```

~11 server launches per node ≈ 3h at the observed ~17 min/arm cadence. Two
nodes give n=2 per arm; rerun for more repeats. Analysis pairs each variant
against its experiment's control on identical traffic; report tok/s, tok/s/GPU
(for the TP4/DP rows), and TTFT/ITL p50/p95/p99 deltas.
