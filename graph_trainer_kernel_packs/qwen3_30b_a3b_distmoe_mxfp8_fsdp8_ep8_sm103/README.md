# Qwen3 30B-A3B DistMoE MXFP8, FSDP 8 x EP 8, GB300 (sm_103)

15 CuTe DSL kernels for 15 fusion regions of one training step, each bitwise identical to the aten ops it
replaces. Pass name: `qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103`.

| | Baseline | With this pack |
|---|---|---|
| GPU time per step, excluding fake-PG collectives (the loop's profiled runs, GPU 0) | 967.9 ms | 679.6 ms (-288.3 ms, -29.8%) |
| loss, steps 1 / 10 | 12.458515167236328 / 5.717247009277344 | identical |
| grad_norm, steps 1 / 10 | 68.36530303955078 / 9.812052726745605 | identical |
| memory at step 10 | 236.0 GiB | 231.3 GiB |

**How these were measured.**
- **Hardware:** one GB300 with the fake process group, profiler on step 10 of 10, CUDA graphs off.
- **GPU time:** the union of kernel intervals in the profiled step (kernels launched with programmatic dependent
  launch overlap their predecessor, so a plain sum double-counts). Copies that only emulate collectives under the
  fake PG (~3 ms) are excluded.
- **Step:** one forward and backward of one 32K-token microbatch.

**Where the speedup comes from.** Two attention glue regions dominate:
- **Attention backward (G-01):** RoPE backward, q/k RMSNorm backward, concatenation with the v gradient, and
  MXFP8 row and column quantization, 30 kernels per layer. One persistent kernel replaces them: 2481 -> 185 us
  per layer.
- **Attention forward (G-02):** the split of the fused wqkv output, q/k RMSNorm and RoPE. 2252 -> 157 us per
  layer.

These two save about 211 ms per step. The rest are router, loss and norm-plus-quantize regions (sec. Kernels).
The finder's roofline analysis of the baseline estimated about 289 ms for fusing all fusible glue; about 12 ms of
glue headroom is left.

## Before a real multi-GPU run

Read this before using the pack outside the fake-PG setup it was verified in:

- **Other ranks.** Every fake-PG run represents SPMD coordinate 0. By construction the pack is rank-agnostic:
  - matchers accept any size, scalar operand and device index;
  - cache keys drop the device index;
  - the references contain only model constants as literals (head dim 128, 32 q and 4 kv heads, 128 experts,
    16 local experts, top-8), no rank-dependent offsets.

  All 15 kernels were also checked exact on device index 1 (imported with another device current, called with
  another device current, and in a run with `LOCAL_RANK=1`). But no other rank's graph was run.
- **152 SMs.** The RMSNorm kernels (G-01, G-15, G-19, G-41, G-45) reproduce the reduction order of the quack
  RMSNorm that this torch build dispatches `aten._fused_rms_norm` and its backward to. That order depends on the
  SM count (16 CTAs per SM). They were verified on a 152-SM GB300; the target checks the compute capability, not
  the SM count. G-01 raises at import on another SM count.
- **Attention backend.** cuDNN's varlen attention backward is nondeterministic even with `debug.deterministic`
  (two identical runs differ in step-1 grad_norm), so `recipe.py` pins FA4. No kernel touches attention, but
  G-01 reads the attention backward's outputs and its specialization includes their layout. With cuDNN the pack
  is untested; if a specialization differs, the pass raises.
- **CUDA graphs.** `recipe.py` keeps the benchmark recipe's CUDA graphs on. The pass loads every kernel at
  compile time on the rank's device, before capture. G-01 uploads static tables at import: imported with one
  device current, a first call on another device inside capture raises.
- **Inputs outside the training contract** (never produced by the recipe):
  - G-07: labels >= vocab size, or negative other than -100, no longer device-assert;
  - G-02: out-of-range positions read outside the RoPE table instead of raising;
  - G-47: out-of-range expert indices wrap (`idx & 127`) where the reference asserts;
  - G-01: strided weights or misaligned inputs raise; G-91 falls back to the reference ops.
- **G-91** divides with a fast path only inside an input range where it is proven exact (checked on 1.07e9
  random and 3.2e6 hard-case pairs); the range check is part of the verified kernel.

## Target

`manifest.json` `target` is checked exactly when the pass runs:

| Field | Value |
|---|---|
| torch | 2.16.0.dev20261105+cu132 (`torch.version.git_version` 084f8661f5bb5de149860f74d82f5313766f4256) |
| CUDA | 13.2 |
| GPU | GB300, compute capability (10, 3), 152 SMs |
| nvidia-cutlass-dsl | 4.8.1 |
| `TORCH_DISABLE_NATIVE_JIT` | unset |

**Built and verified with,** not checked by the pass (their op sources are part of the kernels' cache keys):
torchao 0.19.0.dev20261105+cu132, dist_moe, torchtitan
[agentic_compiler](https://github.com/pytorch/torchtitan/tree/agentic_compiler) `8e8145923`.

**Recipe:** `recipe.py`, torchtitan's `graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu` with:
- one microbatch per step (`num_tokens_per_train_step = -1`; the benchmark recipe accumulates 8);
- FA4 varlen attention;
- `debug.deterministic = True` and seed 42. The specialization keys include the deterministic-algorithms flag;
  without it the pass raises.

**Parallelism:** FSDP 8, EP 8, no PP, 32K-token microbatches, MXFP8 attention projections, MXFP8 Dist-MoE
experts, BF16 LM head.

## Kernels

Calls are per training step, i.e. per joint graph; the pass raises `ValueError` if any count differs. Writer and
verifier names refer to `verification/board.md`. Times are the verifier's measurements: CUDA-graph replays with
rotating inputs and live outputs, interleaved with the reference. Saving = (reference - kernel) x calls.

| Pattern (region) | What | Calls | Ref us | Kernel us | Saving ms | Writer -> verifier |
|---|---|---|---|---|---|---|
| qnfe (G-01) | attention bwd glue: RoPE bwd q/k, q/k RMSNorm bwd (incl. dw), cat with v grad, MXFP8 row+col quant | 48 | 2480.6 | 184.5 | 110.2 | ada -> viggo |
| 4dzo (G-02) | attention fwd glue: split of the wqkv output, q/k RMSNorm, RoPE with position gather | 48 | 2252.4 | 157.1 | 100.6 | ada -> vera |
| s3of (G-06) | router + aux-loss backward | 48 | 381.8 | 12.05 | 17.7 | cyd -> vala |
| 53aq (G-07) | loss backward: `nll_loss_backward` + `log_softmax` backward, [4096,151936] fp32 -> bf16 | 8 | 2556.1 | 518.5 | 16.3 | dex -> vox |
| bme7 (G-90) | loss forward per chunk: bf16 -> fp32 cast, `log_softmax`, `nll_loss`, div | 8 | 2040.5 | 544.2 | 12.0 | dex -> vika |
| bp6b (G-17) | router forward: per-expert token counts | 48 | 125.75 | 3.59 | 5.9 | cyd -> vojta |
| zfby (G-19) | add + RMSNorm bwd (N=2048) + add + MXFP8 row+col quant | 48 | 258.7 | 146.9 | 5.4 | dex -> vesna |
| bj3b (G-91) | router aux-loss forward: row L1 norm, `clamp_min`, div, masked column sum | 48 | 78.54 | 7.01 | 3.4 | bix -> vilma |
| tl5t (G-15) | wqkv dgrad GEMM (call unchanged) + attention_norm RMSNorm bwd + add | 48 | 348.0 | 282.7 | 3.1 | dex -> vada |
| gri3 (G-33) | routed-experts w13 [16,1536,2048] -> `dist_moe.prepare_mxfp8_weight` | 48 | 80.31 | 37.30 | 2.1 | eve -> vola |
| adoe (G-42) | same for w2 [16,2048,768] | 48 | 44.97 | 20.15 | 1.2 | eve -> vola |
| kbxm (G-41) | attention_norm RMSNorm + MXFP8 quant | 48 | 76.53 | 52.09 | 1.2 | ada -> vin |
| na2k (G-45) | residual add + ffn_norm RMSNorm | 48 | 98.26 | 77.25 | 1.0 | ada -> vic |
| vf5c (G-28) | wo forward: MXFP8 quant + scale layout, then the unchanged GEMM | 48 | 241.5 | 221.8 | 0.9 | eve -> vita |
| 5ifh (G-47) | router softmax, top-k score gather, normalization | 48 | 25.23 | 7.95 | 0.8 | cyd -> vlasta |

The verified savings add up to about 282 ms per step; the loop measured 288.3 ms.

**Not in this pack:** G-24 (colwise MXFP8 quantize before the wo wgrad GEMM, 96 calls): a candidate was exact,
but saved only about 2 us per call, since torchao's quantize is near bandwidth. The rest of the step is DistMoE
grouped GEMMs (~304 ms), attention (~157 ms), other GEMMs (~52 ms) and the optimizer (~34 ms).

## How the kernels were verified

Each kernel was written by one agent and checked by a different, freshly started verifier. Only a PASS was
promoted; `verification/board.md` records every handoff, verdict and promotion. Verifiers failed three
submissions, all fixed and re-verified before promotion:
- **G-06 t13:** a softmax-backward sum without its `+0` seed gave -0 where the reference gives +0.
- **G-15 t3, G-19 t5:** non-coherent (`.nc`) loads of partials written by an earlier kernel in the same PDL
  chain.

Every verifier checked:

- **Contract:**
  - signature and output metadata against `specialization_<l2>.json`;
  - the harness (`verification/harness.py`): seeds 0-7 plus real captured inputs, every output byte equal to
    `reference.py`;
  - inputs unmutated, and repeated calls identical.
- **Memory:** output memory poisoned before each call, so unwritten bytes fail; compute-sanitizer memcheck,
  initcheck, racecheck and synccheck, with planted bugs the filters must flag.
- **CUDA graphs:** capture with neighbouring kernels writing the inputs and reading the outputs, replays with
  inputs changed in place, two or more calls in one graph, and the first call inside capture.
- **Adversarial inputs:** each verifier's own suite with deliberately broken copies that must fail. Reduction
  orders (the RMSNorm dw reductions, column sums that only feed a scalar) were tested with exactly representable
  cancelling terms placed so that only the reference's order gives the reference's result.
- **GEMM-adjacent kernels (G-15, G-28):** the `_scaled_mm_v2` call is identical (op, arguments, operand layout
  and alignment) and its output bitwise equal.
- **PDL:** kernels using programmatic dependent launch were run after an early-triggering predecessor that
  writes the inputs, and their SASS was checked for non-coherent loads of data written during the call chain.

**Whole-step check:** loss, grad_norm and load-balance loss of the 10-step runs with the kernels are bitwise equal
to the baseline at both logged steps (1 and 10).

Writer candidates and verifier logs are not published. The loop's audit report is in torchtitan branch
`agentic_compiler`, at `torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8/README.md`.

Re-gate a kernel from the root of a torchtitan checkout:

```
PACK=<site-packages or clone>/graph_trainer_kernel_packs/qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103
PYTHONPATH=. python $PACK/verification/harness.py $PACK/kernels/pat_.../specialization_<l2>.json \
    $PACK/kernels/pat_.../<l2>.py --seeds 0 1 2
```

## Validation

10-step runs with this package at torchtitan `8e8145923` on 2026-11-06, one GB300 with the fake PG:

| Run | Kernel calls | Loss / grad_norm | GPU time or tokens/s |
|---|---|---|---|
| `recipe.py` as shipped (default pass pipeline, CUDA graphs on) plus TensorBoard, device index 0 | 640 to 15 kernels, as in `manifest.json`; 0 lifted calls left on aten | bitwise identical to the recipe without the pack, steps 1 and 10 | 30,980 tokens/s at step 10 |
| the same, device index 1 (`LOCAL_RANK=1`) | 640 to 15 kernels | bitwise identical to the recipe without the pack, steps 1 and 10 | 22,288 tokens/s |
| `recipe.py` without the pack (the reference) | - | - | 24,441 tokens/s |
| the loop's measurement setup (fusion_finder pass pipeline, profiler on step 10, CUDA graphs off), device index 0 | 640 to 15 kernels | bitwise identical to the loop's baseline, steps 1 and 10 | 679.5 ms GPU time per step, excluding fake-PG collectives (loop: 679.6 ms with the kernels, 967.9 ms baseline) |

**Notes on these runs:**
- **Numerics:** all runs also matched the MoE load-balance loss bitwise. The values compared are the TensorBoard
  scalars at full float precision, not the five digits printed to stdout.
- **Throughput:** the tokens/s values are not a clean comparison. The runs shared the node with each other and
  with another user's jobs, which held memory on every GPU. The GPU-time row is the measured speedup.
- **Graph-site counts:** one microbatch per step, so the pass's 640 calls run once per step.

## Files

| Path | Contents |
|---|---|
| `__init__.py` | registers the pass |
| `recipe.py` | the verified recipe |
| `manifest.json` | target, and per kernel its sha256 and calls per joint graph (generated) |
| `regions/` | the 15 regions' topology matchers, independent of shapes and rank; `PLAN` is the lift order (generated) |
| `kernels/` | `pat_<l1>/reference.py`, `specialization_<l2>.json`, `<l2>.py` (generated, byte-identical to the verified files) |
| `verification/` | the swarm's task board and the correctness/timing harness |

Generated with (`<loop>` = the torchtitan report directory above; `iter7.log` is the log of the loop's final
run at torchtitan `8e8145923`). G-90's docstring was edited afterwards to drop a local path:

```
python tools/make_pack.py --regions <loop>/fusion_regions --kernel-cache <loop>/kernel_cache \
    --log iter7.log --out graph_trainer_kernel_packs/qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103 \
    --plan lift_fusion_region_g01,lift_fusion_region_g02,lift_fusion_region_g07,lift_fusion_region_g06,lift_fusion_region_g17,lift_fusion_region_g19,lift_fusion_region_g15,lift_fusion_region_g33,lift_fusion_region_g42,lift_fusion_region_g41,lift_fusion_region_g45,lift_fusion_region_g28,lift_fusion_region_g47,lift_fusion_region_g90,lift_fusion_region_g91
```
