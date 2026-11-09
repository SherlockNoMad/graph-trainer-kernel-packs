# Qwen3 235B-A22B DistMoE MXFP8, FSDP 256 x EP 32, GB300 (sm_103)

19 CuTe DSL kernels for 19 fusion regions of one training step. Each is bitwise identical to the aten ops it
replaces. Pass name: `qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32_sm103`.

Most kernels were ported from the [Qwen3 30B-A3B pack](../qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103/README.md).
The loop that produced this pack took 2 h 41 min, against about 13 h for the 30B pack from scratch. torchtitan's
`reuse_report.md` (see Provenance) covers what transferred.

| | Baseline | With this pack |
|---|---|---|
| tokens/s, 20 steps of `recipe.py` (CUDA graphs on), loop measurement, GPU 3 | 7,698 | 10,631 (+38.1%) |
| GPU time per step, profiled step, excluding fake-PG collectives (loop, GPU 0) | 1036.4 ms | 737.7 ms (-298.7 ms, -28.8%) |
| loss, steps 1 / 10 / 20 | 12.65354061126709 / 9.582934379577637 / 9.238986015319824 | identical |
| grad_norm, steps 1 / 10 / 20 | 204.34934997558594 / 14.810263633728027 / 14.770392417907715 | identical |
| memory at step 20 | 212.2 GiB | 212.6 GiB |

**How these were measured.**
- **Hardware:** one GB300 with the fake process group (NGPU=256).
- **Step:** one forward and backward of one 8192-token microbatch.
- **Throughput:** the tokens/s rows are the loop's 20-step runs on one GPU. The baseline was reproduced back to
  back on that GPU.
- **GPU time:** the union of kernel intervals in the profiled step (profiler on step 10 of 10, CUDA graphs off),
  without the copies that only emulate collectives under the fake PG.

**Where the speedup comes from.**
- **Attention glue around the attention kernel:**
  - backward (G-01): 1197 -> 113 us per layer;
  - forward (G-102): 1062 -> 75 us per layer;
  - together about 195 ms per step.
- **FSDP reduce-scatter bucket pack of the sharded expert gradients (G-105):** 384 -> 46 us, 31.7 ms per step. It
  is a pure layout copy at HBM bandwidth.
- **Router, loss, norm-plus-quantize and bucket regions:** the rest (sec. Kernels).

## Before a real multi-GPU run

Read this before using the pack outside the fake-PG setup it was verified in:

- **Other ranks.** Every fake-PG run represents SPMD coordinate 0. By construction the pack is rank-agnostic:
  - matchers accept any size, scalar operand and device index;
  - cache keys drop the device index.

  All 19 kernels also ran in a full step on device index 1 (`LOCAL_RANK=1`, sec. Validation), with bitwise
  identical results. But no other rank's graph was run, and the FSDP bucket kernels (G-105, G-125, G-133)
  encode this layout's bucket structure: 256-way FSDP, 8-way expert sharding.
- **G-133 (all-gather bucket prep).** The reference writes only this rank's slot of a `torch.empty` bucket; the
  all-gather fills the rest. The kernel does the same: it makes one identical allocation and writes nothing
  outside the slot.
  - Under the fake PG the unwritten tail reaches the model. The runs with the pack are bitwise identical because
    the allocator state is identical.
  - With a real PG the all-gather overwrites the tail.
- **G-01 (attention backward glue)** keeps an exchange buffer and a per-device epoch, and its 152 CTAs wait on each
  other with no timeout. Two conditions follow:
  - Two calls in flight at once on two streams can hang. One compute stream with a full-step CUDA graph, as in
    `recipe.py`, keeps them ordered.
  - At most 133 of the 152 SMs may be held by kernels that wait for later work. NCCL's channels use far fewer.
- **152 SMs and this torch build.** G-01, G-15, G-19, G-41 and G-45 reproduce the reduction order of the quack
  RMSNorm that this torch build dispatches `aten._fused_rms_norm` and its backward to on sm_103 with 152 SMs:
  - forward at N=4096: 2 warps per row;
  - backward: 304 blocks, `reduce_rows(8, 256)`;
  - G-01's q/k RMSNorm backward: 2432 blocks, `reduce_rows(4, 256)`.

  The target checks the torch build and compute capability, not the SM count. G-01 raises at import on another
  SM count.
- **Attention backend.** cuDNN's varlen attention backward is nondeterministic even with `debug.deterministic`,
  so `recipe.py` pins FA4. No kernel touches attention, but G-01 and G-102 read the attention inputs and outputs,
  and their specializations include those layouts.
- **CUDA graphs.** `recipe.py` keeps the benchmark recipe's CUDA graphs on. The pass loads every kernel at compile
  time on the rank's device, before capture. G-01 allocates its tables and exchange buffer at import on the
  current device.
- **Inputs outside the training contract** (never produced by the recipe):
  - misaligned or non-contiguous inputs raise `ValueError` in most kernels;
  - out-of-range expert indices are skipped (G-06, G-17) or wrap (G-47) where the reference asserts;
  - G-07 does not assert on invalid labels.

## Target

`manifest.json` `target` is checked exactly when the pass runs:

| Field | Value |
|---|---|
| torch | 2.16.0a0+git3c53c6f (`torch.version.git_version` 3c53c6fe501fd43f2ae33490b0aee276c6e0db6e) |
| CUDA | 13.0 |
| GPU | GB300, compute capability (10, 3), 152 SMs |
| nvidia-cutlass-dsl | 4.8.0 |
| `TORCH_DISABLE_NATIVE_JIT` | unset |

This is a different torch build and CUDA version from the other packs; each pack checks its own target.

**Built and verified with,** not checked by the pass (their op sources are part of the kernels' cache keys):
- torchao 0.19.0.dev20261103+cu130;
- dist_moe;
- flash-attn-4 4.0.0b37;
- torchtitan [agentic_compiler](https://github.com/pytorch/torchtitan/tree/agentic_compiler) `e96fc894a` (the
  loop runs).

**Recipe:** `recipe.py`, torchtitan's `graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu` with:
- one microbatch per step (`num_tokens_per_train_step = -1`; the benchmark recipe accumulates 16);
- FA4 varlen attention;
- `debug.deterministic = True` and seed 42. The specialization keys include the deterministic-algorithms flag;
  without it the pass raises.

**Parallelism:** FSDP 256, EP 32, no PP, 8K-token microbatches, MXFP8 attention projections, MXFP8 Dist-MoE
experts (FSDP-sharded 8 ways), BF16 LM head.

## Kernels

Calls are per training step, i.e. per joint graph; the pass raises `ValueError` if any count differs. Writer and
verifier names refer to `verification/board.md`, and each verdict is in `verification/verdicts/`. Times are the
verifier's measurements: CUDA-graph replays with rotating inputs and live outputs, interleaved with the reference.
Saving = (reference - kernel) x calls. "From" names the 30B pack kernel the writer started from.

| Pattern (region) | What | Calls | Ref us | Kernel us | Saving ms | From | Writer -> verifier |
|---|---|---|---|---|---|---|---|
| bjai (G-01) | attention bwd glue: RoPE bwd q/k, q/k RMSNorm bwd (incl. dw), cat with v grad, MXFP8 row+col quant | 94 | 1196.9 | 112.7 | 101.9 | qnfe | ada -> viggo |
| xfyh (G-102) | attention fwd glue: split of the wqkv output, q/k RMSNorm, RoPE | 94 | 1062.1 | 74.9 | 92.8 | 4dzo | bix -> vika |
| r5yf (G-105) | FSDP reduce-scatter bucket pack of expert grads w13 [4,3072,4096] + w2 [4,4096,1536] | 94 | 384.1 | 46.5 | 31.7 | new | dex -> vera |
| llq3 (G-06) | router + aux-loss backward | 94 | 196.2 | 5.4 | 17.9 | s3of | cyd -> vala |
| 2nu2 (G-19) | add + RMSNorm bwd (N=4096) + add + MXFP8 row+col quant | 94 | 144.4 | 78.4 | 6.2 | zfby | fay -> vesta |
| 5dmq (G-111) | LM-head grad [151936,4096] padded to the 256-shard layout | 8 | 1100.8 | 354.7 | 6.0 | new | dex -> vito |
| bp6b (G-17) | router forward: per-expert token counts | 94 | 53.3 | 2.9 | 4.7 | bp6b | cyd -> vince |
| pqle (G-115) | expert w13 from the all-gather (bf16, unshard permutation) -> `dist_moe.prepare_mxfp8_weight` | 94 | 68.8 | 23.7 | 4.2 | gri3 | eve -> vesna |
| 3h2b (G-125) | `_pre_bucket_reduce_scatter` of the attention, norm and router grads | 94 | 88.1 | 44.4 | 4.1 | new | dex -> vanya |
| 53aq (G-07) | loss backward per chunk [1024,151936] fp32 -> bf16 | 8 | 645.7 | 131.4 | 4.1 | 53aq | dex -> vox |
| 4j6e (G-15) | wqkv dgrad GEMM (call unchanged) + attention_norm RMSNorm bwd + add | 94 | ~270 | ~227 | 3.6-4.0 | tl5t | fay -> vanda |
| bme7 (G-90) | loss forward per chunk: bf16 -> fp32, `log_softmax`, `nll_loss`, div | 8 | 554.6 | 143.7 | 3.3 | bme7 | dex -> vela |
| nh2i (G-91) | router aux-loss forward: row L1 norm, `clamp_min`, div, masked column sum | 94 | 31.6 | 4.4 | 2.6 | bj3b | cyd -> vilma |
| gbfx (G-129) | same as G-115 for w2 | 94 | 39.0 | 13.5 | 2.4 | adoe | eve -> vola |
| 4a7h (G-133) | FSDP all-gather bucket prep (fp32 -> bf16, rank slot) | 94 | 17.0 | 1.6 | 1.5 | new | eve -> vada |
| 5ifh (G-47) | router softmax, top-k score gather, normalization | 94 | 17.8 | 3.7 | 1.3 | 5ifh | cyd -> vlasta |
| xf3v (G-41) | attention_norm RMSNorm + MXFP8 quant | 94 | 42.3 | 28.7 | 1.3 | kbxm | fay -> vin |
| pnge (G-45) | residual add + ffn_norm RMSNorm | 94 | 53.4 | 40.0 | 1.3 | na2k | fay -> vic |
| vf5c (G-28) | wo forward: MXFP8 quant + scale layout, then the unchanged GEMM | 94 | 185.3 | 175.7 | 0.9 | vf5c | eve -> vita |

The verified savings add up to about 292 ms per step; the loop measured 293 ms (tokens/s) and 298.7 ms (GPU time).

**Not in this pack:** G-24 (colwise MXFP8 quantize before the wgrad GEMMs, 188 calls). The 30B loop found no win
there, so no kernel was written.

## How the kernels were verified

Each kernel was written by one agent and checked by a different, freshly started verifier. Only a PASS was
promoted (`verification/promotions.log` gives each file's sha256). No verifier failed a submission in this run.

Where the reference's reduction order changed with the new shapes, each verifier derived the order independently
and showed that the 30B order gives different bytes:
- the quack RMSNorm configuration at N=4096;
- torch's column-sum split at 8192 rows (G-91);
- its nll reduction at 1024 rows (G-90);
- G-01's dw ownership.

Every verifier checked:
- **Contract:**
  - signature and output metadata against `specialization_<l2>.json`;
  - the harness (`verification/harness.py`): seeds 0-7 plus real captured inputs, every output byte equal to
    `reference.py`;
  - inputs unmutated, and repeated calls identical.
- **Memory:** output memory poisoned before each call, so unwritten bytes fail; compute-sanitizer memcheck,
  initcheck, racecheck and synccheck, with planted bugs the tools must flag.
- **CUDA graphs:** capture with neighbouring kernels writing the inputs and reading the outputs, replays with
  inputs changed in place, two or more calls in one graph, and the first call inside capture.
- **Adversarial inputs:** each verifier's own suite with 8-27 deliberately broken copies that must fail.
  Reduction orders were tested with exactly representable cancelling terms placed so that only the reference's
  order gives the reference's result.
- **GEMM-adjacent kernels (G-15, G-28):** the `_scaled_mm_v2` call is identical (op, arguments, operand layout and
  alignment, same cuBLAS kernel) and its output bitwise equal.
- **PDL:** kernels using programmatic dependent launch were run after an early-triggering predecessor that writes
  the inputs, and their SASS was checked for non-coherent loads of data written during the call chain.
- **G-133:** the harness cannot apply, since the reference leaves most of its output unwritten (see above). The
  verifier checked:
  - the slot and the second output byte for byte;
  - an identical allocation;
  - an allocator-baited tail identical to the reference's.

Writer candidates and verifier logs are not published. The loop's audit report and reuse report are in torchtitan
branch `agentic_compiler`, at `torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8/`
(`README.md`, `reuse_report.md`).

Re-gate a kernel from the root of a torchtitan checkout:

```
PACK=<site-packages or clone>/graph_trainer_kernel_packs/qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32_sm103
PYTHONPATH=. python $PACK/verification/harness.py $PACK/kernels/pat_.../specialization_<l2>.json \
    $PACK/kernels/pat_.../<l2>.py --seeds 0 1 2
```

## Validation

20-step runs of this package at torchtitan `de91a3a2a` (`e96fc894a` plus the loop's results directory) on
2026-11-09, one GB300 per run with the fake PG. Both runs used `recipe.py` as shipped (default pass pipeline, CUDA
graphs on) plus TensorBoard, at the same time on two GPUs:

| Run | Kernel calls | Loss / grad_norm / load-balance loss | tokens/s at step 20 |
|---|---|---|---|
| device index 0 (GPU 3) | 1528 to 19 kernels, as in `manifest.json`; 0 lifted calls left on aten | bitwise identical to the recipe without the pack, steps 1, 10 and 20 | 10,624 |
| device index 1 (`LOCAL_RANK=1`, GPU 1) | 1528 to 19 kernels | bitwise identical to the recipe without the pack, steps 1, 10 and 20 | 10,645 |
| the recipe without the pack (the loop's `timed_baseline`, GPU 3, 2026-11-06) | - | reference | 7,698 |

The loop's own run with these kernels from its kernel cache (`timed_final`, GPU 3) gave 10,631 tokens/s. The
values compared are the TensorBoard scalars at full float precision, not the five digits printed to stdout.

## Provenance

The loop's report is in torchtitan [agentic_compiler](https://github.com/pytorch/torchtitan/tree/agentic_compiler)
`de91a3a2a`, under `torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8/`.

## Files

| Path | Contents |
|---|---|
| `__init__.py` | registers the pass |
| `recipe.py` | the verified recipe |
| `manifest.json` | target, and per kernel its sha256 and calls per joint graph (generated) |
| `regions/` | the 19 regions' topology matchers, independent of shapes and rank; `PLAN` is the lift order (generated) |
| `kernels/` | `pat_<l1>/reference.py`, `specialization_<l2>.json`, `<l2>.py` (generated, byte-identical to the verified files) |
| `verification/` | the swarm's task board, the correctness/timing harness, promotions and the 19 verdicts |

Generated with the following command. `<loop>` is the torchtitan report directory above, and `timed_final.log` is
the log of the loop's final 20-step run. G-90's docstring was edited afterwards to drop a local path.

```
python tools/make_pack.py --regions <loop>/fusion_regions --kernel-cache <loop>/kernel_cache \
    --log timed_final.log --out graph_trainer_kernel_packs/qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32_sm103 \
    --plan lift_fusion_region_g01,lift_fusion_region_g102,lift_fusion_region_g07,lift_fusion_region_g06,lift_fusion_region_g17,lift_fusion_region_g19,lift_fusion_region_g15,lift_fusion_region_g115,lift_fusion_region_g129,lift_fusion_region_g41,lift_fusion_region_g45,lift_fusion_region_g28,lift_fusion_region_g47,lift_fusion_region_g90,lift_fusion_region_g91,lift_fusion_region_g105,lift_fusion_region_g111,lift_fusion_region_g125,lift_fusion_region_g133
```
