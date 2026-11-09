# GPT-OSS 120B MXFP8, FSDP 64 x EP 16, GB300 (sm_103)

16 CuTe DSL kernels for 14 fusion regions of one microbatch's joint graph, each bitwise identical to the aten
ops it replaces. Pass name: `gpt_oss_120b_mxfp8_fsdp64_ep16_sm103`.

| | Deterministic baseline | With this pack |
|---|---|---|
| tokens/s per GPU, step 10 of the benchmark recipe (this package, same session) | 7,474 and 7,801 | 9,422 and 9,514 (1.24x) |
| GPU time per microbatch, excluding fake-PG collectives (profiled 1-microbatch runs) | 1992.3 ms | 1687.4 and 1670.9 ms (1.19x) |
| loss, steps 1 / 10 | 12.747259140014648 / 7.925948143005371 | identical |
| grad_norm, steps 1 / 10 | 13.711431503295898 / 12.714406967163086 | identical |
| memory at step 10 | 202.62 GiB | 201.84 GiB |

**How these were measured.**
- **Hardware:** one GB300 with the fake process group, CUDA graphs off (the recipe's all-to-all dispatcher
  host-syncs).
- **tokens/s:** this package's `recipe.py`, alternating on one GPU with the deterministic recipe (see
  Validation).
- **Non-deterministic recipe:** the loop's runs measured about 15,000 tokens/s for the recipe without
  deterministic algorithms, so with this pack the deterministic recipe is still about 0.6x of it.
- **GPU time:** profiled 1-microbatch runs of the loop's `lifted` config with this pack's 14 regions, on one
  GPU in the same session.
- **Fake-PG collectives:** copies that only emulate communication (about 35 ms per microbatch), excluded from
  GPU time.

**Where the speedup comes from.** SwiGLU-with-clamp forward and backward (G-017, G-007) save 135.5 ms per
microbatch, the deterministic `index_put` regions of MoE dispatch and combine (G-011, G-023, G-025, G-036)
105.7 ms, RoPE (G-019, G-020) 71.1 ms, and six smaller regions 49.6 ms (verified per-kernel savings, see
Kernels).

## Before a real multi-GPU run

Read this before using the pack outside the fake-PG setup it was verified in:

- **Cooperative kernels.** Four kernels (G-011, G-023, G-025, G-036) launch whole-GPU cooperative grids
  with a persistent grid barrier, one CTA per SM.
  - The launch waits until all SMs are free. With a single stream (fake PG) that is harmless.
  - With real NCCL kernels overlapping, they serialize against communication.
  - They can deadlock if a kernel holding SMs on another stream waits on work queued after them. The
    verifiers reproduced both effects with side-stream kernels.
  - Untested on a real multi-GPU job.
- **Other ranks.** Every fake-PG run represents SPMD coordinate 0. By construction the pack is rank-agnostic:
  - matchers accept any size, scalar operand and device index;
  - cache keys drop the device index;
  - the references contain only model constants as literals (hidden size 2880, 8 local experts, 128 experts,
    top-4, 64 heads), no rank-dependent offsets.

  But no other rank's graph was run.
- **G-028** reproduces aten's row and column reduction orders for this GPU and torch build, with no runtime
  check. `PYTORCH_SUM_INNER_TREE=1` changes the reference's row sum and breaks parity; it is unset in the
  target.
- **CUDA graphs.** Several kernels synchronize the device at import or on their first call: G-031, G-039,
  G-041 and G-048 at import, G-023 on the first call. They cannot make a first call inside CUDA-graph
  capture. The recipe has CUDA graphs off.

## Target

`manifest.json` `target` is checked exactly when the pass runs:

| Field | Value |
|---|---|
| torch | 2.16.0a0+git3c53c6f (`torch.version.git_version` 3c53c6fe501fd43f2ae33490b0aee276c6e0db6e) |
| CUDA | 13.0 |
| GPU | GB300, compute capability (10, 3) |
| nvidia-cutlass-dsl | 4.8.0 |
| `TORCH_DISABLE_NATIVE_JIT` | unset |

**Built and verified with:** torchtitan [agentic_compiler](https://github.com/pytorch/torchtitan/tree/agentic_compiler)
`4f955c1c8`; validated again at `bd4300eb9` after G-005 and G-014 were removed. This version is not checked
by the pass. The pass needs its `substitute_kernels` to key inputs with symbolic sizes by description:
GPT-OSS's routed-row count is a new unbacked symbol in every layer.

**Recipe:** `recipe.py`, torchtitan's `graph_trainer_gpt_oss_120b_mxfp8_64gpu` with `debug.deterministic = True`
and seed 42.
- **Parallelism and batch:** FSDP 64, EP 16, no PP; GBS 1280, MBS 4, sequence length 4096 (5 microbatches of
  16,384 tokens per step).
- **Precision:** MXFP8 attention projections and lm_head; BF16 experts.
- **Graph:** the gradient-accumulation graph builder handles the microbatches. The pass runs once on the
  microbatch's joint graph.
- **Determinism is required:** the kernels reproduce deterministic-mode numerics, and the specialization keys
  include the deterministic-algorithms flag. Without it the pass raises.

## Kernels

Calls are per joint graph; the pass raises `ValueError` if any count differs. Writer and verifier names refer to
`verification/board.md`. Times are the verifier's interleaved A/B measurements on real inputs, per call:
- **Eager:** includes host overhead, which is what training sees without CUDA graphs.
- **Graph:** one call per CUDA-graph replay.
- **Saving:** (reference graph - kernel graph) x calls.

| Pattern (region) | What | Calls | Ref us eager / graph | Kernel us eager / graph | Saving ms | Writer -> verifier |
|---|---|---|---|---|---|---|
| yjxz (G-007) | SwiGLU-with-clamp backward | 36 | 3371 / 3301 | 552 / 534 | 99.6 | sol -> vera |
| jvof (G-011) | MoE combine backward: gather + mul + row sum + `index_put` | 36 | 1787 / 1702 | 142.9 / 145.2 | 56.0 | mox, pip -> joy |
| w7az (G-017) | SwiGLU-with-clamp forward | 36 | 1454 / 1381 | 415 / 384 | 35.9 | sol -> vito |
| ojkx (G-019 q) | RoPE backward, q | 36 | 993 / 940 | 57.9 / 46.3 | 32.2 | sol -> una |
| mv2l atcl (G-020 q) | RoPE forward, q | 36 | 954 / 915 | 62.5 / 48.4 | 31.2 | sol -> rho |
| 5r57 (G-023) | `new_zeros` + `index_put` + add | 36 | 843 / 718 | 95.6 / 97.0 | 22.4 | pip -> bea |
| jn7t (G-025) | `new_zeros` + `index_put` | 36 | 758 / 686 | 136 / 133 | 19.9 | pip -> cal |
| cuks (G-028) | attention-sink backward | 36 | 481 / 438 | 91.5 / 71.7 | 13.2 | sol -> ana |
| jofj (G-031) | w13 expert-bias expand | 36 | 780 / 526 | 216.6 / 213.9 | 11.2 | rex -> gus |
| 53aq (G-037) | loss backward (`log_softmax` + `nll`) | 8 | 1630 / 1586 | 371 / 346 | 9.9 | sol -> eli |
| 2uey (G-036) | `new_empty` + `index_put_` | 36 | 481 / 343 | 134.5 / 136.5 | 7.4 | pip -> dov |
| s7of (G-039) | w2 expert-bias expand | 36 | 522 / 300 | 111.7 / 123.4 | 6.4 | rex -> lux |
| 4bre (G-041) | router backward | 36 | 576 / 145 | 17 / 11 | 4.8 | rex -> noa |
| f6c7 (G-048) | routed-experts `mul.Scalar` + `mul` | 36 | 294 / 271 | 161 / 158 | 4.1 | rex -> kit |
| mloe (G-019 k) | RoPE backward, k | 36 | 260 / 118 | 25.7 / 10.1 | 3.9 | sol -> zed |
| mv2l ajqa (G-020 k) | RoPE forward, k | 36 | 176 / 116 | 24.8 / 9.8 | 3.8 | sol -> kai |

The verified savings add up to 361.9 ms per microbatch; the profiled runs measured 313.1 ms (1992.3 -> 1679.2
ms, the mean of two runs).

**Not in this pack:**
- **G-005 and G-014**, removed on 2026-11-09: the expert-bias gradients' deterministic `index_add` (G-005,
  two kernels) and the MoE combine forward's `deterministic_scatter_add` (G-014). Their kernels passed the
  same verification, but their saving (1075 ms per microbatch verified, 1139 ms measured) mostly reflects how
  slow torch's deterministic sort-then-ordered-accumulate path is for these two ops, not what fusion buys, so
  they are left out. They remain in the earlier version of this pack (commit `073bba4`).
- **G-046** (aux-loss forward): verified, but its region absorbed an all-reduce bucket-pack `cat` that exists
  only after FSDP bucketing. The gradient-accumulation builder buckets after the extra passes, so the region
  never matches the benchmark graph.
- **Other remaining time:** of the 1679 ms, about 1140 ms in G-005's and G-014's deterministic ops and about
  333 ms in GEMMs and attention.

## How the kernels were verified

Each kernel was written by one agent and checked by a different, freshly started verifier. Only a PASS was
promoted; `verification/board.md` records every handoff, verdict and promotion. G-011 took three candidates:
- **f1 FAIL:** a grid-barrier word stayed un-zeroed after a first call under capture.
- **f2 FAIL:** a two-thread race.
- **f3 PASS.**

Every verifier checked:

- **Contract:**
  - signature and output metadata against `specialization_<l2>.json`;
  - the harness (`verification/harness.py`): random seeds plus real captured inputs, every output byte equal to
    `reference.py`;
  - inputs unmutated, and repeated calls identical.
- **Memory:** output memory poisoned before each call, so unwritten bytes fail; compute-sanitizer memcheck,
  initcheck, racecheck and synccheck.
- **CUDA graphs:** replays with inputs changed in place.
- **Adversarial inputs:** each verifier's own suite, with deliberately broken copies that must fail.
  - The pointwise kernels (G-007, G-017, G-019, G-020, G-048) were swept over all 2^32 input bit-pattern pairs.
  - Reductions were tested with order-sensitive inputs.
- **Index semantics:** the `index_put` kernels reproduce torch's deterministic path: negative indices
  wrapped, a stable radix sort, and duplicates added in source order, each add rounded to bf16 from +0.
  - The generators `verification/gen_*.py` produce valid index inputs for the harness: routing-like, heavy
    duplicates, empty destinations, negative indices.
- **PDL:** kernels using programmatic dependent launch were run after an early-triggering, small-grid
  predecessor.

**Whole-step check:** loss, grad_norm and load-balance loss of the 10-step runs with the kernels are bitwise
equal to the deterministic baseline at both logged steps (1 and 10).

Writer candidates and verifier logs are not published. The loop's audit report is in torchtitan branch
`agentic_compiler`, at `torchtitan/experiments/graph_trainer/agentic_compiler/gpt_oss_120b_mxfp8_64gpu/README.md`.

Re-gate a kernel from the root of a torchtitan checkout:

```
PACK=<site-packages or clone>/graph_trainer_kernel_packs/gpt_oss_120b_mxfp8_fsdp64_ep16_sm103
PYTHONPATH=. python $PACK/verification/harness.py $PACK/kernels/pat_.../specialization_<l2>.json \
    $PACK/kernels/pat_.../<l2>.py --seeds 0 1 2 --gen $PACK/verification/gen_rex.py
```

Index-valued problems need `--gen`:
- G-011, G-023, G-025 and G-036: `gen_mox.py`;
- G-031, G-039 and G-041: `gen_rex.py`.

## Validation

10-step runs of `recipe.py` with this package at torchtitan `bd4300eb9`, with TensorBoard added, on 2026-11-09.
All ran on GPU 0 of one GB300 with the fake PG, in two rounds of three alternating runs:

| Run | Kernel calls | Loss / grad_norm vs the deterministic recipe | tokens/s, step 10 |
|---|---|---|---|
| `recipe.py`, round 1 | 548 to 16 kernels, as in `manifest.json`; 0 lifted calls left on aten | bitwise identical, steps 1 and 10 | 9,422 |
| `recipe.py`, round 2 | 548 to 16 kernels | bitwise identical, steps 1 and 10 | 9,514 |
| the deterministic recipe without the pack (the reference), rounds 1 / 2 | - | bitwise identical to each other and to the 2026-11-06 reference | 7,474 / 7,801 |
| the earlier version with G-005 and G-014 (`073bba4`), rounds 1 / 2 | 656 to 19 kernels | bitwise identical, steps 1 and 10 | 24,733 / 24,678 |

Profiled 1-microbatch runs on GPU 3 in the same session (GPU time excluding fake-PG collectives): no regions
1992.3 ms; this pack's 14 regions 1687.4 and 1670.9 ms; the earlier version's 16 regions 540.2 and 539.6 ms.
The 2026-11-06 validation of the earlier version also ran on device index 1 (`LOCAL_RANK=1`), bitwise; that
was not repeated for this version.

**Notes on these runs:**
- **Throughput:** other users' processes ran on two of the node's other GPUs during these runs; the two
  reference runs differ by 4%.
- **Numerics:** all runs also matched the MoE load-balance loss bitwise. The values compared are the
  TensorBoard scalars at full float precision, not the five digits printed to stdout.
- **Graph-site counts:** the pass runs once on the microbatch's joint graph. Each of the 5 microbatches per
  step runs the compiled graph, so a step makes 5x the calls listed.


## Files

| Path | Contents |
|---|---|
| `__init__.py` | registers the pass |
| `recipe.py` | the verified recipe |
| `manifest.json` | target, and per kernel its sha256 and calls per joint graph (generated) |
| `regions/` | the 14 regions' topology matchers, independent of shapes and rank; `PLAN` is the lift order (generated) |
| `kernels/` | `pat_<l1>/reference.py`, `specialization_<l2>.json`, `<l2>.py` (generated, byte-identical to the verified files) |
| `verification/` | the swarm's task board (it also records the removed G-005 and G-014), the correctness/timing harness, and the input generators |

Generated with (`<loop>` = the torchtitan report directory above; `full_final.log` is the loop's final
benchmark run):

```
python tools/make_pack.py --regions <loop>/fusion_regions --kernel-cache <loop>/kernel_cache \
    --log full_final.log --out graph_trainer_kernel_packs/gpt_oss_120b_mxfp8_fsdp64_ep16_sm103 \
    --plan lift_fusion_region_g005,lift_fusion_region_g007,lift_fusion_region_g011,lift_fusion_region_g014,lift_fusion_region_g019,lift_fusion_region_g020,lift_fusion_region_g017,lift_fusion_region_g023,lift_fusion_region_g025,lift_fusion_region_g028,lift_fusion_region_g036,lift_fusion_region_g037,lift_fusion_region_g031,lift_fusion_region_g039,lift_fusion_region_g048,lift_fusion_region_g041
```

G-005 and G-014 were then removed by hand: their `kernels/` directories, `regions/` files, `PLAN` entries and
manifest entries, and `verification/gen_index_add.py`.
