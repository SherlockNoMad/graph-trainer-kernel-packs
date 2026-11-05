# DeepSeek-V3 671B DistMoE MXFP8, FSDP 256 x EP 64, GB300 (sm_103)

20 CuTe DSL kernels for 13 fusion regions of one training step, each bitwise identical to the aten ops it
replaces. Pass name: `dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103`.

| | Baseline | With this pack |
|---|---|---|
| GPU time per step, excluding fake-PG collectives | 583.9 ms | 503.4 ms (-80.5 ms, -13.8%) |
| GPU time per step, total | 758.8 ms | 678.4 ms |
| loss, steps 1 / 10 | 12.281303405761719 / 12.479232788085938 | identical |
| grad_norm, steps 1 / 10 | 63.61985778808594 / 4.849545478820801 | identical |
| memory at step 10 | 139.45 GiB | 139.37 GiB |

Measured on one GB300 (GPU 0 for both runs) with the fake process group, step 10 of 10, profiler on, CUDA
graphs off. Fake-PG collectives are copies that only emulate communication, so the first row is the
meaningful one. Memory varies by about 0.6 GiB between identical runs.

## Target

`manifest.json` `target` is checked exactly when the pass runs:

| Field | Value |
|---|---|
| torch | 2.16.0a0+git3c53c6f (`torch.version.git_version` 3c53c6fe501fd43f2ae33490b0aee276c6e0db6e) |
| CUDA | 13.0 |
| GPU | GB300, compute capability (10, 3) |
| nvidia-cutlass-dsl | 4.8.0 |
| `TORCH_DISABLE_NATIVE_JIT` | unset |

Built and verified with, not checked by the pass (their op sources are part of the kernels' cache keys):
torchao 0.19.0.dev20261103+cu130, [dist_moe](https://github.com/meta-pytorch/dist_moe) 285763b, torchtitan
[agentic_compiler](https://github.com/pytorch/torchtitan/tree/agentic_compiler) 1730fb2b8.

The recipe is `recipe.py`: torchtitan's `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8` on FSDP 256 x EP 64,
no PP, one microbatch per step, `debug.deterministic = True`, `scratch_capacity_factor = 64` (needed under
the fake PG). Only this recipe under the fake PG on one GPU has been run; a real 256-GPU job runs the same
graph, but is untested.

## Kernels

Reference and kernel times are the verifier's amortized measurements (several calls per CUDA graph,
interleaved with the reference). Calls are per training step, i.e. per joint graph; the pass raises
`ValueError` if any count differs. Writer and verifier names refer to `verification/board.md`.

| Pattern (region) | What | Calls | Ref us | Kernel us | Saving ms/step | Writer -> verifier |
|---|---|---|---|---|---|---|
| gjn6 (G-001) | router forward: group-limited top-8, aux loss | 58 | 244.9 | 6.22 | 13.8 | cyd -> vic |
| erxi (G-006) | router backward | 58 | 201.1 | 5.84 | 11.3 | ada -> vera |
| h4xd (G-020) | router recompute | 58 | 86.7 | 6.23 | 4.7 | bix -> vik |
| k6pn (G-035) | fused_mla_kv_backward + mxfp8 row/col quant | 61 | 193 | 101 | 5.6 | dex -> vox |
| z47c (G-047 v0) | rmsnorm + 2x rowwise quant | 61 | 52.2 | 18.2 | 2.1 | eve -> val |
| use7 (G-047 v1) | rmsnorm -> y, rstd + 2x quant | 61 | 54.1 | 26.4 | 1.7 | eve -> val |
| dre3 (G-082 v2) | add + rmsnorm + quant | 61 | 65.1 | 40.8 | 1.5 | eve -> val |
| c2qd (G-082 v0) | add + rmsnorm + quant | 58 | 65.2 | 40.0 | 1.5 | eve -> val |
| 5yin (G-082 v1) | add + rmsnorm + quant | 3 | 65.0 | 32.7 | 0.1 | eve -> val |
| a6rv (G-049 v1) | rmsnorm bwd + add + row/col quant | 61 | 92.7 | 56.6 | 2.2 | fay -> vin |
| 3ybb (G-049 v0) | same, other output order | 60 | 92.9 | 56.6 | 2.2 | fay -> vin |
| 5nj2 (G-007) | GEMM (unchanged) + 8-row-block permutation replacing a slow cat | 58 | 232.8 | 76.8 | 9.0 | gil -> vlad |
| ponk (G-003) | GEMM (unchanged) + FSDP reduce-scatter bucket pack | 58 | 494.4 | 396.7 | 5.7 | gil -> vera2 |
| czaf (G-028) | routed-experts reduce-scatter bucket pack | 58 | 173.6 | 102.5 | 4.1 | gil -> vex |
| lfu5 a2jgmo (G-015) | fused quantize + scale layout, then unchanged GEMM | 61 | 30.4 | 26.0 | 0.3 | eve -> vola |
| lfu5 adm7po (G-015) | same | 61 | 44.8 | 38.5 | 0.4 | eve -> vola |
| lfu5 autudf (G-015) | same | 61 | 187.2 | 168.1 | 1.2 | eve -> vola |
| bqhy a42i3e (G-015) | same, other output order | 58 | 89.7 | 81.7 | 0.5 | eve -> vola |
| x6t6 (G-021) | `dist_moe.prepare_mxfp8_weight`, w13 [4,4096,7168] | 116 | 102.0 | 51.4 | 5.9 | hana -> vita |
| 5uzo (G-061) | `dist_moe.prepare_mxfp8_weight`, w2 [4,7168,2048] | 116 | 45.1 | 28.3 | 1.9 | hana -> vita |

The verified savings add up to about 76 ms per step; the rest of the measured 80.5 ms is run-to-run variation.

Two rare G-015 specializations have no kernel (bqhy afd5ld, 3 calls; lfu5 aglgyz, 1 call): their lifted calls
run the original aten ops. Regions G-031 and G-073 of the loop's last plan are left out, since no kernel was
promoted for them: torchao's colwise quantize is already near bandwidth, and the FSDP all-gather bucket's
output has bytes no op writes, so a bytewise check is ill-defined.

**Numerics dependency.** The norm kernels (z47c, use7, dre3, c2qd, 5yin, a6rv, 3ybb) reproduce the reduction
order of the quack RMSNorm that this torch build vendors and dispatches `aten._fused_rms_norm` (and its
backward) to. Setting `TORCH_DISABLE_NATIVE_JIT` routes the reference elsewhere, so the target requires it
unset.

## How the kernels were verified

Each kernel was written by one agent and checked by a different, freshly started verifier; only a PASS was
promoted (`verification/board.md` records every handoff, verdict and promotion). Every verifier checked:

- **Contract:** signature and output metadata against `specialization_<l2>.json`; the harness
  (`verification/harness.py`): seeds 0-7 plus real captured inputs, every output byte equal to
  `reference.py`; inputs unmutated; repeated calls identical.
- **Memory:** output memory poisoned before each call, so unwritten bytes fail; compute-sanitizer memcheck,
  initcheck, racecheck and synccheck.
- **CUDA graphs:** capture with neighbouring kernels writing the inputs and reading the outputs, replays
  with inputs changed in place, two calls in one graph.
- **Adversarial inputs:** each verifier's own suite (special values, ties, order-sensitive cancellation),
  with deliberately broken copies that must fail.
- **GEMM-adjacent kernels:** the `_scaled_mm_v2` call is identical (op, arguments, operand layout, cuBLAS
  kernel) and its output bitwise equal.
- **PDL:** kernels using programmatic dependent launch were run after an early-triggering small-grid
  predecessor (this caught one real ordering bug, fixed before promotion).

Then the whole step: loss and grad_norm of a 10-step run with the kernels are bitwise equal to the baseline
at both logged steps (1 and 10). Writer candidates and verifier logs (about 1 GB) are not published; the
loop's audit report is in torchtitan branch `agentic_compiler`,
`torchtitan/experiments/graph_trainer/agentic_compiler/dsv3_671b_distmoe_mxfp8/README.md`.

## Validation

Two 10-step runs with this package, its pass as the only extra pass, on 2026-11-05:

| Run | Kernel calls | Loss / grad_norm vs baseline | GPU time, excl. fake-PG collectives |
|---|---|---|---|
| measurement setup above (fusion_finder pass pipeline, profiler, CUDA graphs off) | 1247 to 20 kernels, as in `manifest.json` | bitwise identical, steps 1 and 10 | 503.4 ms (baseline 583.9 ms) |
| `recipe.py` as shipped (default pass pipeline, CUDA graphs on) plus TensorBoard | 1247 to 20 kernels | bitwise identical, steps 1 and 10 | not profiled |

Both runs also matched the MoE load-balance loss bitwise. The loss and grad_norm values compared are the
TensorBoard scalars at full float precision, not the five digits printed to stdout.

## Files

| Path | Contents |
|---|---|
| `__init__.py` | registers the pass |
| `recipe.py` | the verified recipe |
| `manifest.json` | target, and per kernel its sha256 and calls per step (generated) |
| `regions/` | the 13 regions' topology matchers; `PLAN` is the lift order (generated) |
| `kernels/` | `pat_<l1>/reference.py`, `specialization_<l2>.json`, `<l2>.py` (generated, byte-identical to the verified files) |
| `verification/` | the swarm's task board and the correctness/timing harness |

Generated with:

```
python tools/make_pack.py --regions <loop>/fusion_regions --kernel-cache <loop>/kernel_cache \
    --log <loop>/iter9.log --out graph_trainer_kernel_packs/dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103 \
    --plan lift_fusion_region_g006,lift_fusion_region_g020,lift_fusion_region_g001,lift_fusion_region_g047,lift_fusion_region_g082,lift_fusion_region_g049,lift_fusion_region_g035,lift_fusion_region_g007,lift_fusion_region_g003,lift_fusion_region_g015,lift_fusion_region_g028,lift_fusion_region_g021,lift_fusion_region_g061
```
