# Kernel problems

The kernel problems that torchtitan's fusion_finder skill wrote for four GraphTrainer training steps on
GB300 (sm_103). A problem is one specialization of a fusion region lifted from the step's joint graph: the
region's aten code as `reference`, on inputs of exact shapes, strides and dtypes, with input cases saved from
the training step. A kernel solves it if it matches `reference` bit for bit on every case.

Each set has the layout of a fusion_finder pack (`graph_trainer_kernel_packs/fusion_finder_pack.py`) without
the `kernel.py` files, so a solved set becomes a pack (see "Solving a set").

| Set | Model and layout | Problems (regions) | Calls per step | Measured | HBM roofline | Solved |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `dsv3_debug_fsdp8_sm103` | DeepSeek-V3 debug model; FSDP 8 | 13 (12) | 65 | 15.4 ms | 1.7 ms | 13, pack [`dsv3_debug_fsdp8_sm103`](../graph_trainer_kernel_packs/dsv3_debug_fsdp8_sm103/README.md) |
| `qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103` | Qwen3 30B-A3B, DistMoE, MXFP8; FSDP 8 x EP 8, 32K tokens | 19 (18) | 736 | 343.3 ms | 55.7 ms | 17, pack [`qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103_fusion_finder`](../graph_trainer_kernel_packs/qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103_fusion_finder/README.md) |
| `dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103` | DeepSeek-V3 671B, DistMoE, MXFP8; FSDP 256 x EP 64, no PP, 4K tokens | 50 (30) | 2,354 | 163.2 ms | 59.0 ms | - |
| `gpt_oss_120b_mxfp8_fsdp64_ep16_sm103` | GPT-OSS 120B, MXFP8 attention + lm_head; FSDP 64 x EP 16, 16K tokens | 22 (21) | 664 | 1,599.9 ms | 86.3 ms | - |

All four are one microbatch per step with `debug.deterministic = True`, run on one GPU with the fake process
group (rank 0 of the layout). Measured is the GPU time of the problems' calls in the profiled step of
fusion_finder's capture (CUDA graphs off); HBM roofline is the time to move the bytes they read, and the bytes
they write for use outside, at the measured HBM bandwidth. In each set one problem has neither (`null` in
`manifest.json`): none of its calls could be matched to a timed instance of the capture.

## Layout

```
<set>/
  manifest.json            target environment; per problem (id), by gap: region, calls per step,
                           measured_us and ideal_us per step
  regions.json             fusion_finder's regions of the capture (ids, ops, modules, reference code)
  kernels/
    check_kernel.py        the bitwise gate, shared by all problems of the set
    pattern_<l1>/<l2>/     one problem: problem.py (reference and the KernelBench interface),
                           spec.json (contract, input cases, the calls it comes from, the GPU kernels
                           the reference ran), inputs.pt (the saved integer/bool/complex inputs)
```

`<l1>` keys a region's structure (problems of one region at other sizes share it), `<l2>` the exact code and
inputs. The keys are the ones the packs look kernels up by: a `kernel.py` solving a problem here is the
kernel at the same key in a pack.

## Checking a kernel

Write `kernel_function`, with the signature of `reference`, in `kernel.py` in the problem's directory, then
from the root of a torchtitan checkout (the references call torchtitan, torchao and `dist_moe` ops):

```
PYTHONPATH=. python <set>/kernels/check_kernel.py <set>/kernels/pattern_<l1>/<l2> \
    <set>/kernels/pattern_<l1>/<l2>/kernel.py
```

It runs every case with floating-point inputs from two seeds (the second puts exact zeros of either sign in
half of them) and requires the outputs and in-place writes to match `reference` bit for bit. The references
reproduce the numerics of the build in `manifest.json` `target`; run in that environment.

## Solving a set

torchtitan's autokernel skill takes `<set>/kernels` as its problem set and promotes each verified kernel to
`kernel.py` in its problem. `tools/make_fusion_finder_pack.py --capture <set> --problems <set>/kernels --out
graph_trainer_kernel_packs/<pack>` then builds a pack of the solved problems; the pack's `__init__.py`,
`recipe.py` and `README.md` are written by hand.
