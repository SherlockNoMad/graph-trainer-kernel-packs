# dsv3_debug_fsdp8_sm103

DeepSeek-V3 debug model (torchtitan's GraphTrainer debug recipe, FSDP 8), one microbatch per step, on
GB300 (sm_103). 13 CuTe DSL kernels replace the 12 fusion regions fusion_finder lifts from the training
step (65 calls per step), bitwise identical to the ops they replace.

The first pack in the fusion_finder format (`fusion_finder_pack.py`): torchtitan's fusion_finder skill
found and lifted the regions and wrote each specialization as a kernel problem; the autokernel skill
wrote each kernel, and a separate, newly started agent verified it before it was promoted.

## Results

GPU time per step (profiled step 10, CUDA graphs off so kernels can be attributed to graph nodes;
one GB300, fake process group):

| | Lifted, no kernels | With the pack | Change |
| --- | ---: | ---: | ---: |
| GPU busy | 44.35 ms | 29.89 ms | -32.6% |
| glue ops | 18.29 ms | 3.93 ms | -78.5% |
| the 12 lifted regions | 16.29 ms | 1.93 ms | 8.4x |

The regions' HBM roofline is 1.8 ms. Tokens per second with the recipe as it runs (CUDA graphs on),
median of steps 5-20: 257,533 -> 316,902 (+23%). The debug model is small: most of its step is
attention and launch latency.

Per kernel, time per step in training (`node_table.py` on both runs):

| Problem (region) | What it is | Calls | Lifted | Kernel | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| r000 | MLA RoPE recompute, conj, RMS norm | 6 | 3.100 ms | 0.382 ms | 8.1x |
| r001 | attention RoPE forward + RMS norm | 6 | 1.879 ms | 0.208 ms | 9.0x |
| r003 | RoPE backward | 6 | 1.693 ms | 0.189 ms | 9.0x |
| r002 | router gate backward | 5 | 1.592 ms | 0.121 ms | 13.1x |
| r004 | RoPE backward with head sum | 6 | 1.609 ms | 0.259 ms | 6.2x |
| r007 | attention k layout | 6 | 1.193 ms | 0.158 ms | 7.6x |
| r005 | router top-k, aux loss, token sort | 5 | 1.043 ms | 0.050 ms | 20.8x |
| r006 | router aux-loss backward | 4 | 1.113 ms | 0.188 ms | 5.9x |
| r009 | routed-expert combine, residual, RMS norm | 5 | 0.877 ms | 0.084 ms | 10.5x |
| r012 | SwiGLU backward (routed experts) | 5 | 0.758 ms | 0.095 ms | 8.0x |
| r010 | ffn_norm backward | 5 | 0.720 ms | 0.109 ms | 6.6x |
| r011 (`arvsvny2`) | SwiGLU backward (shared experts) | 5 | 0.506 ms | 0.064 ms | 7.9x |
| r011 (`akm7fmaf`) | SwiGLU backward (dense FFN) | 1 | 0.205 ms | 0.027 ms | 7.6x |

## Numerics

Loss and grad_norm of the recipe with and without the pack are bitwise identical over 20 steps
(seed 42, deterministic, `scripts/loss_compare.py --assert-equal`). Every kernel passes
`kernels/check_kernel.py <problem dir> <problem dir>/kernel.py`, its bitwise gate on the input cases
saved from the training step.

Some kernels depend on facts about their index inputs that hold in this model but are not in the
contract: r002 needs its scatter index to be a permutation (the token dispatcher's sort), r009 and r010
need exactly 3 routed rows per token (top-k 3), and r000 and r001 need RoPE positions in [0, 2048).

## Use

Requirements:
- torchtitan main with fusion_finder's kernel problems (pytorch/torchtitan#6171) and the follow-up that
  adds `kernel_problems.register_problem_kernels`; neither is in main yet. The kernels are stored under
  the problems' `pattern_<l1>/<l2>` keys, so a change to the keys needs a regenerated pack.
- The exact environment in `manifest.json` `target`.
- `recipe.py`, or a recipe that produces the same graph.

From the root of the torchtitan checkout:

```
PYTHONPATH=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts:. \
NGPU=8 LOCAL_RANK=0 python -m torchtitan.train \
    --module graph_trainer_kernel_packs.dsv3_debug_fsdp8_sm103.recipe \
    --config dsv3_debug_fsdp8 --comm-backend fake
```

## Layout

```
manifest.json            target environment; per kernel (problem id): region, sha256, calls per step
regions.json             fusion_finder's regions: the patterns the lift passes match, in lift order
kernels/
  check_kernel.py        the bitwise gate
  pattern_<l1>/<l2>/     one kernel problem: problem.py (reference), spec.json (contract, cases,
                         calls), inputs.pt (saved integer/bool/complex inputs), kernel.py
```
