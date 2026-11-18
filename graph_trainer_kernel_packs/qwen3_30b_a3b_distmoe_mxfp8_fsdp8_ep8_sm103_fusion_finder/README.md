# qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103_fusion_finder

Qwen3 30B-A3B with MXFP8 Dist-MoE experts and MXFP8 attention projections, FSDP 8 x EP 8, one 32K-token
microbatch per step, on GB300 (sm_103). 17 CuTe DSL kernels replace 16 fusion regions that fusion_finder
lifts from the training step (687 calls per step), bitwise identical to the ops they replace.

The model and recipe of `qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103`, made in the fusion_finder format
(`fusion_finder_pack.py`): torchtitan's fusion_finder skill found and lifted the regions and wrote each
specialization as a kernel problem; the autokernel skill wrote each kernel, and a separate, newly started
agent verified it before it was promoted.

## Results

GPU time per step (profiled step 10, CUDA graphs off so kernels can be attributed to graph nodes; one GB300,
fake process group):

| | Lifted, no kernels | With the pack | Change |
| --- | ---: | ---: | ---: |
| GPU busy | 974.2 ms | 693.4 ms | -28.8% |
| glue ops | 355.1 ms | 74.4 ms | -79.0% |
| the 16 regions with kernels | 347.5 ms | 66.8 ms | 5.2x |

The HBM roofline of 16 of the regions (r015's has no byte count) is 55.4 ms. With the recipe as it runs (CUDA
graphs on), median of steps 5-20, against the recipe without the pack:

| | Without the pack | With the pack | Change |
| --- | ---: | ---: | ---: |
| tokens per second | 33,165 | 45,375 | +36.8% |
| peak memory (step 20) | 236.0 GiB | 231.2 GiB | -4.8 GiB |

Per kernel, time per step in training (`node_table.py` on both profiled runs):

| Problem (region) | What it is | Calls | Lifted | Kernel | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| r001 | attention forward: qkv split, q/k RMSNorm, RoPE | 48 | 111.551 ms | 7.174 ms | 15.5x |
| r000 | attention backward: RoPE backward, q/k RMSNorm backward, qkv-gradient cat | 48 | 122.591 ms | 17.454 ms | 7.0x |
| r002 | MoE router gate backward | 48 | 18.846 ms | 0.707 ms | 26.7x |
| r003 | cross-entropy loss backward, one 4096-token chunk | 8 | 20.766 ms | 4.445 ms | 4.7x |
| r004 | MoE router forward: scores, top-8, token sort | 48 | 11.498 ms | 0.843 ms | 13.6x |
| r006 | cross-entropy loss forward, one 4096-token chunk | 7 | 14.292 ms | 4.969 ms | 2.9x |
| r007 | ffn_norm backward + residual add (the MXFP8 quantization for wo stays torch ops) | 48 | 12.265 ms | 8.526 ms | 1.4x |
| r011 | attention_norm backward + residual add | 47 | 6.514 ms | 3.873 ms | 1.7x |
| r008 | attention_norm forward: residual add, RMSNorm, MXFP8 quantization | 47 | 6.534 ms | 4.173 ms | 1.6x |
| r009 | routed-expert weight cast + Dist-MoE MXFP8 weight quantization | 48 | 4.005 ms | 1.966 ms | 2.0x |
| r005 | cross-entropy loss forward, the last chunk | 1 | 2.043 ms | 0.712 ms | 2.9x |
| r012 | MoE router aux loss | 48 | 1.386 ms | 0.149 ms | 9.3x |
| r013 | routed-expert weight cast + Dist-MoE MXFP8 weight quantization | 48 | 2.203 ms | 1.020 ms | 2.2x |
| r015 | ffn_norm forward: residual add, RMSNorm | 49 | 4.747 ms | 3.686 ms | 1.3x |
| r018 | MXFP8 rowwise quantization (attention forward) | 48 | 3.411 ms | 2.820 ms | 1.2x |
| r016 (`a4oz3djo`) | MXFP8 colwise quantization (wqkv backward) | 48 | 3.113 ms | 2.836 ms | 1.1x |
| r016 (`aaha4wlh`) | MXFP8 colwise quantization (wo backward) | 48 | 1.722 ms | 1.451 ms | 1.2x |

Two regions above the profit floor keep their aten ops: r010 (1 call, 0.78 ms) was not attempted, and r017's
kernel (0.65 ms per step) was exact but did not beat the reference consistently in timing.

fusion_finder does not lift a region if lifting it would raise the step's peak memory: e.g. the 8 chunks of
the chunked loss, joined only through the sum of their losses, would be one call holding all chunks' logits
(+21 GiB); they are one call per chunk (r005, r006).

## Numerics

Loss and grad_norm of the recipe with and without the pack are bitwise identical over 20 steps (seed 42,
deterministic, CUDA graphs on, `scripts/loss_compare.py --assert-equal`): loss 12.458515167236328 /
5.717247009277344 / 5.424350261688232 and grad_norm 68.36530303955078 / 9.812052726745605 / 9.32946491241455
at steps 1 / 10 / 20. Every kernel passes `kernels/check_kernel.py <problem dir> <problem dir>/kernel.py`, its
bitwise gate on the input cases saved from the training step (with exact zeros of either sign in the
floating-point inputs on odd seeds).

Some kernels depend on facts about their inputs that hold in this model but are not in the contract: r005 and
r006 need labels in [0, 151936) or -100, r002 expert ids in [0, 128), and r001 RoPE positions in
[-4096, 4096); where aten raises a device assert on other values, these kernels do not check. The RMSNorm
kernels (r000, r007, r011) reproduce the reduction order of the quack RMSNorm that this torch build runs,
which depends on the SM count; they were verified on a 152-SM GB300, and the target checks the compute
capability, not the SM count.

## Use

Requirements:
- torchtitan main with fusion_finder's kernel problems (pytorch/torchtitan#6171) and the follow-up that adds
  `kernel_problems.register_problem_kernels`; neither is in main yet.
  The kernels are stored under the problems' `pattern_<l1>/<l2>` keys, so a change to the keys needs a
  regenerated pack.
- The optional `dist_moe` package (`torchtitan/models/common/dist_moe/README.md`).
- The exact environment in `manifest.json` `target`.
- `recipe.py`, or a recipe that produces the same graph. The Megatron-matched benchmark recipe it starts from
  is not in torchtitan main, so `recipe.py` holds a copy.
- One fusion_finder pack per process: the lift passes are named by region id, so a second pack's passes would
  replace them (the pack's call-count check then raises).

From the root of the torchtitan checkout:

```
PYTHONPATH=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts:. \
NGPU=8 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True python -m torchtitan.train \
    --module graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103_fusion_finder.recipe \
    --config qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8 --comm-backend fake
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
