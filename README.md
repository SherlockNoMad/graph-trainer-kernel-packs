# graph-trainer-kernel-packs

Kernel packs for [torchtitan](https://github.com/pytorch/torchtitan) GraphTrainer. A pack is one registered
graph pass for one model, parallelism and hardware. It replaces regions of the training step's joint graph
with handwritten kernels that compute **bitwise identical** results, so loss and grad_norm do not change.

| Pack | Model and layout | Hardware | GPU time per step | Numerics |
|---|---|---|---|---|
| [`dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103`](graph_trainer_kernel_packs/dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103/README.md) | DeepSeek-V3 671B, DistMoE, MXFP8; FSDP 256 x EP 64 | GB300 | -13.8% (583.9 -> 503.4 ms, excluding fake-PG collectives) | bitwise identical |

## Use

Requirements:

- torchtitan with `substitute_kernels` ([pytorch/torchtitan#5838](https://github.com/pytorch/torchtitan/pull/5838)),
  which is not in `main` yet: branch [`agentic_compiler`](https://github.com/pytorch/torchtitan/tree/agentic_compiler)
  at `d89f7f53e` or later, where its cache keys are the same on every rank. A pack looks its kernels up
  with these keys, so a later change to them needs a regenerated pack.
- The exact environment in the pack's `manifest.json` `target` (torch build, CUDA, GPU, cutlass-dsl).
- The pack's recipe, or one that produces the same graph: same model config and parallelism, and
  `debug.deterministic = True`.

```
pip install git+https://github.com/SherlockNoMad/graph-trainer-kernel-packs
```

Each pack ships the recipe it was verified with. From the root of the torchtitan checkout:

```
NGPU=256 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m torchtitan.train \
    --module graph_trainer_kernel_packs.dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103.recipe \
    --config dsv3_671b_distmoe_mxfp8_fsdp256_ep64 --comm-backend fake
```

Or, in your own recipe:

```python
from graph_trainer_kernel_packs.dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103 import (
    dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103,  # importing registers the pass
)

config.compile.extra_passes = [dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103.__name__]
```

## How a pack works

```
joint graph (after GraphTrainer's own passes, before Inductor)
  -> check the environment against manifest.json "target"                  ValueError on mismatch
  -> lift each region in regions.PLAN order: matching subgraphs become call_module fusion_G<id>
  -> key each lifted call like substitute_kernels (#5838):
       l1 = hash of the submodule's code (device indices dropped) and its custom ops' sources
       l2 = specialization (AOTAutograd cache key of the submodule on its inputs, on device index 0)
  -> kernels/pat_<l1>/<l2>.py exists: check its sha256, call its run() instead   ValueError on sha mismatch
     otherwise: the lifted call keeps running its original aten ops
  -> every kernel called exactly as often as manifest.json records             ValueError otherwise
```

Every rank lifts the same regions and finds the same kernels: matchers accept any size, scalar operand and
device index (ranks differ in device index, bucket rank and FSDP shard offsets), and neither key depends on
the device index. Nothing is written to the pack at runtime. A wrong environment, graph or specialization
fails loudly at compile time instead of silently running slower or differently.

**Known limitation.** l2 is the AOTAutograd cache key, which also covers settings a handwritten kernel does
not depend on: `torch.use_deterministic_algorithms` and `fill_uninitialized_memory`, and Inductor config
(e.g. `TORCHINDUCTOR_MAX_AUTOTUNE=1` changes it). With the FX graph cache disabled the key cannot be computed.
Runs that differ in these settings get no kernel hits, so the pass raises. Keying specializations by input
shapes, strides and dtypes instead is a follow-up for #5838.

## How packs are made

The kernels come from an agentic compiler loop in torchtitan (branch `agentic_compiler`):

1. The **fusion_region_finder** skill profiles a training step, finds fusable regions of the joint graph, and
   generates a topology matcher ("lift pass") per region.
2. **substitute_kernels** (#5838) turns each lifted region into a kernel problem: `reference.py` plus a
   specialization spec per input shape.
3. The **autokernel** skill ([#5839](https://github.com/pytorch/torchtitan/pull/5839)) runs a swarm: an agent
   writes a CuTe DSL kernel, a different freshly started agent verifies it (bytewise against `reference.py`,
   poisoned output memory, compute-sanitizer, CUDA graphs, adversarial inputs), and only verified kernels
   are promoted.
4. A training run with the plan and the kernels must give bitwise identical loss and grad_norm.

`tools/make_pack.py` turns the loop's output (regions, kernel cache, and the log of a run with all kernels hit)
into a pack's `manifest.json`, `regions/` and `kernels/`. The pack's `__init__.py`, `recipe.py` and `README.md`
are written by hand; see the existing pack.

## Layout

```
graph_trainer_kernel_packs/
  lifting.py      region lifting runtime (vendored from torchtitan's fusion_region_finder skill)
  pack.py         apply_pack(): target check, lifting, lookup-only substitution, call-count check
  <pack>/         one directory per pack: __init__.py (registers the pass), recipe.py, README.md,
                  manifest.json, regions/, kernels/, verification/
tools/make_pack.py
```

Generated region files and kernels are kept exactly as produced and verified; don't reformat them (a kernel's
sha256 is checked on load).

## License

BSD 3-Clause, see [LICENSE](LICENSE). The code derives from torchtitan, which uses the same license.
