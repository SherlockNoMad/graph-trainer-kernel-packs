# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply a kernel pack to a GraphTrainer joint graph.

A pack directory holds::

    manifest.json             # target environment; each kernel's sha256 and calls per graph
    regions/                  # topology matchers; regions.PLAN lists them in lift order
    kernels/pat_<l1>/         # torchtitan substitute_kernels (PR #5838) cache entries
      reference.py            # the aten ops the kernels of this pattern were verified against
      specialization_<l2>.json
      <l2>.py                 # verified kernel defining run(...)

``apply_pack`` lifts every instance of each region into ``call_module fusion_<id>``, keys
each call with #5838's ``cache_keys`` (code and custom-op sources -> l1, input
specialization -> l2) and replaces it with the kernel stored under that key. Calls whose
key has no kernel keep running their aten ops. Unlike ``substitute_with_kernels_pass``,
nothing is written to the pack.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.substitute_kernels import cache_keys

from graph_trainer_kernel_packs.lifting import lift_matches

logger = logging.getLogger(__name__)


def current_target() -> dict[str, Any]:
    """The environment a pack is verified in; ``apply_pack`` requires an exact match."""
    return {
        # The kernels reproduce this build's aten numerics bit for bit.
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability()),
        "nvidia-cutlass-dsl": metadata.version("nvidia-cutlass-dsl"),
        # When set, ops such as aten._fused_rms_norm leave torch's native (vendored
        # quack) kernels, which changes the numerics the pack's kernels reproduce.
        "TORCH_DISABLE_NATIVE_JIT": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
    }


def _load_kernel(path: Path, sha256: str) -> Callable[..., Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its sha256 in manifest.json")
    name = f"_kernel_pack_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.run


def apply_pack(
    gm: fx.GraphModule, pack_dir: Path, plan: list[ModuleType]
) -> fx.GraphModule:
    """Lift ``plan``'s regions in order, then substitute the pack's kernels.

    Raises ValueError if the environment differs from the manifest's target, or if a
    kernel is not called exactly as often as the manifest records: either means this is
    not the graph, or not the specialization keys, that the pack was verified on.
    """
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    found = current_target()
    if found != manifest["target"]:
        mismatch = {
            k: (found.get(k), v)
            for k, v in manifest["target"].items()
            if found.get(k) != v
        }
        raise ValueError(
            f"kernel pack {pack_dir.name} targets another environment, "
            f"(found, expected): {mismatch}"
        )
    for region in plan:
        gm = lift_matches(gm, region.REGION_ID, region.MATCHERS)

    region_ids = {region.REGION_ID for region in plan}
    kernels = manifest["kernels"]
    runs: dict[str, Callable[..., Any]] = {}
    calls = dict.fromkeys(kernels, 0)
    replaced: set[str] = set()
    kept = 0
    for node in list(gm.graph.find_nodes(op="call_module")):
        if node.meta.get("fusion_region") not in region_ids:
            continue
        submod = gm.get_submodule(node.target)
        l1, l2, _, _ = cache_keys(submod, [arg.meta["val"] for arg in node.args])
        key = f"{l1}/{l2}.py"
        if key not in kernels:
            kept += 1
            continue
        if key not in runs:
            runs[key] = _load_kernel(pack_dir / "kernels" / key, kernels[key]["sha256"])
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(runs[key], node.args)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced.add(node.target)
        calls[key] += 1

    wrong = {
        k: (n, kernels[k]["calls"])
        for k, n in calls.items()
        if n != kernels[k]["calls"]
    }
    if wrong:
        raise ValueError(
            f"kernel pack {pack_dir.name}: kernel calls differ from the verified graph, "
            f"(found, expected): {wrong}. The graph or the specialization keys are not "
            "the pack's; check the model config, parallelism, debug.deterministic and "
            "Inductor config against the pack's recipe."
        )
    still_called = {n.target for n in gm.graph.find_nodes(op="call_module")}
    for target in replaced - still_called:
        gm.delete_submodule(target)
    gm.recompile()
    logger.info(
        "kernel pack %s: %d calls to %d kernels; %d lifted calls without a kernel "
        "run their aten ops",
        pack_dir.name,
        sum(calls.values()),
        len(kernels),
        kept,
    )
    return gm
