# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel packs made by torchtitan's fusion_finder and autokernel skills.

A pack directory holds::

    manifest.json               # target environment; per kernel its sha256 and calls per graph
    regions.json                # fusion_finder's regions: the lift passes' patterns, in lift order
    kernels/pattern_<l1>/<l2>/  # fusion_finder kernel problems, keyed like substitute_kernels
      problem.py                # the reference the kernel is verified against
      spec.json                 # its contract, input cases and the calls it replaces
      kernel.py                 # the verified kernel_function

``fusion_finder_passes(pack_dir)`` returns the graph passes to list in
``compile.extra_passes``: fusion_finder's lift pass per region of ``regions.json`` (in
its order), its ``use_problem_kernels`` pass on ``kernels/`` (each lifted call whose
problem has a ``kernel.py`` calls its ``kernel_function``), and a check that the
environment is the manifest's target and that each kernel is the verified file and is
called exactly as often as the manifest records. Calls without a kernel keep running
their aten ops. Nothing is written to the pack at runtime.

The lift and kernel passes come from the fusion_finder skill of the torchtitan checkout
(``torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts``), which
must be on ``PYTHONPATH``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.registry import register_graph_pass

logger = logging.getLogger(__name__)


def current_target() -> dict[str, Any]:
    """The environment a pack is verified in; the pack requires an exact match.
    (``pack.current_target`` of the packs for torchtitan's ``agentic_compiler``
    branch, whose ``pack`` module imports ``substitute_kernels`` from its old path.)"""
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


def _check(gm: fx.GraphModule, pack_dir: Path) -> fx.GraphModule:
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
    kernels_dir = (pack_dir / "kernels").resolve()
    calls: Counter[str] = Counter()
    for node in gm.graph.find_nodes(op="call_module"):
        kernel_function = getattr(gm.get_submodule(node.target), "kernel_function", None)
        if kernel_function is None:
            continue
        path = Path(kernel_function.__code__.co_filename).resolve()
        calls[str(path.parent.relative_to(kernels_dir))] += 1
    for problem_id in calls:
        path = kernels_dir / problem_id / "kernel.py"
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha256 != manifest["kernels"].get(problem_id, {}).get("sha256"):
            raise ValueError(f"{path} does not match its sha256 in manifest.json")
    expected = {k: v["calls"] for k, v in manifest["kernels"].items()}
    if dict(calls) != expected:
        wrong = {
            k: (calls.get(k, 0), expected.get(k, 0))
            for k in calls.keys() | expected.keys()
            if calls.get(k, 0) != expected.get(k, 0)
        }
        raise ValueError(
            f"kernel pack {pack_dir.name}: kernel calls differ from the verified graph, "
            f"(found, expected): {wrong}. The graph or the specialization keys are not "
            "the pack's; check the model config, parallelism, debug.deterministic and "
            "Inductor config against the pack's recipe."
        )
    logger.info(
        f"kernel pack {pack_dir.name}: {sum(calls.values())} calls to {len(calls)} "
        "kernels, as verified"
    )
    return gm


def fusion_finder_passes(pack_dir: Path) -> list[str]:
    """Register the pack's passes and return their names, in the order
    ``compile.extra_passes`` must list them."""
    from kernel_problems import register_problem_kernels
    from lift_regions import register_lift_passes

    def check(gm: fx.GraphModule, example_inputs: Any = None) -> fx.GraphModule:
        return _check(gm, pack_dir)

    check.__name__ = check.__qualname__ = f"check_{pack_dir.name}"
    register_graph_pass(check)
    return [
        *register_lift_passes(str(pack_dir)),
        register_problem_kernels(str(pack_dir / "kernels")),
        check.__name__,
    ]
