# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a fusion_finder kernel pack from a capture and its kernel problems.

    python tools/make_fusion_finder_pack.py \\
        --capture <capture dir> --problems <problems dir> \\
        --out graph_trainer_kernel_packs/<pack>

``--capture`` is the fusion_finder capture whose ``regions.json`` the lift passes
replay; ``--problems`` the problems written from it (``kernel_problems.py``), where
autokernel promoted a verified ``kernel.py`` into each problem it solved. Each solved
problem is copied whole (``problem.py``, ``spec.json``, ``inputs.pt``, ``kernel.py``)
under ``kernels/``, with the problem set's ``check_kernel.py``, so every kernel can be
re-checked against its reference. ``manifest.json`` records the target environment
(run this in it) and per kernel its sha256 and calls per graph (``spec.json``). The
pack's ``__init__.py``, ``recipe.py`` and ``README.md`` are written by hand.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from graph_trainer_kernel_packs.fusion_finder_pack import current_target

_PROBLEM_FILES = ("problem.py", "spec.json", "inputs.pt", "kernel.py")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    kernels_dir = a.out / "kernels"
    if kernels_dir.exists():
        shutil.rmtree(kernels_dir)
    kernels_dir.mkdir(parents=True)
    shutil.copyfile(a.capture / "regions.json", a.out / "regions.json")
    shutil.copyfile(a.problems / "check_kernel.py", kernels_dir / "check_kernel.py")
    kernels = {}
    for kernel in sorted(a.problems.glob("pattern_*/*/kernel.py")):
        problem_id = str(kernel.parent.relative_to(a.problems))
        spec = json.loads((kernel.parent / "spec.json").read_text())
        (kernels_dir / problem_id).mkdir(parents=True)
        for name in _PROBLEM_FILES:
            shutil.copyfile(kernel.parent / name, kernels_dir / problem_id / name)
        kernels[problem_id] = {
            "region": spec["source"]["region"],
            "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "calls": len(spec["source"]["calls"]),
        }
    manifest = {"target": current_target(), "kernels": kernels}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{a.out}: {len(kernels)} kernels, {sum(k['calls'] for k in kernels.values())} calls")


if __name__ == "__main__":
    main()
