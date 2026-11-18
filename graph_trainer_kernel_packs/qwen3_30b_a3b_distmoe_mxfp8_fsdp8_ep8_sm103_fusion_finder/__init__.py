# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel pack for Qwen3 30B-A3B, MXFP8 Dist-MoE, FSDP 8 x EP 8, on GB300 (sm_103), made by
torchtitan's fusion_finder and autokernel skills (see ``fusion_finder_pack``).

Importing this package registers the pack's graph passes; enable them with
``config.compile.extra_passes = PASSES`` (``recipe.py`` does). See README.md for the
target environment and the evidence.
"""

from pathlib import Path

from graph_trainer_kernel_packs.fusion_finder_pack import fusion_finder_passes

PASSES = fusion_finder_passes(Path(__file__).parent)
