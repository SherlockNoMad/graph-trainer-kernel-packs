# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel pack for Qwen3 30B-A3B DistMoE MXFP8 on FSDP 8 x EP 8, GB300 (sm_103).

Importing this package registers the GraphTrainer pass of the same name; enable it with
``config.compile.extra_passes = ["qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103"]``
(``recipe.py`` does). See README.md for the target environment and the evidence.
"""

from pathlib import Path

from torchtitan.experiments.graph_trainer.registry import register_graph_pass

from graph_trainer_kernel_packs.pack import apply_pack

from .regions import PLAN


@register_graph_pass
def qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103(gm, example_inputs=None):
    return apply_pack(gm, Path(__file__).parent, PLAN)
