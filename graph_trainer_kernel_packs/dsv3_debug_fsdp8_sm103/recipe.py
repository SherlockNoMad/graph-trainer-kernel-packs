# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: torchtitan's GraphTrainer DeepSeek-V3 debug
model recipe (FSDP 8) with one microbatch per step, deterministic, run on one GPU with
the fake process group.

From the root of a torchtitan checkout (see README.md for the commit), with the
fusion_finder skill's scripts on PYTHONPATH:

    PYTHONPATH=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts:. \\
    NGPU=8 LOCAL_RANK=0 python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.dsv3_debug_fsdp8_sm103.recipe \\
        --config dsv3_debug_fsdp8 --comm-backend fake
"""

from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)

from . import PASSES


def dsv3_debug_fsdp8():
    config = graph_trainer_deepseek_v3_debugmodel()
    # One microbatch per step, the graph the kernels were verified on.
    config.training.num_tokens_per_train_step = -1
    # The kernels reproduce the deterministic code paths bit for bit; seed 42
    # reproduces the losses in README.md.
    config.debug.deterministic = True
    config.debug.seed = 42
    config.compile.extra_passes = PASSES
    return config
