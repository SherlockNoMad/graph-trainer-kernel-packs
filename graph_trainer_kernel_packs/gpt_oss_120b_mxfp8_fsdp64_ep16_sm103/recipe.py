# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: torchtitan's GPT-OSS 120B MXFP8 benchmark recipe
(FSDP 64 x EP 16, 5 microbatches per step), deterministic, run on one GPU with the fake
process group.

From the root of a torchtitan checkout with PR #5838 and symbolic-size cache keys
(branch agentic_compiler at 4f955c1c8 or later):

    NGPU=64 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.gpt_oss_120b_mxfp8_fsdp64_ep16_sm103.recipe \\
        --config gpt_oss_120b_mxfp8_fsdp64_ep16 --comm-backend fake
"""

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    graph_trainer_gpt_oss_120b_mxfp8_64gpu,
)

from . import gpt_oss_120b_mxfp8_fsdp64_ep16_sm103 as kernel_pack_pass


def gpt_oss_120b_mxfp8_fsdp64_ep16():
    config = graph_trainer_gpt_oss_120b_mxfp8_64gpu()
    # The kernels reproduce the deterministic index_put path, and their specialization
    # keys include the deterministic-algorithms setting; seed 42 reproduces the losses
    # in README.md.
    config.debug.deterministic = True
    config.debug.seed = 42
    config.compile.extra_passes = [kernel_pack_pass.__name__]
    return config
