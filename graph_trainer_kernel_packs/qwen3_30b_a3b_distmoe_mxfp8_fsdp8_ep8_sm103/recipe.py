# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: torchtitan's Qwen3 30B-A3B MXFP8 Dist-MoE
benchmark recipe (FSDP 8 x EP 8) with one 32K-token microbatch per step and FA4 varlen
attention, deterministic, run on one GPU with the fake process group.

From the root of a torchtitan checkout with PR #5838 (branch agentic_compiler at
8e8145923 or later):

    NGPU=8 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103.recipe \\
        --config qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8 --comm-backend fake
"""

from torchtitan.models.common.attention.attention import VarlenInnerAttention

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
)

from . import qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103 as kernel_pack_pass


def qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8():
    config = graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu()
    # One microbatch per step, the graph the kernels were verified on.
    config.training.num_tokens_per_train_step = -1
    # cuDNN's varlen attention backward is nondeterministic even with
    # debug.deterministic; FA4 is not. No kernel touches attention.
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    # The specialization keys include the deterministic-algorithms setting; seed 42
    # reproduces the losses in README.md.
    config.debug.deterministic = True
    config.debug.seed = 42
    config.compile.extra_passes = [kernel_pack_pass.__name__]
    return config
