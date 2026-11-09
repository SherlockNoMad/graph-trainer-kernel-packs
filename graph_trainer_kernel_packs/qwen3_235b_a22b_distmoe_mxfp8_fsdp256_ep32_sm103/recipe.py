# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: torchtitan's Qwen3 235B-A22B MXFP8 Dist-MoE
benchmark recipe (FSDP 256 x EP 32) with one 8192-token microbatch per step and FA4 varlen
attention, deterministic, run on one GPU with the fake process group.

From the root of a torchtitan checkout with PR #5838 (branch agentic_compiler at
e96fc894a or later):

    NGPU=256 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32_sm103.recipe \\
        --config qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32 --comm-backend fake
"""

from torchtitan.models.common.attention.attention import VarlenInnerAttention

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu,
)

from . import qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32_sm103 as kernel_pack_pass


def qwen3_235b_a22b_distmoe_mxfp8_fsdp256_ep32():
    config = graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu()
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
