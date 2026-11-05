# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: torchtitan's 671B MXFP8 DistMoE GraphTrainer recipe
on FSDP 256 x EP 64, run on one GPU with the fake process group.

From the root of a torchtitan checkout with PR #5838 (e.g. branch agentic_compiler):

    NGPU=256 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103.recipe \\
        --config dsv3_671b_distmoe_mxfp8_fsdp256_ep64 --comm-backend fake
"""

from torchtitan_recipes.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
)

from . import dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103 as kernel_pack_pass


def dsv3_671b_distmoe_mxfp8_fsdp256_ep64():
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.pipeline_parallel_degree = 1
    # One microbatch per step: deferring gradient reduction changes nothing, and
    # GraphTrainer without PP rejects it.
    config.parallelism.fsdp_defer_gradient_reduction = False
    config.training.num_tokens_per_train_step = -1
    # Under the fake PG every EP peer aliases this rank, so routing imbalance does not
    # average out and can exceed the recipe's receive capacity; EP x covers the worst case.
    config.dist_moe_runtime.scratch_capacity_factor = float(
        config.parallelism.expert_parallel_degree
    )
    # The kernels' specialization keys include the deterministic-algorithms settings;
    # seed 42 reproduces the losses in README.md.
    config.debug.deterministic = True
    config.debug.seed = 42
    config.compile.extra_passes = [kernel_pack_pass.__name__]
    return config
