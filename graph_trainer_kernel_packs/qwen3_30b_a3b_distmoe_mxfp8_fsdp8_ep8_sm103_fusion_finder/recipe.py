# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recipe the pack is verified with: the Qwen3 30B-A3B MXFP8 Dist-MoE benchmark
recipe matched to Megatron-Bridge (FSDP 8 x EP 8, round-robin routing, no recompute)
with GraphTrainer, one 32K-token microbatch per step and FA4 varlen attention,
deterministic, run on one GPU with the fake process group.

The benchmark recipe is not in torchtitan main, so it is copied here (from
``torchtitan_recipes/graph_trainer/megatron_bridge_benchmarks.py`` on torchtitan's
``agentic_compiler`` branch). From the root of a torchtitan checkout (see README.md for
the commit), with the fusion_finder skill's scripts on PYTHONPATH:

    PYTHONPATH=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts:. \\
    NGPU=8 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python -m torchtitan.train \\
        --module graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103_fusion_finder.recipe \\
        --config qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8 --comm-backend fake
"""

import dataclasses

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, MXFP8LinearConverter
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.models.common.attention.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.moe import (
    RoundRobinTokenChoiceTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity

from . import PASSES

_SEQ_LEN = 4096
_MICRO_BATCH_SIZE = 8


def _benchmark_config() -> Trainer.Config:
    """``qwen3_30b_a3b_dist_moe_mxfp8_8gpu`` of the benchmark recipes."""
    import dist_moe

    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=8,
    )
    model_config = build_model_config(
        "30B-A3B",
        seq_len=_SEQ_LEN,
        parallelism=parallelism,
        converters=[MXFP8LinearConverter.Config(fqns=["attention"])],
    )
    # Route with exact balance, like Megatron's moe_router_force_load_balancing.
    router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
    for _fqn, router, parent, attr in list(
        model_config.traverse(TokenChoiceTopKRouter.Config)
    ):
        assert parent is not None and isinstance(attr, str)
        setattr(
            parent,
            attr,
            RoundRobinTokenChoiceTopKRouter.Config(
                **{f.name: getattr(router, f.name) for f in router_fields}
            ),
        )
    num_tokens_per_microbatch = _MICRO_BATCH_SIZE * _SEQ_LEN
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(num_tokens_per_microbatch),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", moment_dtype="bfloat16")]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            num_tokens_per_train_step=512 * _SEQ_LEN,
            steps=50,
            # Until Dist-MoE replaces it, the stock all-to-all dispatcher
            # synchronizes with the host.
            disable_accelerator_graphs=True,
        ),
        precision=PrecisionConfig(
            dtype="float32", param_dtype="bfloat16", reduce_dtype="bfloat16"
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
    )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=1.0,
        scratch_capacity_factor=1.5,
    )
    config = apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                # GraphTrainer traces WGRAD as a functional graph output.
                inplace_wgrad_accum=False,
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )
    # Dist-MoE dispatch never synchronizes with the host: capture the step in
    # CUDA graphs.
    config.training.disable_accelerator_graphs = False
    config.model.local_compile.regions.append("moe_aux_loss")
    return config


def qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8():
    config = to_graph_trainer_config(_benchmark_config(), GraphTrainerQwen3Model.Config)
    config.compile = GraphTrainerCompileConfig(memory_policy="none")
    # One microbatch per step, the graph the kernels were verified on.
    config.training.num_tokens_per_train_step = -1
    # cuDNN's varlen attention backward is nondeterministic even with
    # debug.deterministic; FA4 is not. No kernel touches attention.
    for _fqn, inner, _parent, _attr in config.model.traverse(VarlenInnerAttention.Config):
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    # The kernels reproduce the deterministic code paths bit for bit; seed 42
    # reproduces the losses in README.md.
    config.debug.deterministic = True
    config.debug.seed = 42
    config.compile.extra_passes = PASSES
    return config
