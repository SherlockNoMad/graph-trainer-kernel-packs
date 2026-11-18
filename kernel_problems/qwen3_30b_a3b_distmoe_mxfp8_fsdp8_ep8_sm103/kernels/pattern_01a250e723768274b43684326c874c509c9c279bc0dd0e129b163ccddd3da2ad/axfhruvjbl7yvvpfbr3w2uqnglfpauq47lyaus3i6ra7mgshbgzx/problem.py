"""Kernel problem pattern_01a250e723768274b43684326c874c509c9c279bc0dd0e129b163ccddd3da2ad/axfhruvjbl7yvvpfbr3w2uqnglfpauq47lyaus3i6ra7mgshbgzx: region r009 of outputs/fusion_finder/qwen3_30b.

48 calls per step, forward, in layers.0.feed_forward.routed_experts, layers.1.feed_forward.routed_experts, layers.2.feed_forward.routed_experts ....
Measured 82.1 us per call in the capture; ideal (HBM roofline) 36.7 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import dist_moe._blockscaled


def reference(in0):
    v1 = torch.ops.aten._to_copy.default(in0, dtype = torch.bfloat16);  in0 = None
    v2 = torch.ops.aten.reshape.default(v1, [16, 1536, 2048]);  v1 = None
    v3 = torch.ops.dist_moe.prepare_mxfp8_weight.default(v2);  v2 = None
    v4 = v3[0]
    v5 = v3[1]
    v6 = v3[2];  v3 = None
    v7 = torch.ops.aten.new_empty.default(v4, [0], dtype = torch.uint8, pin_memory = False)
    return (v4, v5, v6, v7)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0):
        return reference(in0)


def get_input_cases(seed=0):
    """The inputs of each case (calls of the training run, see spec.json): the
    integer, bool and complex inputs as saved, floating-point inputs generated from
    ``seed``."""
    path = Path(__file__).parents[2] / "check_kernel.py"
    spec = importlib.util.spec_from_file_location("check_kernel", path)
    check_kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_kernel)
    return check_kernel.load_cases(Path(__file__).parent, seed)


def get_inputs():
    return get_input_cases()[0]


def get_init_inputs():
    return []
