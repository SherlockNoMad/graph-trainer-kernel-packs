"""Kernel problem pattern_ebf7b12c439d5cb6360a7c5052da525e5d3cb5dff67008cd3b9a37297526f00c/aqay2xqjf2fa7hnogo4zbzwt5cvtpjt37us3t6bzsp47galeaipp: region r007 of outputs/fusion_finder/dsv3_671b.

58 calls per step, recompute, in layers.60.feed_forward.routed_experts, layers.59.feed_forward.routed_experts, layers.58.feed_forward.routed_experts ....
Measured 102.4 us per call in the capture; ideal (HBM roofline) 51.8 us.

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
    v1 = torch.ops.dist_moe.prepare_mxfp8_weight.default(in0);  in0 = None
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2];  v1 = None
    v5 = torch.ops.aten.new_empty.default(v2, [0], dtype = torch.uint8, pin_memory = False)
    return (v2, v3, v4, v5)


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
