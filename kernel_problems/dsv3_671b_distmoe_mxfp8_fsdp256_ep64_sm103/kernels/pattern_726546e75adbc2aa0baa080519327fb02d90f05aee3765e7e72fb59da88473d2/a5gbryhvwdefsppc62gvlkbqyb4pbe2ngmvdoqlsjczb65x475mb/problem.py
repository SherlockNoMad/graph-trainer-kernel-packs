"""Kernel problem pattern_726546e75adbc2aa0baa080519327fb02d90f05aee3765e7e72fb59da88473d2/a5gbryhvwdefsppc62gvlkbqyb4pbe2ngmvdoqlsjczb65x475mb: region r008 of outputs/fusion_finder/dsv3_671b.

58 calls per step, forward, in layers.3.feed_forward.routed_experts, layers.4.feed_forward.routed_experts, layers.5.feed_forward.routed_experts ....
Measured 45.0 us per call in the capture; ideal (HBM roofline) 25.9 us.

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
    return (v2, v3, v4)


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
