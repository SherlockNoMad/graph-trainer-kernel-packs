"""Kernel problem pattern_157215941e5f259a125f6454ea782064cba5d5ef9821b6ee441c921ad976c67d/aclkmyqyehk7tbc3bxi3nqp6juxowwqypwsntowoapc34obllkoj: region r003 of outputs/fusion_finder/dsv3_debug.

6 calls per step, backward, in layers.5.attention.rope, layers.4.attention.rope, layers.3.attention.rope ....
Measured 282.0 us per call in the capture; ideal (HBM roofline) 29.6 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1):
    v1 = torch.ops.aten.squeeze.dim(in0, 0);  in0 = None
    v2 = torch.ops.aten.transpose.int(v1, 0, 1);  v1 = None
    v3 = torch.ops.aten.slice.Tensor(v2, 2, 0, 128)
    v4 = torch.ops.aten.slice.Tensor(v2, 2, 128, 192);  v2 = None
    v5 = torch.ops.aten._to_copy.default(v4, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v4 = None
    v6 = torch.ops.aten.reshape.default(v5, [16384, 16, 32, 2]);  v5 = None
    v7 = torch.ops.aten.view_as_complex.default(v6);  v6 = None
    v9 = torch.ops.aten.mul.Tensor(v7, in1);  v7 = in1 = None
    v10 = torch.ops.aten.view_as_real.default(v9);  v9 = None
    v11 = torch.ops.aten.reshape.default(v10, [16384, 16, 64]);  v10 = None
    v12 = torch.ops.aten._to_copy.default(v11, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v11 = None
    v13 = torch.ops.aten.cat.default([v3, v12], 2);  v3 = v12 = None
    v14 = torch.ops.aten.reshape.default(v13, [16384, 3072]);  v13 = None
    v15 = torch.ops.aten.t.default(v14)
    return (v14, v15)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1):
        return reference(in0, in1)


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
