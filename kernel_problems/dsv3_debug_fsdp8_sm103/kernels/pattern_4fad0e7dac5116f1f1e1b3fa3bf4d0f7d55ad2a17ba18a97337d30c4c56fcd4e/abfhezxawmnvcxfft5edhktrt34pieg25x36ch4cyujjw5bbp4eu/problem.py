"""Kernel problem pattern_4fad0e7dac5116f1f1e1b3fa3bf4d0f7d55ad2a17ba18a97337d30c4c56fcd4e/abfhezxawmnvcxfft5edhktrt34pieg25x36ch4cyujjw5bbp4eu: region r004 of outputs/fusion_finder/dsv3_debug.

6 calls per step, backward, in layers.5.attention.rope, layers.4.attention.rope, layers.3.attention.rope ....
Measured 267.9 us per call in the capture; ideal (HBM roofline) 44.4 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2):
    v1 = torch.ops.aten.squeeze.dim(in0, 0);  in0 = None
    v2 = torch.ops.aten.transpose.int(v1, 0, 1);  v1 = None
    v4 = torch.ops.aten.squeeze.dim(in1, 0);  in1 = None
    v5 = torch.ops.aten.transpose.int(v4, 0, 1);  v4 = None
    v6 = torch.ops.aten.slice.Tensor(v5, 2, 0, 128)
    v7 = torch.ops.aten.slice.Tensor(v5, 2, 128, 192);  v5 = None
    v8 = torch.ops.aten.sum.dim_IntList(v7, [1], True);  v7 = None
    v9 = torch.ops.aten.squeeze.dim(v8, 1);  v8 = None
    v10 = torch.ops.aten.cat.default([v6, v2], 2);  v6 = v2 = None
    v11 = torch.ops.aten.reshape.default(v10, [16384, 4096]);  v10 = None
    v12 = torch.ops.aten.t.default(v11)
    v13 = torch.ops.aten.unsqueeze.default(v9, 1);  v9 = None
    v14 = torch.ops.aten._to_copy.default(v13, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v13 = None
    v15 = torch.ops.aten.reshape.default(v14, [16384, 1, 32, 2]);  v14 = None
    v16 = torch.ops.aten.view_as_complex.default(v15);  v15 = None
    v18 = torch.ops.aten.mul.Tensor(v16, in2);  v16 = in2 = None
    v19 = torch.ops.aten.view_as_real.default(v18);  v18 = None
    v20 = torch.ops.aten.reshape.default(v19, [16384, 1, 64]);  v19 = None
    v21 = torch.ops.aten._to_copy.default(v20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v20 = None
    v22 = torch.ops.aten.squeeze.dim(v21, 1);  v21 = None
    return (v11, v12, v22)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2):
        return reference(in0, in1, in2)


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
