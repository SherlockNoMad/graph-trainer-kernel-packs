"""Kernel problem pattern_014c7f1530e0bb28a17fd596bc431e1bc5bcbb406ca5db8838c071272f7455e8/atgxg3zzd7hd3r3ty5v54y5zrdt3m7r76bm3t7dbg4cvepclff3i: region r008 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.attention.inner_attention, layers.34.attention.inner_attention, layers.33.attention.inner_attention ....
Measured 443.2 us per call in the capture; ideal (HBM roofline) 58.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3):
    v1 = torch.ops.aten.reshape.default(in0, [16384, 64, 64]);  in0 = None
    v3 = torch.ops.aten.mul.Tensor(v1, in1);  in1 = None
    v5 = torch.ops.aten.mul.Tensor(v1, in2);  v1 = in2 = None
    v6 = torch.ops.aten.sum.dim_IntList(v3, [2], True);  v3 = None
    v7 = torch.ops.aten._to_copy.default(v6, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v6 = None
    v8 = torch.ops.aten.squeeze.dim(v7, -1);  v7 = None
    v10 = torch.ops.aten.alias.default(in3);  in3 = None
    v11 = torch.ops.aten.sigmoid_backward.default(v8, v10);  v8 = v10 = None
    v12 = torch.ops.aten.neg.default(v11);  v11 = None
    v13 = torch.ops.aten.sum.dim_IntList(v12, [0], True);  v12 = None
    v14 = torch.ops.aten._to_copy.default(v13, dtype = torch.bfloat16);  v13 = None
    v15 = torch.ops.aten.reshape.default(v14, [64]);  v14 = None
    return (v5, v15)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3):
        return reference(in0, in1, in2, in3)


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
