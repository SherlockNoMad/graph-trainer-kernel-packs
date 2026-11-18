"""Kernel problem pattern_a2d3fe9f709b9af3a01153486443bffd882a9a19f9ff4f24d62b4952aaa8704c/axsm33lvaz5myg6zuitrbwa5kuvedthorzb6r5balq4ruvvg4b46: region r021 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward, in layers.60.attention.wkv_a, layers.59.attention.wkv_a, layers.58.attention.wkv_a ....
Measured 24.6 us per call in the capture; ideal (HBM roofline) 2.6 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3, in4):
    v4 = torch.ops.aten._fused_rms_norm_backward.default(in0, in1, [512], in2, in3, [True, True]);  in0 = in1 = in2 = in3 = None
    v5 = v4[0]
    v6 = v4[1];  v4 = None
    v8 = torch.ops.aten.cat.default([v5, in4], 1);  v5 = in4 = None
    v9 = torch.ops.torchao.mxfp8_quantize.default(v8, True, True, 32, 32, 'e4m3', 'rceil');  v8 = None
    v10 = v9[0]
    v11 = v9[1]
    v12 = v9[2]
    v13 = v9[3];  v9 = None
    v14 = torch.ops.torchao.triton_mx_block_rearrange.default(v12);  v12 = None
    v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v13);  v13 = None
    v16 = torch.ops.aten.t.default(v11);  v11 = None
    return (v6, v10, v14, v15, v16)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4):
        return reference(in0, in1, in2, in3, in4)


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
