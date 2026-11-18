"""Kernel problem pattern_83d7f044213d670a499e6b29b7794ecbb588f129260b926a00c32f019c7452f4/a7xu3da7xqoa6tk64vdlqgrp6rapompagot5pqzf634vpqxtdkkx: region r014 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.attention.inner_attention, layers.1.attention.inner_attention, layers.2.attention.inner_attention ....
Measured 210.4 us per call in the capture; ideal (HBM roofline) 50.4 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2):
    v1 = torch.ops.aten.alias.default(in0)
    v3 = torch.ops.aten.transpose.int(in1, 0, 1);  in1 = None
    v5 = torch.ops.aten.sub.Tensor(v3, in2);  v3 = in2 = None
    v6 = torch.ops.aten.sigmoid.default(v5);  v5 = None
    v7 = torch.ops.aten.alias.default(v6)
    v8 = torch.ops.aten.unsqueeze.default(v6, -1);  v6 = None
    v9 = torch.ops.aten._to_copy.default(v8, dtype = torch.bfloat16);  v8 = None
    v10 = torch.ops.aten.mul.Tensor(in0, v9);  in0 = None
    v11 = torch.ops.aten.reshape.default(v10, [16384, -1]);  v10 = None
    v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, False, 32, 1, 'e4m3', 'rceil')
    v13 = v12[0]
    v14 = v12[2];  v12 = None
    v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
    return (v1, v7, v9, v11, v13, v15)


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
