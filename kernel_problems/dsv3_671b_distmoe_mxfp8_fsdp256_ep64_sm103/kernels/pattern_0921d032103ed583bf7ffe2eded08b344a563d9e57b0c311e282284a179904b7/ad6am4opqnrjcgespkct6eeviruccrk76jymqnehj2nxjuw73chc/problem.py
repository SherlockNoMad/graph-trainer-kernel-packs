"""Kernel problem pattern_0921d032103ed583bf7ffe2eded08b344a563d9e57b0c311e282284a179904b7/ad6am4opqnrjcgespkct6eeviruccrk76jymqnehj2nxjuw73chc: region r006 of outputs/fusion_finder/dsv3_671b.

57 calls per step, forward, in layers.4.attention.wkv_a, layers.5.attention.wkv_a, layers.6.attention.wkv_a ....
Measured 115.1 us per call in the capture; ideal (HBM roofline) 42.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3):
    v2 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
    v4 = torch.ops.aten.add.Tensor(in2, v2);  in2 = v2 = None
    v6 = torch.ops.aten._fused_rms_norm.default(v4, [7168], in3, 1e-06);  in3 = None
    v7 = v6[0];  v6 = None
    v8 = torch.ops.torchao.mxfp8_quantize.default(v7, True, False, 32, 1, 'e4m3', 'rceil')
    v9 = v8[0]
    v10 = v8[2];  v8 = None
    v11 = torch.ops.torchao.triton_mx_block_rearrange.default(v10);  v10 = None
    v12 = torch.ops.torchao.mxfp8_quantize.default(v7, True, False, 32, 1, 'e4m3', 'rceil');  v7 = None
    v13 = v12[0]
    v14 = v12[2];  v12 = None
    v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
    return (v4, v9, v11, v13, v15)


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
