"""Kernel problem pattern_b36ad5e2c46f5b1ee4ff2bf82cdaaca3077f3e2f11eb893ffdb170c06e80d0c5/aiq7ufb4euw4kildrhmzh4ohzxpz24i7opmrh3fpk5ynaqdf6bdw: region r028 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward/recompute, in layers.60.attention.wq_b, layers.59.attention.wq_b, layers.58.attention.wq_b ....
Measured 19.2 us per call in the capture; ideal (HBM roofline) 5.5 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1):
    v2 = torch.ops.aten._fused_rms_norm.default(in0, [1536], in1, 1e-06);  in0 = in1 = None
    v3 = v2[0]
    v4 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
    v5 = v4[0]
    v6 = v4[2];  v4 = None
    v7 = torch.ops.torchao.triton_mx_block_rearrange.default(v6);  v6 = None
    v8 = v2[1];  v2 = None
    v9 = torch.ops.torchao.mxfp8_quantize.default(v3, False, True, 1, 32, 'e4m3', 'rceil');  v3 = None
    v10 = v9[1]
    v11 = v9[3];  v9 = None
    v12 = torch.ops.torchao.triton_mx_block_rearrange.default(v11);  v11 = None
    v13 = torch.ops.aten.alias.default(v8);  v8 = None
    v14 = torch.ops.aten.alias.default(v13);  v13 = None
    return (v5, v7, v10, v12, v14)


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
