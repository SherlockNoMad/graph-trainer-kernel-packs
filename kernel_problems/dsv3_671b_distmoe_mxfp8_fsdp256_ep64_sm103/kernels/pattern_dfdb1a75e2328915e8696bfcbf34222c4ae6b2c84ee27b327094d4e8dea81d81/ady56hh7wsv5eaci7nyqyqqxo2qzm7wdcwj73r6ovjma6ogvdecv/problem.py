"""Kernel problem pattern_dfdb1a75e2328915e8696bfcbf34222c4ae6b2c84ee27b327094d4e8dea81d81/ady56hh7wsv5eaci7nyqyqqxo2qzm7wdcwj73r6ovjma6ogvdecv: region r009 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward/recompute, in layers.60.attention.wq_a, layers.59.attention.wq_a, layers.58.attention.wq_a ....
Measured 99.2 us per call in the capture; ideal (HBM roofline) 34.9 us.

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
    v2 = torch.ops.aten._fused_rms_norm.default(in0, [7168], in1, 1e-06);  in0 = in1 = None
    v3 = v2[0]
    v4 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
    v5 = v4[0]
    v6 = v4[2];  v4 = None
    v7 = torch.ops.torchao.triton_mx_block_rearrange.default(v6);  v6 = None
    v8 = v2[1];  v2 = None
    v9 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
    v10 = v9[0]
    v11 = v9[2];  v9 = None
    v12 = torch.ops.torchao.triton_mx_block_rearrange.default(v11);  v11 = None
    v13 = torch.ops.torchao.mxfp8_quantize.default(v3, False, True, 1, 32, 'e4m3', 'rceil')
    v14 = v13[1]
    v15 = v13[3];  v13 = None
    v16 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
    v17 = torch.ops.torchao.mxfp8_quantize.default(v3, False, True, 1, 32, 'e4m3', 'rceil');  v3 = None
    v18 = v17[1]
    v19 = v17[3];  v17 = None
    v20 = torch.ops.torchao.triton_mx_block_rearrange.default(v19);  v19 = None
    v21 = torch.ops.aten.alias.default(v8);  v8 = None
    v22 = torch.ops.aten.alias.default(v21);  v21 = None
    return (v5, v7, v10, v12, v14, v16, v18, v20, v22)


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
