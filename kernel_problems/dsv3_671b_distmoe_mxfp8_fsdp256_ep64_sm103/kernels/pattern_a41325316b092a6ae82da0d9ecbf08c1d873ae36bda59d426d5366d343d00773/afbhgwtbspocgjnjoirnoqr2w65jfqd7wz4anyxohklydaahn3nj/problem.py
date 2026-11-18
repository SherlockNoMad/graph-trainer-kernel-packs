"""Kernel problem pattern_a41325316b092a6ae82da0d9ecbf08c1d873ae36bda59d426d5366d343d00773/afbhgwtbspocgjnjoirnoqr2w65jfqd7wz4anyxohklydaahn3nj: region r005 of outputs/fusion_finder/dsv3_671b.

58 calls per step, backward, in layers.60.attention.wo, layers.59.attention.wo, layers.58.attention.wo ....
Measured 146.9 us per call in the capture; ideal (HBM roofline) 68.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3, in4, in5, in6):
    v2 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
    v4 = torch.ops.aten.add.Tensor(v2, in2);  v2 = in2 = None
    v8 = torch.ops.aten._fused_rms_norm_backward.default(v4, in3, [7168], in4, in5, [True, True]);  v4 = in3 = in4 = in5 = None
    v9 = v8[0]
    v11 = torch.ops.aten.add.Tensor(in6, v9);  in6 = v9 = None
    v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, True, 32, 32, 'e4m3', 'rceil')
    v13 = v12[0]
    v14 = v12[1]
    v15 = torch.ops.aten.t.default(v14);  v14 = None
    v16 = v12[2]
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
    v18 = v12[3];  v12 = None
    v19 = torch.ops.torchao.triton_mx_block_rearrange.default(v18);  v18 = None
    v20 = v8[1];  v8 = None
    return (v11, v13, v15, v17, v19, v20)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6):
        return reference(in0, in1, in2, in3, in4, in5, in6)


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
