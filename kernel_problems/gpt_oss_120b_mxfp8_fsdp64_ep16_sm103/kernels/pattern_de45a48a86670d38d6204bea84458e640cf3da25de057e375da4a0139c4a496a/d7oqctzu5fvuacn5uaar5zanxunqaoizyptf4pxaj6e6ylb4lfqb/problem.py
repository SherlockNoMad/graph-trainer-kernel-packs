"""Kernel problem pattern_de45a48a86670d38d6204bea84458e640cf3da25de057e375da4a0139c4a496a/d7oqctzu5fvuacn5uaar5zanxunqaoizyptf4pxaj6e6ylb4lfqb: region r007 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.attention.wo, layers.34.attention.wo, layers.33.attention.wo ....
Measured 924.8 us per call in the capture; ideal (HBM roofline) 136.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3, in4, in5, in6):
    v1 = torch.ops.aten.new_zeros.default(in0, [16384, 2880], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v3 = torch.ops.aten.index_put.default(v1, [in1], in0, True);  v1 = in1 = in0 = None
    v5 = torch.ops.aten.add.Tensor(v3, in2);  v3 = in2 = None
    v7 = torch.ops.aten.alias.default(in3);  in3 = None
    v10 = torch.ops.aten._fused_rms_norm_backward.default(v5, in4, [2880], v7, in5, [True, True]);  v5 = in4 = v7 = in5 = None
    v11 = v10[0]
    v12 = v10[1];  v10 = None
    v14 = torch.ops.aten.add.Tensor(in6, v11);  in6 = v11 = None
    v15 = torch.ops.aten.sum.dim_IntList(v14, [0])
    v16 = torch.ops.torchao.mxfp8_quantize.default(v14, True, True, 32, 32, 'e4m3', 'rceil')
    v17 = v16[0]
    v18 = v16[1]
    v19 = v16[2]
    v20 = v16[3];  v16 = None
    v21 = torch.ops.torchao.triton_mx_block_rearrange.default(v19);  v19 = None
    v22 = torch.ops.torchao.triton_mx_block_rearrange.default(v20);  v20 = None
    v23 = torch.ops.aten.t.default(v18);  v18 = None
    return (v12, v14, v15, v17, v21, v22, v23)


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
