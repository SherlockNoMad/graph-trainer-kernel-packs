"""Kernel problem pattern_afda0aa6f89f7a1cf325ae36e7027617a0e1cf1c129e18590f92a24654f300a0/acddqoujm5shnv5dplshpvjwmhekkidxzihqlrtwnfnjqyh3kfcg: region r005 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.attention.rope, layers.34.attention.rope, layers.33.attention.rope ....
Measured 1255.1 us per call in the capture; ideal (HBM roofline) 51.0 us.

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


def reference(in0, in1, in2, in3, in4):
    v1 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in0 = None
    v3 = torch.ops.aten._to_copy.default(in1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in1 = None
    v5 = torch.ops.aten.mul.Tensor(v1, in2)
    v6 = torch.ops.aten.slice.Tensor(v5, 2, 0, 32)
    v7 = torch.ops.aten.slice.Tensor(v5, 2, 32, 64);  v5 = None
    v8 = torch.ops.aten.neg.default(v6);  v6 = None
    v9 = torch.ops.aten.slice_backward.default(v8, [16384, 8, 64], 2, 32, 9223372036854775807, 1);  v8 = None
    v10 = torch.ops.aten.slice_backward.default(v7, [16384, 8, 64], 2, 0, 32, 1);  v7 = None
    v11 = torch.ops.aten.add.Tensor(v9, v10);  v9 = v10 = None
    v13 = torch.ops.aten.mul.Tensor(v1, in3);  v1 = None
    v14 = torch.ops.aten.add.Tensor(v11, v13);  v11 = v13 = None
    v15 = torch.ops.aten._to_copy.default(v14, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v14 = None
    v16 = torch.ops.aten.mul.Tensor(v3, in2);  in2 = None
    v17 = torch.ops.aten.slice.Tensor(v16, 2, 0, 32)
    v18 = torch.ops.aten.slice.Tensor(v16, 2, 32, 64);  v16 = None
    v19 = torch.ops.aten.neg.default(v17);  v17 = None
    v20 = torch.ops.aten.slice_backward.default(v19, [16384, 64, 64], 2, 32, 9223372036854775807, 1);  v19 = None
    v21 = torch.ops.aten.slice_backward.default(v18, [16384, 64, 64], 2, 0, 32, 1);  v18 = None
    v22 = torch.ops.aten.add.Tensor(v20, v21);  v20 = v21 = None
    v23 = torch.ops.aten.mul.Tensor(v3, in3);  v3 = in3 = None
    v24 = torch.ops.aten.add.Tensor(v22, v23);  v22 = v23 = None
    v25 = torch.ops.aten._to_copy.default(v24, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v24 = None
    v27 = torch.ops.aten.reshape.default(in4, [16384, 8, 1, 64]);  in4 = None
    v28 = torch.ops.aten.reshape.default(v15, [16384, 8, 1, 64]);  v15 = None
    v29 = torch.ops.aten.reshape.default(v25, [16384, 8, 8, 64]);  v25 = None
    v30 = torch.ops.aten.cat.default([v29, v28, v27], 2);  v29 = v28 = v27 = None
    v31 = torch.ops.aten.reshape.default(v30, [16384, 5120]);  v30 = None
    v32 = torch.ops.aten.sum.dim_IntList(v31, [0])
    v33 = torch.ops.torchao.mxfp8_quantize.default(v31, True, True, 32, 32, 'e4m3', 'rceil');  v31 = None
    v34 = v33[0]
    v35 = v33[1]
    v36 = v33[2]
    v37 = v33[3];  v33 = None
    v38 = torch.ops.torchao.triton_mx_block_rearrange.default(v36);  v36 = None
    v39 = torch.ops.torchao.triton_mx_block_rearrange.default(v37);  v37 = None
    v40 = torch.ops.aten.t.default(v35);  v35 = None
    return (v32, v34, v38, v39, v40)


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
