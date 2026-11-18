"""Kernel problem pattern_ff076517a275d8a8a99316de1bc48645132ad0c210cf51ed5ba9d51bb024030a/d24u23dxe5c4t6n6d64fh7qramk24qinees7xd67akpqt6p4l4o6: region r000 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.feed_forward.routed_experts, layers.34.feed_forward.routed_experts, layers.33.feed_forward.routed_experts ....
Measured 18480.3 us per call in the capture; ideal (HBM roofline) 543.7 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
    v1 = torch.ops.aten.mul.Scalar(in0, 1);  in0 = None
    v3 = torch.ops.aten.mul.Tensor(in1, v1);  v1 = None
    v5 = torch.ops.aten.mul.Scalar(in2, 1);  in2 = None
    v6 = torch.ops.aten.mul.Tensor(in1, v5);  v5 = None
    v7 = torch.ops.aten.add.Tensor(in1, v3);  in1 = v3 = None
    v9 = torch.ops.aten.mul.Tensor(v7, in3);  in3 = None
    v11 = torch.ops.aten.mul.Tensor(v7, in4);  v7 = in4 = None
    v13 = torch.ops.aten.alias.default(in5);  in5 = None
    v14 = torch.ops.aten.sigmoid_backward.default(v9, v13);  v9 = v13 = None
    v15 = torch.ops.aten.mul.Tensor(v14, 1.702);  v14 = None
    v16 = torch.ops.aten.add.Tensor(v11, v15);  v11 = v15 = None
    v17 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v19 = torch.ops.aten.gt.Scalar(in6, -7.0)
    v20 = torch.ops.aten.lt.Scalar(in6, 7.0);  in6 = None
    v21 = torch.ops.aten.logical_and_.default(v19, v20);  v19 = v20 = None
    v22 = torch.ops.aten.where.self(v21, v6, v17);  v21 = v6 = v17 = None
    v23 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v25 = torch.ops.aten.lt.Scalar(in7, 7.0);  in7 = None
    v26 = torch.ops.aten.where.self(v25, v16, v23);  v25 = v16 = v23 = None
    v27 = torch.ops.aten.stack.default([v26, v22], -2);  v26 = v22 = None
    v29 = torch.ops.aten.reshape.default(v27, [in8, 5760]);  in8 = None
    v30 = torch.ops.aten.new_zeros.default(v29, [9, 5760], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v32 = torch.ops.aten.index_add.default(v30, 0, in9, v29);  v30 = in9 = v29 = None
    v33 = torch.ops.aten.slice.Tensor(v32, 0, 0, 8);  v32 = None
    v34 = torch.ops.aten.reshape.default(v33, [8, 2, 2880]);  v33 = None
    v36 = torch.ops.aten.reshape.default(v27, [in10, 5760]);  v27 = in10 = None
    v37 = torch.ops.aten.transpose.int(v36, -2, -1)
    return (v34, v36, v37)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10)


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
