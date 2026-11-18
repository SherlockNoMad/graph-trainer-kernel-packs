"""Kernel problem pattern_e44d5ecb45eba92a8031ea84cf737c1a55d5c3308611c527099d13c008c9eab2/azgjfmra5kmdc6gjp7uxryx7nqyv675opn64z3h7ddtrjab4caqf: region r001 of outputs/fusion_finder/dsv3_debug.

6 calls per step, forward, in layers.0.attention.rope, layers.1.attention.rope, layers.2.attention.rope ....
Measured 313.7 us per call in the capture; ideal (HBM roofline) 34.5 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4):
    v1 = torch.ops.aten.reshape.default(in0, [16384, -1, 192]);  in0 = None
    v3 = torch.ops.aten.split_with_sizes.default(in1, [512, 64], -1);  in1 = None
    v4 = v3[0]
    v5 = v3[1];  v3 = None
    v6 = torch.ops.aten.unsqueeze.default(v5, 1);  v5 = None
    v7 = torch.ops.aten.split_with_sizes.default(v1, [128, 64], -1);  v1 = None
    v8 = v7[0]
    v9 = v7[1];  v7 = None
    v12 = torch.ops.aten.index.Tensor(in2, [in3]);  in2 = in3 = None
    v13 = torch.ops.aten.reshape.default(v12, [16384, 1, 32]);  v12 = None
    v14 = torch.ops.aten._to_copy.default(v9, dtype = torch.float32);  v9 = None
    v15 = torch.ops.aten.reshape.default(v14, [16384, 16, -1, 2]);  v14 = None
    v16 = torch.ops.aten.view_as_complex.default(v15);  v15 = None
    v17 = torch.ops.aten.mul.Tensor(v16, v13);  v16 = None
    v18 = torch.ops.aten.view_as_real.default(v17);  v17 = None
    v19 = torch.ops.aten.reshape.default(v18, [16384, 16, 64]);  v18 = None
    v20 = torch.ops.aten._to_copy.default(v19, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v19 = None
    v21 = torch.ops.aten._to_copy.default(v6, dtype = torch.float32);  v6 = None
    v22 = torch.ops.aten.reshape.default(v21, [16384, 1, -1, 2]);  v21 = None
    v23 = torch.ops.aten.view_as_complex.default(v22);  v22 = None
    v24 = torch.ops.aten.mul.Tensor(v23, v13);  v23 = v13 = None
    v25 = torch.ops.aten.view_as_real.default(v24);  v24 = None
    v26 = torch.ops.aten.reshape.default(v25, [16384, 1, 64]);  v25 = None
    v27 = torch.ops.aten._to_copy.default(v26, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v26 = None
    v28 = torch.ops.aten.cat.default([v8, v20], -1);  v8 = v20 = None
    v30 = torch.ops.aten._fused_rms_norm.default(v4, [512], in4, 1e-06);  v4 = in4 = None
    v31 = v30[0];  v30 = None
    v32 = torch.ops.aten.squeeze.dim(v27, 1);  v27 = None
    v33 = torch.ops.aten.unsqueeze.default(v32, 1);  v32 = None
    v34 = torch.ops.aten.expand.default(v33, [-1, 16, -1]);  v33 = None
    v35 = torch.ops.aten.transpose.int(v28, 0, 1);  v28 = None
    v36 = torch.ops.aten.unsqueeze.default(v35, 0);  v35 = None
    return (v31, v34, v36)


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
