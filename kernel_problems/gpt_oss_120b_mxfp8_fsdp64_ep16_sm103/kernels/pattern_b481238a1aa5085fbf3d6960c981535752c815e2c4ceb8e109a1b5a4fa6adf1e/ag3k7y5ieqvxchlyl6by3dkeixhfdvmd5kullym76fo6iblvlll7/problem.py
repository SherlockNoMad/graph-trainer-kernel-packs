"""Kernel problem pattern_b481238a1aa5085fbf3d6960c981535752c815e2c4ceb8e109a1b5a4fa6adf1e/ag3k7y5ieqvxchlyl6by3dkeixhfdvmd5kullym76fo6iblvlll7: region r006 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.attention.rope, layers.1.attention.rope, layers.2.attention.rope ....
Measured 1113.3 us per call in the capture; ideal (HBM roofline) 49.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2):
    v1 = torch.ops.aten.reshape.default(in0, [16384, -1, 10, 64]);  in0 = None
    v2 = torch.ops.aten.split_with_sizes.default(v1, [8, 1, 1], -2);  v1 = None
    v3 = v2[0]
    v4 = v2[1]
    v5 = v2[2];  v2 = None
    v6 = torch.ops.aten.clone.default(v3, memory_format = torch.contiguous_format);  v3 = None
    v7 = torch.ops.aten.reshape.default(v6, [16384, 64, 64]);  v6 = None
    v8 = torch.ops.aten.reshape.default(v4, [16384, 8, 64]);  v4 = None
    v9 = torch.ops.aten.clone.default(v8, memory_format = torch.contiguous_format);  v8 = None
    v10 = torch.ops.aten.reshape.default(v5, [16384, 8, 64]);  v5 = None
    v11 = torch.ops.aten.clone.default(v10, memory_format = torch.contiguous_format);  v10 = None
    v14 = torch.ops.aten.index.Tensor(in1, [in2]);  in1 = in2 = None
    v15 = torch.ops.aten.reshape.default(v14, [16384, 1, 128]);  v14 = None
    v16 = torch.ops.aten.slice.Tensor(v15, 2, 0, 64)
    v17 = torch.ops.aten.slice.Tensor(v15, 2, 64, 9223372036854775807);  v15 = None
    v18 = torch.ops.aten._to_copy.default(v7, dtype = torch.float32);  v7 = None
    v19 = torch.ops.aten.mul.Tensor(v18, v16)
    v20 = torch.ops.aten.slice.Tensor(v18, 2, 0, 32)
    v21 = torch.ops.aten.slice.Tensor(v18, 2, 32, 9223372036854775807);  v18 = None
    v22 = torch.ops.aten.neg.default(v21);  v21 = None
    v23 = torch.ops.aten.cat.default([v22, v20], -1);  v22 = v20 = None
    v24 = torch.ops.aten.mul.Tensor(v23, v17);  v23 = None
    v25 = torch.ops.aten.add.Tensor(v19, v24);  v19 = v24 = None
    v26 = torch.ops.aten._to_copy.default(v9, dtype = torch.float32);  v9 = None
    v27 = torch.ops.aten.mul.Tensor(v26, v16)
    v28 = torch.ops.aten.slice.Tensor(v26, 2, 0, 32)
    v29 = torch.ops.aten.slice.Tensor(v26, 2, 32, 9223372036854775807);  v26 = None
    v30 = torch.ops.aten.neg.default(v29);  v29 = None
    v31 = torch.ops.aten.cat.default([v30, v28], -1);  v30 = v28 = None
    v32 = torch.ops.aten.mul.Tensor(v31, v17);  v31 = None
    v33 = torch.ops.aten.add.Tensor(v27, v32);  v27 = v32 = None
    v34 = torch.ops.aten._to_copy.default(v25, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v25 = None
    v35 = torch.ops.aten._to_copy.default(v33, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v33 = None
    return (v11, v16, v17, v34, v35)


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
