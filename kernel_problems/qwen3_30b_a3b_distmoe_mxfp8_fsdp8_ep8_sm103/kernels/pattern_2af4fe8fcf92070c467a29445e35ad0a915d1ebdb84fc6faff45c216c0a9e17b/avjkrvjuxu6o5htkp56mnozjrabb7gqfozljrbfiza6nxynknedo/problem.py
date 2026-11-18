"""Kernel problem pattern_2af4fe8fcf92070c467a29445e35ad0a915d1ebdb84fc6faff45c216c0a9e17b/avjkrvjuxu6o5htkp56mnozjrabb7gqfozljrbfiza6nxynknedo: region r001 of outputs/fusion_finder/qwen3_30b.

48 calls per step, forward, in layers.0.attention.rope, layers.1.attention.rope, layers.2.attention.rope ....
Measured 2307.9 us per call in the capture; ideal (HBM roofline) 189.6 us.

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
    v1 = torch.ops.aten.reshape.default(in0, [32768, -1, 10, 128]);  in0 = None
    v2 = torch.ops.aten.split_with_sizes.default(v1, [8, 1, 1], -2);  v1 = None
    v3 = v2[0]
    v4 = v2[1]
    v5 = v2[2];  v2 = None
    v6 = torch.ops.aten.clone.default(v3, memory_format = torch.contiguous_format);  v3 = None
    v7 = torch.ops.aten.reshape.default(v6, [32768, 32, 128]);  v6 = None
    v8 = torch.ops.aten.reshape.default(v4, [32768, 4, 128]);  v4 = None
    v9 = torch.ops.aten.clone.default(v8, memory_format = torch.contiguous_format);  v8 = None
    v10 = torch.ops.aten.reshape.default(v5, [32768, 4, 128]);  v5 = None
    v11 = torch.ops.aten.clone.default(v10, memory_format = torch.contiguous_format);  v10 = None
    v13 = torch.ops.aten._fused_rms_norm.default(v7, [128], in1, 1e-06);  in1 = None
    v14 = v13[0]
    v15 = v13[1];  v13 = None
    v16 = torch.ops.aten.alias.default(v15);  v15 = None
    v18 = torch.ops.aten._fused_rms_norm.default(v9, [128], in2, 1e-06);  in2 = None
    v19 = v18[0]
    v20 = v18[1];  v18 = None
    v21 = torch.ops.aten.alias.default(v20);  v20 = None
    v24 = torch.ops.aten.index.Tensor(in3, [in4]);  in3 = in4 = None
    v25 = torch.ops.aten.reshape.default(v24, [32768, 1, 256]);  v24 = None
    v26 = torch.ops.aten.slice.Tensor(v25, 2, 0, 128)
    v27 = torch.ops.aten.slice.Tensor(v25, 2, 128, 9223372036854775807);  v25 = None
    v28 = torch.ops.aten._to_copy.default(v14, dtype = torch.float32);  v14 = None
    v29 = torch.ops.aten.mul.Tensor(v28, v26)
    v30 = torch.ops.aten.slice.Tensor(v28, 2, 0, 64)
    v31 = torch.ops.aten.slice.Tensor(v28, 2, 64, 9223372036854775807);  v28 = None
    v32 = torch.ops.aten.neg.default(v31);  v31 = None
    v33 = torch.ops.aten.cat.default([v32, v30], -1);  v32 = v30 = None
    v34 = torch.ops.aten.mul.Tensor(v33, v27);  v33 = None
    v35 = torch.ops.aten.add.Tensor(v29, v34);  v29 = v34 = None
    v36 = torch.ops.aten._to_copy.default(v19, dtype = torch.float32);  v19 = None
    v37 = torch.ops.aten.mul.Tensor(v36, v26)
    v38 = torch.ops.aten.slice.Tensor(v36, 2, 0, 64)
    v39 = torch.ops.aten.slice.Tensor(v36, 2, 64, 9223372036854775807);  v36 = None
    v40 = torch.ops.aten.neg.default(v39);  v39 = None
    v41 = torch.ops.aten.cat.default([v40, v38], -1);  v40 = v38 = None
    v42 = torch.ops.aten.mul.Tensor(v41, v27);  v41 = None
    v43 = torch.ops.aten.add.Tensor(v37, v42);  v37 = v42 = None
    v44 = torch.ops.aten._to_copy.default(v35, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v35 = None
    v45 = torch.ops.aten._to_copy.default(v43, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v43 = None
    return (v7, v9, v11, v16, v21, v26, v27, v44, v45)


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
