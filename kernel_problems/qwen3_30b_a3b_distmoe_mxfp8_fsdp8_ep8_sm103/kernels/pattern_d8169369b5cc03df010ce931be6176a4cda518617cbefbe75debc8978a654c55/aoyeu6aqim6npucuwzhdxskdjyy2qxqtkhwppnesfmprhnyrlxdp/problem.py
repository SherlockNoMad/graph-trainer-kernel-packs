"""Kernel problem pattern_d8169369b5cc03df010ce931be6176a4cda518617cbefbe75debc8978a654c55/aoyeu6aqim6npucuwzhdxskdjyy2qxqtkhwppnesfmprhnyrlxdp: region r017 of outputs/fusion_finder/qwen3_30b.

48 calls per step, backward, in layers.47.attention.qkv_linear.wqkv, layers.46.attention.qkv_linear.wqkv, layers.45.attention.qkv_linear.wqkv ....
Measured 14.0 us per call in the capture; ideal (HBM roofline) 2.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0):
    v1 = torch.ops.aten.split_with_sizes.default(in0, [1310720, 1048576, 16, 16, 32768, 256, 256]);  in0 = None
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = v1[5]
    v8 = v1[6];  v1 = None
    v9 = torch.ops.aten.view.default(v2, [640, 2048]);  v2 = None
    v10 = torch.ops.aten.view.default(v3, [256, 4096]);  v3 = None
    v11 = torch.ops.aten.view.default(v4, [16]);  v4 = None
    v12 = torch.ops.aten.view.default(v5, [16]);  v5 = None
    v13 = torch.ops.aten.view.default(v6, [16, 2048]);  v6 = None
    v14 = torch.ops.aten.view.default(v7, [256]);  v7 = None
    v15 = torch.ops.aten.view.default(v8, [256]);  v8 = None
    v16 = torch.ops.aten._to_copy.default(v9, dtype = torch.float32);  v9 = None
    v17 = torch.ops.aten._to_copy.default(v10, dtype = torch.float32);  v10 = None
    v18 = torch.ops.aten._to_copy.default(v11, dtype = torch.float32);  v11 = None
    v19 = torch.ops.aten._to_copy.default(v12, dtype = torch.float32);  v12 = None
    v20 = torch.ops.aten._to_copy.default(v13, dtype = torch.float32);  v13 = None
    v21 = torch.ops.aten._to_copy.default(v14, dtype = torch.float32);  v14 = None
    v22 = torch.ops.aten._to_copy.default(v15, dtype = torch.float32);  v15 = None
    return (v16, v17, v18, v19, v20, v21, v22)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0):
        return reference(in0)


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
