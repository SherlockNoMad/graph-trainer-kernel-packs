"""Kernel problem pattern_768ef1855e519a9496064134691e9c1b8a1b3a2bea36f438a6103753a6910197/a3i4ygdp3mshurshv75y7hzehvf77dwuf4n65jrspajm3dov3vou: region r023 of outputs/fusion_finder/dsv3_671b.

58 calls per step, backward, in layers.60.attention.wq_a, layers.59.attention.wq_a, layers.58.attention.wq_a ....
Measured 22.6 us per call in the capture; ideal (HBM roofline) 0.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0):
    v1 = torch.ops.aten.split_with_sizes.default(in0, [43008, 6, 147456, 21504, 2, 65536, 458752, 28, 28, 7168, 114688, 57344]);  in0 = None
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = v1[5]
    v8 = v1[6]
    v9 = v1[7]
    v10 = v1[8]
    v11 = v1[9]
    v12 = v1[10]
    v13 = v1[11];  v1 = None
    v14 = torch.ops.aten.view.default(v2, [6, 7168]);  v2 = None
    v15 = torch.ops.aten.view.default(v3, [6]);  v3 = None
    v16 = torch.ops.aten.view.default(v4, [96, 1536]);  v4 = None
    v17 = torch.ops.aten.view.default(v5, [3, 7168]);  v5 = None
    v18 = torch.ops.aten.view.default(v6, [2]);  v6 = None
    v19 = torch.ops.aten.view.default(v7, [128, 512]);  v7 = None
    v20 = torch.ops.aten.view.default(v8, [28, 16384]);  v8 = None
    v21 = torch.ops.aten.view.default(v9, [28]);  v9 = None
    v22 = torch.ops.aten.view.default(v10, [28]);  v10 = None
    v23 = torch.ops.aten.view.default(v11, [1, 7168]);  v11 = None
    v24 = torch.ops.aten.view.default(v12, [2, 8, 7168]);  v12 = None
    v25 = torch.ops.aten.view.default(v13, [28, 2048]);  v13 = None
    v26 = torch.ops.aten._to_copy.default(v14, dtype = torch.float32);  v14 = None
    v27 = torch.ops.aten._to_copy.default(v15, dtype = torch.float32);  v15 = None
    v28 = torch.ops.aten._to_copy.default(v16, dtype = torch.float32);  v16 = None
    v29 = torch.ops.aten._to_copy.default(v17, dtype = torch.float32);  v17 = None
    v30 = torch.ops.aten._to_copy.default(v18, dtype = torch.float32);  v18 = None
    v31 = torch.ops.aten._to_copy.default(v19, dtype = torch.float32);  v19 = None
    v32 = torch.ops.aten._to_copy.default(v20, dtype = torch.float32);  v20 = None
    v33 = torch.ops.aten._to_copy.default(v21, dtype = torch.float32);  v21 = None
    v34 = torch.ops.aten._to_copy.default(v22, dtype = torch.float32);  v22 = None
    v35 = torch.ops.aten._to_copy.default(v23, dtype = torch.float32);  v23 = None
    v36 = torch.ops.aten._to_copy.default(v24, dtype = torch.float32);  v24 = None
    v37 = torch.ops.aten._to_copy.default(v25, dtype = torch.float32);  v25 = None
    return (v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37)


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
