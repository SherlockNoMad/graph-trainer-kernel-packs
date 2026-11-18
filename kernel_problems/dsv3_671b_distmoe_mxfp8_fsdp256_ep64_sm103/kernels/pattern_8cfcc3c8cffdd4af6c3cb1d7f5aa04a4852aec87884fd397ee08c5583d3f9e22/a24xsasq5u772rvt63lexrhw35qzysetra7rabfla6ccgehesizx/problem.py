"""Kernel problem pattern_8cfcc3c8cffdd4af6c3cb1d7f5aa04a4852aec87884fd397ee08c5583d3f9e22/a24xsasq5u772rvt63lexrhw35qzysetra7rabfla6ccgehesizx: region r002 of outputs/fusion_finder/dsv3_671b.

58 calls per step, backward, in layers.60.feed_forward.router.gate, layers.59.feed_forward.router.gate, layers.58.feed_forward.router.gate ....
Measured 179.8 us per call in the capture; ideal (HBM roofline) 2.2 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4, in5, in6):
    v2 = torch.ops.aten.masked_fill.Scalar(in0, in1, 0);  in0 = in1 = None
    v3 = torch.ops.aten.mul.Tensor(v2, 2.5);  v2 = None
    v4 = torch.ops.aten.neg.default(v3)
    v6 = torch.ops.aten.mul.Tensor(v4, in2);  v4 = in2 = None
    v8 = torch.ops.aten.div.Tensor(v3, in3);  v3 = in3 = None
    v9 = torch.ops.aten.sum.dim_IntList(v6, [1], True);  v6 = None
    v10 = torch.ops.aten.expand.default(v9, [4096, 8]);  v9 = None
    v11 = torch.ops.aten.add.Tensor(v8, v10);  v8 = v10 = None
    v12 = torch.ops.aten.new_zeros.default(v11, [4096, 256])
    v14 = torch.ops.aten.scatter_add.default(v12, -1, in4, v11);  v12 = in4 = v11 = None
    v16 = torch.ops.aten.add.Tensor(in5, v14);  in5 = v14 = None
    v18 = torch.ops.aten.sigmoid_backward.default(v16, in6);  v16 = in6 = None
    v19 = torch.ops.aten._to_copy.default(v18, dtype = torch.bfloat16)
    v20 = torch.ops.aten.view.dtype(v19, torch.int16)
    v21 = torch.ops.aten.alias.default(v20);  v20 = None
    v22 = torch.ops.aten._to_copy.default(v21, dtype = torch.int32);  v21 = None
    v23 = torch.ops.aten.__lshift__.Scalar(v22, 16);  v22 = None
    v24 = torch.ops.aten.view.dtype(v23, torch.float32);  v23 = None
    v25 = torch.ops.aten.alias.default(v24);  v24 = None
    v26 = torch.ops.aten.sub.Tensor(v18, v25);  v18 = v25 = None
    v27 = torch.ops.aten._to_copy.default(v26, dtype = torch.bfloat16)
    v28 = torch.ops.aten.view.dtype(v27, torch.int16)
    v29 = torch.ops.aten.alias.default(v28);  v28 = None
    v30 = torch.ops.aten._to_copy.default(v29, dtype = torch.int32);  v29 = None
    v31 = torch.ops.aten.__lshift__.Scalar(v30, 16);  v30 = None
    v32 = torch.ops.aten.view.dtype(v31, torch.float32);  v31 = None
    v33 = torch.ops.aten.alias.default(v32);  v32 = None
    v34 = torch.ops.aten.sub.Tensor(v26, v33);  v26 = v33 = None
    v35 = torch.ops.aten._to_copy.default(v34, dtype = torch.bfloat16);  v34 = None
    v36 = torch.ops.aten.cat.default([v19, v27, v35], 1);  v19 = v27 = v35 = None
    v37 = torch.ops.aten.split.Tensor(v36, 256, 1)
    v38 = v37[0]
    v39 = v37[1]
    v40 = v37[2];  v37 = None
    v41 = torch.ops.aten.permute.default(v38, [1, 0]);  v38 = None
    v42 = torch.ops.aten.permute.default(v39, [1, 0]);  v39 = None
    v43 = torch.ops.aten.permute.default(v40, [1, 0]);  v40 = None
    return (v36, v41, v42, v43)


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
