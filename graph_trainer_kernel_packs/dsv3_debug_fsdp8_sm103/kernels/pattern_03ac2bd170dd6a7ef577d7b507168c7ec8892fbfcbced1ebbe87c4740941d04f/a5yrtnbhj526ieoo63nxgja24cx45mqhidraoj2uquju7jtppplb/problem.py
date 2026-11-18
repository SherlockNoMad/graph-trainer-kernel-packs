"""Kernel problem pattern_03ac2bd170dd6a7ef577d7b507168c7ec8892fbfcbced1ebbe87c4740941d04f/a5yrtnbhj526ieoo63nxgja24cx45mqhidraoj2uquju7jtppplb: region r002 of outputs/fusion_finder/dsv3_debug.

5 calls per step, backward, in layers.5.feed_forward.router.gate, layers.4.feed_forward.router.gate, layers.3.feed_forward.router.gate ....
Measured 322.7 us per call in the capture; ideal (HBM roofline) 11.3 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7):
    v2 = torch.ops.aten.mul.Tensor(in0, in1);  in0 = in1 = None
    v3 = torch.ops.aten.sum.dim_IntList(v2, [1], True);  v2 = None
    v4 = torch.ops.aten.reshape.default(v3, [49152]);  v3 = None
    v5 = torch.ops.aten.new_zeros.default(v4, [49152], dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v7 = torch.ops.aten.index_put.default(v5, [in2], v4, True);  v5 = in2 = v4 = None
    v8 = torch.ops.aten.reshape.default(v7, [16384, 3]);  v7 = None
    v9 = torch.ops.aten.mul.Tensor(v8, 1.0);  v8 = None
    v10 = torch.ops.aten.neg.default(v9)
    v12 = torch.ops.aten.mul.Tensor(v10, in3);  v10 = in3 = None
    v14 = torch.ops.aten.div.Tensor(v9, in4);  v9 = in4 = None
    v15 = torch.ops.aten.sum.dim_IntList(v12, [1], True);  v12 = None
    v16 = torch.ops.aten.expand.default(v15, [16384, 3]);  v15 = None
    v17 = torch.ops.aten.add.Tensor(v14, v16);  v14 = v16 = None
    v18 = torch.ops.aten.new_zeros.default(v17, [16384, 8])
    v20 = torch.ops.aten.scatter_add.default(v18, -1, in5, v17);  v18 = in5 = v17 = None
    v22 = torch.ops.aten.add.Tensor(in6, v20);  in6 = v20 = None
    v24 = torch.ops.aten.sigmoid_backward.default(v22, in7);  v22 = in7 = None
    v25 = torch.ops.aten._to_copy.default(v24, dtype = torch.bfloat16)
    v26 = torch.ops.aten.view.dtype(v25, torch.int16)
    v27 = torch.ops.aten.alias.default(v26);  v26 = None
    v28 = torch.ops.aten._to_copy.default(v27, dtype = torch.int32);  v27 = None
    v29 = torch.ops.aten.__lshift__.Scalar(v28, 16);  v28 = None
    v30 = torch.ops.aten.view.dtype(v29, torch.float32);  v29 = None
    v31 = torch.ops.aten.alias.default(v30);  v30 = None
    v32 = torch.ops.aten.sub.Tensor(v24, v31);  v24 = v31 = None
    v33 = torch.ops.aten._to_copy.default(v32, dtype = torch.bfloat16)
    v34 = torch.ops.aten.view.dtype(v33, torch.int16)
    v35 = torch.ops.aten.alias.default(v34);  v34 = None
    v36 = torch.ops.aten._to_copy.default(v35, dtype = torch.int32);  v35 = None
    v37 = torch.ops.aten.__lshift__.Scalar(v36, 16);  v36 = None
    v38 = torch.ops.aten.view.dtype(v37, torch.float32);  v37 = None
    v39 = torch.ops.aten.alias.default(v38);  v38 = None
    v40 = torch.ops.aten.sub.Tensor(v32, v39);  v32 = v39 = None
    v41 = torch.ops.aten._to_copy.default(v40, dtype = torch.bfloat16);  v40 = None
    v42 = torch.ops.aten.cat.default([v25, v33, v41], 1);  v25 = v33 = v41 = None
    v43 = torch.ops.aten.split.Tensor(v42, 8, 1)
    v44 = v43[0]
    v45 = v43[1]
    v46 = v43[2];  v43 = None
    v47 = torch.ops.aten.permute.default(v44, [1, 0]);  v44 = None
    v48 = torch.ops.aten.permute.default(v45, [1, 0]);  v45 = None
    v49 = torch.ops.aten.permute.default(v46, [1, 0]);  v46 = None
    return (v42, v47, v48, v49)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7)


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
