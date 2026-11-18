"""Kernel problem pattern_36cf508d2a71caf3c93d12c0bab13be72d84abb976149d49bb22e7be7eb9b39c/allkr7ngqnppciiqmf2xuumhyfbkfdiicd2u3y64jyvjb55jy2dv: region r000 of outputs/fusion_finder/dsv3_671b.

58 calls per step, forward, in layers.3.feed_forward.router, layers.4.feed_forward.router, layers.5.feed_forward.router ....
Measured 319.4 us per call in the capture; ideal (HBM roofline) 0.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from math import inf
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7):
    v1 = torch.ops.aten.sigmoid.default(in0);  in0 = None
    v3 = torch.ops.aten.add.Tensor(v1, in1);  in1 = None
    v4 = torch.ops.aten.reshape.default(v3, [4096, 8, 32]);  v3 = None
    v5 = torch.ops.aten.topk.default(v4, 2)
    v6 = v5[0];  v5 = None
    v7 = torch.ops.aten.sum.dim_IntList(v6, [-1]);  v6 = None
    v8 = torch.ops.aten.topk.default(v7, 4, -1, True, False)
    v9 = v8[1];  v8 = None
    v10 = torch.ops.aten.ones_like.default(v7, dtype = torch.bool, pin_memory = False);  v7 = None
    v11 = torch.ops.aten.scatter_.value(v10, -1, v9, False);  v10 = v9 = None
    v12 = torch.ops.aten.unsqueeze.default(v11, -1);  v11 = None
    v13 = torch.ops.aten.masked_fill.Scalar(v4, v12, -inf);  v4 = v12 = None
    v14 = torch.ops.aten.reshape.default(v13, [4096, 256]);  v13 = None
    v15 = torch.ops.aten.topk.default(v14, 8, -1, True, False);  v14 = None
    v16 = v15[1];  v15 = None
    v17 = torch.ops.aten.gather.default(v1, -1, v16)
    v18 = torch.ops.aten.sum.dim_IntList(v17, [-1], True)
    v19 = torch.ops.aten.add.Tensor(v18, 1e-20);  v18 = None
    v20 = torch.ops.aten.div.Tensor(v17, v19);  v17 = v19 = None
    v21 = torch.ops.aten.mul.Tensor(v20, 2.5);  v20 = None
    v22 = torch.ops.aten.zeros_like.default(v1, dtype = torch.bool, pin_memory = False)
    v23 = torch.ops.aten.scatter_.value(v22, -1, v16, True);  v22 = None
    v25 = torch.ops.aten.bitwise_not.default(in2);  in2 = None
    v26 = torch.ops.aten.bitwise_and.Tensor(v23, v25);  v23 = v25 = None
    v27 = torch.ops.aten.sum.dim_IntList(v26, [0])
    v29 = torch.ops.aten.add_.Tensor(in3, v27);  in3 = v27 = v29 = None
    v30 = torch.ops.aten._to_copy.default(v26, dtype = torch.float32)
    v31 = torch.ops.aten.sum.dim_IntList(v30, [0]);  v30 = None
    v32 = torch.ops.aten.linalg_vector_norm.default(v1, 1, [-1], True)
    v33 = torch.ops.aten.clamp_min.default(v32, 1e-12);  v32 = None
    v34 = torch.ops.aten.expand.default(v33, [4096, 256]);  v33 = None
    v35 = torch.ops.aten.div.Tensor(v1, v34);  v1 = v34 = None
    v37 = torch.ops.aten.bitwise_not.default(in4);  in4 = None
    v38 = torch.ops.aten.mul.Tensor(v35, v37);  v35 = v37 = None
    v39 = torch.ops.aten.sum.dim_IntList(v38, [0]);  v38 = None
    v40 = torch.ops.aten.linalg_vector_norm.default(v31, 1, [0], True)
    v41 = torch.ops.aten.clamp_min.default(v40, 1e-12);  v40 = None
    v42 = torch.ops.aten.expand.default(v41, [256]);  v41 = None
    v43 = torch.ops.aten.div.Tensor(v31, v42);  v31 = v42 = None
    v44 = torch.ops.aten.mul.Tensor(v43, 256);  v43 = None
    v45 = torch.ops.aten.mul.Tensor(v44, v39);  v44 = v39 = None
    v46 = torch.ops.aten.sum.default(v45);  v45 = None
    v48 = torch.ops.aten.clamp_min.default(in5, 1);  in5 = None
    v49 = torch.ops.aten.reciprocal.default(v48);  v48 = None
    v50 = torch.ops.aten.mul.Tensor(v49, 1.0);  v49 = None
    v51 = torch.ops.aten.mul.Tensor(v46, v50);  v46 = v50 = None
    v53 = torch.ops.aten.add_.Tensor(in6, v51);  in6 = v51 = v53 = None
    v55 = torch.ops.aten.masked_fill.Scalar(v16, in7, -1)
    v56 = torch.ops.aten.masked_fill.Scalar(v21, in7, 0.0);  v21 = in7 = None
    return (v16, v26, v55, v56)


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
