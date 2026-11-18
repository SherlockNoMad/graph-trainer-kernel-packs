"""Kernel problem pattern_1156f9a94b59fc4d8aacdcc54ba30fda45e8b451dd96be3ee447fd6a2190718c/ag25dcznlyfkce2yeitm5ukdi37tvergm55oo6joy33wozlqmoze: region r004 of outputs/fusion_finder/qwen3_30b.

48 calls per step, forward, in layers.0.feed_forward.router, layers.1.feed_forward.router, layers.2.feed_forward.router ....
Measured 236.8 us per call in the capture; ideal (HBM roofline) 5.8 us.

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
    v1 = torch.ops.aten._softmax.default(in0, -1, False);  in0 = None
    v2 = torch.ops.aten.alias.default(v1)
    v3 = torch.ops.aten.arange.default(262144, dtype = torch.int64, device = device(type='cuda'), pin_memory = False)
    v4 = torch.ops.aten.reshape.default(v3, [32768, 8]);  v3 = None
    v5 = torch.ops.aten.remainder.Scalar(v4, 128);  v4 = None
    v6 = torch.ops.aten.gather.default(v1, -1, v5)
    v7 = torch.ops.aten.sum.dim_IntList(v6, [-1], True)
    v8 = torch.ops.aten.add.Tensor(v7, 1e-20);  v7 = None
    v9 = torch.ops.aten.div.Tensor(v6, v8)
    v10 = torch.ops.aten.mul.Tensor(v9, 1.0);  v9 = None
    v11 = torch.ops.aten.zeros_like.default(v1, dtype = torch.bool, pin_memory = False)
    v12 = torch.ops.aten.scatter_.value(v11, -1, v5, True);  v11 = None
    v14 = torch.ops.aten.bitwise_not.default(in1);  in1 = None
    v15 = torch.ops.aten.bitwise_and.Tensor(v12, v14);  v12 = v14 = None
    v16 = torch.ops.aten.sum.dim_IntList(v15, [0])
    v18 = torch.ops.aten.add_.Tensor(in2, v16);  in2 = v16 = v18 = None
    v19 = torch.ops.aten._to_copy.default(v15, dtype = torch.float32);  v15 = None
    v20 = torch.ops.aten.sum.dim_IntList(v19, [0]);  v19 = None
    v21 = torch.ops.aten.view.default(v20, [-1]);  v20 = None
    v22 = torch.ops.aten.linalg_vector_norm.default(v1, 1, [-1], True)
    v23 = torch.ops.aten.clamp_min.default(v22, 1e-12)
    v24 = torch.ops.aten.expand.default(v23, [32768, 128]);  v23 = None
    v25 = torch.ops.aten.div.Tensor(v1, v24)
    v27 = torch.ops.aten.bitwise_not.default(in3);  in3 = None
    v28 = torch.ops.aten.mul.Tensor(v25, v27);  v25 = None
    v29 = torch.ops.aten.sum.dim_IntList(v28, [0]);  v28 = None
    v31 = torch.ops.aten.masked_fill.Scalar(v5, in4, -1)
    v32 = torch.ops.aten.masked_fill.Scalar(v10, in4, 0.0);  v10 = in4 = None
    return (v1, v2, v5, v6, v8, v21, v22, v24, v27, v29, v31, v32)


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
