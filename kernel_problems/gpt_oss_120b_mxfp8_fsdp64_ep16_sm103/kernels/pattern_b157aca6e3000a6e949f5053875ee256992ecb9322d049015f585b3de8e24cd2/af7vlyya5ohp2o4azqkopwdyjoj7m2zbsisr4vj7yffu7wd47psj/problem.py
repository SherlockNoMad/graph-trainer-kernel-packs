"""Kernel problem pattern_b157aca6e3000a6e949f5053875ee256992ecb9322d049015f585b3de8e24cd2/af7vlyya5ohp2o4azqkopwdyjoj7m2zbsisr4vj7yffu7wd47psj: region r011 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.router, layers.1.feed_forward.router, layers.2.feed_forward.router ....
Measured 213.5 us per call in the capture; ideal (HBM roofline) 2.8 us.

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
    v1 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32);  in0 = None
    v3 = torch.ops.aten.add.Tensor(in1, v1);  in1 = v1 = None
    v4 = torch.ops.aten._softmax.default(v3, -1, False);  v3 = None
    v5 = torch.ops.aten.alias.default(v4)
    v6 = torch.ops.aten.arange.default(65536, dtype = torch.int64, device = device(type='cuda'), pin_memory = False)
    v7 = torch.ops.aten.reshape.default(v6, [16384, 4]);  v6 = None
    v8 = torch.ops.aten.remainder.Scalar(v7, 128);  v7 = None
    v9 = torch.ops.aten.gather.default(v4, -1, v8)
    v10 = torch.ops.aten.sum.dim_IntList(v9, [-1], True)
    v11 = torch.ops.aten.add.Tensor(v10, 1e-20);  v10 = None
    v12 = torch.ops.aten.div.Tensor(v9, v11)
    v13 = torch.ops.aten.mul.Tensor(v12, 1.0);  v12 = None
    v14 = torch.ops.aten.zeros_like.default(v4, dtype = torch.bool, pin_memory = False)
    v15 = torch.ops.aten.scatter_.value(v14, -1, v8, True);  v14 = None
    v17 = torch.ops.aten.bitwise_not.default(in2);  in2 = None
    v18 = torch.ops.aten.bitwise_and.Tensor(v15, v17);  v17 = None
    v19 = torch.ops.aten.sum.dim_IntList(v18, [0])
    v21 = torch.ops.aten.add_.Tensor(in3, v19);  in3 = v19 = v21 = None
    v22 = torch.ops.aten._to_copy.default(v18, dtype = torch.float32);  v18 = None
    v23 = torch.ops.aten.sum.dim_IntList(v22, [0]);  v22 = None
    v24 = torch.ops.aten.view.default(v23, [-1]);  v23 = None
    v25 = torch.ops.aten.linalg_vector_norm.default(v4, 1, [-1], True)
    v26 = torch.ops.aten.clamp_min.default(v25, 1e-12)
    v27 = torch.ops.aten.expand.default(v26, [16384, 128]);  v26 = None
    v28 = torch.ops.aten.div.Tensor(v4, v27)
    v30 = torch.ops.aten.bitwise_not.default(in4);  in4 = None
    v31 = torch.ops.aten.mul.Tensor(v28, v30);  v28 = None
    v32 = torch.ops.aten.sum.dim_IntList(v31, [0]);  v31 = None
    v33 = torch.ops.aten.sum.dim_IntList(v15, [0]);  v15 = None
    v34 = torch.ops.aten.reshape.default(v8, [-1])
    v35 = torch.ops.aten._to_copy.default(v34, dtype = torch.uint8);  v34 = None
    v36 = torch.ops.aten.sort.stable(v35, stable = True);  v35 = None
    v37 = v36[1];  v36 = None
    v38 = torch.ops.aten.reshape.default(v13, [-1]);  v13 = None
    v39 = torch.ops.aten.index.Tensor(v38, [v37]);  v38 = None
    v40 = torch.ops.aten.floor_divide.default(v37, 4)
    v41 = torch.ops.aten.reshape.default(v33, [16, -1])
    v42 = torch.ops.aten.reshape.default(v33, [16, -1]);  v33 = None
    v43 = torch.ops.aten.sum.dim_IntList(v42, [1]);  v42 = None
    v44 = torch.ops.aten.reshape.default(v39, [-1, 1]);  v39 = None
    v45 = torch.ops.aten.reshape.default(v40, [-1, 1])
    v46 = torch.ops.aten.expand.default(v45, [-1, 2880]);  v45 = None
    return (v4, v5, v8, v9, v11, v24, v25, v27, v30, v32, v37, v40, v41, v43, v44, v46)


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
