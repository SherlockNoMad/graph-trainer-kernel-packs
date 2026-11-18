"""Kernel problem pattern_47742e295090b19e0b074523f63306a3b97791f686cfda773a72852fe6b63d9c/av76scgjjc4wshdtwapt5hoxji3gzzxhc3rchm2ubeva6fnndlq2: region r005 of outputs/fusion_finder/dsv3_debug.

5 calls per step, forward, in layers.1.feed_forward.router.aux_loss, layers.2.feed_forward.router.aux_loss, layers.3.feed_forward.router.aux_loss ....
Measured 214.5 us per call in the capture; ideal (HBM roofline) 5.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7):
    v1 = torch.ops.aten.sigmoid.default(in0);  in0 = None
    v3 = torch.ops.aten.add.Tensor(v1, in1);  in1 = None
    v4 = torch.ops.aten.topk.default(v3, 3, -1, True, False);  v3 = None
    v5 = v4[1];  v4 = None
    v6 = torch.ops.aten.gather.default(v1, -1, v5)
    v7 = torch.ops.aten.sum.dim_IntList(v6, [-1], True)
    v8 = torch.ops.aten.add.Tensor(v7, 1e-20);  v7 = None
    v9 = torch.ops.aten.div.Tensor(v6, v8);  v6 = v8 = None
    v10 = torch.ops.aten.mul.Tensor(v9, 1.0);  v9 = None
    v11 = torch.ops.aten.zeros_like.default(v1, dtype = torch.bool, pin_memory = False)
    v12 = torch.ops.aten.scatter_.value(v11, -1, v5, True);  v11 = None
    v14 = torch.ops.aten.bitwise_not.default(in2);  in2 = None
    v15 = torch.ops.aten.bitwise_and.Tensor(v12, v14);  v14 = None
    v16 = torch.ops.aten.sum.dim_IntList(v15, [0])
    v18 = torch.ops.aten.add_.Tensor(in3, v16);  in3 = v16 = v18 = None
    v19 = torch.ops.aten._to_copy.default(v15, dtype = torch.float32)
    v20 = torch.ops.aten.sum.dim_IntList(v19, [0]);  v19 = None
    v21 = torch.ops.aten.linalg_vector_norm.default(v1, 1, [-1], True)
    v22 = torch.ops.aten.clamp_min.default(v21, 1e-12);  v21 = None
    v23 = torch.ops.aten.expand.default(v22, [16384, 8]);  v22 = None
    v24 = torch.ops.aten.div.Tensor(v1, v23);  v1 = v23 = None
    v26 = torch.ops.aten.bitwise_not.default(in4);  in4 = None
    v27 = torch.ops.aten.mul.Tensor(v24, v26);  v24 = v26 = None
    v28 = torch.ops.aten.sum.dim_IntList(v27, [0]);  v27 = None
    v29 = torch.ops.aten.linalg_vector_norm.default(v20, 1, [0], True)
    v30 = torch.ops.aten.clamp_min.default(v29, 1e-12);  v29 = None
    v31 = torch.ops.aten.expand.default(v30, [8]);  v30 = None
    v32 = torch.ops.aten.div.Tensor(v20, v31);  v20 = v31 = None
    v33 = torch.ops.aten.mul.Tensor(v32, 8);  v32 = None
    v34 = torch.ops.aten.mul.Tensor(v33, v28);  v33 = v28 = None
    v35 = torch.ops.aten.sum.default(v34);  v34 = None
    v37 = torch.ops.aten.clamp_min.default(in5, 1);  in5 = None
    v38 = torch.ops.aten.reciprocal.default(v37);  v37 = None
    v39 = torch.ops.aten.mul.Tensor(v38, 1.0);  v38 = None
    v40 = torch.ops.aten.mul.Tensor(v35, v39);  v35 = v39 = None
    v42 = torch.ops.aten.add_.Tensor(in6, v40);  in6 = v40 = v42 = None
    v43 = torch.ops.aten.sum.dim_IntList(v12, [0]);  v12 = None
    v44 = torch.ops.aten.reshape.default(v5, [-1])
    v45 = torch.ops.aten._to_copy.default(v44, dtype = torch.uint8);  v44 = None
    v46 = torch.ops.aten.sort.stable(v45, stable = True);  v45 = None
    v47 = v46[1];  v46 = None
    v48 = torch.ops.aten.reshape.default(v10, [-1]);  v10 = None
    v49 = torch.ops.aten.index.Tensor(v48, [v47]);  v48 = None
    v50 = torch.ops.aten.floor_divide.default(v47, 3);  v47 = None
    v52 = torch.ops.aten.index.Tensor(in7, [v50]);  in7 = None
    v53 = torch.ops.aten.cumsum.default(v43, 0, dtype = torch.int32)
    v54 = torch.ops.aten.reshape.default(v49, [-1, 1]);  v49 = None
    v55 = torch.ops.aten.reshape.default(v50, [-1, 1]);  v50 = None
    v56 = torch.ops.aten.expand.default(v55, [-1, 256]);  v55 = None
    return (v5, v15, v43, v52, v53, v54, v56)


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
