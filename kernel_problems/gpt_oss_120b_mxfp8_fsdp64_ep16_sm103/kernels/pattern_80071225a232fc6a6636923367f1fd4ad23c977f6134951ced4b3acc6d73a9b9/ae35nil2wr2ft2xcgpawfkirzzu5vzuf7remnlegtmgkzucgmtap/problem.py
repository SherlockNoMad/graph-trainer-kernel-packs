"""Kernel problem pattern_80071225a232fc6a6636923367f1fd4ad23c977f6134951ced4b3acc6d73a9b9/ae35nil2wr2ft2xcgpawfkirzzu5vzuf7remnlegtmgkzucgmtap: region r021 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.routed_experts, layers.1.feed_forward.routed_experts, layers.2.feed_forward.routed_experts ....
Measured 32.4 us per call in the capture; ideal (HBM roofline) 0.0 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0):
    v1 = torch.ops.aten.reshape.default(in0, [-1]);  in0 = None
    v2 = torch.ops.aten.reshape.default(v1, [16, -1])
    v3 = torch.ops.aten.sum.dim_IntList(v2, [1]);  v2 = None
    v4 = torch.ops.aten.reshape.default(v1, [16, 8])
    v5 = torch.ops.aten.cumsum.default(v1, 0)
    v6 = torch.ops.aten.sub.Tensor(v5, v1);  v5 = v1 = None
    v7 = torch.ops.aten.reshape.default(v6, [16, 8]);  v6 = None
    v8 = torch.ops.aten.t.default(v4)
    v9 = torch.ops.aten.clone.default(v8, memory_format = torch.contiguous_format);  v8 = None
    v10 = torch.ops.aten.reshape.default(v9, [128]);  v9 = None
    v11 = torch.ops.aten.t.default(v7);  v7 = None
    v12 = torch.ops.aten.clone.default(v11, memory_format = torch.contiguous_format);  v11 = None
    v13 = torch.ops.aten.reshape.default(v12, [128]);  v12 = None
    v14 = torch.ops.aten.cumsum.default(v10, 0)
    v15 = torch.ops.aten.sub.Tensor(v14, v10);  v14 = None
    v16 = torch.ops.aten.sum.dim_IntList(v4, [0]);  v4 = None
    v17 = torch.ops.aten.cumsum.default(v16, 0, dtype = torch.int32);  v16 = None
    v18 = torch.ops.aten.new_zeros.default(v17, [1], pin_memory = False)
    v19 = torch.ops.aten.cat.default([v18, v17]);  v18 = None
    v20 = torch.ops.aten.slice.Tensor(v19, -1, 1, 9)
    v21 = torch.ops.aten.slice.Tensor(v19, -1, 0, 8);  v19 = None
    v22 = torch.ops.aten.sub.Tensor(v20, v21);  v20 = v21 = None
    v23 = torch.ops.aten.select.int(v17, 0, -1)
    v24 = torch.ops.aten.new_zeros.default(v17, [1], pin_memory = False)
    v25 = torch.ops.aten.cat.default([v24, v17]);  v24 = None
    v26 = torch.ops.aten.slice.Tensor(v25, -1, 1, 9)
    v27 = torch.ops.aten.slice.Tensor(v25, -1, 0, 8);  v25 = None
    v28 = torch.ops.aten.sub.Tensor(v26, v27);  v26 = v27 = None
    v29 = torch.ops.aten.select.int(v17, 0, -1)
    return (v3, v10, v13, v15, v17, v22, v23, v28, v29)


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
