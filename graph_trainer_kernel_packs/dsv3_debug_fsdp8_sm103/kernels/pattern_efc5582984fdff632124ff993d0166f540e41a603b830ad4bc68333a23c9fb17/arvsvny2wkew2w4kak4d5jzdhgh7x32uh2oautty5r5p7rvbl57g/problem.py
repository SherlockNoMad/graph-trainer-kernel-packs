"""Kernel problem pattern_efc5582984fdff632124ff993d0166f540e41a603b830ad4bc68333a23c9fb17/arvsvny2wkew2w4kak4d5jzdhgh7x32uh2oautty5r5p7rvbl57g: region r011 of outputs/fusion_finder/dsv3_debug.

5 calls per step, backward/recompute, in layers.5.feed_forward.shared_experts, layers.4.feed_forward.shared_experts, layers.3.feed_forward.shared_experts ....
Measured 103.6 us per call in the capture; ideal (HBM roofline) 14.5 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1):
    v1 = torch.ops.aten.reshape.default(in0, [16384, 2, 512]);  in0 = None
    v2 = torch.ops.aten.unbind.int(v1, -2);  v1 = None
    v3 = v2[0]
    v4 = v2[1];  v2 = None
    v5 = torch.ops.aten.silu.default(v3)
    v6 = torch.ops.aten.mul.Tensor(v5, v4)
    v8 = torch.ops.aten.mul.Tensor(in1, v5);  v5 = None
    v9 = torch.ops.aten.mul.Tensor(in1, v4);  in1 = v4 = None
    v10 = torch.ops.aten.silu_backward.default(v9, v3);  v9 = v3 = None
    v11 = torch.ops.aten.stack.default([v10, v8], -2);  v10 = v8 = None
    v12 = torch.ops.aten.reshape.default(v11, [16384, 1024]);  v11 = None
    v13 = torch.ops.aten.t.default(v12)
    return (v6, v12, v13)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1):
        return reference(in0, in1)


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
