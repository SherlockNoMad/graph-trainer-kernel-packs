"""Kernel problem pattern_23554a5cca25974d0375de1f3dd37ba641b98ee4363c56ee8af9a4dcdc52a834/dvcvbew4gaieloiv66jlt3ibkmrygt3ocm4jwk4qp6dzhar5s5bw: region r015 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.routed_experts, layers.1.feed_forward.routed_experts, layers.2.feed_forward.routed_experts ....
Measured 263.5 us per call in the capture; ideal (HBM roofline) 109.0 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3):
    v1 = torch.ops.aten.arange.default(in0, device = device(type='cuda'), pin_memory = False)
    v3 = torch.ops.aten.add.Tensor(in1, v1);  in1 = v1 = None
    v5 = torch.ops.aten.sub.Tensor(v3, in2);  v3 = in2 = None
    v7 = torch.ops.aten.index.Tensor(in3, [v5]);  in3 = None
    v8 = torch.ops.aten.arange.default(in0, dtype = torch.int64, device = device(type='cuda'), pin_memory = False);  in0 = None
    v9 = torch.ops.aten.empty_like.default(v5, pin_memory = False)
    v10 = torch.ops.aten.scatter_.src(v9, 0, v5, v8);  v9 = v8 = None
    return (v5, v7, v10)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3):
        return reference(in0, in1, in2, in3)


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
