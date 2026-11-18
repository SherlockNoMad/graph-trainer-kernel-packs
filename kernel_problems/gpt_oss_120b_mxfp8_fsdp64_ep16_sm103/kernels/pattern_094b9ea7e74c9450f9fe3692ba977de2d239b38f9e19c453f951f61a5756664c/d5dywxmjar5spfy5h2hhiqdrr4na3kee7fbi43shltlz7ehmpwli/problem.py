"""Kernel problem pattern_094b9ea7e74c9450f9fe3692ba977de2d239b38f9e19c453f951f61a5756664c/d5dywxmjar5spfy5h2hhiqdrr4na3kee7fbi43shltlz7ehmpwli: region r016 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.feed_forward.routed_experts, layers.34.feed_forward.routed_experts, layers.33.feed_forward.routed_experts ....
Measured 258.2 us per call in the capture; ideal (HBM roofline) 108.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2):
    v1 = torch.ops.aten.arange.default(in0, dtype = torch.int64, device = device(type='cuda'), pin_memory = False);  in0 = None
    v3 = torch.ops.aten.empty_like.default(in1, pin_memory = False)
    v4 = torch.ops.aten.scatter_.src(v3, 0, in1, v1);  v3 = in1 = v1 = None
    v6 = torch.ops.aten.index.Tensor(in2, [v4]);  in2 = v4 = None
    return (v6,)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2):
        return reference(in0, in1, in2)


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
