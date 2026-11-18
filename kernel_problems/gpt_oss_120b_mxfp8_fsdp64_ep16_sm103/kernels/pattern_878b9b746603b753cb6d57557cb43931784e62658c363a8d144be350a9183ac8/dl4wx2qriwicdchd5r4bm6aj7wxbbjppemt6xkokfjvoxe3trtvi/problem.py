"""Kernel problem pattern_878b9b746603b753cb6d57557cb43931784e62658c363a8d144be350a9183ac8/dl4wx2qriwicdchd5r4bm6aj7wxbbjppemt6xkokfjvoxe3trtvi: region r001 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, backward, in layers.35.feed_forward.routed_experts, layers.34.feed_forward.routed_experts, layers.33.feed_forward.routed_experts ....
Measured 14175.7 us per call in the capture; ideal (HBM roofline) 108.9 us.

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
    v1 = torch.ops.aten.arange.default(in0, dtype = torch.int64, device = device(type='cuda'), pin_memory = False);  in0 = None
    v3 = torch.ops.aten.empty_like.default(in1, pin_memory = False)
    v4 = torch.ops.aten.scatter_.src(v3, 0, in1, v1);  v3 = in1 = v1 = None
    v6 = torch.ops.aten.index.Tensor(in2, [v4]);  in2 = v4 = None
    v7 = torch.ops.aten.new_zeros.default(v6, [9, 2880], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v9 = torch.ops.aten.index_add.default(v7, 0, in3, v6);  v7 = in3 = None
    v10 = torch.ops.aten.slice.Tensor(v9, 0, 0, 8);  v9 = None
    v11 = torch.ops.aten.transpose.int(v6, -2, -1)
    return (v6, v10, v11)


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
