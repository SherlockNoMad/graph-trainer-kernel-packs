"""Kernel problem pattern_77b3cbfb4623cf6392b0755fe89ff18caacc44ad4960cfd78562418e68c13bd9/aunprf4levra5pxs5ypjot4mabfm2wm2rs3gqclf4bat2i7hbdn3: region r018 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.routed_experts, layers.1.feed_forward.routed_experts, layers.2.feed_forward.routed_experts ....
Measured 139.1 us per call in the capture; ideal (HBM roofline) 68.0 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1):
    v2 = torch.ops.aten.index.Tensor(in0, [in1]);  in0 = in1 = None
    return (v2,)


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
