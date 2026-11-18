"""Kernel problem pattern_e7d0aa78efd174144ecde477abd0a1dc357117102648326f2965bf8497626dea/aarsanokqze2m5p6fvippzdkavrpg22kxsftmnyeajupgbo7f4jr: region r010 of outputs/fusion_finder/qwen3_30b.

1 call per step, backward, in layers.0.attention_norm.
Measured 782.0 us per call in the capture; ideal (HBM roofline) 166.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4, in5):
    v1 = torch.ops.aten.alias.default(in0);  in0 = None
    v5 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [2048], v1, in3, [True, True]);  in1 = in2 = v1 = in3 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v9 = torch.ops.aten.add.Tensor(in4, v6);  in4 = v6 = None
    v11 = torch.ops.aten.embedding_dense_backward.default(v9, in5, 151936, -1, False);  v9 = in5 = None
    return (v7, v11)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5):
        return reference(in0, in1, in2, in3, in4, in5)


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
