"""Kernel problem pattern_d781b78ec9972a2558419b26ffb0d31b883e4188ed75d9a5286eeba4d99ff194/azrd6bvgq26f6gdg7taedl267y7jzepdnvmpfth3hwoqabftk5cb: region r011 of outputs/fusion_finder/qwen3_30b.

47 calls per step, backward, in layers.47.attention_norm, layers.46.attention_norm, layers.45.attention_norm ....
Measured 140.7 us per call in the capture; ideal (HBM roofline) 96.6 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4):
    v1 = torch.ops.aten.alias.default(in0);  in0 = None
    v5 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [2048], v1, in3, [True, True]);  in1 = in2 = v1 = in3 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v9 = torch.ops.aten.add.Tensor(in4, v6);  in4 = v6 = None
    return (v7, v9)


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
