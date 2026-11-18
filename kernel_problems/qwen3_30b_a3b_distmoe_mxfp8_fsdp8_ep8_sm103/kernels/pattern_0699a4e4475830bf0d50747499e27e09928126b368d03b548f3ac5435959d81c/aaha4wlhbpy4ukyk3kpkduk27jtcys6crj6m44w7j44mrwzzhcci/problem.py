"""Kernel problem pattern_0699a4e4475830bf0d50747499e27e09928126b368d03b548f3ac5435959d81c/aaha4wlhbpy4ukyk3kpkduk27jtcys6crj6m44w7j44mrwzzhcci: region r016 of outputs/fusion_finder/qwen3_30b.

48 calls per step, backward, in layers.47.attention.qkv_linear.wqkv, layers.46.attention.qkv_linear.wqkv, layers.45.attention.qkv_linear.wqkv ....
Measured 35.3 us per call in the capture; ideal (HBM roofline) 29.6 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0):
    v1 = torch.ops.torchao.mxfp8_quantize.default(in0, False, True, 1, 32, 'e4m3', 'rceil');  in0 = None
    v2 = v1[1]
    v3 = v1[3];  v1 = None
    v4 = torch.ops.torchao.triton_mx_block_rearrange.default(v3);  v3 = None
    return (v2, v4)


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
