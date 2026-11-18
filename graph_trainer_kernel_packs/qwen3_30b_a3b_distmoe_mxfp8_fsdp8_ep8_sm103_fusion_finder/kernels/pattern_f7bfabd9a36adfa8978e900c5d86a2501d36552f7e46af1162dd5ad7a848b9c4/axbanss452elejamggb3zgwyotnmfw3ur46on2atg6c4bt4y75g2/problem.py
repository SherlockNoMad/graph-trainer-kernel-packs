"""Kernel problem pattern_f7bfabd9a36adfa8978e900c5d86a2501d36552f7e46af1162dd5ad7a848b9c4/axbanss452elejamggb3zgwyotnmfw3ur46on2atg6c4bt4y75g2: region r018 of outputs/fusion_finder/qwen3_30b.

48 calls per step, forward, in layers.0.attention.wo, layers.1.attention.wo, layers.2.attention.wo ....
Measured 70.4 us per call in the capture; ideal (HBM roofline) 59.1 us.

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
    v1 = torch.ops.aten.alias.default(in0)
    v2 = torch.ops.aten.reshape.default(in0, [32768, -1]);  in0 = None
    v3 = torch.ops.torchao.mxfp8_quantize.default(v2, True, False, 32, 1, 'e4m3', 'rceil')
    v4 = v3[0]
    v5 = v3[2];  v3 = None
    v6 = torch.ops.torchao.triton_mx_block_rearrange.default(v5);  v5 = None
    return (v1, v2, v4, v6)


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
