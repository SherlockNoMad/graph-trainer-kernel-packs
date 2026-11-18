"""Kernel problem pattern_0699a4e4475830bf0d50747499e27e09928126b368d03b548f3ac5435959d81c/afsdfsnukpbxmlgqbr6nagh56m7rffjdqrf6bmuvys7qwfnwzifq: region r031 of outputs/fusion_finder/dsv3_671b.

61 calls per step, forward, in layers.0.attention.wo, layers.1.attention.wo, layers.2.attention.wo ....
Measured 38.9 us per call in the capture; ideal (HBM roofline) 29.6 us.

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
    v1 = torch.ops.aten.reshape.default(in0, [4096, -1]);  in0 = None
    v2 = torch.ops.torchao.mxfp8_quantize.default(v1, True, False, 32, 1, 'e4m3', 'rceil');  v1 = None
    v3 = v2[0]
    v4 = v2[2];  v2 = None
    v5 = torch.ops.torchao.triton_mx_block_rearrange.default(v4);  v4 = None
    return (v3, v5)


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
