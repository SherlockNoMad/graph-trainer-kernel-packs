"""Kernel problem pattern_284c8fbe17944a889302168e60387b8e8ab1986cdeb830d2fe047c0258f5f3b1/af4bxx2bdfiq5i2lsqxirqwtk2thmi3o7p2j4aa6hlt3c4wjm5oe: region r024 of outputs/fusion_finder/dsv3_671b.

3 calls per step, forward, in layers.0.feed_forward.w2, layers.1.feed_forward.w2, layers.2.feed_forward.w2.
Measured 62.3 us per call in the capture; ideal (HBM roofline) 58.2 us.

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
    v1 = torch.ops.torchao.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale.default(in0);  in0 = None
    v2 = v1[0]
    v3 = torch.ops.aten.t.default(v2);  v2 = None
    v4 = v1[1];  v1 = None
    return (v3, v4)


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
