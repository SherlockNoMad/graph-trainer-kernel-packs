"""Kernel problem pattern_9ecac9669da356cb5f78a89903e3e56f3aaae96a66a441ee66fe95b8340af373/as5ddgxdi2mw6lgk77jzalsmeruf5adllatsojekzklgqjvxsblv: region r025 of outputs/fusion_finder/dsv3_671b.

61 calls per step, recompute, in layers.60.attention.wkv_a, layers.59.attention.wkv_a, layers.58.attention.wkv_a ....
Measured 4.8 us per call in the capture; ideal (HBM roofline) 1.8 us.

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
    v3 = v1[1]
    v4 = v1[2];  v1 = None
    v5 = torch.ops.aten.t.default(v2)
    return (v2, v3, v4, v5)


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
