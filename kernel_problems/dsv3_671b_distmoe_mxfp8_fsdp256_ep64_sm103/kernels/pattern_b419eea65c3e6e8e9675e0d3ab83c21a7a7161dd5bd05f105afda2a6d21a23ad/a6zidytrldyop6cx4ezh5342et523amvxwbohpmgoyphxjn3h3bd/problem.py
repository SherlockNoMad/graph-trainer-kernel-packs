"""Kernel problem pattern_b419eea65c3e6e8e9675e0d3ab83c21a7a7161dd5bd05f105afda2a6d21a23ad/a6zidytrldyop6cx4ezh5342et523amvxwbohpmgoyphxjn3h3bd: region r027 of outputs/fusion_finder/dsv3_671b.

58 calls per step, forward, in layers.3.feed_forward.shared_experts.w2, layers.4.feed_forward.shared_experts.w2, layers.5.feed_forward.shared_experts.w2 ....
Measured 19.9 us per call in the capture; ideal (HBM roofline) 7.4 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels
import torchtitan_recipes.overrides.fused_swiglu


def reference(in0):
    v1 = torch.ops.aten.reshape.default(in0, [4096, 2, 2048]);  in0 = None
    v2 = torch.ops.torchtitan.silu_and_mul.default(v1);  v1 = None
    v3 = torch.ops.torchao.mxfp8_quantize.default(v2, True, True, 32, 32, 'e4m3', 'rceil');  v2 = None
    v4 = v3[0]
    v5 = v3[2];  v3 = None
    v6 = torch.ops.torchao.triton_mx_block_rearrange.default(v5);  v5 = None
    return (v4, v6)


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
