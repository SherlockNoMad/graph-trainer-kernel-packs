"""Kernel problem pattern_e8da1a71205a516218367a404e8577e16e52a2e31fbd198b793c1bfa64fc5de2/ahfinmrd4sdonynjdtz7p63sszpzrgtuavu7ihnsal3sfb6vhaak: region r011 of outputs/fusion_finder/dsv3_671b.

3 calls per step, backward/recompute, in layers.2.feed_forward.w13, layers.1.feed_forward.w13, layers.0.feed_forward.w13.
Measured 356.9 us per call in the capture; ideal (HBM roofline) 134.1 us.

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


def reference(in0, in1):
    v1 = torch.ops.aten.reshape.default(in0, [4096, 2, 18432]);  in0 = None
    v2 = torch.ops.torchtitan.silu_and_mul.default(v1)
    v3 = torch.ops.torchao.mxfp8_quantize.default(v2, True, True, 32, 32, 'e4m3', 'rceil');  v2 = None
    v4 = v3[1]
    v5 = v3[3];  v3 = None
    v6 = torch.ops.torchao.triton_mx_block_rearrange.default(v5);  v5 = None
    v8 = torch.ops.torchtitan.silu_and_mul_backward.default(in1, v1);  in1 = v1 = None
    v9 = torch.ops.aten.reshape.default(v8, [4096, 36864]);  v8 = None
    v10 = torch.ops.torchao.mxfp8_quantize.default(v9, True, True, 32, 32, 'e4m3', 'rceil');  v9 = None
    v11 = v10[0]
    v12 = v10[1]
    v13 = torch.ops.aten.t.default(v12);  v12 = None
    v14 = v10[2]
    v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
    v16 = v10[3];  v10 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
    return (v4, v6, v11, v13, v15, v17)


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
