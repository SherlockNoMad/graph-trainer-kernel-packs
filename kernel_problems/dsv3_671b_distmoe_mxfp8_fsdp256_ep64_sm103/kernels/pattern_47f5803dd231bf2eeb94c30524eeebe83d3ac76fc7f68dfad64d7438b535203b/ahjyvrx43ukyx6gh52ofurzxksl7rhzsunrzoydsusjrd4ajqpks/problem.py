"""Kernel problem pattern_47f5803dd231bf2eeb94c30524eeebe83d3ac76fc7f68dfad64d7438b535203b/ahjyvrx43ukyx6gh52ofurzxksl7rhzsunrzoydsusjrd4ajqpks: region r013 of outputs/fusion_finder/dsv3_671b.

3 calls per step, backward/recompute, in layers.2.feed_forward.w13, layers.1.feed_forward.w13, layers.0.feed_forward.w13.
Measured 79.4 us per call in the capture; ideal (HBM roofline) 42.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2):
    v2 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
    v4 = torch.ops.aten._fused_rms_norm.default(v2, [7168], in2, 1e-06);  in2 = None
    v5 = v4[0]
    v6 = torch.ops.torchao.mxfp8_quantize.default(v5, True, False, 32, 1, 'e4m3', 'rceil')
    v7 = v6[0]
    v8 = v6[2];  v6 = None
    v9 = torch.ops.torchao.triton_mx_block_rearrange.default(v8);  v8 = None
    v10 = torch.ops.torchao.mxfp8_quantize.default(v5, False, True, 1, 32, 'e4m3', 'rceil');  v5 = None
    v11 = v10[1]
    v12 = v10[3];  v10 = None
    v13 = torch.ops.torchao.triton_mx_block_rearrange.default(v12);  v12 = None
    v14 = v4[1];  v4 = None
    v15 = torch.ops.aten.alias.default(v14);  v14 = None
    v16 = torch.ops.aten.alias.default(v15);  v15 = None
    return (v2, v7, v9, v11, v13, v16)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2):
        return reference(in0, in1, in2)


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
