"""Kernel problem pattern_9fac1011d292deac8298f175cc00b38013ef47009b2833c6f869db2591b9c8a8/axh333fk2j7nwcew2exsum2dtnznvk4gtnm77tysnf7c5iv2rzdr: region r020 of outputs/fusion_finder/dsv3_671b.

3 calls per step, forward, in layers.0.feed_forward.w13, layers.1.feed_forward.w13, layers.2.feed_forward.w13.
Measured 64.6 us per call in the capture; ideal (HBM roofline) 29.8 us.

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
    v5 = v4[0];  v4 = None
    v6 = torch.ops.torchao.mxfp8_quantize.default(v5, True, False, 32, 1, 'e4m3', 'rceil');  v5 = None
    v7 = v6[0]
    v8 = v6[2];  v6 = None
    v9 = torch.ops.torchao.triton_mx_block_rearrange.default(v8);  v8 = None
    return (v2, v7, v9)


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
