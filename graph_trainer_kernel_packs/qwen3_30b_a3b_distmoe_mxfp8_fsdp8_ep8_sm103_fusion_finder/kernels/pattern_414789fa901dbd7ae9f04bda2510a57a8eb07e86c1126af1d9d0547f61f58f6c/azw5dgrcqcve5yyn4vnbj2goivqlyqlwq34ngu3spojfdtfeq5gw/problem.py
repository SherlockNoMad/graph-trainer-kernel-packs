"""Kernel problem pattern_414789fa901dbd7ae9f04bda2510a57a8eb07e86c1126af1d9d0547f61f58f6c/azw5dgrcqcve5yyn4vnbj2goivqlyqlwq34ngu3spojfdtfeq5gw: region r008 of outputs/fusion_finder/qwen3_30b.

47 calls per step, forward, in layers.1.attention_norm, layers.2.attention_norm, layers.3.attention_norm ....
Measured 139.6 us per call in the capture; ideal (HBM roofline) 87.5 us.

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
    v4 = torch.ops.aten._fused_rms_norm.default(v2, [2048], in2, 1e-06);  in2 = None
    v5 = v4[0]
    v6 = v4[1];  v4 = None
    v7 = torch.ops.aten.alias.default(v6);  v6 = None
    v8 = torch.ops.torchao.mxfp8_quantize.default(v5, True, False, 32, 1, 'e4m3', 'rceil')
    v9 = v8[0]
    v10 = v8[2];  v8 = None
    v11 = torch.ops.torchao.triton_mx_block_rearrange.default(v10);  v10 = None
    return (v2, v5, v7, v9, v11)


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
