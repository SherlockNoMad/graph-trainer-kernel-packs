"""Kernel problem pattern_3ac130daa4ea7ef97ee43af5b3dff9228e29c214081bead8e59934042459a993/a32vn73djnay6nf4hiaaemswp3zuylgm7hxcaymxpzqopufwnbph: region r026 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward, in layers.60.attention.wq_a, layers.59.attention.wq_a, layers.58.attention.wq_a ....
Measured 23.8 us per call in the capture; ideal (HBM roofline) 7.4 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3):
    v4 = torch.ops.aten._fused_rms_norm_backward.default(in0, in1, [1536], in2, in3, [True, True]);  in0 = in1 = in2 = in3 = None
    v5 = v4[0]
    v6 = v4[1];  v4 = None
    v7 = torch.ops.torchao.mxfp8_quantize.default(v5, True, True, 32, 32, 'e4m3', 'rceil');  v5 = None
    v8 = v7[0]
    v9 = v7[1]
    v10 = v7[2]
    v11 = v7[3];  v7 = None
    v12 = torch.ops.torchao.triton_mx_block_rearrange.default(v10);  v10 = None
    v13 = torch.ops.torchao.triton_mx_block_rearrange.default(v11);  v11 = None
    v14 = torch.ops.aten.t.default(v9);  v9 = None
    return (v6, v8, v12, v13, v14)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3):
        return reference(in0, in1, in2, in3)


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
