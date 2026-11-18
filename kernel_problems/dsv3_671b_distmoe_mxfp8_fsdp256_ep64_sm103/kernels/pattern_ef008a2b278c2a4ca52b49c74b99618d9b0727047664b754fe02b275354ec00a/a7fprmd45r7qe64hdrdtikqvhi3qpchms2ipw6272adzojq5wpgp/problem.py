"""Kernel problem pattern_ef008a2b278c2a4ca52b49c74b99618d9b0727047664b754fe02b275354ec00a/a7fprmd45r7qe64hdrdtikqvhi3qpchms2ipw6272adzojq5wpgp: region r020 of outputs/fusion_finder/dsv3_671b.

62 calls per step, forward, in layers.1.attention.wq_a, layers.2.attention.wq_a, layers.3.attention.wq_a ....
Measured 64.9 us per call in the capture; ideal (HBM roofline) 38.3 us.

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
    v6 = torch.ops.torchao.mxfp8_quantize.default(v5, True, False, 32, 1, 'e4m3', 'rceil')
    v7 = v6[0]
    v8 = v6[2];  v6 = None
    v9 = torch.ops.torchao.triton_mx_block_rearrange.default(v8);  v8 = None
    return (v2, v5, v7, v9)


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
