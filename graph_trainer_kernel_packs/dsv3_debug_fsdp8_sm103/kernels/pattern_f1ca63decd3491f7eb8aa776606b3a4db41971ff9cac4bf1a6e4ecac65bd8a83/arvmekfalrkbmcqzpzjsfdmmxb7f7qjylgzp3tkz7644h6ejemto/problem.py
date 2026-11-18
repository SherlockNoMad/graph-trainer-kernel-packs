"""Kernel problem pattern_f1ca63decd3491f7eb8aa776606b3a4db41971ff9cac4bf1a6e4ecac65bd8a83/arvmekfalrkbmcqzpzjsfdmmxb7f7qjylgzp3tkz7644h6ejemto: region r010 of outputs/fusion_finder/dsv3_debug.

5 calls per step, backward, in layers.5.ffn_norm, layers.4.ffn_norm, layers.3.ffn_norm ....
Measured 138.1 us per call in the capture; ideal (HBM roofline) 9.7 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7):
    v1 = torch.ops.aten.new_zeros.default(in0, [16384, 256], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
    v3 = torch.ops.aten.index_put.default(v1, [in1], in0, True);  v1 = in1 = in0 = None
    v5 = torch.ops.aten.add.Tensor(in2, v3);  in2 = v3 = None
    v7 = torch.ops.aten.add.Tensor(v5, in3);  v5 = in3 = None
    v11 = torch.ops.aten._fused_rms_norm_backward.default(v7, in4, [256], in5, in6, [True, True]);  v7 = in4 = in5 = in6 = None
    v12 = v11[0]
    v13 = v11[1];  v11 = None
    v14 = torch.ops.aten._to_copy.default(v13, dtype = torch.float32);  v13 = None
    v16 = torch.ops.aten.add.Tensor(in7, v12);  in7 = v12 = None
    v17 = torch.ops.aten.t.default(v16)
    return (v14, v16, v17)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7)


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
