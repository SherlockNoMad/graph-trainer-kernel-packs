"""Kernel problem pattern_891c0c9c0ac143fbcde90bc36b61ec136f68230ba8e88f12c795f3774b88a44b/a4s4prszprwdrml5plva4lkzm4zftuapysvsolgjwr2cn7lltdxd: region r015 of outputs/fusion_finder/dsv3_671b.

1 call per step, backward, in lm_head.
Measured 2620.9 us per call in the capture; ideal (HBM roofline) 771.6 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3, in4, in5):
    v2 = torch.ops.aten.div.Tensor(in0, in1);  in0 = in1 = None
    v6 = torch.ops.aten.nll_loss_backward.default(v2, in2, in3, None, 2, -100, in4);  v2 = in2 = in3 = in4 = None
    v8 = torch.ops.aten.alias.default(in5);  in5 = None
    v9 = torch.ops.aten._log_softmax_backward_data.default(v6, v8, 1, torch.float32);  v6 = v8 = None
    v10 = torch.ops.aten._to_copy.default(v9, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v9 = None
    v11 = torch.ops.torchao.mxfp8_quantize.default(v10, True, True, 32, 32, 'e4m3', 'rceil');  v10 = None
    v12 = v11[0]
    v13 = v11[1]
    v14 = v11[2]
    v15 = v11[3];  v11 = None
    v16 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
    v18 = torch.ops.aten.t.default(v13);  v13 = None
    return (v12, v16, v17, v18)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5):
        return reference(in0, in1, in2, in3, in4, in5)


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
