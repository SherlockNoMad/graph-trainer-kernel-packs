"""Kernel problem pattern_7d3a61803f3add81ae14f560d08863e19bab17b71c8831fc8537c6fbddf14e97/abyywhawrvjvy4yfginrkoexpmrtk442ncq457v3e2zmtcom6t7g: region r010 of outputs/fusion_finder/dsv3_671b.

60 calls per step, backward, in layers.59.feed_forward.shared_experts.w2, layers.58.feed_forward.shared_experts.w2, layers.57.feed_forward.shared_experts.w2 ....
Measured 123.7 us per call in the capture; ideal (HBM roofline) 59.7 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1, in2, in3, in4, in5):
    v2 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
    v6 = torch.ops.aten._fused_rms_norm_backward.default(v2, in2, [7168], in3, in4, [True, True]);  v2 = in2 = in3 = in4 = None
    v7 = v6[0]
    v8 = v6[1];  v6 = None
    v10 = torch.ops.aten.add.Tensor(in5, v7);  in5 = v7 = None
    v11 = torch.ops.torchao.mxfp8_quantize.default(v10, True, True, 32, 32, 'e4m3', 'rceil')
    v12 = v11[0]
    v13 = v11[1]
    v14 = v11[2]
    v15 = v11[3];  v11 = None
    v16 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
    v18 = torch.ops.aten.t.default(v13);  v13 = None
    return (v8, v10, v12, v16, v17, v18)


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
