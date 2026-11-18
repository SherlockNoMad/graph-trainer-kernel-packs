"""Kernel problem pattern_7d3a61803f3add81ae14f560d08863e19bab17b71c8831fc8537c6fbddf14e97/a4vjva3kddnc7cydjresst6oc3svstrqtyqrbn5of6tu6xp73pi2: region r007 of outputs/fusion_finder/qwen3_30b.

48 calls per step, backward, in layers.47.attention.wo, layers.46.attention.wo, layers.45.attention.wo ....
Measured 252.1 us per call in the capture; ideal (HBM roofline) 136.4 us.

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
    v4 = torch.ops.aten.alias.default(in2);  in2 = None
    v7 = torch.ops.aten._fused_rms_norm_backward.default(v2, in3, [2048], v4, in4, [True, True]);  v2 = in3 = v4 = in4 = None
    v8 = v7[0]
    v9 = v7[1];  v7 = None
    v11 = torch.ops.aten.add.Tensor(in5, v8);  in5 = v8 = None
    v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, True, 32, 32, 'e4m3', 'rceil')
    v13 = v12[0]
    v14 = v12[1]
    v15 = v12[2]
    v16 = v12[3];  v12 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
    v18 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
    v19 = torch.ops.aten.t.default(v14);  v14 = None
    return (v9, v11, v13, v17, v18, v19)


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
