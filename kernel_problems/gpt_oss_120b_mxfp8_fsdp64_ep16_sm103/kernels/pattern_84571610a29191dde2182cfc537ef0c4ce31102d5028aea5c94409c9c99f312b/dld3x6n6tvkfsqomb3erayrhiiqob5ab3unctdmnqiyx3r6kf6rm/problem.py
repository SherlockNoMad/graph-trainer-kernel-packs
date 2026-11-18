"""Kernel problem pattern_84571610a29191dde2182cfc537ef0c4ce31102d5028aea5c94409c9c99f312b/dld3x6n6tvkfsqomb3erayrhiiqob5ab3unctdmnqiyx3r6kf6rm: region r004 of outputs/fusion_finder/gpt_oss_120b.

35 calls per step, forward, in layers.1.attention_norm, layers.2.attention_norm, layers.3.attention_norm ....
Measured 1583.9 us per call in the capture; ideal (HBM roofline) 116.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels
import torchtitan.ops.scatter_add


def reference(in0, in1, in2, in3, in4, in5):
    v2 = torch.ops.aten.mul.Tensor(in0, in1);  in0 = in1 = None
    v3 = torch.ops.aten._to_copy.default(v2, dtype = torch.bfloat16);  v2 = None
    v6 = torch.ops.torchtitan.deterministic_scatter_add.default(in2, in3, v3);  in2 = in3 = v3 = None
    v8 = torch.ops.aten.add.Tensor(in4, v6);  in4 = v6 = None
    v10 = torch.ops.aten._fused_rms_norm.default(v8, [2880], in5, 1e-05);  in5 = None
    v11 = v10[0]
    v12 = v10[1];  v10 = None
    v13 = torch.ops.aten.alias.default(v12);  v12 = None
    v14 = torch.ops.torchao.mxfp8_quantize.default(v11, True, False, 32, 1, 'e4m3', 'rceil')
    v15 = v14[0]
    v16 = v14[2];  v14 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
    return (v8, v11, v13, v15, v17)


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
