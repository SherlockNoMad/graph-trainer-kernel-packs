"""Kernel problem pattern_05759d92dcffeb169928d27dca4ee61308bc315ed6180d32bd5d3ce2c13b5004/axjjlqbvxalnpbxeakybln3ltdxngi7qbehydjdsb2dy2vd2nhpa: region r004 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward, in layers.60.attention.inner_attention, layers.59.attention.inner_attention, layers.58.attention.inner_attention ....
Measured 298.4 us per call in the capture; ideal (HBM roofline) 188.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels
import torchtitan_recipes.overrides.fused_mla


def reference(in0, in1, in2, in3, in4, in5, in6):
    v1 = torch.ops.aten.unsqueeze.default(in0, 0);  in0 = None
    v3 = torch.ops.aten.unsqueeze.default(in1, 0);  in1 = None
    v6 = torch.ops.torchtitan.fused_mla_kv_backward.default(v3, v1, in2, in3, 128, 64);  v3 = v1 = in2 = in3 = None
    v7 = v6[0]
    v8 = v6[1];  v6 = None
    v9 = torch.ops.aten.squeeze.dim(v8, 0);  v8 = None
    v10 = torch.ops.aten.squeeze.dim(v7, 0);  v7 = None
    v11 = torch.ops.aten.reshape.default(v10, [4096, 32768]);  v10 = None
    v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, True, 32, 32, 'e4m3', 'rceil');  v11 = None
    v13 = v12[0]
    v14 = v12[1]
    v15 = v12[2]
    v16 = v12[3];  v12 = None
    v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
    v18 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
    v19 = torch.ops.aten.t.default(v14);  v14 = None
    v21 = torch.ops.aten.unsqueeze.default(in4, 0);  in4 = None
    v24 = torch.ops.torchtitan.fused_mla_q_rope_.default(v21, in5, in6, 128, True);  v21 = in5 = in6 = None
    v25 = torch.ops.aten.squeeze.dim(v24, 0);  v24 = None
    v26 = torch.ops.aten.reshape.default(v25, [4096, 24576]);  v25 = None
    v27 = torch.ops.torchao.mxfp8_quantize.default(v26, True, True, 32, 32, 'e4m3', 'rceil');  v26 = None
    v28 = v27[0]
    v29 = v27[1]
    v30 = v27[2]
    v31 = v27[3];  v27 = None
    v32 = torch.ops.torchao.triton_mx_block_rearrange.default(v30);  v30 = None
    v33 = torch.ops.torchao.triton_mx_block_rearrange.default(v31);  v31 = None
    v34 = torch.ops.aten.t.default(v29);  v29 = None
    return (v9, v13, v17, v18, v19, v28, v32, v33, v34)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6):
        return reference(in0, in1, in2, in3, in4, in5, in6)


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
