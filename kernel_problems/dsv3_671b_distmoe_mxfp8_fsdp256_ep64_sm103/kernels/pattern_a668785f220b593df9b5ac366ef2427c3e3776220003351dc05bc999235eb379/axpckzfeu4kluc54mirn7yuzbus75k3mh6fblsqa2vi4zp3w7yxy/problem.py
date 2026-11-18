"""Kernel problem pattern_a668785f220b593df9b5ac366ef2427c3e3776220003351dc05bc999235eb379/axpckzfeu4kluc54mirn7yuzbus75k3mh6fblsqa2vi4zp3w7yxy: region r029 of outputs/fusion_finder/dsv3_671b.

61 calls per step, backward/recompute, in layers.60.attention.inner_attention.wkv_b, layers.59.attention.inner_attention.wkv_b, layers.58.attention.inner_attention.wkv_b ....
Measured 14.1 us per call in the capture; ideal (HBM roofline) 1.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchao.prototype.mx_formats.kernels


def reference(in0, in1):
    v1 = torch.ops.aten.split_with_sizes.default(in0, [512, 64], -1);  in0 = None
    v2 = v1[0]
    v3 = v1[1];  v1 = None
    v5 = torch.ops.aten._fused_rms_norm.default(v2, [512], in1, 1e-06);  in1 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v8 = torch.ops.torchao.mxfp8_quantize.default(v6, True, True, 32, 32, 'e4m3', 'rceil');  v6 = None
    v9 = v8[0]
    v10 = v8[2]
    v11 = v8[1]
    v12 = v8[3];  v8 = None
    v13 = torch.ops.torchao.triton_mx_block_rearrange.default(v10);  v10 = None
    v14 = torch.ops.aten.unsqueeze.default(v3, 0);  v3 = None
    v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v12);  v12 = None
    v16 = torch.ops.aten.alias.default(v7);  v7 = None
    v17 = torch.ops.aten.alias.default(v16);  v16 = None
    return (v2, v9, v11, v13, v14, v15, v17)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1):
        return reference(in0, in1)


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
