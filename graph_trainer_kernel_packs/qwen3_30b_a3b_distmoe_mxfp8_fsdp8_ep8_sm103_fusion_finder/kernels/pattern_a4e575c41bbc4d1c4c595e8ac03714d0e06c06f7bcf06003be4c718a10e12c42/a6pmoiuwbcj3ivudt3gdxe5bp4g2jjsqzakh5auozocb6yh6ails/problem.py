"""Kernel problem pattern_a4e575c41bbc4d1c4c595e8ac03714d0e06c06f7bcf06003be4c718a10e12c42/a6pmoiuwbcj3ivudt3gdxe5bp4g2jjsqzakh5auozocb6yh6ails: region r000 of outputs/fusion_finder/qwen3_30b.

48 calls per step, backward, in layers.47.attention.rope, layers.46.attention.rope, layers.45.attention.rope ....
Measured 2555.1 us per call in the capture; ideal (HBM roofline) 192.0 us.

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


def reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
    v1 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in0 = None
    v3 = torch.ops.aten._to_copy.default(in1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in1 = None
    v5 = torch.ops.aten.mul.Tensor(v1, in2)
    v6 = torch.ops.aten.slice.Tensor(v5, 2, 0, 64)
    v7 = torch.ops.aten.slice.Tensor(v5, 2, 64, 128);  v5 = None
    v8 = torch.ops.aten.neg.default(v6);  v6 = None
    v9 = torch.ops.aten.slice_backward.default(v8, [32768, 4, 128], 2, 64, 9223372036854775807, 1);  v8 = None
    v10 = torch.ops.aten.slice_backward.default(v7, [32768, 4, 128], 2, 0, 64, 1);  v7 = None
    v11 = torch.ops.aten.add.Tensor(v9, v10);  v9 = v10 = None
    v13 = torch.ops.aten.mul.Tensor(v1, in3);  v1 = None
    v14 = torch.ops.aten.add.Tensor(v11, v13);  v11 = v13 = None
    v15 = torch.ops.aten._to_copy.default(v14, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v14 = None
    v16 = torch.ops.aten.mul.Tensor(v3, in2);  in2 = None
    v17 = torch.ops.aten.slice.Tensor(v16, 2, 0, 64)
    v18 = torch.ops.aten.slice.Tensor(v16, 2, 64, 128);  v16 = None
    v19 = torch.ops.aten.neg.default(v17);  v17 = None
    v20 = torch.ops.aten.slice_backward.default(v19, [32768, 32, 128], 2, 64, 9223372036854775807, 1);  v19 = None
    v21 = torch.ops.aten.slice_backward.default(v18, [32768, 32, 128], 2, 0, 64, 1);  v18 = None
    v22 = torch.ops.aten.add.Tensor(v20, v21);  v20 = v21 = None
    v23 = torch.ops.aten.mul.Tensor(v3, in3);  v3 = in3 = None
    v24 = torch.ops.aten.add.Tensor(v22, v23);  v22 = v23 = None
    v25 = torch.ops.aten._to_copy.default(v24, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v24 = None
    v27 = torch.ops.aten.alias.default(in4);  in4 = None
    v30 = torch.ops.aten._fused_rms_norm_backward.default(v15, in5, [128], v27, in6, [True, True]);  v15 = in5 = v27 = in6 = None
    v31 = v30[0]
    v32 = v30[1];  v30 = None
    v34 = torch.ops.aten.alias.default(in7);  in7 = None
    v37 = torch.ops.aten._fused_rms_norm_backward.default(v25, in8, [128], v34, in9, [True, True]);  v25 = in8 = v34 = in9 = None
    v38 = v37[0]
    v39 = v37[1];  v37 = None
    v41 = torch.ops.aten.reshape.default(in10, [32768, 4, 1, 128]);  in10 = None
    v42 = torch.ops.aten.reshape.default(v31, [32768, 4, 1, 128]);  v31 = None
    v43 = torch.ops.aten.reshape.default(v38, [32768, 4, 8, 128]);  v38 = None
    v44 = torch.ops.aten.cat.default([v43, v42, v41], 2);  v43 = v42 = v41 = None
    v45 = torch.ops.aten.reshape.default(v44, [32768, 5120]);  v44 = None
    v46 = torch.ops.torchao.mxfp8_quantize.default(v45, True, True, 32, 32, 'e4m3', 'rceil');  v45 = None
    v47 = v46[0]
    v48 = v46[1]
    v49 = v46[2]
    v50 = v46[3];  v46 = None
    v51 = torch.ops.torchao.triton_mx_block_rearrange.default(v49);  v49 = None
    v52 = torch.ops.torchao.triton_mx_block_rearrange.default(v50);  v50 = None
    v53 = torch.ops.aten.t.default(v48);  v48 = None
    return (v32, v39, v47, v51, v52, v53)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10)


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
