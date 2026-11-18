"""Kernel problem pattern_0fdba30d05e0cc3cef49e887a5f83a823ac123b9cd4cc7cf3b40dc959438eb2d/al6uziplhf75gjmghb2m7xu6muouze23n6t74cmk3qav3lnrm6h3: region r000 of outputs/fusion_finder/dsv3_debug.

6 calls per step, backward/recompute, in layers.5.attention.rope, layers.4.attention.rope, layers.3.attention.rope ....
Measured 518.8 us per call in the capture; ideal (HBM roofline) 69.2 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5):
    v1 = torch.ops.aten.reshape.default(in0, [16384, -1, 192]);  in0 = None
    v3 = torch.ops.aten.split_with_sizes.default(in1, [512, 64], -1);  in1 = None
    v4 = v3[1]
    v5 = v3[0];  v3 = None
    v6 = torch.ops.aten.unsqueeze.default(v4, 1);  v4 = None
    v7 = torch.ops.aten.split_with_sizes.default(v1, [128, 64], -1);  v1 = None
    v8 = v7[0]
    v9 = v7[1];  v7 = None
    v12 = torch.ops.aten.index.Tensor(in2, [in3]);  in2 = in3 = None
    v13 = torch.ops.aten.reshape.default(v12, [16384, 1, 32]);  v12 = None
    v14 = torch.ops.aten._to_copy.default(v9, dtype = torch.float32);  v9 = None
    v15 = torch.ops.aten.reshape.default(v14, [16384, 16, -1, 2]);  v14 = None
    v16 = torch.ops.aten.view_as_complex.default(v15);  v15 = None
    v17 = torch.ops.aten.mul.Tensor(v16, v13);  v16 = None
    v18 = torch.ops.aten.view_as_real.default(v17);  v17 = None
    v19 = torch.ops.aten.reshape.default(v18, [16384, 16, 64]);  v18 = None
    v20 = torch.ops.aten._to_copy.default(v19, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v19 = None
    v21 = torch.ops.aten._to_copy.default(v6, dtype = torch.float32);  v6 = None
    v22 = torch.ops.aten.reshape.default(v21, [16384, 1, -1, 2]);  v21 = None
    v23 = torch.ops.aten.view_as_complex.default(v22);  v22 = None
    v24 = torch.ops.aten.mul.Tensor(v23, v13);  v23 = None
    v25 = torch.ops.aten.view_as_real.default(v24);  v24 = None
    v26 = torch.ops.aten.reshape.default(v25, [16384, 1, 64]);  v25 = None
    v27 = torch.ops.aten._to_copy.default(v26, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v26 = None
    v28 = torch.ops.aten.cat.default([v8, v20], -1);  v8 = v20 = None
    v29 = torch.ops.aten.squeeze.dim(v27, 1);  v27 = None
    v31 = torch.ops.aten.reshape.default(in4, [16384, -1, 256]);  in4 = None
    v32 = torch.ops.aten.split_with_sizes.default(v31, [128, 128], -1);  v31 = None
    v33 = v32[0]
    v34 = v32[1];  v32 = None
    v35 = torch.ops.aten.unsqueeze.default(v29, 1);  v29 = None
    v36 = torch.ops.aten.expand.default(v35, [-1, 16, -1]);  v35 = None
    v37 = torch.ops.aten.cat.default([v33, v36], -1);  v33 = v36 = None
    v38 = torch.ops.aten.transpose.int(v28, 0, 1);  v28 = None
    v39 = torch.ops.aten.unsqueeze.default(v38, 0);  v38 = None
    v40 = torch.ops.aten.transpose.int(v37, 0, 1);  v37 = None
    v41 = torch.ops.aten.unsqueeze.default(v40, 0);  v40 = None
    v42 = torch.ops.aten.transpose.int(v34, 0, 1);  v34 = None
    v43 = torch.ops.aten.unsqueeze.default(v42, 0);  v42 = None
    v45 = torch.ops.aten._fused_rms_norm.default(v5, [512], in5, 1e-06);  in5 = None
    v46 = v45[0]
    v47 = v45[1];  v45 = None
    v48 = torch.ops.aten.alias.default(v47);  v47 = None
    v49 = torch.ops.aten.alias.default(v48);  v48 = None
    v50 = torch.ops.aten._conj.default(v13)
    v51 = torch.ops.aten.clone.default(v50);  v50 = None
    v52 = torch.ops.aten._conj.default(v13);  v13 = None
    v53 = torch.ops.aten.clone.default(v52);  v52 = None
    return (v5, v39, v41, v43, v46, v49, v51, v53)


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
