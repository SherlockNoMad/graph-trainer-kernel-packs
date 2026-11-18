"""Kernel problem pattern_981adcd534dac44bdd178933c8002dbfc46c04f288f2718e4b5268ab67d17c29/dactgok6mjbgu2jqbr7dmmovxt6h5nwkmtue4ry7cwapxht2lvs3: region r020 of outputs/fusion_finder/gpt_oss_120b.

1 call per step, forward, in lm_head.
Measured 1640.3 us per call in the capture; ideal (HBM roofline) 129.7 us.

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
    v14 = torch.ops.aten.split_with_sizes.default(v11, [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])
    v15 = v14[0]
    v16 = v14[1]
    v17 = v14[2]
    v18 = v14[3]
    v19 = v14[4]
    v20 = v14[5]
    v21 = v14[6]
    v22 = v14[7];  v14 = None
    v23 = torch.ops.aten.alias.default(v15);  v15 = None
    v24 = torch.ops.aten.alias.default(v16);  v16 = None
    v25 = torch.ops.aten.alias.default(v17);  v17 = None
    v26 = torch.ops.aten.alias.default(v18);  v18 = None
    v27 = torch.ops.aten.alias.default(v19);  v19 = None
    v28 = torch.ops.aten.alias.default(v20);  v20 = None
    v29 = torch.ops.aten.alias.default(v21);  v21 = None
    v30 = torch.ops.aten.alias.default(v22);  v22 = None
    v31 = torch.ops.aten.zeros_like.default(v11, dtype = torch.bfloat16, pin_memory = False)
    v32 = torch.ops.aten.new_zeros.default(v11, [], dtype = torch.float32, pin_memory = False);  v11 = None
    v33 = torch.ops.torchao.mxfp8_quantize.default(v23, True, False, 32, 1, 'e4m3', 'rceil')
    v34 = v33[0]
    v35 = v33[2];  v33 = None
    v36 = torch.ops.torchao.triton_mx_block_rearrange.default(v35);  v35 = None
    v37 = torch.ops.aten.slice.Tensor(v31, 0, 2048, 4096)
    v38 = torch.ops.torchao.mxfp8_quantize.default(v24, True, False, 32, 1, 'e4m3', 'rceil')
    v39 = v38[0]
    v40 = v38[2];  v38 = None
    v41 = torch.ops.torchao.triton_mx_block_rearrange.default(v40);  v40 = None
    v42 = torch.ops.aten.slice.Tensor(v31, 0, 4096, 6144)
    v43 = torch.ops.torchao.mxfp8_quantize.default(v25, True, False, 32, 1, 'e4m3', 'rceil')
    v44 = v43[0]
    v45 = v43[2];  v43 = None
    v46 = torch.ops.torchao.triton_mx_block_rearrange.default(v45);  v45 = None
    v47 = torch.ops.aten.slice.Tensor(v31, 0, 6144, 8192)
    v48 = torch.ops.torchao.mxfp8_quantize.default(v26, True, False, 32, 1, 'e4m3', 'rceil')
    v49 = v48[0]
    v50 = v48[2];  v48 = None
    v51 = torch.ops.torchao.triton_mx_block_rearrange.default(v50);  v50 = None
    v52 = torch.ops.aten.slice.Tensor(v31, 0, 8192, 10240)
    v53 = torch.ops.torchao.mxfp8_quantize.default(v27, True, False, 32, 1, 'e4m3', 'rceil')
    v54 = v53[0]
    v55 = v53[2];  v53 = None
    v56 = torch.ops.torchao.triton_mx_block_rearrange.default(v55);  v55 = None
    v57 = torch.ops.aten.slice.Tensor(v31, 0, 10240, 12288)
    v58 = torch.ops.torchao.mxfp8_quantize.default(v28, True, False, 32, 1, 'e4m3', 'rceil')
    v59 = v58[0]
    v60 = v58[2];  v58 = None
    v61 = torch.ops.torchao.triton_mx_block_rearrange.default(v60);  v60 = None
    v62 = torch.ops.aten.slice.Tensor(v31, 0, 12288, 14336)
    v63 = torch.ops.torchao.mxfp8_quantize.default(v29, True, False, 32, 1, 'e4m3', 'rceil')
    v64 = v63[0]
    v65 = v63[2];  v63 = None
    v66 = torch.ops.torchao.triton_mx_block_rearrange.default(v65);  v65 = None
    v67 = torch.ops.aten.slice.Tensor(v31, 0, 14336, 16384)
    v68 = torch.ops.torchao.mxfp8_quantize.default(v30, True, False, 32, 1, 'e4m3', 'rceil')
    v69 = v68[0]
    v70 = v68[2];  v68 = None
    v71 = torch.ops.torchao.triton_mx_block_rearrange.default(v70);  v70 = None
    v72 = torch.ops.aten.slice.Tensor(v31, 0, 0, 2048)
    return (v8, v13, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v34, v36, v37, v39, v41, v42, v44, v46, v47, v49, v51, v52, v54, v56, v57, v59, v61, v62, v64, v66, v67, v69, v71, v72)


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
