"""Kernel problem pattern_49355e705b459265df40610e8e7f753387384754be60f13f6f5e8a7797cd6305/dhlsz5txs7zd7zdl2bi2uk4t66yocihbfdamsjhc24fkmzvzxa3l: region r019 of outputs/fusion_finder/gpt_oss_120b.

1 call per step, backward, in layers.35.feed_forward.router.gate.
Measured 2129.4 us per call in the capture; ideal (HBM roofline) 154.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18):
    v1 = torch.ops.aten.alias.default(in0);  in0 = None
    v5 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [2880], v1, in3, [True, True]);  in1 = in2 = v1 = in3 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v9 = torch.ops.aten.gather.default(v6, 0, in4);  in4 = None
    v10 = torch.ops.aten._to_copy.default(v9, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v9 = None
    v12 = torch.ops.aten.mul.Tensor(v10, in5);  in5 = None
    v14 = torch.ops.aten.mul.Tensor(v10, in6);  v10 = in6 = None
    v15 = torch.ops.aten._to_copy.default(v14, dtype = torch.bfloat16);  v14 = None
    v16 = torch.ops.aten.sum.dim_IntList(v12, [1], True);  v12 = None
    v17 = torch.ops.aten.reshape.default(v16, [65536]);  v16 = None
    v18 = torch.ops.aten.new_zeros.default(v17, [65536], dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v20 = torch.ops.aten.index_put.default(v18, [in7], v17, True);  v18 = in7 = v17 = None
    v21 = torch.ops.aten.reshape.default(v20, [16384, 4]);  v20 = None
    v23 = torch.ops.aten.ones_like.default(in8, pin_memory = False);  in8 = None
    v25 = torch.ops.aten.mul.Tensor(v23, in9);  v23 = in9 = None
    v26 = torch.ops.aten.expand.default(v25, [128]);  v25 = None
    v28 = torch.ops.aten.mul.Tensor(v26, in10);  v26 = in10 = None
    v29 = torch.ops.aten.unsqueeze.default(v28, 0);  v28 = None
    v30 = torch.ops.aten.expand.default(v29, [16384, 128]);  v29 = None
    v32 = torch.ops.aten.mul.Tensor(v30, in11);  v30 = in11 = None
    v33 = torch.ops.aten.neg.default(v32)
    v36 = torch.ops.aten.div.Tensor(in12, in13)
    v37 = torch.ops.aten.div.Tensor(v36, in13);  v36 = None
    v38 = torch.ops.aten.mul.Tensor(v33, v37);  v33 = v37 = None
    v39 = torch.ops.aten.div.Tensor(v32, in13);  v32 = in13 = None
    v40 = torch.ops.aten.sum.dim_IntList(v38, [1], True);  v38 = None
    v42 = torch.ops.aten.gt.Scalar(in14, 1e-12);  in14 = None
    v43 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v44 = torch.ops.aten.where.self(v42, v40, v43);  v42 = v40 = v43 = None
    v45 = torch.ops.aten.sgn.default(in12);  in12 = None
    v46 = torch.ops.aten.mul.Tensor(v45, v44);  v45 = v44 = None
    v47 = torch.ops.aten.add.Tensor(v39, v46);  v39 = v46 = None
    v48 = torch.ops.aten.mul.Tensor(v21, 1.0);  v21 = None
    v49 = torch.ops.aten.neg.default(v48)
    v52 = torch.ops.aten.div.Tensor(in15, in16);  in15 = None
    v53 = torch.ops.aten.div.Tensor(v52, in16);  v52 = None
    v54 = torch.ops.aten.mul.Tensor(v49, v53);  v49 = v53 = None
    v55 = torch.ops.aten.div.Tensor(v48, in16);  v48 = in16 = None
    v56 = torch.ops.aten.sum.dim_IntList(v54, [1], True);  v54 = None
    v57 = torch.ops.aten.expand.default(v56, [16384, 4]);  v56 = None
    v58 = torch.ops.aten.add.Tensor(v55, v57);  v55 = v57 = None
    v59 = torch.ops.aten.new_zeros.default(v58, [16384, 128])
    v61 = torch.ops.aten.scatter_add.default(v59, -1, in17, v58);  v59 = in17 = v58 = None
    v62 = torch.ops.aten.add.Tensor(v47, v61);  v47 = v61 = None
    v64 = torch.ops.aten.alias.default(in18);  in18 = None
    v65 = torch.ops.aten._softmax_backward_data.default(v62, v64, -1, torch.float32);  v62 = v64 = None
    v66 = torch.ops.aten.sum.dim_IntList(v65, [0], True)
    v67 = torch.ops.aten.reshape.default(v66, [128]);  v66 = None
    v68 = torch.ops.aten._to_copy.default(v67, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v67 = None
    v69 = torch.ops.aten._to_copy.default(v65, dtype = torch.bfloat16)
    v70 = torch.ops.aten.view.dtype(v69, torch.int16)
    v71 = torch.ops.aten.alias.default(v70);  v70 = None
    v72 = torch.ops.aten._to_copy.default(v71, dtype = torch.int32);  v71 = None
    v73 = torch.ops.aten.__lshift__.Scalar(v72, 16);  v72 = None
    v74 = torch.ops.aten.view.dtype(v73, torch.float32);  v73 = None
    v75 = torch.ops.aten.alias.default(v74);  v74 = None
    v76 = torch.ops.aten.sub.Tensor(v65, v75);  v65 = v75 = None
    v77 = torch.ops.aten._to_copy.default(v76, dtype = torch.bfloat16)
    v78 = torch.ops.aten.view.dtype(v77, torch.int16)
    v79 = torch.ops.aten.alias.default(v78);  v78 = None
    v80 = torch.ops.aten._to_copy.default(v79, dtype = torch.int32);  v79 = None
    v81 = torch.ops.aten.__lshift__.Scalar(v80, 16);  v80 = None
    v82 = torch.ops.aten.view.dtype(v81, torch.float32);  v81 = None
    v83 = torch.ops.aten.alias.default(v82);  v82 = None
    v84 = torch.ops.aten.sub.Tensor(v76, v83);  v76 = v83 = None
    v85 = torch.ops.aten._to_copy.default(v84, dtype = torch.bfloat16);  v84 = None
    v86 = torch.ops.aten.cat.default([v69, v77, v85], 1);  v69 = v77 = v85 = None
    v87 = torch.ops.aten.split.Tensor(v86, 128, 1)
    v88 = v87[0]
    v89 = v87[1]
    v90 = v87[2];  v87 = None
    v91 = torch.ops.aten.permute.default(v88, [1, 0]);  v88 = None
    v92 = torch.ops.aten.permute.default(v89, [1, 0]);  v89 = None
    v93 = torch.ops.aten.permute.default(v90, [1, 0]);  v90 = None
    return (v6, v7, v15, v68, v86, v91, v92, v93)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18)


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
