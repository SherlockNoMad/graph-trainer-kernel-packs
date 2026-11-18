"""Kernel problem pattern_e2f4d946a469eb54e4a48ed6d3f852a9251704813da5efb8279a812a33c5bbc7/am3pp6qdtx3vuuzqpt5rbp7ep3alx2kvmvssm2thkrdehyytbi5t: region r002 of outputs/fusion_finder/qwen3_30b.

48 calls per step, backward, in layers.47.feed_forward.router.gate, layers.46.feed_forward.router.gate, layers.45.feed_forward.router.gate ....
Measured 391.8 us per call in the capture; ideal (HBM roofline) 9.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
    v2 = torch.ops.aten.masked_fill.Scalar(in0, in1, 0);  in0 = in1 = None
    v4 = torch.ops.aten.ones_like.default(in2, pin_memory = False);  in2 = None
    v6 = torch.ops.aten.mul.Tensor(v4, in3);  v4 = in3 = None
    v7 = torch.ops.aten.expand.default(v6, [128]);  v6 = None
    v9 = torch.ops.aten.mul.Tensor(v7, in4);  v7 = in4 = None
    v10 = torch.ops.aten.unsqueeze.default(v9, 0);  v9 = None
    v11 = torch.ops.aten.expand.default(v10, [32768, 128]);  v10 = None
    v13 = torch.ops.aten.mul.Tensor(v11, in5);  v11 = in5 = None
    v14 = torch.ops.aten.neg.default(v13)
    v17 = torch.ops.aten.div.Tensor(in6, in7)
    v18 = torch.ops.aten.div.Tensor(v17, in7);  v17 = None
    v19 = torch.ops.aten.mul.Tensor(v14, v18);  v14 = v18 = None
    v20 = torch.ops.aten.div.Tensor(v13, in7);  v13 = in7 = None
    v21 = torch.ops.aten.sum.dim_IntList(v19, [1], True);  v19 = None
    v23 = torch.ops.aten.gt.Scalar(in8, 1e-12);  in8 = None
    v24 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v25 = torch.ops.aten.where.self(v23, v21, v24);  v23 = v21 = v24 = None
    v26 = torch.ops.aten.sgn.default(in6);  in6 = None
    v27 = torch.ops.aten.mul.Tensor(v26, v25);  v26 = v25 = None
    v28 = torch.ops.aten.add.Tensor(v20, v27);  v20 = v27 = None
    v29 = torch.ops.aten.mul.Tensor(v2, 1.0);  v2 = None
    v30 = torch.ops.aten.neg.default(v29)
    v33 = torch.ops.aten.div.Tensor(in9, in10);  in9 = None
    v34 = torch.ops.aten.div.Tensor(v33, in10);  v33 = None
    v35 = torch.ops.aten.mul.Tensor(v30, v34);  v30 = v34 = None
    v36 = torch.ops.aten.div.Tensor(v29, in10);  v29 = in10 = None
    v37 = torch.ops.aten.sum.dim_IntList(v35, [1], True);  v35 = None
    v38 = torch.ops.aten.expand.default(v37, [32768, 8]);  v37 = None
    v39 = torch.ops.aten.add.Tensor(v36, v38);  v36 = v38 = None
    v40 = torch.ops.aten.new_zeros.default(v39, [32768, 128])
    v42 = torch.ops.aten.scatter_add.default(v40, -1, in11, v39);  v40 = in11 = v39 = None
    v43 = torch.ops.aten.add.Tensor(v28, v42);  v28 = v42 = None
    v45 = torch.ops.aten.alias.default(in12);  in12 = None
    v46 = torch.ops.aten._softmax_backward_data.default(v43, v45, -1, torch.float32);  v43 = v45 = None
    v47 = torch.ops.aten._to_copy.default(v46, dtype = torch.bfloat16)
    v48 = torch.ops.aten.view.dtype(v47, torch.int16)
    v49 = torch.ops.aten.alias.default(v48);  v48 = None
    v50 = torch.ops.aten._to_copy.default(v49, dtype = torch.int32);  v49 = None
    v51 = torch.ops.aten.__lshift__.Scalar(v50, 16);  v50 = None
    v52 = torch.ops.aten.view.dtype(v51, torch.float32);  v51 = None
    v53 = torch.ops.aten.alias.default(v52);  v52 = None
    v54 = torch.ops.aten.sub.Tensor(v46, v53);  v46 = v53 = None
    v55 = torch.ops.aten._to_copy.default(v54, dtype = torch.bfloat16)
    v56 = torch.ops.aten.view.dtype(v55, torch.int16)
    v57 = torch.ops.aten.alias.default(v56);  v56 = None
    v58 = torch.ops.aten._to_copy.default(v57, dtype = torch.int32);  v57 = None
    v59 = torch.ops.aten.__lshift__.Scalar(v58, 16);  v58 = None
    v60 = torch.ops.aten.view.dtype(v59, torch.float32);  v59 = None
    v61 = torch.ops.aten.alias.default(v60);  v60 = None
    v62 = torch.ops.aten.sub.Tensor(v54, v61);  v54 = v61 = None
    v63 = torch.ops.aten._to_copy.default(v62, dtype = torch.bfloat16);  v62 = None
    v64 = torch.ops.aten.cat.default([v47, v55, v63], 1);  v47 = v55 = v63 = None
    v65 = torch.ops.aten.split.Tensor(v64, 128, 1)
    v66 = v65[0]
    v67 = v65[1]
    v68 = v65[2];  v65 = None
    v69 = torch.ops.aten.permute.default(v66, [1, 0]);  v66 = None
    v70 = torch.ops.aten.permute.default(v67, [1, 0]);  v67 = None
    v71 = torch.ops.aten.permute.default(v68, [1, 0]);  v68 = None
    return (v64, v69, v70, v71)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12)


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
