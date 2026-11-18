"""Kernel problem pattern_64b322d0868f8bd60187abb4c0cd5d804e7b5da9eda9afaa3de86d3a7e98cd92/dbmeg4477wnuzkft3usjjhjjzfp5enstkkpunnirjsnaq3n36zqk: region r002 of outputs/fusion_finder/gpt_oss_120b.

35 calls per step, backward, in layers.34.feed_forward.router.gate, layers.33.feed_forward.router.gate, layers.32.feed_forward.router.gate ....
Measured 2172.9 us per call in the capture; ideal (HBM roofline) 181.3 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

from torch import device
import torch


def reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19):
    v1 = torch.ops.aten.alias.default(in0);  in0 = None
    v5 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [2880], v1, in3, [True, True]);  in1 = in2 = v1 = in3 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v9 = torch.ops.aten.add.Tensor(in4, v6);  in4 = v6 = None
    v11 = torch.ops.aten.gather.default(v9, 0, in5);  in5 = None
    v12 = torch.ops.aten._to_copy.default(v11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v11 = None
    v14 = torch.ops.aten.mul.Tensor(v12, in6);  in6 = None
    v16 = torch.ops.aten.mul.Tensor(v12, in7);  v12 = in7 = None
    v17 = torch.ops.aten._to_copy.default(v16, dtype = torch.bfloat16);  v16 = None
    v18 = torch.ops.aten.sum.dim_IntList(v14, [1], True);  v14 = None
    v19 = torch.ops.aten.reshape.default(v18, [65536]);  v18 = None
    v20 = torch.ops.aten.new_zeros.default(v19, [65536], dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v22 = torch.ops.aten.index_put.default(v20, [in8], v19, True);  v20 = in8 = v19 = None
    v23 = torch.ops.aten.reshape.default(v22, [16384, 4]);  v22 = None
    v25 = torch.ops.aten.ones_like.default(in9, pin_memory = False);  in9 = None
    v27 = torch.ops.aten.mul.Tensor(v25, in10);  v25 = in10 = None
    v28 = torch.ops.aten.expand.default(v27, [128]);  v27 = None
    v30 = torch.ops.aten.mul.Tensor(v28, in11);  v28 = in11 = None
    v31 = torch.ops.aten.unsqueeze.default(v30, 0);  v30 = None
    v32 = torch.ops.aten.expand.default(v31, [16384, 128]);  v31 = None
    v34 = torch.ops.aten.mul.Tensor(v32, in12);  v32 = in12 = None
    v35 = torch.ops.aten.neg.default(v34)
    v38 = torch.ops.aten.div.Tensor(in13, in14)
    v39 = torch.ops.aten.div.Tensor(v38, in14);  v38 = None
    v40 = torch.ops.aten.mul.Tensor(v35, v39);  v35 = v39 = None
    v41 = torch.ops.aten.div.Tensor(v34, in14);  v34 = in14 = None
    v42 = torch.ops.aten.sum.dim_IntList(v40, [1], True);  v40 = None
    v44 = torch.ops.aten.gt.Scalar(in15, 1e-12);  in15 = None
    v45 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v46 = torch.ops.aten.where.self(v44, v42, v45);  v44 = v42 = v45 = None
    v47 = torch.ops.aten.sgn.default(in13);  in13 = None
    v48 = torch.ops.aten.mul.Tensor(v47, v46);  v47 = v46 = None
    v49 = torch.ops.aten.add.Tensor(v41, v48);  v41 = v48 = None
    v50 = torch.ops.aten.mul.Tensor(v23, 1.0);  v23 = None
    v51 = torch.ops.aten.neg.default(v50)
    v54 = torch.ops.aten.div.Tensor(in16, in17);  in16 = None
    v55 = torch.ops.aten.div.Tensor(v54, in17);  v54 = None
    v56 = torch.ops.aten.mul.Tensor(v51, v55);  v51 = v55 = None
    v57 = torch.ops.aten.div.Tensor(v50, in17);  v50 = in17 = None
    v58 = torch.ops.aten.sum.dim_IntList(v56, [1], True);  v56 = None
    v59 = torch.ops.aten.expand.default(v58, [16384, 4]);  v58 = None
    v60 = torch.ops.aten.add.Tensor(v57, v59);  v57 = v59 = None
    v61 = torch.ops.aten.new_zeros.default(v60, [16384, 128])
    v63 = torch.ops.aten.scatter_add.default(v61, -1, in18, v60);  v61 = in18 = v60 = None
    v64 = torch.ops.aten.add.Tensor(v49, v63);  v49 = v63 = None
    v66 = torch.ops.aten.alias.default(in19);  in19 = None
    v67 = torch.ops.aten._softmax_backward_data.default(v64, v66, -1, torch.float32);  v64 = v66 = None
    v68 = torch.ops.aten.sum.dim_IntList(v67, [0], True)
    v69 = torch.ops.aten.reshape.default(v68, [128]);  v68 = None
    v70 = torch.ops.aten._to_copy.default(v69, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v69 = None
    v71 = torch.ops.aten._to_copy.default(v67, dtype = torch.bfloat16)
    v72 = torch.ops.aten.view.dtype(v71, torch.int16)
    v73 = torch.ops.aten.alias.default(v72);  v72 = None
    v74 = torch.ops.aten._to_copy.default(v73, dtype = torch.int32);  v73 = None
    v75 = torch.ops.aten.__lshift__.Scalar(v74, 16);  v74 = None
    v76 = torch.ops.aten.view.dtype(v75, torch.float32);  v75 = None
    v77 = torch.ops.aten.alias.default(v76);  v76 = None
    v78 = torch.ops.aten.sub.Tensor(v67, v77);  v67 = v77 = None
    v79 = torch.ops.aten._to_copy.default(v78, dtype = torch.bfloat16)
    v80 = torch.ops.aten.view.dtype(v79, torch.int16)
    v81 = torch.ops.aten.alias.default(v80);  v80 = None
    v82 = torch.ops.aten._to_copy.default(v81, dtype = torch.int32);  v81 = None
    v83 = torch.ops.aten.__lshift__.Scalar(v82, 16);  v82 = None
    v84 = torch.ops.aten.view.dtype(v83, torch.float32);  v83 = None
    v85 = torch.ops.aten.alias.default(v84);  v84 = None
    v86 = torch.ops.aten.sub.Tensor(v78, v85);  v78 = v85 = None
    v87 = torch.ops.aten._to_copy.default(v86, dtype = torch.bfloat16);  v86 = None
    v88 = torch.ops.aten.cat.default([v71, v79, v87], 1);  v71 = v79 = v87 = None
    v89 = torch.ops.aten.split.Tensor(v88, 128, 1)
    v90 = v89[0]
    v91 = v89[1]
    v92 = v89[2];  v89 = None
    v93 = torch.ops.aten.permute.default(v90, [1, 0]);  v90 = None
    v94 = torch.ops.aten.permute.default(v91, [1, 0]);  v91 = None
    v95 = torch.ops.aten.permute.default(v92, [1, 0]);  v92 = None
    return (v7, v9, v17, v70, v88, v93, v94, v95)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19):
        return reference(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19)


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
