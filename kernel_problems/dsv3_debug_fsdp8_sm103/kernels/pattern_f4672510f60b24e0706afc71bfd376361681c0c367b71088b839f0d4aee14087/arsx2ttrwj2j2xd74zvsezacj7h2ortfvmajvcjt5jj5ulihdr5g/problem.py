"""Kernel problem pattern_f4672510f60b24e0706afc71bfd376361681c0c367b71088b839f0d4aee14087/arsx2ttrwj2j2xd74zvsezacj7h2ortfvmajvcjt5jj5ulihdr5g: region r006 of outputs/fusion_finder/dsv3_debug.

4 calls per step, backward/recompute, in layers.4.feed_forward.router.aux_loss, layers.3.feed_forward.router.aux_loss, layers.2.feed_forward.router.aux_loss ....
Measured 287.9 us per call in the capture; ideal (HBM roofline) 27.0 us.

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
    v2 = torch.ops.aten._fused_rms_norm.default(in0, [256], in1, 1e-06)
    v3 = v2[0]
    v4 = v2[1];  v2 = None
    v7 = torch.ops.aten.add.Tensor(in2, in3);  in2 = in3 = None
    v8 = torch.ops.aten.alias.default(v4);  v4 = None
    v9 = torch.ops.aten.alias.default(v8);  v8 = None
    v10 = torch.ops.aten._fused_rms_norm_backward.default(v7, in0, [256], v9, in1, [True, True]);  v7 = in0 = v9 = in1 = None
    v11 = v10[0]
    v12 = v10[1];  v10 = None
    v13 = torch.ops.aten._to_copy.default(v12, dtype = torch.float32);  v12 = None
    v15 = torch.ops.aten.add.Tensor(in4, v11);  in4 = v11 = None
    v16 = torch.ops.aten.t.default(v15)
    v19 = torch.ops.aten.add.Tensor(in5, in6);  in5 = in6 = None
    v21 = torch.ops.aten._fused_rms_norm.default(v19, [256], in7, 1e-06);  in7 = None
    v22 = v21[0]
    v23 = v21[1];  v21 = None
    v25 = torch.ops.aten.reshape.default(in8, [-1])
    v26 = torch.ops.aten._to_copy.default(v25, dtype = torch.uint8);  v25 = None
    v27 = torch.ops.aten.sort.stable(v26, stable = True);  v26 = None
    v28 = v27[1];  v27 = None
    v29 = torch.ops.aten.floor_divide.default(v28, 3)
    v30 = torch.ops.aten.reshape.default(v29, [-1, 1])
    v31 = torch.ops.aten.expand.default(v30, [-1, 256]);  v30 = None
    v32 = torch.ops.aten.gather.default(v15, 0, v31);  v31 = None
    v33 = torch.ops.aten._to_copy.default(v32, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v32 = None
    v34 = torch.ops.aten.index.Tensor(v22, [v29])
    v36 = torch.ops.aten.sigmoid.default(in9);  in9 = None
    v37 = torch.ops.aten.gather.default(v36, -1, in8);  in8 = None
    v38 = torch.ops.aten.sum.dim_IntList(v37, [-1], True)
    v39 = torch.ops.aten.add.Tensor(v38, 1e-20);  v38 = None
    v40 = torch.ops.aten.div.Tensor(v37, v39)
    v41 = torch.ops.aten.mul.Tensor(v40, 1.0);  v40 = None
    v42 = torch.ops.aten.reshape.default(v41, [-1]);  v41 = None
    v43 = torch.ops.aten.index.Tensor(v42, [v28]);  v42 = None
    v44 = torch.ops.aten.reshape.default(v43, [-1, 1]);  v43 = None
    v45 = torch.ops.aten.mul.Tensor(v33, v44);  v44 = None
    v46 = torch.ops.aten._to_copy.default(v45, dtype = torch.bfloat16);  v45 = None
    v47 = torch.ops.aten.transpose.int(v46, -2, -1)
    v49 = torch.ops.aten._to_copy.default(in10, dtype = torch.float32);  in10 = None
    v50 = torch.ops.aten.sum.dim_IntList(v49, [0]);  v49 = None
    v51 = torch.ops.aten.linalg_vector_norm.default(v36, 1, [-1], True)
    v52 = torch.ops.aten.clamp_min.default(v51, 1e-12)
    v53 = torch.ops.aten.expand.default(v52, [16384, 8]);  v52 = None
    v54 = torch.ops.aten.div.Tensor(v36, v53)
    v56 = torch.ops.aten.bitwise_not.default(in11);  in11 = None
    v57 = torch.ops.aten.mul.Tensor(v54, v56);  v54 = None
    v58 = torch.ops.aten.sum.dim_IntList(v57, [0]);  v57 = None
    v59 = torch.ops.aten.linalg_vector_norm.default(v50, 1, [0], True)
    v60 = torch.ops.aten.clamp_min.default(v59, 1e-12);  v59 = None
    v61 = torch.ops.aten.expand.default(v60, [8]);  v60 = None
    v62 = torch.ops.aten.div.Tensor(v50, v61);  v50 = v61 = None
    v63 = torch.ops.aten.mul.Tensor(v62, 8);  v62 = None
    v64 = torch.ops.aten.mul.Tensor(v63, v58);  v58 = None
    v65 = torch.ops.aten.sum.default(v64);  v64 = None
    v67 = torch.ops.aten.clamp_min.default(in12, 1);  in12 = None
    v68 = torch.ops.aten.reciprocal.default(v67);  v67 = None
    v69 = torch.ops.aten.mul.Tensor(v68, 1.0);  v68 = None
    v70 = torch.ops.aten.mul.Tensor(v69, 0.001);  v69 = None
    v71 = torch.ops.aten.mul.Tensor(v65, v70);  v65 = None
    v72 = torch.ops.aten.ones_like.default(v71, pin_memory = False);  v71 = None
    v73 = torch.ops.aten.mul.Tensor(v72, v70);  v72 = v70 = None
    v74 = torch.ops.aten.expand.default(v73, [8]);  v73 = None
    v75 = torch.ops.aten.mul.Tensor(v74, v63);  v74 = v63 = None
    v76 = torch.ops.aten.unsqueeze.default(v75, 0);  v75 = None
    v77 = torch.ops.aten.expand.default(v76, [16384, 8]);  v76 = None
    v78 = torch.ops.aten.mul.Tensor(v77, v56);  v77 = v56 = None
    v79 = torch.ops.aten.neg.default(v78)
    v80 = torch.ops.aten.div.Tensor(v36, v53)
    v81 = torch.ops.aten.div.Tensor(v80, v53);  v80 = None
    v82 = torch.ops.aten.mul.Tensor(v79, v81);  v79 = v81 = None
    v83 = torch.ops.aten.div.Tensor(v78, v53);  v78 = v53 = None
    v84 = torch.ops.aten.sum.dim_IntList(v82, [1], True);  v82 = None
    v85 = torch.ops.aten.gt.Scalar(v51, 1e-12);  v51 = None
    v86 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v87 = torch.ops.aten.where.self(v85, v84, v86);  v85 = v84 = v86 = None
    v88 = torch.ops.aten.sgn.default(v36)
    v89 = torch.ops.aten.mul.Tensor(v88, v87);  v88 = v87 = None
    v90 = torch.ops.aten.add.Tensor(v83, v89);  v83 = v89 = None
    v91 = torch.ops.aten.div.Tensor(v37, v39);  v37 = None
    v92 = torch.ops.aten.div.Tensor(v91, v39);  v91 = None
    v93 = torch.ops.aten.alias.default(v36);  v36 = None
    v94 = torch.ops.aten.alias.default(v93);  v93 = None
    v95 = torch.ops.aten.alias.default(v23);  v23 = None
    v96 = torch.ops.aten.alias.default(v95);  v95 = None
    return (v3, v13, v15, v16, v19, v22, v28, v29, v33, v34, v39, v46, v47, v90, v92, v94, v96)


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
