"""Kernel problem pattern_927270379598229cd924653f56bdf733162017ea0efcb1f72db03e3fe2bb5168/aapana3iaytkl4s2eqvbxql4wydcm3hwwqccwagt3ejn7ulwbroo: region r003 of outputs/fusion_finder/dsv3_671b.

58 calls per step, backward/recompute, in layers.60.feed_forward.router.aux_loss, layers.59.feed_forward.router.aux_loss, layers.58.feed_forward.router.aux_loss ....
Measured 143.0 us per call in the capture; ideal (HBM roofline) 2.0 us.

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
    v1 = torch.ops.aten.sigmoid.default(in0);  in0 = None
    v3 = torch.ops.aten.gather.default(v1, -1, in1);  in1 = None
    v4 = torch.ops.aten.sum.dim_IntList(v3, [-1], True)
    v5 = torch.ops.aten.add.Tensor(v4, 1e-20);  v4 = None
    v6 = torch.ops.aten.div.Tensor(v3, v5)
    v7 = torch.ops.aten.mul.Tensor(v6, 2.5);  v6 = None
    v9 = torch.ops.aten.masked_fill.Scalar(v7, in2, 0.0);  v7 = in2 = None
    v11 = torch.ops.aten._to_copy.default(in3, dtype = torch.float32);  in3 = None
    v12 = torch.ops.aten.sum.dim_IntList(v11, [0]);  v11 = None
    v13 = torch.ops.aten.linalg_vector_norm.default(v1, 1, [-1], True)
    v14 = torch.ops.aten.clamp_min.default(v13, 1e-12)
    v15 = torch.ops.aten.expand.default(v14, [4096, 256]);  v14 = None
    v16 = torch.ops.aten.div.Tensor(v1, v15)
    v18 = torch.ops.aten.bitwise_not.default(in4);  in4 = None
    v19 = torch.ops.aten.mul.Tensor(v16, v18);  v16 = None
    v20 = torch.ops.aten.sum.dim_IntList(v19, [0]);  v19 = None
    v21 = torch.ops.aten.linalg_vector_norm.default(v12, 1, [0], True)
    v22 = torch.ops.aten.clamp_min.default(v21, 1e-12);  v21 = None
    v23 = torch.ops.aten.expand.default(v22, [256]);  v22 = None
    v24 = torch.ops.aten.div.Tensor(v12, v23);  v12 = v23 = None
    v25 = torch.ops.aten.mul.Tensor(v24, 256);  v24 = None
    v26 = torch.ops.aten.mul.Tensor(v25, v20);  v20 = None
    v27 = torch.ops.aten.sum.default(v26);  v26 = None
    v29 = torch.ops.aten.clamp_min.default(in5, 1);  in5 = None
    v30 = torch.ops.aten.reciprocal.default(v29);  v29 = None
    v31 = torch.ops.aten.mul.Tensor(v30, 1.0);  v30 = None
    v32 = torch.ops.aten.mul.Tensor(v31, 0.001);  v31 = None
    v33 = torch.ops.aten.mul.Tensor(v27, v32);  v27 = None
    v34 = torch.ops.aten.ones_like.default(v33, pin_memory = False);  v33 = None
    v35 = torch.ops.aten.mul.Tensor(v34, v32);  v34 = v32 = None
    v36 = torch.ops.aten.expand.default(v35, [256]);  v35 = None
    v37 = torch.ops.aten.mul.Tensor(v36, v25);  v36 = v25 = None
    v38 = torch.ops.aten.unsqueeze.default(v37, 0);  v37 = None
    v39 = torch.ops.aten.expand.default(v38, [4096, 256]);  v38 = None
    v40 = torch.ops.aten.mul.Tensor(v39, v18);  v39 = v18 = None
    v41 = torch.ops.aten.neg.default(v40)
    v42 = torch.ops.aten.div.Tensor(v1, v15)
    v43 = torch.ops.aten.div.Tensor(v42, v15);  v42 = None
    v44 = torch.ops.aten.mul.Tensor(v41, v43);  v41 = v43 = None
    v45 = torch.ops.aten.div.Tensor(v40, v15);  v40 = v15 = None
    v46 = torch.ops.aten.sum.dim_IntList(v44, [1], True);  v44 = None
    v47 = torch.ops.aten.gt.Scalar(v13, 1e-12);  v13 = None
    v48 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
    v49 = torch.ops.aten.where.self(v47, v46, v48);  v47 = v46 = v48 = None
    v50 = torch.ops.aten.sgn.default(v1)
    v51 = torch.ops.aten.mul.Tensor(v50, v49);  v50 = v49 = None
    v52 = torch.ops.aten.add.Tensor(v45, v51);  v45 = v51 = None
    v53 = torch.ops.aten.div.Tensor(v3, v5);  v3 = None
    v54 = torch.ops.aten.div.Tensor(v53, v5);  v53 = None
    v55 = torch.ops.aten.alias.default(v1);  v1 = None
    v56 = torch.ops.aten.alias.default(v55);  v55 = None
    return (v5, v9, v52, v54, v56)


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
