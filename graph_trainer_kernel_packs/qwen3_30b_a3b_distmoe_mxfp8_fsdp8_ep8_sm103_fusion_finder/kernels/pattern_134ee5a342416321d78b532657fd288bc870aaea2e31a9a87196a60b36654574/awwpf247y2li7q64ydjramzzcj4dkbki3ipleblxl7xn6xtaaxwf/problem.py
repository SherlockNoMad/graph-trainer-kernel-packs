"""Kernel problem pattern_134ee5a342416321d78b532657fd288bc870aaea2e31a9a87196a60b36654574/awwpf247y2li7q64ydjramzzcj4dkbki3ipleblxl7xn6xtaaxwf: region r012 of outputs/fusion_finder/qwen3_30b.

48 calls per step, forward, in layers.0.feed_forward.router.aux_loss, layers.1.feed_forward.router.aux_loss, layers.2.feed_forward.router.aux_loss ....
Measured 28.0 us per call in the capture; ideal (HBM roofline) 0.0 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4):
    v1 = torch.ops.aten.clamp_min.default(in0, 1);  in0 = None
    v2 = torch.ops.aten.reciprocal.default(v1);  v1 = None
    v3 = torch.ops.aten.mul.Tensor(v2, 1.0);  v2 = None
    v4 = torch.ops.aten.mul.Tensor(v3, 0.001)
    v6 = torch.ops.aten.split_with_sizes.default(in1, [128]);  in1 = None
    v7 = v6[0];  v6 = None
    v8 = torch.ops.aten.view.default(v7, [128]);  v7 = None
    v10 = torch.ops.aten.add_.Tensor(in2, v8);  in2 = v8 = None
    v11 = torch.ops.aten.clone.default(v10);  v10 = None
    v12 = torch.ops.aten.linalg_vector_norm.default(v11, 1, [0], True)
    v13 = torch.ops.aten.clamp_min.default(v12, 1e-12);  v12 = None
    v14 = torch.ops.aten.expand.default(v13, [128]);  v13 = None
    v15 = torch.ops.aten.div.Tensor(v11, v14);  v11 = v14 = None
    v16 = torch.ops.aten.mul.Tensor(v15, 128);  v15 = None
    v18 = torch.ops.aten.mul.Tensor(v16, in3);  in3 = None
    v19 = torch.ops.aten.sum.default(v18);  v18 = None
    v20 = torch.ops.aten.mul.Tensor(v19, v4)
    v21 = torch.ops.aten.mul.Tensor(v19, v3);  v19 = v3 = None
    v23 = torch.ops.aten.add_.Tensor(in4, v21);  in4 = v21 = v23 = None
    return (v4, v16, v20)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4):
        return reference(in0, in1, in2, in3, in4)


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
