"""Kernel problem pattern_59c97116ae464dff62026ee0feb97935e38a375930a296284868978f0e2aa8be/ahd3jcbqicmu7ue3eplwodwxba642wsmpq6w6o43tpe6s5jt2goq: region r003 of outputs/fusion_finder/qwen3_30b.

8 calls per step, backward, in loss.
Measured 2596.1 us per call in the capture; ideal (HBM roofline) 895.3 us.

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
    v2 = torch.ops.aten.div.Tensor(in0, in1);  in0 = in1 = None
    v6 = torch.ops.aten.nll_loss_backward.default(v2, in2, in3, None, 2, -100, in4);  v2 = in2 = in3 = in4 = None
    v8 = torch.ops.aten.alias.default(in5);  in5 = None
    v9 = torch.ops.aten._log_softmax_backward_data.default(v6, v8, 1, torch.float32);  v6 = v8 = None
    v10 = torch.ops.aten._to_copy.default(v9, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v9 = None
    v11 = torch.ops.aten.t.default(v10)
    return (v10, v11)


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
