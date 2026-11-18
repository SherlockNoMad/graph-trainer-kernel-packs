"""Kernel problem pattern_66ddca5025af039fc8c82be8721516b8eab9441dfa02c1157c9dea847dbc3746/apoumr2kj2eufyzc5opa3cwyr5gzngm4zkntr7mrme5kr2ydj3o3: region r005 of outputs/fusion_finder/qwen3_30b.

1 call per step, forward, in loss.
Measured 2046.3 us per call in the capture; ideal (HBM roofline) 537.2 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3):
    v1 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32);  in0 = None
    v2 = torch.ops.aten._log_softmax.default(v1, 1, False);  v1 = None
    v3 = torch.ops.aten.alias.default(v2)
    v5 = torch.ops.aten.nll_loss_forward.default(v2, in1, None, 2, -100);  in1 = None
    v6 = v5[0]
    v7 = v5[1];  v5 = None
    v9 = torch.ops.aten.div.Tensor(v6, in2);  v6 = in2 = None
    v10 = torch.ops.aten.alias.default(v9)
    v12 = torch.ops.aten.add.Tensor(in3, v10);  in3 = v10 = None
    v13 = torch.ops.aten.ones_like.default(v9, pin_memory = False, memory_format = torch.preserve_format);  v9 = None
    v14 = torch.ops.aten.alias.default(v12);  v12 = None
    v15 = torch.ops.aten.clone.default(v14);  v14 = None
    return (v2, v3, v7, v13, v15)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3):
        return reference(in0, in1, in2, in3)


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
