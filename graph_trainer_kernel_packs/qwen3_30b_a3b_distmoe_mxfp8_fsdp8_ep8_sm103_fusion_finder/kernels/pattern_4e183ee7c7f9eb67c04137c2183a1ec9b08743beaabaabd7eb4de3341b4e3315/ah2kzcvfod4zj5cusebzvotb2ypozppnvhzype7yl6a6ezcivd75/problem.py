"""Kernel problem pattern_4e183ee7c7f9eb67c04137c2183a1ec9b08743beaabaabd7eb4de3341b4e3315/ah2kzcvfod4zj5cusebzvotb2ypozppnvhzype7yl6a6ezcivd75: region r015 of outputs/fusion_finder/qwen3_30b.

49 calls per step, forward, in layers.0.ffn_norm, layers.1.ffn_norm, layers.2.ffn_norm ....
Measured 96.0 us per call in the capture; ideal (HBM roofline) 77.3 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2):
    v2 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
    v4 = torch.ops.aten._fused_rms_norm.default(v2, [2048], in2, 1e-06);  in2 = None
    v5 = v4[0]
    v6 = v4[1];  v4 = None
    v7 = torch.ops.aten.alias.default(v6);  v6 = None
    return (v2, v5, v7)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2):
        return reference(in0, in1, in2)


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
