"""Kernel problem pattern_d0e85a34b56d3d3863e6d14d3e5c6143b707769021430d5fa7ba3d7df8151ad3/awdbw6kh6f6vayimz26gnvo46psgkdw4i3xblbe72273vf6thwdp: region r009 of outputs/fusion_finder/dsv3_debug.

5 calls per step, forward, in layers.1.feed_forward.routed_experts, layers.2.feed_forward.routed_experts, layers.3.feed_forward.routed_experts ....
Measured 173.5 us per call in the capture; ideal (HBM roofline) 9.8 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch
import torchtitan.ops.scatter_add


def reference(in0, in1, in2, in3, in4, in5, in6):
    v2 = torch.ops.aten.mul.Tensor(in0, in1);  in0 = in1 = None
    v3 = torch.ops.aten._to_copy.default(v2, dtype = torch.bfloat16);  v2 = None
    v6 = torch.ops.torchtitan.deterministic_scatter_add.default(in2, in3, v3);  in2 = in3 = v3 = None
    v8 = torch.ops.aten.add.Tensor(v6, in4);  v6 = in4 = None
    v10 = torch.ops.aten.add.Tensor(in5, v8);  in5 = v8 = None
    v12 = torch.ops.aten._fused_rms_norm.default(v10, [256], in6, 1e-06);  in6 = None
    v13 = v12[0];  v12 = None
    return (v10, v13)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1, in2, in3, in4, in5, in6):
        return reference(in0, in1, in2, in3, in4, in5, in6)


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
