"""Kernel problem pattern_3c0c0c3537276f8b2acafd28b8760f2f440f611758de472894e86dee247b3e90/ajlmnqp4dkgv2qio2kh56s3kwh5xquc4ksqechpq3u3fqchkd52h: region r007 of outputs/fusion_finder/dsv3_debug.

6 calls per step, forward, in layers.0.attention.inner_attention, layers.1.attention.inner_attention, layers.2.attention.inner_attention ....
Measured 198.5 us per call in the capture; ideal (HBM roofline) 34.1 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1):
    v1 = torch.ops.aten.reshape.default(in0, [16384, -1, 256]);  in0 = None
    v2 = torch.ops.aten.split_with_sizes.default(v1, [128, 128], -1);  v1 = None
    v3 = v2[0]
    v4 = v2[1];  v2 = None
    v6 = torch.ops.aten.cat.default([v3, in1], -1);  v3 = in1 = None
    v7 = torch.ops.aten.transpose.int(v6, 0, 1);  v6 = None
    v8 = torch.ops.aten.unsqueeze.default(v7, 0);  v7 = None
    v9 = torch.ops.aten.transpose.int(v4, 0, 1);  v4 = None
    v10 = torch.ops.aten.unsqueeze.default(v9, 0);  v9 = None
    return (v8, v10)


class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, in0, in1):
        return reference(in0, in1)


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
