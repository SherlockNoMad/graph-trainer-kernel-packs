"""Kernel problem pattern_bef6d4287c74b8727d66d6f7fd0840b9f391726ebe593b4ec023c0516f9d2932/dwfgvwu4gphfwihhe5k7ikjvto2ladtytouzzaxrqommzzpdo5is: region r009 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.routed_experts.w2, layers.1.feed_forward.routed_experts.w2, layers.2.feed_forward.routed_experts.w2 ....
Measured 468.9 us per call in the capture; ideal (HBM roofline) 108.9 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4, in5):
    v2 = torch.ops.aten.rsub.Scalar(in0, in1);  in0 = None
    v3 = torch.ops.aten.unsqueeze.default(v2, 0);  v2 = None
    v5 = torch.ops.aten.new_zeros.default(in2, [1, 2880], pin_memory = False)
    v6 = torch.ops.aten.cat.default([in2, v5]);  in2 = v5 = None
    v8 = torch.ops.aten.cat.default([in3, v3]);  in3 = v3 = None
    v9 = torch.ops.aten._to_copy.default(v8, dtype = torch.int64);  v8 = None
    v10 = torch.ops.aten.repeat_interleave.Tensor(v9, output_size = in1);  v9 = in1 = None
    v11 = torch.ops.aten.index_select.default(v6, 0, v10);  v6 = None
    v13 = torch.ops.aten.add.Tensor(in4, v11);  in4 = v11 = None
    v15 = torch.ops.aten.index.Tensor(v13, [in5]);  v13 = in5 = None
    return (v10, v15)


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
