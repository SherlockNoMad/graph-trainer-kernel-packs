"""Kernel problem pattern_8711cc2e5e2481ba930c96e98e1ba0e88b3f0135a0ad4cb350538b1cea0568da/df57lv2q7qjawmdy2atkzicvxix5xwi7jzyxkgmo7j4wgx6stgy4: region r003 of outputs/fusion_finder/gpt_oss_120b.

36 calls per step, forward, in layers.0.feed_forward.routed_experts.w13, layers.1.feed_forward.routed_experts.w13, layers.2.feed_forward.routed_experts.w13 ....
Measured 1948.0 us per call in the capture; ideal (HBM roofline) 489.4 us.

``reference`` is the computation a kernel replaces. Write ``kernel_function``,
with the signature of ``reference``, in ``kernel.py`` next to this file. It must
match ``reference`` bit for bit, including in-place writes to its inputs, on
every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.
"""

import importlib.util
from pathlib import Path

import torch


def reference(in0, in1, in2, in3, in4):
    v2 = torch.ops.aten.reshape.default(in0, [in1, 2, 2880]);  in0 = None
    v4 = torch.ops.aten.rsub.Scalar(in2, in1);  in2 = None
    v5 = torch.ops.aten.unsqueeze.default(v4, 0);  v4 = None
    v7 = torch.ops.aten.new_zeros.default(in3, [1, 5760], pin_memory = False)
    v8 = torch.ops.aten.cat.default([in3, v7]);  in3 = v7 = None
    v10 = torch.ops.aten.cat.default([in4, v5]);  in4 = v5 = None
    v11 = torch.ops.aten._to_copy.default(v10, dtype = torch.int64);  v10 = None
    v12 = torch.ops.aten.repeat_interleave.Tensor(v11, output_size = in1);  v11 = None
    v13 = torch.ops.aten.index_select.default(v8, 0, v12);  v8 = None
    v14 = torch.ops.aten.reshape.default(v13, [in1, 2, 2880]);  v13 = in1 = None
    v15 = torch.ops.aten.add.Tensor(v2, v14);  v2 = v14 = None
    v16 = torch.ops.aten.unbind.int(v15, -2);  v15 = None
    v17 = v16[0]
    v18 = v16[1];  v16 = None
    v19 = torch.ops.aten.clamp.default(v17, None, 7.0)
    v20 = torch.ops.aten.clamp.default(v18, -7.0, 7.0)
    v21 = torch.ops.aten.mul.Tensor(v19, 1.702)
    v22 = torch.ops.aten.sigmoid.default(v21);  v21 = None
    v23 = torch.ops.aten.alias.default(v22)
    v24 = torch.ops.aten.mul.Tensor(v19, v22)
    v25 = torch.ops.aten.addcmul.default(v24, v24, v20)
    return (v12, v17, v18, v19, v20, v22, v23, v24, v25)


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
