"""RoPE backward of q (DeepSeek-V3 debug model, MLA attention), one CuTe DSL kernel.

reference, per token t: x = in0[0, :, t] (16 heads x 192: 128 nope | 64 rope), f = in1[t] (32 complex64)
  out[t, h] = [x[h, :128] | bf16(complex(f32(x[h, 128:])) * f)]   ([16384, 3072]), returns (out, out.t()).
in0's [1, 16, T, 192] view has strides (., 192, 3072, 1): physically [T, 16, 192], the layout of out.

One CTA (128 threads) per token; each thread copies 2 nope vectors (16 B) and computes one 16 B rope vector.
complex64 mul as aten (c10::complex operator*=, nvcc contracts): re = fma(a, c, -(b*d)),
im = fma(b, c, a*d); bf16 rounding cvt.rn.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
NH = 16
QW = 192  # q / out head width
THREADS = 128


def _asm(ret_type, args, text, cons, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret_type,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _ir(v, loc, ip):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


@dsl_user_op
def ld_v4(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_v4_cached(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [_ir(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


_ROPE = (
    "{\n"
    ".reg .b32 a, b, t, re, im;\n"
    "shl.b32 a, $1, 16;\n"
    "and.b32 b, $1, 0xFFFF0000;\n"
    "mul.rn.f32 t, b, $3;\n"
    "neg.f32 t, t;\n"
    "fma.rn.f32 re, a, $2, t;\n"
    "mul.rn.f32 t, a, $3;\n"
    "fma.rn.f32 im, b, $2, t;\n"
    "cvt.rn.bf16x2.f32 $0, im, re;\n"
    "}"
)


@dsl_user_op
def rope_word(x, c, d, *, loc=None, ip=None):
    """bf16x2 (re, im) times complex64 (c, d) -> bf16x2, aten complex64 mul numerics."""
    return Int32(
        _asm(T.i32(), [_ir(v, loc, ip) for v in (x, c, d)], _ROPE, "=r,r,r,r", loc, ip)
    )


class RopeBwd:
    @cute.jit
    def __call__(self, in0, in1f, out, stream):
        self.kernel(in0, in1f, out).launch(grid=[NT, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, in0: cute.Tensor, in1f: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        row = t * (NH * QW)
        rh = tid // 8
        rj = tid % 8
        f0 = ld_v4(in1f.iterator + (t * 64 + rj * 8))
        f1 = ld_v4(in1f.iterator + (t * 64 + rj * 8 + 4))
        ro = row + rh * QW + 128 + rj * 8
        xr = ld_v4(in0.iterator + ro)
        copies = []
        for i in cutlass.range_constexpr(2):
            c = tid + i * THREADS
            o = row + (c // 16) * QW + (c % 16) * 8
            copies.append((o, ld_v4(in0.iterator + o)))
        for o, v in copies:
            st_v4(out.iterator + o, v)
        fr = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]]
        st_v4(out.iterator + ro, [rope_word(xr[k], fr[2 * k], fr[2 * k + 1]) for k in range(4)])


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            RopeBwd(),
            _fake(cutlass.BFloat16, NT * NH * QW),
            _fake(cutlass.Float32, NT * 64),
            _fake(cutlass.BFloat16, NT * NH * QW),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1):
    x = _aligned(in0.squeeze(0).transpose(0, 1))  # [T, 16, 192], a view for the training layout
    f = _aligned(torch.view_as_real(in1.contiguous()))
    out = torch.empty((NT, NH * QW), dtype=torch.bfloat16, device=in0.device)
    _compiled(in0.device.index)(x.view(-1), f.view(-1), out.view(-1))
    return (out, out.t())
