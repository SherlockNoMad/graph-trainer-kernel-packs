"""Cross-entropy loss backward of one 4096-token chunk (Qwen3 30B-A3B, vocab 151936), one CuTe DSL kernel.

reference (problem.py), per row i with target t = in3[i] and log-probs lp = in5[i] (in2 is the same tensor,
nll_loss_backward only reads its shape):
  g = in0 / in1                                  (f32 / int64 -> div.rn(in0, cvt.rn.f32.s64(in1)))
  gin = nll_loss_backward(g, sum reduction)      (zeros, gin[t] = -g unless t == -100)
  s = sum(gin)                                   (aten adds +0 to the single nonzero: s = 0 + (-g), or +0)
  dx = _log_softmax_backward_data(gin, lp) = gin - expf(lp) * s   (nvcc contracts: fma(-e, s, gin))
  out0 = bf16(dx) (cvt.rn); out1 = out0.t()

One CTA (512 threads) per row; each thread handles 8 consecutive columns per step (2 x 16 B loads of lp,
one 16 B store of bf16). expf is libdevice's __nv_expf sequence (the one nvcc inlines into aten's kernel),
written as explicit PTX so nothing is re-contracted.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 4096
V = 151936
OCT = V // 8  # 8-column groups per row
THREADS = 512
FULL = OCT // THREADS  # 37
TAIL = OCT - FULL * THREADS  # 48
IGNORE_INDEX = -100


def _asm(ret, args, code, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret,
        args,
        code,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def ld_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.nc.s64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def st_v4_b32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Int32(x), loc, ip) for x in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def grad_scalars(x0, x1, t, *, loc=None, ip=None):
    """gneg = -(in0 / f32(in1)); s = t != -100 ? 0 + gneg : 0."""
    code = (
        "{\n.reg .f32 d, g;\n.reg .pred p;\n"
        "cvt.rn.f32.s64 d, $3;\n"
        "div.rn.f32 g, $2, d;\n"
        "neg.f32 $0, g;\n"
        f"setp.ne.s64 p, $4, {IGNORE_INDEX};\n"
        "add.rn.f32 g, 0f00000000, $0;\n"
        "selp.f32 $1, g, 0f00000000, p;\n}"
    )
    res = _asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [_v(Float32(x0), loc, ip), _v(Int64(x1), loc, ip), _v(Int64(t), loc, ip)],
        code,
        "=f,=f,f,l,l",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(2)]


def _elem(x, r, k):
    """r = fma(-expf(x), s, gin) with gin = (d == k) ? gneg : +0; s = $12, gneg = $13, d = $14."""
    return (
        f"fma.rn.sat.f32 t, {x}, 0f3BBB989D, 0f3F000000;\n"
        "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
        "sub.rn.f32 f, 0f4B40007F, j;\n"
        f"fma.rn.f32 f, {x}, 0f3FB8AA3B, f;\n"
        f"fma.rn.f32 f, {x}, 0f32A57060, f;\n"
        "mov.b32 jb, j;\nshl.b32 jb, jb, 23;\nmov.b32 sc, jb;\n"
        "ex2.approx.ftz.f32 e, f;\n"
        "mul.rn.f32 e, e, sc;\n"
        "neg.f32 e, e;\n"
        f"setp.eq.s32 p, $14, {k};\n"
        "selp.f32 gin, $13, 0f00000000, p;\n"
        f"fma.rn.f32 {r}, e, $12, gin;\n"
    )


_OCT_CODE = (
    "{\n.reg .f32 t, j, f, sc, e, gin, r0, r1;\n.reg .b32 jb;\n.reg .pred p;\n"
    + "".join(
        _elem(f"${4 + 2 * q}", "r0", 2 * q) + _elem(f"${5 + 2 * q}", "r1", 2 * q + 1) + f"cvt.rn.bf16x2.f32 ${q}, r1, r0;\n"
        for q in range(4)
    )
    + "}"
)


@dsl_user_op
def bwd_oct(x, s, gneg, d, *, loc=None, ip=None):
    """8 log-probs -> 4 bf16x2 words of the loss gradient."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_v(Float32(v), loc, ip) for v in x] + [_v(Float32(s), loc, ip), _v(Float32(gneg), loc, ip), _v(Int32(d), loc, ip)],
        _OCT_CODE,
        "=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,r",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


class LossBwd:
    @cute.jit
    def __call__(self, lp, tgt, g0, g1, out, stream):
        self.kernel(lp, tgt, g0, g1, out).launch(grid=[ROWS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, lp: cute.Tensor, tgt: cute.Tensor, g0: cute.Tensor, g1: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        t = ld_s64(tgt.iterator.toint() + Int64(row) * 8)
        gneg, s = grad_scalars(ld_f32(g0.iterator.toint()), ld_s64(g1.iterator.toint()), t)
        t32 = Int32(t)
        lp_row = lp.iterator.toint() + Int64(row) * (V * 4)
        out_row = out.iterator.toint() + Int64(row) * (V * 2)
        for k in cutlass.range(FULL, unroll=4):
            o = tid + k * THREADS
            a = ld_v4_f32(lp_row + Int64(o) * 32)
            b = ld_v4_f32(lp_row + Int64(o) * 32 + 16)
            w = bwd_oct([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]], s, gneg, t32 - o * 8)
            st_v4_b32(out_row + Int64(o) * 16, w)
        if tid < TAIL:
            o = tid + FULL * THREADS
            a = ld_v4_f32(lp_row + Int64(o) * 32)
            b = ld_v4_f32(lp_row + Int64(o) * 32 + 16)
            w = bwd_oct([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]], s, gneg, t32 - o * 8)
            st_v4_b32(out_row + Int64(o) * 16, w)


_CACHE = {}


def _fake(dtype, n, align):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            LossBwd(),
            _fake(cutlass.Float32, ROWS * V, 16),
            _fake(cutlass.Int64, ROWS, 8),
            _fake(cutlass.Float32, 1, 4),
            _fake(cutlass.Int64, 1, 8),
            _fake(cutlass.BFloat16, ROWS * V, 16),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def kernel_function(in0, in1, in2, in3, in4, in5):
    assert in5.shape == (ROWS, V) and in5.is_contiguous() and in5.data_ptr() % 16 == 0
    assert in3.shape == (ROWS,) and in3.is_contiguous()
    out = torch.empty((ROWS, V), dtype=torch.bfloat16, device=in5.device)
    _compiled(in5.device.index)(in5.view(-1), in3, in0.view(1), in1.view(1), out.view(-1))
    return (out, out.t())
