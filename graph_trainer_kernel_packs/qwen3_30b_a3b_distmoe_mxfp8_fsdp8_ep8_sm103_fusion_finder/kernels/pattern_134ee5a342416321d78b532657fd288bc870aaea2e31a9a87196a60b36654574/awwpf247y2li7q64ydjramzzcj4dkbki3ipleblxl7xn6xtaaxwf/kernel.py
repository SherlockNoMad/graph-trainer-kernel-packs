"""MoE router aux loss forward (Qwen3 30B-A3B, 128 experts), one CuTe DSL kernel (one warp).

reference (E = 128 experts, all f32 except in0):
  c = 1.0f / float(max(in0, 1)) (in0 int64 0-dim); v3 = c * 1.0f; out0 = v4 = v3 * 0.001f
  in2 += in1[:128]                                  (in place)
  l1 = sum(|in2|) (linalg_vector_norm ord 1); n = clamp_min(l1, 1e-12)
  out1 = v16 = (in2 / n) * 128
  s = sum(v16 * in3); out2 = s * v4; in4 += s * v3  (in place)

Lane x owns elements 4x .. 4x + 3. Every elementwise op is one f32 op rounded like aten's kernel (div.rn
for the true divisions, the int64 -> f32 cast cvt.rn).

Reduction order: aten reduce_kernel<512, 1> (ATen/native/cuda/Reduce.cuh of this build) for a contiguous
128-float reduction to one output: dim0 = 128 >= 128 -> vectorize_input, input_vec_size 4, so one warp
of 32 threads; thread x loads float4 x (elements 4x..4x+3, shift 0: v11 and v18 are fresh aligned
allocations), accumulators acc[i] = 0 op x[4x + i], combined ((acc0 + acc1) + acc2) + acc3, then
WARP_SHFL_DOWN at offsets 16, 8, 4, 2, 1 (decreasing on CUDA); lane 0 holds the result. The xor
butterfly at the same offsets gives lane 0 the same sums and every lane the same bits (f32 add is
commutative). For the L1 norm op is acc + |x| (NormOneOps), for the sum acc + x. Checked against
aten.sum and aten.linalg_vector_norm on 3000 random 128-vectors with exponents spread over 2^-40..2^40:
0 mismatches.
clamp_min (aten clamp_min_scalar_kernel_impl) keeps NaN: max.NaN.f32 (l1 can only be a canonical NaN).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

E = 128


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _p(ptr, loc, ip):
    return ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


def _i32(text, args, loc, ip):
    return Int32(_asm(T.i32(), [_v(a, loc, ip) for a in args], text, "=r" + ",r" * len(args), loc, ip))


@dsl_user_op
def ld_b32(ptr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_p(ptr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def st_b32(ptr, v, *, loc=None, ip=None):
    _asm(None, [_p(ptr, loc, ip), _v(v, loc, ip)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def scale_of(ptr, *, loc=None, ip=None):
    """(v3, v4) bits: v3 = (1.0f / float(max(in0, 1))) * 1.0f, v4 = v3 * 0.001f."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_p(ptr, loc, ip)],
        "{\n.reg .s64 n;\n.reg .f32 f, c;\n"
        "ld.global.s64 n, [$2];\n"
        "max.s64 n, n, 1;\n"
        "cvt.rn.f32.s64 f, n;\n"
        "div.rn.f32 c, 0f3F800000, f;\n"
        "mul.rn.f32 c, c, 0f3F800000;\n"
        "mov.b32 $0, c;\n"
        "mul.rn.f32 c, c, 0f3A83126F;\n"
        "mov.b32 $1, c;\n}",
        "=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(2)]


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def fmul(a, b, *, loc=None, ip=None):
    return _i32("mul.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def thread_sum4(a0, a1, a2, a3, absolute: bool, *, loc=None, ip=None):
    """((((0 op a0) + (0 op a1)) + (0 op a2)) + (0 op a3)), op = + |.| or +."""
    lines = ["{\n.reg .f32 x, s, t;\n"]
    for k in range(4):
        lines.append(f"mov.b32 x, ${k + 1};\n")
        if absolute:
            lines.append("abs.f32 x, x;\n")
        lines.append(f"add.rn.f32 {'s' if k == 0 else 't'}, 0f00000000, x;\n")
        if k:
            lines.append("add.rn.f32 s, s, t;\n")
    lines.append("mov.b32 $0, s;\n}")
    return _i32("".join(lines), (a0, a1, a2, a3), loc, ip)


@dsl_user_op
def clamp_min_eps(x, *, loc=None, ip=None):
    """clamp_min(x, 1e-12f), NaN kept."""
    return _i32(
        "{\n.reg .f32 a;\nmov.b32 a, $1;\nmax.NaN.f32 a, a, 0f2B8CBCCC;\nmov.b32 $0, a;\n}", (x,), loc, ip
    )


@dsl_user_op
def div_mul128(u, n, *, loc=None, ip=None):
    """(u / n) * 128 (two roundings)."""
    return _i32(
        "{\n.reg .f32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\ndiv.rn.f32 a, a, b;\n"
        "mul.rn.f32 a, a, 0f43000000;\nmov.b32 $0, a;\n}",
        (u, n),
        loc,
        ip,
    )


def _warp_sum(s):
    for off in (16, 8, 4, 2, 1):
        s = fadd(s, cute.arch.shuffle_sync_bfly(s, offset=off))
    return s


class AuxLoss:
    @cute.jit
    def __call__(self, m0, m1, m2, m3, m4, o0, o1, o2, stream):
        self.kernel(m0, m1, m2, m3, m4, o0, o1, o2).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, m0, m1, m2, m3, m4, o0, o1, o2):
        x, _, _ = cute.arch.thread_idx()
        v3, v4 = scale_of(m0.iterator)
        u = []
        for k in cutlass.range_constexpr(4):
            a = ld_b32(m2.iterator + (4 * x + k))
            b = ld_b32(m1.iterator + (4 * x + k))
            u.append(fadd(a, b))
        w = []
        for k in cutlass.range_constexpr(4):
            st_b32(m2.iterator + (4 * x + k), u[k])
            w.append(ld_b32(m3.iterator + (4 * x + k)))
        l1 = _warp_sum(thread_sum4(u[0], u[1], u[2], u[3], True))
        n = clamp_min_eps(l1)
        p = []
        for k in cutlass.range_constexpr(4):
            v16 = div_mul128(u[k], n)
            st_b32(o1.iterator + (4 * x + k), v16)
            p.append(fmul(v16, w[k]))
        s = _warp_sum(thread_sum4(p[0], p[1], p[2], p[3], False))
        if x == 0:
            st_b32(o0.iterator, v4)
            st_b32(o2.iterator, fmul(s, v4))
            st_b32(m4.iterator, fadd(ld_b32(m4.iterator), fmul(s, v3)))


_CACHE = {}


def _fake(dtype, n, align=4):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        f32 = cutlass.Float32
        fn = cute.compile(
            AuxLoss(),
            _fake(cutlass.Int64, 1, 8),
            _fake(f32, E),
            _fake(f32, E),
            _fake(f32, E),
            _fake(f32, 1),
            _fake(f32, 1),
            _fake(f32, E),
            _fake(f32, 1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def kernel_function(in0, in1, in2, in3, in4):
    assert in0.shape == () and in0.dtype == torch.int64
    assert in1.shape[0] >= E and in1.dtype == torch.float32
    assert in2.shape == (E,) and in3.shape == (E,) and in4.shape == ()
    assert in2.is_contiguous() and in2.dtype == torch.float32 and in4.dtype == torch.float32
    dev = in2.device
    out0 = torch.empty((), dtype=torch.float32, device=dev)
    out1 = torch.empty((E,), dtype=torch.float32, device=dev)
    out2 = torch.empty((), dtype=torch.float32, device=dev)
    _compiled(dev.index)(
        in0.view(1),
        in1[:E].contiguous(),
        in2,
        in3.contiguous(),
        in4.view(1),
        out0.view(1),
        out1,
        out2.view(1),
    )
    return (out0, out1, out2)
