"""SwiGLU recompute + backward of a dense FFN (DeepSeek-V3 debug model), one CuTe DSL kernel.

reference (bf16 in, bf16 out, every aten op computes in float32 and rounds to bf16):
  g = in0[:, :1024], u = in0[:, 1024:], dy = in1
  v5 = silu(g); v6 = v5 * u; v8 = dy * v5; v9 = dy * u; v10 = silu_backward(v9, g)
  v12 = [v10 | v8] (16384 x 2048); returns (v6, v12, v12.t())

One pass: each thread reads 16 B of g, u and dy (8 columns of one row), computes in registers and
writes 16 B of v6, v10 and v8 (v12.t() is a view). aten's float math (ActivationSiluKernel.cu, opmath
float32) is reproduced op for op in PTX with explicit rounding, so nothing is contracted unless
torch's nvcc build contracted it (checked at f32 precision against aten's float32 kernels for every
bf16 x, see the candidate's tools/f32check.py):
  e = expf(-x)                  CUDA's accurate expf (libdevice __nv_expf as compiled into torch):
                                t = sat(fma(-x, 0x3BBB989D, 0.5)); j = fma.rm(t, 252, 12582913);
                                f = fma(-x, log2e_lo, fma(-x, log2e_hi, 12583039 - j)); 2^j * ex2(f)
  d = 1 + e                     fma(2^j, ex2(f), 1) (the power-of-two product is exact, so this
                                equals the rounded product plus 1)
  silu:          x / d          (div.rn)
  silu_backward: s = 1 / d (rcp.rn); (dy * s) * fma(x, 1 - s, 1)
  products of two bf16 are exact in f32; every aten op output is rounded to bf16 (cvt.rn).

Division fast path: ptxas's div.rn.f32 is MUFU.RCP r0; r = fma(r0, fma(-d, r0, 1), r0);
q = x * r; q = fma(r, fma(-d, q, x), q), guarded by FCHK, and its rcp.rn.f32 fast path computes the
same r guarded by an exponent check. Both depend only on the bf16 x (65536 values): for every bf16 x
except |x| >= 64, +-0, inf and NaN the unguarded sequence gives the same q (and bf16(q)) and s bits
as div.rn / rcp.rn (exhaustive test, tools/fastcheck.py), so one shared reciprocal serves both. A
thread whose 8 g values include any of those specials takes the exact div.rn / rcp.rn path.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 16384
N = 1024  # columns of g, u, dy, v6
VEC = 8  # bf16 per 16-byte access
VPR = N // VEC  # vectors per row
NVEC = M * VPR
THREADS = 128
ITEMS = 1  # vectors per thread

_HEAD = (
    ".reg .b32 x0, x1, n0, n1, u0, u1, y0, y1, t0, t1, j0, j1, f0, f1, d0, d1, b0, b1;\n"
    ".reg .b32 q0, q1, w5, v0, v1, w9, z0, z1, s0, s1, r0, r1, p0, p1, h0, h1, m0, m1;\n"
    ".reg .b64 X, NX, U, Y, TT, J, F, S, E, D, ND, R0, EE, QA, REM, Q, V5, P6, P8, P9, DY, SS, OMS, TB, R;\n"
    ".reg .b64 CS, CB, CN, CL, CLO, ONE2;\n"
    "mov.b64 CS, {0f437C0000, 0f437C0000};\n"
    "mov.b64 CB, {0f4B400001, 0f4B400001};\n"
    "mov.b64 CN, {0f4B40007F, 0f4B40007F};\n"
    "mov.b64 CL, {0f3FB8AA3B, 0f3FB8AA3B};\n"
    "mov.b64 CLO, {0f32A57060, 0f32A57060};\n"
    "mov.b64 ONE2, {0f3F800000, 0f3F800000};\n"
    "shl.b32 x0, $3, 16;\n"
    "and.b32 x1, $3, 0xFFFF0000;\n"
    "shl.b32 u0, $4, 16;\n"
    "and.b32 u1, $4, 0xFFFF0000;\n"
    "shl.b32 y0, $5, 16;\n"
    "and.b32 y1, $5, 0xFFFF0000;\n"
    "mov.b64 X, {x0, x1};\n"
    "mov.b64 U, {u0, u1};\n"
    "mov.b64 Y, {y0, y1};\n"
    # e = expf(-x) = 2^j * ex2(f); D = 1 + e
    "neg.f32 n0, x0;\n"
    "neg.f32 n1, x1;\n"
    "mov.b64 NX, {n0, n1};\n"
    "fma.rn.sat.f32 t0, n0, 0f3BBB989D, 0f3F000000;\n"
    "fma.rn.sat.f32 t1, n1, 0f3BBB989D, 0f3F000000;\n"
    "mov.b64 TT, {t0, t1};\n"
    "fma.rm.f32x2 J, TT, CS, CB;\n"
    "sub.rn.f32x2 F, CN, J;\n"
    "fma.rn.f32x2 F, NX, CL, F;\n"
    "fma.rn.f32x2 F, NX, CLO, F;\n"
    "mov.b64 {j0, j1}, J;\n"
    "shl.b32 j0, j0, 23;\n"
    "shl.b32 j1, j1, 23;\n"
    "mov.b64 S, {j0, j1};\n"
    "mov.b64 {f0, f1}, F;\n"
    "ex2.approx.ftz.f32 f0, f0;\n"
    "ex2.approx.ftz.f32 f1, f1;\n"
    "mov.b64 E, {f0, f1};\n"
    "fma.rn.f32x2 D, S, E, ONE2;\n"
    "mov.b64 {d0, d1}, D;\n"
)

_TAIL = (
    # v5 = bf16(Q); v6 = bf16(v5 * u), v8 = bf16(dy * v5), v9 = bf16(dy * u)
    "mov.b64 {q0, q1}, Q;\n"
    "cvt.rn.bf16x2.f32 w5, q1, q0;\n"
    "shl.b32 v0, w5, 16;\n"
    "and.b32 v1, w5, 0xFFFF0000;\n"
    "mov.b64 V5, {v0, v1};\n"
    "mul.rn.f32x2 P6, V5, U;\n"
    "mov.b64 {p0, p1}, P6;\n"
    "cvt.rn.bf16x2.f32 $0, p1, p0;\n"
    "mul.rn.f32x2 P8, Y, V5;\n"
    "mov.b64 {h0, h1}, P8;\n"
    "cvt.rn.bf16x2.f32 $2, h1, h0;\n"
    "mul.rn.f32x2 P9, Y, U;\n"
    "mov.b64 {z0, z1}, P9;\n"
    "cvt.rn.bf16x2.f32 w9, z1, z0;\n"
    "shl.b32 z0, w9, 16;\n"
    "and.b32 z1, w9, 0xFFFF0000;\n"
    "mov.b64 DY, {z0, z1};\n"
    # silu_backward with SS = 1 / D: v10 = bf16((dy * s) * fma(x, 1 - s, 1))
    "sub.rn.f32x2 OMS, ONE2, SS;\n"
    "fma.rn.f32x2 TB, X, OMS, ONE2;\n"
    "mul.rn.f32x2 R, DY, SS;\n"
    "mul.rn.f32x2 R, R, TB;\n"
    "mov.b64 {r0, r1}, R;\n"
    "cvt.rn.bf16x2.f32 $1, r1, r0;\n"
)

# Exact path: div.rn / rcp.rn.
_EXACT = (
    "{\n"
    + _HEAD
    + "div.rn.f32 q0, x0, d0;\n"
    "div.rn.f32 q1, x1, d1;\n"
    "mov.b64 Q, {q0, q1};\n"
    "rcp.rn.f32 s0, d0;\n"
    "rcp.rn.f32 s1, d1;\n"
    "mov.b64 SS, {s0, s1};\n"
    + _TAIL
    + "}"
)

# Fast path: ptxas's div.rn / rcp.rn fast-path sequences without their guards, one shared reciprocal.
_FAST = (
    "{\n"
    + _HEAD
    + "rcp.approx.ftz.f32 r0, d0;\n"
    "rcp.approx.ftz.f32 r1, d1;\n"
    "mov.b64 R0, {r0, r1};\n"
    "neg.f32 m0, d0;\n"
    "neg.f32 m1, d1;\n"
    "mov.b64 ND, {m0, m1};\n"
    "fma.rn.f32x2 EE, ND, R0, ONE2;\n"
    "fma.rn.f32x2 SS, R0, EE, R0;\n"
    "mul.rn.f32x2 QA, X, SS;\n"
    "fma.rn.f32x2 REM, ND, QA, X;\n"
    "fma.rn.f32x2 Q, SS, REM, QA;\n"
    + _TAIL
    + "}"
)


def _pair(text, g, u, dy, loc, ip):
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32)>"),
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (g, u, dy)],
        text,
        "=r,=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(3))


@dsl_user_op
def pair_exact(g, u, dy, *, loc=None, ip=None):
    """Two elements: g, u, dy bf16x2 -> (v6, v10, v8) bf16x2, exact for every input."""
    return _pair(_EXACT, g, u, dy, loc, ip)


@dsl_user_op
def pair_fast(g, u, dy, *, loc=None, ip=None):
    """Two elements as pair_exact, exact unless a g element is special (see special_bits)."""
    return _pair(_FAST, g, u, dy, loc, ip)


@dsl_user_op
def special_bits(w0, w1, w2, w3, *, loc=None, ip=None):
    """Nonzero iff a bf16 half of the four words has |x| >= 64 (bits 0x4280, includes inf and NaN)
    or is +-0. Per half (no carry crosses halves): a = |h|; bit 15 of a + 0x3D80 is set iff
    a >= 0x4280; bit 15 of a + 0x7FFF is clear iff a == 0."""
    text = "{\n.reg .b32 a, b, c, f;\nmov.b32 f, 0;\n"
    for k in range(4):
        text += (
            f"and.b32 a, ${k + 1}, 0x7FFF7FFF;\n"
            "add.u32 b, a, 0x3D803D80;\n"
            "add.u32 c, a, 0x7FFF7FFF;\n"
            "not.b32 c, c;\n"
            "or.b32 b, b, c;\n"
            "or.b32 f, f, b;\n"
        )
    text += "and.b32 $0, f, 0x80008000;\n}"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(v).ir_value(loc=loc, ip=ip) for v in (w0, w1, w2, w3)],
            text,
            "=r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_v4(ptr, *, loc=None, ip=None):
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [Int32(v).ir_value(loc=loc, ip=ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class SwiGLUBwd:
    @cute.jit
    def __call__(self, mIn0, mIn1, mOut0, mOut1, stream):
        self.kernel(mIn0, mIn1, mOut0, mOut1).launch(
            grid=[NVEC // (THREADS * ITEMS), 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIn0: cute.Tensor, mIn1: cute.Tensor, mOut0: cute.Tensor, mOut1: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        loads = []
        for it in cutlass.range_constexpr(ITEMS):
            idx = (bid * ITEMS + it) * THREADS + tid
            row = idx // VPR
            col = (idx % VPR) * VEC
            o2 = row * (2 * N) + col  # g (and v10) offset in the 2N-wide rows
            g = ld_v4(mIn0.iterator + o2)
            u = ld_v4(mIn0.iterator + (o2 + N))
            y = ld_v4(mIn1.iterator + idx * VEC)
            loads.append((idx, o2, g, u, y))
        for idx, o2, g, u, y in loads:
            if special_bits(g[0], g[1], g[2], g[3]) != Int32(0):
                w6, w10, w8 = [], [], []
                for k in cutlass.range_constexpr(4):
                    a, b, c = pair_exact(g[k], u[k], y[k])
                    w6.append(a)
                    w10.append(b)
                    w8.append(c)
                st_v4(mOut0.iterator + idx * VEC, w6)
                st_v4(mOut1.iterator + o2, w10)
                st_v4(mOut1.iterator + (o2 + N), w8)
            else:
                w6, w10, w8 = [], [], []
                for k in cutlass.range_constexpr(4):
                    a, b, c = pair_fast(g[k], u[k], y[k])
                    w6.append(a)
                    w10.append(b)
                    w8.append(c)
                st_v4(mOut0.iterator + idx * VEC, w6)
                st_v4(mOut1.iterator + o2, w10)
                st_v4(mOut1.iterator + (o2 + N), w8)


_CACHE = {}


def _fake(n):
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, (n,), stride=(1,), assumed_align=16)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            SwiGLUBwd(),
            _fake(M * 2 * N),
            _fake(M * N),
            _fake(M * N),
            _fake(M * 2 * N),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def kernel_function(in0, in1):
    assert in0.shape == (M, 2 * N) and in1.shape == (M, N)
    x = in0.contiguous()
    dy = in1.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    if dy.data_ptr() % 16:
        dy = dy.clone()
    v6 = torch.empty((M, N), dtype=torch.bfloat16, device=in0.device)
    v12 = torch.empty((M, 2 * N), dtype=torch.bfloat16, device=in0.device)
    _compiled(in0.device.index)(x.view(-1), dy.view(-1), v6.view(-1), v12.view(-1))
    return (v6, v12, v12.t())
