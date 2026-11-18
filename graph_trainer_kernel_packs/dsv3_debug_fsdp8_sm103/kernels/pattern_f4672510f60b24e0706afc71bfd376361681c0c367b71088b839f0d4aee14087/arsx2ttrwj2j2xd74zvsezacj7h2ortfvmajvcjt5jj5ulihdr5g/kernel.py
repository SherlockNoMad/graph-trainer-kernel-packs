"""MoE router aux-loss recompute + backward of the DeepSeek-V3 debug model (16384 tokens, top-3 of 8 experts):
two CuTe DSL kernels around _fused_rms_norm_backward, which stays the same torch call.

_fused_rms_norm / _fused_rms_norm_backward run torch's vendored quack CuTe kernels on this build (RMSNorm,
RMSNormBackward + reduce_rows). The two forward norms are fused here in quack's order (BRIEF.md shared findings:
N = 256 -> threads_per_row 32, vecsize 8; lane sum of squares in column order from 0, butterfly xor 16 .. 1,
rstd = rsqrt.approx.ftz(s / 256 + eps), y = bf16(w * (rstd * x))); the backward (dw reduce over rows) stays the
same torch call on bitwise-identical inputs.
Kernel A: v7 = in2 + in3 and v19 = in5 + in6 (bf16 adds, as aten: bf16(f32 + f32)), v3/v4 = rmsnorm(in0, in1),
v22/v23 = rmsnorm(v19, in7), and per-CTA histograms of the top-k ids in8 and of the column counts of in10.
Kernel B (after them), one CTA per 128 tokens:
  - stable sort of the 49152 ids by counting (expert offset + histogram prefix + in-CTA rank): v28, v29;
  - v15 = bf16(in4 + v11); v33 = f32(v15)[v29], v46 = bf16(v33 * w[v28]), v34 = v22[v29] written by row;
  - the router gradient per token: v36 = sigmoid(in9) (libdevice expf, exact), v39, v92, and
    v90 = v78 / L + sgn(v36) * where(L1 > 1e-12, sum_8(-v78 * v36 / L / L), 0) with aten's reduce orders
    (3-sum (a + c) + b, 8-sums tree (0+4, 2+6), (1+5, 3+7)); v63 comes from the in10 column counts (exact integers).
  v58 (the dim-0 float sum) only feeds ones_like, so it is not computed.
All float ops are explicit .rn PTX.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


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


_SIGMOID = (
    "{\n.reg .f32 x, t, j, f, s, e, d;\n.reg .b32 jb;\n"
    "neg.f32 x, $1;\n"
    "fma.rn.sat.f32 t, x, 0f3BBB989D, 0f3F000000;\n"
    "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
    "sub.rn.f32 f, 0f4B40007F, j;\n"
    "fma.rn.f32 f, x, 0f3FB8AA3B, f;\n"
    "fma.rn.f32 f, x, 0f32A57060, f;\n"
    "mov.b32 jb, j;\nshl.b32 jb, jb, 23;\nmov.b32 s, jb;\n"
    "ex2.approx.ftz.f32 e, f;\n"
    "fma.rn.f32 d, s, e, 0f3F800000;\n"
    "rcp.rn.f32 $0, d;\n}"
)


@dsl_user_op
def sigmoid_f32(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(x), loc, ip)], _SIGMOID, "=f,f", loc=loc, ip=ip))


def _binf(op):
    @dsl_user_op
    def f(a, b, *, loc=None, ip=None):
        return Float32(
            _asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], f"{op} $0, $1, $2;", "=f,f,f", loc=loc, ip=ip)
        )

    return f


add_rn = _binf("add.rn.f32")
mul_rn = _binf("mul.rn.f32")
div_rn = _binf("div.rn.f32")


@dsl_user_op
def abs_f32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], "abs.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def clamp_min_f32(a, lo, *, loc=None, ip=None):
    """aten clamp_min: NaN stays NaN, else max(a, lo)."""
    code = "{\n.reg .pred p;\n.reg .f32 m;\nsetp.nan.f32 p, $1, $1;\nmax.f32 m, $1, $2;\nselp.f32 $0, $1, m, p;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(lo), loc, ip)], code, "=f,f,f", loc=loc, ip=ip))


@dsl_user_op
def topk_key(a, *, loc=None, ip=None):
    """TopKTypeConfig<float>::convert: order-preserving uint32 key, NaN -> 0xffffffff."""
    code = (
        "{\n.reg .pred p;\n.reg .b32 x, m;\nmov.b32 x, $1;\nshr.s32 m, x, 31;\nor.b32 m, m, 0x80000000;\n"
        "xor.b32 $0, x, m;\nsetp.nan.f32 p, $1, $1;\n@p mov.b32 $0, 0xffffffff;\n}"
    )
    return Uint32(_asm(T.i32(), [_v(Float32(a), loc, ip)], code, "=r,f", loc=loc, ip=ip))


@dsl_user_op
def cvt_f32_s64(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(a), loc, ip)], "cvt.rn.f32.s64 $0, $1;", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def cvt_f32_s32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(a), loc, ip)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def ld_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_u8(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", side=True, loc=loc, ip=ip))


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", side=True, loc=loc, ip=ip))


@dsl_user_op
def ldnv_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ldnv_b64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_b64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.b64 $0, [$1];", "=l,l", side=True, loc=loc, ip=ip))


@dsl_user_op
def st_b64(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int64(val), loc, ip)], "st.global.b64 [$0], $1;", "l,l", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_b32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.b32 [$0], $1;", "l,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_u8(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.u8 [$0], $1;", "l,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_f32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Float32(val), loc, ip)], "st.global.f32 [$0], $1;", "l,f", side=True, loc=loc, ip=ip)


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
def f2i(a, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Float32(a), loc, ip)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip))


@dsl_user_op
def i2f(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(a), loc, ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


def _sel(c, a, b):
    return type(a)(cutlass.select_(c, a, b))


def _spread4(nib):
    """4 bits -> one byte (0/1) per bit."""
    return (nib * Int32(0x00204081)) & Int32(0x01010101)


def _addr(t):
    return t.iterator.toint()



sub_rn = _binf("sub.rn.f32")


@dsl_user_op
def neg_f32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], "neg.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def ld_v2_b32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(2)]


@dsl_user_op
def ld_nc_b64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.nc.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def st_v8_b32(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip)] + [_v(Int32(x), loc, ip) for x in w],
         "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};", "l,r,r,r,r,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_nc_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_nc_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    """Two floats -> packed bf16 (round to nearest even), lo in the low half."""
    return Int32(_asm(T.i32(), [_v(Float32(hi), loc, ip), _v(Float32(lo), loc, ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;",
                      "=r,f,f", loc=loc, ip=ip))


def _bf_lo(w):
    return i2f(w << Int32(16))


def _bf_hi(w):
    return i2f(w & Int32(-65536))


NT = 16384
NS = NT * 3
H = 256
TPB = 128  # tokens per CTA
NCTA = NT // TPB
THREADS = 256
KA_ROWS = 16
KA_CTAS = NT // KA_ROWS
KB_THREADS = 1024


def _tree8(x):
    """aten block_x_reduce over 8 lanes, offsets 4, 2, 1 (ident 0 added first)."""
    z = [add_rn(Float32(0.0), x[i]) for i in range(8)]
    a = [add_rn(z[i], z[i + 4]) for i in range(4)]
    b = [add_rn(a[i], a[i + 2]) for i in range(2)]
    return add_rn(b[0], b[1])


def _add_bf16x8(a, b):
    return [bf16x2(add_rn(_bf_lo(a[q]), _bf_lo(b[q])), add_rn(_bf_hi(a[q]), _bf_hi(b[q]))) for q in range(4)]


def _onehot16(e):
    """id e -> 1 in 16-bit field e % 4 of word e // 4 (as (lo, hi) Int64 pair)."""
    sh = Int64(16) * Int64(e % 4)
    one = Int64(1) << sh
    return _sel(e < 4, one, Int64(0)), _sel(e >= 4, one, Int64(0))


_SUMSQ8 = (
    "{\n.reg .b32 a, s;\nmov.b32 s, 0f00000000;\n"
    + "".join(
        f"shl.b32 a, ${k + 1}, 16;\nfma.rn.f32 s, a, a, s;\nand.b32 a, ${k + 1}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
        for k in range(4)
    )
    + "mov.b32 $0, s;\n}"
)


@dsl_user_op
def sumsq8(w, *, loc=None, ip=None):
    """(((0 + x0^2) + x1^2) + ...) + x7^2 of 8 bf16 (4 words, element order). bf16 squares are exact in f32."""
    return Float32(_asm(T.f32(), [_v(Int32(x), loc, ip) for x in w], _SUMSQ8, "=f" + ",r" * 4, loc=loc, ip=ip))


@dsl_user_op
def rstd256(s, *, loc=None, ip=None):
    code = "{\n.reg .f32 m;\ndiv.rn.f32 m, $1, 0f43800000;\nadd.rn.f32 m, m, 0f358637BD;\nrsqrt.approx.ftz.f32 $0, m;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(s), loc, ip)], code, "=f,f", loc=loc, ip=ip))


def _rmsnorm_row(x, g):
    """quack RMSNorm, N = 256 (threads_per_row 32, vecsize 8; lane i owns columns 8i .. 8i + 7): bf16 out words, rstd."""
    sq = sumsq8(x)
    for off in (16, 8, 4, 2, 1):
        sq = add_rn(sq, cute.arch.shuffle_sync_bfly(sq, off))
    r = rstd256(sq)
    y = [bf16x2(mul_rn(_bf_lo(g[q]), mul_rn(r, _bf_lo(x[q]))), mul_rn(_bf_hi(g[q]), mul_rn(r, _bf_hi(x[q])))) for q in range(4)]
    return y, r


class RouterAuxBwd:
    @cute.jit
    def __call__(self, in0, in1, in2, in3, in5, in6, in7, in8, in10, v3, v4, v7, v19, v22, v23, scratch, stream):
        self.ka(in0, in1, in2, in3, in5, in6, in7, in8, in10, v3, v4, v7, v19, v22, v23, scratch).launch(
            grid=[KA_CTAS, 1, 1], block=[THREADS, 1, 1], smem=256, stream=stream)

    @cute.kernel
    def ka(self, in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor, in3: cute.Tensor, in5: cute.Tensor,
           in6: cute.Tensor, in7: cute.Tensor, in8: cute.Tensor, in10: cute.Tensor, v3: cute.Tensor, v4: cute.Tensor,
           v7: cute.Tensor, v19: cute.Tensor, v22: cute.Tensor, v23: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sh = smem.allocate_tensor(Int64, cute.make_layout(12), 16)  # [warp 4][lo, hi, m]
        lane = tid % 32
        wid = tid // 32
        base = Int64(bid * KA_ROWS * H * 2)
        # warp w, step u: row bid * KA_ROWS + u * 8 + w; lane owns columns 8 lane .. 8 lane + 7 (quack's split)
        offs = [base + Int64(u * THREADS + tid) * 16 for u in range(KA_ROWS * H * 2 // 16 // THREADS)]
        g1 = ld_v4_b32(_addr(in1) + Int64(lane) * 16)
        g7 = ld_v4_b32(_addr(in7) + Int64(lane) * 16)
        ld = [(ld_v4_b32(_addr(in0) + o), ld_v4_b32(_addr(in2) + o), ld_v4_b32(_addr(in3) + o), ld_v4_b32(_addr(in5) + o),
               ld_v4_b32(_addr(in6) + o)) for o in offs]
        for u in cutlass.range_constexpr(len(offs)):
            a0, a2, a3, a5, a6 = ld[u]
            vo = offs[u]
            row = Int64(bid * KA_ROWS + u * (THREADS // 32) + wid)
            st_v4_b32(_addr(v7) + vo, _add_bf16x8(a2, a3))
            x19 = _add_bf16x8(a5, a6)
            st_v4_b32(_addr(v19) + vo, x19)
            y22, r23 = _rmsnorm_row(x19, g7)
            st_v4_b32(_addr(v22) + vo, y22)
            y3, r4 = _rmsnorm_row(a0, g1)
            st_v4_b32(_addr(v3) + vo, y3)
            if lane == 0:
                st_f32(_addr(v23) + row * 4, r23)
                st_f32(_addr(v4) + row * 4, r4)
        hist_cta = bid < NCTA
        if tid < TPB and hist_cta:
            t = Int64(bid * TPB + tid)
            lo = Int64(0)
            hi = Int64(0)
            for j in cutlass.range_constexpr(3):
                a, b = _onehot16(Int32(ld_nc_b64(_addr(in8) + t * 24 + 8 * j)))
                lo = lo + a
                hi = hi + b
            m = ld_nc_b64(_addr(in10) + t * 8)
            for o in (16, 8, 4, 2, 1):
                lo = lo + cute.arch.shuffle_sync_bfly(lo, o)
                hi = hi + cute.arch.shuffle_sync_bfly(hi, o)
                m = m + cute.arch.shuffle_sync_bfly(m, o)
            if lane == 0:
                sh[wid * 3] = lo
                sh[wid * 3 + 1] = hi
                sh[wid * 3 + 2] = m
        cute.arch.sync_threads()
        if tid < 3 and hist_cta:
            st_b64(_addr(scratch) + Int64(bid * 3 + tid) * 8, sh[tid] + sh[3 + tid] + sh[6 + tid] + sh[9 + tid])


class RouterAuxBwdB:
    @cute.jit
    def __call__(self, in4, in8, in9, in11, in12, v11, v12, v22, scratch, v13, v15, v28, v29, v33, v34, v39, v46,
                 v90, v92, v94, stream):
        self.kb(in4, in8, in9, in11, in12, v11, v12, v22, scratch, v13, v15, v28, v29, v33, v34, v39, v46, v90, v92,
                v94).launch(grid=[NCTA, 1, 1], block=[KB_THREADS, 1, 1], max_number_threads=[KB_THREADS, 1, 1], smem=8192, stream=stream)

    @cute.kernel
    def kb(self, in4: cute.Tensor, in8: cute.Tensor, in9: cute.Tensor, in11: cute.Tensor, in12: cute.Tensor,
           v11: cute.Tensor, v12: cute.Tensor, v22: cute.Tensor, scratch: cute.Tensor, v13: cute.Tensor,
           v15: cute.Tensor, v28: cute.Tensor, v29: cute.Tensor, v33: cute.Tensor, v34: cute.Tensor,
           v39: cute.Tensor, v46: cute.Tensor, v90: cute.Tensor, v92: cute.Tensor, v94: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sred = smem.allocate_tensor(Int64, cute.make_layout(4 * 16), 16)  # [warp][16], warps < 4
        sbase = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        sv63 = smem.allocate_tensor(Float32, cute.make_layout(8), 16)
        sscan = smem.allocate_tensor(Int64, cute.make_layout(64), 16)
        spos = smem.allocate_tensor(Int32, cute.make_layout(TPB * 4), 16)
        sw = smem.allocate_tensor(Float32, cute.make_layout(TPB * 4), 16)
        lane = tid % 32
        wid = tid // 32

        if bid == 0 and tid < H:
            hb = ld_nc_b32(_addr(v12) + Int64(tid // 2) * 4)
            st_f32(_addr(v13) + Int64(tid) * 4, _sel(tid % 2 == 0, _bf_lo(hb), _bf_hi(hb)))

        # expert prefix / totals (16-bit fields) and in10 column totals (8-bit fields per CTA)
        if tid < TPB:
            ci = tid % NCTA
            lo = ld_b64(_addr(scratch) + Int64(ci * 3) * 8)
            hi = ld_b64(_addr(scratch) + Int64(ci * 3 + 1) * 8)
            mm = ld_b64(_addr(scratch) + Int64(ci * 3 + 2) * 8)
            before = tid < bid
            vals = []
            for e in cutlass.range_constexpr(8):
                w = lo if e < 4 else hi
                cnt = (w >> Int64(16 * (e % 4))) & Int64(0xFFFF)
                vals.append((cnt << Int64(32)) + _sel(before, cnt, Int64(0)))
            for e in cutlass.range_constexpr(8):
                vals.append((mm >> Int64(8 * e)) & Int64(255))
            for q in cutlass.range_constexpr(16):
                v = vals[q]
                for o in (16, 8, 4, 2, 1):
                    v = v + cute.arch.shuffle_sync_bfly(v, o)
                if lane == 0:
                    sred[wid * 16 + q] = v
        cute.arch.sync_threads()
        if tid == 0:
            tot = [sred[q] + sred[16 + q] + sred[32 + q] + sred[48 + q] for q in range(16)]
            off = Int32(0)
            for e in cutlass.range_constexpr(8):
                sbase[e] = off + Int32(tot[e] & Int64(0xFFFFFFFF))
                off = off + Int32(tot[e] >> Int64(32))
            v50 = [cvt_f32_s64(tot[8 + e]) for e in range(8)]
            v60 = clamp_min_f32(_tree8([abs_f32(v50[e]) for e in range(8)]), Float32(1e-12))
            for e in cutlass.range_constexpr(8):
                sv63[e] = mul_rn(div_rn(v50[e], v60), Float32(8.0))
        cute.arch.sync_threads()

        # per token (threads < 128): router gradient, sort rank
        tt = tid % TPB
        t = Int64(bid * TPB + tt)
        ex = [Int32(ld_nc_b64(_addr(in8) + t * 24 + 8 * j)) for j in range(3)]
        lo = Int64(0)
        hi = Int64(0)
        for j in cutlass.range_constexpr(3):
            a, b = _onehot16(ex[j])
            lo = lo + a
            hi = hi + b
        inc_lo = lo
        inc_hi = hi
        for o in (1, 2, 4, 8, 16):
            ul = cute.arch.shuffle_sync_up(inc_lo, o)
            uh = cute.arch.shuffle_sync_up(inc_hi, o)
            inc_lo = _sel(lane >= o, inc_lo + ul, inc_lo)
            inc_hi = _sel(lane >= o, inc_hi + uh, inc_hi)
        if lane == 31:
            sscan[wid * 2] = inc_lo
            sscan[wid * 2 + 1] = inc_hi
        cute.arch.sync_threads()
        if tid < TPB:
            pre_lo = Int64(0)
            pre_hi = Int64(0)
            for ww in cutlass.range_constexpr(TPB // 32 - 1):
                pre_lo = pre_lo + _sel(Int32(ww) < wid, sscan[ww * 2], Int64(0))
                pre_hi = pre_hi + _sel(Int32(ww) < wid, sscan[ww * 2 + 1], Int64(0))
            ex_lo = inc_lo - lo + pre_lo
            ex_hi = inc_hi - hi + pre_hi
            # router math
            x9 = ld_v4_f32(_addr(in9) + t * 32) + ld_v4_f32(_addr(in9) + t * 32 + 16)
            m11 = ld_u8(_addr(in11) + t)
            n12 = ldnv_b64(_addr(in12))
            s = [sigmoid_f32(x9[e]) for e in range(8)]
            g = []
            for j in cutlass.range_constexpr(3):
                v = s[0]
                for e in cutlass.range_constexpr(1, 8):
                    v = _sel(ex[j] == e, s[e], v)
                g.append(v)
            d39 = add_rn(add_rn(add_rn(g[0], g[2]), g[1]), Float32(1e-20))
            st_f32(_addr(v39) + t * 4, d39)
            for j in cutlass.range_constexpr(3):
                st_f32(_addr(v92) + t * 12 + 4 * j, div_rn(div_rn(g[j], d39), d39))
            st_v4_b32(_addr(v94) + t * 32, [f2i(s[e]) for e in range(4)])
            st_v4_b32(_addr(v94) + t * 32 + 16, [f2i(s[e]) for e in range(4, 8)])
            n12 = _sel(n12 < Int64(1), Int64(1), n12)
            v70 = mul_rn(mul_rn(div_rn(Float32(1.0), cvt_f32_s64(n12)), Float32(1.0)), Float32(0.001))
            v73 = mul_rn(Float32(1.0), v70)
            v51 = _tree8([abs_f32(s[e]) for e in range(8)])
            v53 = clamp_min_f32(v51, Float32(1e-12))
            nm = _sel(m11 == Int32(0), Float32(1.0), Float32(0.0))
            v78 = [mul_rn(mul_rn(v73, sv63[e]), nm) for e in range(8)]
            v82 = [mul_rn(neg_f32(v78[e]), div_rn(div_rn(s[e], v53), v53)) for e in range(8)]
            v84 = _tree8(v82)
            v87 = _sel(v51 > Float32(1e-12), v84, Float32(0.0))
            o90 = []
            for e in cutlass.range_constexpr(8):
                sg = _sel(s[e] > Float32(0.0), Float32(1.0), _sel(s[e] < Float32(0.0), Float32(-1.0), Float32(0.0)))
                o90.append(add_rn(div_rn(v78[e], v53), mul_rn(sg, v87)))
            st_v4_b32(_addr(v90) + t * 32, [f2i(o90[e]) for e in range(4)])
            st_v4_b32(_addr(v90) + t * 32 + 16, [f2i(o90[e]) for e in range(4, 8)])
            # stable-sort positions: (expert, flat index t*3+j)
            for j in cutlass.range_constexpr(3):
                e = ex[j]
                w = _sel(e < 4, ex_lo, ex_hi)
                rank = Int32((w >> (Int64(16) * Int64(e % 4))) & Int64(0xFFFF))
                for jj in cutlass.range_constexpr(j):
                    rank = rank + _sel(ex[jj] == e, Int32(1), Int32(0))
                pos = sbase[e] + rank
                st_b64(_addr(v28) + Int64(pos) * 8, t * 3 + j)
                st_b64(_addr(v29) + Int64(pos) * 8, t)
                spos[tt * 4 + j] = pos
                sw[tt * 4 + j] = mul_rn(div_rn(g[j], d39), Float32(1.0))
        cute.arch.sync_threads()

        # rows: warp w handles tokens 4 w .. 4 w + 3 (loads first), 8 columns per lane
        for bt in cutlass.range_constexpr(1):
            batch = []
            for u in cutlass.range_constexpr(4):
                tk = wid * 4 + u
                roff = Int64(bid * TPB + tk) * (H * 2) + Int64(lane) * 16
                batch.append((tk, roff, ld_v4_b32(_addr(in4) + roff), ld_v4_b32(_addr(v11) + roff), ld_v4_b32(_addr(v22) + roff)))
            for tk, roff, a4, a11, r22 in batch:
                r15 = _add_bf16x8(a4, a11)
                st_v4_b32(_addr(v15) + roff, r15)
                f = []
                for q in cutlass.range_constexpr(4):
                    f.append(_bf_lo(r15[q]))
                    f.append(_bf_hi(r15[q]))
                for j in cutlass.range_constexpr(3):
                    pos = Int64(spos[tk * 4 + j])
                    wj = sw[tk * 4 + j]
                    o32 = _addr(v33) + pos * (H * 4) + Int64(lane) * 32
                    st_v8_b32(o32, [f2i(f[q]) for q in range(8)])
                    st_v4_b32(_addr(v46) + pos * (H * 2) + Int64(lane) * 16,
                              [bf16x2(mul_rn(f[2 * q], wj), mul_rn(f[2 * q + 1], wj)) for q in range(4)])
                    st_v4_b32(_addr(v34) + pos * (H * 2) + Int64(lane) * 16, r22)

_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fns = _CACHE.get(dev)
    if fns is None:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        bf = lambda n, a=16: _fake(cutlass.BFloat16, n, a)
        fa = cute.compile(
            RouterAuxBwd(), bf(NT * H), bf(H), bf(NT * H), bf(NT * H), bf(NT * H), bf(NT * H), bf(H),
            _fake(cutlass.Int64, NS, 8), _fake(cutlass.Uint8, NT * 8, 8), bf(NT * H), _fake(cutlass.Float32, NT, 4),
            bf(NT * H), bf(NT * H), bf(NT * H), _fake(cutlass.Float32, NT, 4), _fake(cutlass.Int64, NCTA * 3, 8), stream,
            options="--enable-tvm-ffi")
        fb = cute.compile(
            RouterAuxBwdB(), bf(NT * H), _fake(cutlass.Int64, NS, 8), _fake(cutlass.Float32, NT * 8),
            _fake(cutlass.Uint8, NT, 1), _fake(cutlass.Int64, 1, 8), bf(NT * H), bf(H, 4), bf(NT * H),
            _fake(cutlass.Int64, NCTA * 3, 8), _fake(cutlass.Float32, H, 4), bf(NT * H), _fake(cutlass.Int64, NS, 8),
            _fake(cutlass.Int64, NS, 8), _fake(cutlass.Float32, NS * H), bf(NS * H), _fake(cutlass.Float32, NT, 4),
            bf(NS * H), _fake(cutlass.Float32, NT * 8), _fake(cutlass.Float32, NT * 3, 4), _fake(cutlass.Float32, NT * 8),
            stream, options="--enable-tvm-ffi")
        fns = (fa, fb)
        _CACHE[dev] = fns
    return fns


def _a16(t):
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
    dev = in0.device
    fa, fb = _compiled(dev.index)
    e = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=dev)
    v7 = e((NT, H), torch.bfloat16)
    v19 = e((NT, H), torch.bfloat16)
    scratch = e((NCTA * 3,), torch.int64)
    in8c = in8.contiguous()
    v3 = e((NT, H), torch.bfloat16)
    v4 = e((NT, 1), torch.float32)
    v22 = e((NT, H), torch.bfloat16)
    v23 = e((NT, 1), torch.float32)
    fa(_a16(in0).view(-1), _a16(in1), _a16(in2).view(-1), _a16(in3).view(-1), _a16(in5).view(-1), _a16(in6).view(-1),
       _a16(in7), in8c.view(-1), _a16(in10).view(-1).view(torch.uint8), v3.view(-1), v4.view(-1), v7.view(-1),
       v19.view(-1), v22.view(-1), v23.view(-1), scratch)
    v11, v12 = torch.ops.aten._fused_rms_norm_backward.default(v7, in0, [H], v4, in1, [True, True])
    v13 = e((H,), torch.float32)
    v15 = e((NT, H), torch.bfloat16)
    v28 = e((NS,), torch.int64)
    v29 = e((NS,), torch.int64)
    v33 = e((NS, H), torch.float32)
    v34 = e((NS, H), torch.bfloat16)
    v39 = e((NT, 1), torch.float32)
    v46 = e((NS, H), torch.bfloat16)
    v90 = e((NT, 8), torch.float32)
    v92 = e((NT, 3), torch.float32)
    v94 = e((NT, 8), torch.float32)
    fb(_a16(in4).view(-1), in8c.view(-1), _a16(in9).view(-1), in11.reshape(-1).view(torch.uint8), in12.reshape(1),
       _a16(v11).view(-1), v12.contiguous(), _a16(v22).view(-1), scratch, v13, v15.view(-1), v28, v29, v33.view(-1),
       v34.view(-1), v39.view(-1), v46.view(-1), v90.view(-1), v92.view(-1), v94.view(-1))
    return (v3, v13, v15, v15.t(), v19, v22, v28, v29, v33, v34, v39, v46, v46.t(), v90, v92, v94, v23)
