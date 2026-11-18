"""Qwen3 30B-A3B attention RoPE forward with the q/k RMS norms, one CuTe DSL kernel.

reference, per token t (in0[t] = 4 kv groups x 10 heads x 128: 8 q heads, 1 k head, 1 v head):
  out0 = q (32 x 128), out1 = k (4 x 128), out2 = v (4 x 128): pre-norm copies
  out3, out4 = rstd of _fused_rms_norm(q / k, [128], in1 / in2, 1e-6) (f32)
  f = in3[in4[t]] (256 f32), out5 = f[:128] (cos), out6 = f[128:] (sin), views of one T x 256 buffer
  out7 / out8 = bf16(rope(f32(yq / yk))), y = the bf16 RMS norm output:
    rope(y)[d] = y[d] * cos[d] + (-y[d + 64]) * sin[d]   (d < 64)
    rope(y)[d] = y[d] * cos[d] + y[d - 64] * sin[d]      (d >= 64)
  every product and the sum rounded to f32 separately (separate aten mul / neg / cat / add kernels).

Layout: one warp per token, lane = 8 g + j (g = kv group, j = 0..7). Lane (g, j) owns columns
[8j, 8j + 8) and [64 + 8j, 72 + 8j) of each of the 10 heads of group g, so the rotate-half partner of
every element is in the same thread, and the cos/sin for those columns stay in registers for all 9
normed heads. Lane (g, j) also writes 8 floats of the gathered freqs row, and the q rstd of head 8g+j.

RMS norm numerics: aten._fused_rms_norm (bf16, N=128) dispatches to torch's vendored quack RMSNorm
(torch/_vendor/quack/rmsnorm.py): threads_per_row 16 (N <= 128), vecsize 8, so quack lane i owns
columns [8i, 8i + 8); it sums squares in column order from 0 (squares of bf16 are exact in f32, so
fma == mul + add), then butterfly shfl.xor 8, 4, 2, 1. The xor-8 step adds chunks i and i ^ 8, which
here are the two chunks of one lane (f32 add is commutative), and xor 4, 2, 1 run across the 8 lanes
of the group. rstd = rsqrt.approx.ftz(s / 128 + eps); y = bf16((x * rstd) * w).

Assumptions: in4 values are in [-4096, 4096) (aten indexing raises otherwise); negative positions wrap
like aten indexing. Only the low 32 bits of each int64 position are read.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TOK = 32768
G = 4  # kv groups (k / v heads)
HPG = 10  # heads per group in in0: 8 q, 1 k, 1 v
QPG = 8
D = 128
ROW = G * HPG * D  # 5120
NQ = G * QPG  # 32
NPOS = 4096
FW = 256  # freqs row: 128 cos | 128 sin
THREADS = 128
TOK_PER_CTA = THREADS // 32
EPS_BITS = "0f358637BD"  # float32(1e-6)


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


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


def _struct(n):
    return ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * n) + ")>")


@dsl_user_op
def ld_v4_stream(ptr, *, loc=None, ip=None):
    res = _asm(
        _struct(4),
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
        _struct(4),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_pos(ptr, *, loc=None, ip=None):
    """positions[t] (int64, low word), negative wrapped like aten indexing."""
    return Int32(
        _asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "{\n.reg .s32 p, q;\n.reg .pred n;\n"
            "ld.global.nc.b32 p, [$1];\n"
            "setp.lt.s32 n, p, 0;\n"
            f"add.s32 q, p, {NPOS};\n"
            "selp.s32 $0, q, p, n;\n}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [_v(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def st_b32(ptr, v, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), _v(v, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def sel_eq(a, b, x, y, *, loc=None, ip=None):
    """x if a == b else y."""
    return Int32(
        _asm(
            T.i32(),
            [_v(a, loc, ip), _v(b, loc, ip), _v(x, loc, ip), _v(y, loc, ip)],
            "{\n.reg .pred p;\nsetp.eq.s32 p, $1, $2;\nselp.b32 $0, $3, $4, p;\n}",
            "=r,r,r,r,r",
            loc,
            ip,
        )
    )


_SUMSQ16 = (
    "{\n.reg .b32 a, s, u;\nmov.b32 s, 0f00000000;\n"
    + "".join(
        f"shl.b32 a, ${k + 1}, 16;\nfma.rn.f32 s, a, a, s;\n"
        f"and.b32 a, ${k + 1}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
        for k in range(4)
    )
    + "mov.b32 u, 0f00000000;\n"
    + "".join(
        f"shl.b32 a, ${k + 5}, 16;\nfma.rn.f32 u, a, a, u;\n"
        f"and.b32 a, ${k + 5}, 0xFFFF0000;\nfma.rn.f32 u, a, a, u;\n"
        for k in range(4)
    )
    + "add.rn.f32 $0, s, u;\n}"
)


@dsl_user_op
def sumsq_two_chunks(w, *, loc=None, ip=None):
    """(sum of squares of chunk lo) + (sum of squares of chunk hi), each 8 bf16 (4 words) summed in
    element order from 0: quack's lane sum and its butterfly xor-8 step."""
    return Int32(_asm(T.i32(), [_v(v, loc, ip) for v in w], _SUMSQ16, "=r" + ",r" * 8, loc, ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_v(a, loc, ip), _v(b, loc, ip)], "add.rn.f32 $0, $1, $2;", "=r,r,r", loc, ip)
    )


@dsl_user_op
def rstd_of(s, *, loc=None, ip=None):
    text = (
        "{\n.reg .b32 m;\n"
        "div.rn.f32 m, $1, 0f43000000;\n"  # / 128
        f"add.rn.f32 m, m, {EPS_BITS};\n"
        "rsqrt.approx.ftz.f32 $0, m;\n}"
    )
    return Int32(_asm(T.i32(), [_v(s, loc, ip)], text, "=r,r", loc, ip))


# One pair of words: x lo word (columns 8j + 2k, +1), x hi word (64 + 8j + 2k, +1), w lo / hi words,
# rstd, then cos lo (2 f32), cos hi, sin lo, sin hi. Out: roped lo word, roped hi word.
_NORM_ROPE = (
    "{\n"
    ".reg .b32 x0, x1, w0, w1, ylo, yhi, a0, a1, b0, b1, t, u, l0, l1, h0, h1;\n"
    # y lo = bf16((x * rstd) * w)
    "shl.b32 x0, $2, 16;\nand.b32 x1, $2, 0xFFFF0000;\n"
    "shl.b32 w0, $4, 16;\nand.b32 w1, $4, 0xFFFF0000;\n"
    "mul.rn.f32 x0, x0, $6;\nmul.rn.f32 x1, x1, $6;\n"
    "mul.rn.f32 x0, x0, w0;\nmul.rn.f32 x1, x1, w1;\n"
    "cvt.rn.bf16x2.f32 ylo, x1, x0;\n"
    # y hi
    "shl.b32 x0, $3, 16;\nand.b32 x1, $3, 0xFFFF0000;\n"
    "shl.b32 w0, $5, 16;\nand.b32 w1, $5, 0xFFFF0000;\n"
    "mul.rn.f32 x0, x0, $6;\nmul.rn.f32 x1, x1, $6;\n"
    "mul.rn.f32 x0, x0, w0;\nmul.rn.f32 x1, x1, w1;\n"
    "cvt.rn.bf16x2.f32 yhi, x1, x0;\n"
    # f32(y)
    "shl.b32 a0, ylo, 16;\nand.b32 a1, ylo, 0xFFFF0000;\n"
    "shl.b32 b0, yhi, 16;\nand.b32 b1, yhi, 0xFFFF0000;\n"
    # lo: a * cos_lo + (-b) * sin_lo
    "mul.rn.f32 t, a0, $7;\nmul.rn.f32 u, b0, $11;\nneg.f32 u, u;\nadd.rn.f32 l0, t, u;\n"
    "mul.rn.f32 t, a1, $8;\nmul.rn.f32 u, b1, $12;\nneg.f32 u, u;\nadd.rn.f32 l1, t, u;\n"
    # hi: b * cos_hi + a * sin_hi
    "mul.rn.f32 t, b0, $9;\nmul.rn.f32 u, a0, $13;\nadd.rn.f32 h0, t, u;\n"
    "mul.rn.f32 t, b1, $10;\nmul.rn.f32 u, a1, $14;\nadd.rn.f32 h1, t, u;\n"
    "cvt.rn.bf16x2.f32 $0, l1, l0;\n"
    "cvt.rn.bf16x2.f32 $1, h1, h0;\n"
    "}"
)


@dsl_user_op
def norm_rope(xl, xh, wl, wh, r, cl0, cl1, ch0, ch1, sl0, sl1, sh0, sh1, *, loc=None, ip=None):
    args = [xl, xh, wl, wh, r, cl0, cl1, ch0, ch1, sl0, sl1, sh0, sh1]
    res = _asm(_struct(2), [_v(a, loc, ip) for a in args], _NORM_ROPE, "=r,=r" + ",r" * 13, loc, ip)
    return _unpack(res, 2, loc, ip)


class RopeFwd:
    @cute.jit
    def __call__(self, mX, mWq, mWk, mF, mP, oQ, oK, oV, oRq, oRk, oF, oQr, oKr, stream):
        self.kernel(mX, mWq, mWk, mF, mP, oQ, oK, oV, oRq, oRk, oF, oQr, oKr).launch(
            grid=[TOK // TOK_PER_CTA, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mX, mWq, mWk, mF, mP, oQ, oK, oV, oRq, oRk, oF, oQr, oKr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        t = bidx * TOK_PER_CTA + tidx // 32
        g = lane // 8
        j = lane % 8

        # inputs: x of the 10 heads of group g (two 16 B chunks each)
        xbase = t * ROW + g * (HPG * D) + j * 8
        xs = []
        for h in cutlass.range_constexpr(HPG):
            lo = ld_v4_stream(mX.iterator + (xbase + h * D))
            hi = ld_v4_stream(mX.iterator + (xbase + h * D + 64))
            xs.append((lo, hi))
        pos = ld_pos(mP.iterator + t)
        fb = pos * FW + j * 8
        segs = []
        for s in cutlass.range_constexpr(4):  # cos lo, cos hi, sin lo, sin hi
            a = ld_v4_cached(mF.iterator + (fb + s * 64))
            b = ld_v4_cached(mF.iterator + (fb + s * 64 + 4))
            segs.append(a + b)
        wq = ld_v4_cached(mWq.iterator + j * 8) + ld_v4_cached(mWq.iterator + (j * 8 + 64))
        wk = ld_v4_cached(mWk.iterator + j * 8) + ld_v4_cached(mWk.iterator + (j * 8 + 64))

        # pre-norm copies of q, k, v
        qbase = t * (NQ * D) + g * (QPG * D) + j * 8
        kbase = t * (G * D) + g * D + j * 8
        for h in cutlass.range_constexpr(QPG):
            st_v4(oQ.iterator + (qbase + h * D), xs[h][0])
            st_v4(oQ.iterator + (qbase + h * D + 64), xs[h][1])
        st_v4(oK.iterator + kbase, xs[QPG][0])
        st_v4(oK.iterator + (kbase + 64), xs[QPG][1])
        st_v4(oV.iterator + kbase, xs[QPG + 1][0])
        st_v4(oV.iterator + (kbase + 64), xs[QPG + 1][1])

        # gathered freqs row: lane (g, j) writes columns [64 g + 8 j, +8) = segment g of its columns
        fo = []
        for k in cutlass.range_constexpr(8):
            fo.append(
                sel_eq(g, 0, segs[0][k], sel_eq(g, 1, segs[1][k], sel_eq(g, 2, segs[2][k], segs[3][k])))
            )
        fob = t * FW + g * 64 + j * 8
        st_v4(oF.iterator + fob, fo[0:4])
        st_v4(oF.iterator + (fob + 4), fo[4:8])

        cl, ch, sl, sh = segs
        rq = Int32(0)
        for h in cutlass.range_constexpr(QPG + 1):
            lo, hi = xs[h]
            w = wq if h < QPG else wk
            s = sumsq_two_chunks(lo + hi)
            for off in cutlass.range_constexpr(3):
                s = fadd(s, cute.arch.shuffle_sync_bfly(s, offset=4 >> off))
            r = rstd_of(s)
            ol = []
            oh = []
            for k in cutlass.range_constexpr(4):
                o = norm_rope(
                    lo[k], hi[k], w[k], w[k + 4], r,
                    cl[2 * k], cl[2 * k + 1], ch[2 * k], ch[2 * k + 1],
                    sl[2 * k], sl[2 * k + 1], sh[2 * k], sh[2 * k + 1],
                )
                ol.append(o[0])
                oh.append(o[1])
            if cutlass.const_expr(h < QPG):
                st_v4(oQr.iterator + (qbase + h * D), ol)
                st_v4(oQr.iterator + (qbase + h * D + 64), oh)
                rq = sel_eq(j, h, r, rq)
            else:
                st_v4(oKr.iterator + kbase, ol)
                st_v4(oKr.iterator + (kbase + 64), oh)
                if j == 0:
                    st_b32(oRk.iterator + (t * G + g), r)
        st_b32(oRq.iterator + (t * NQ + g * QPG + j), rq)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        bf, f32 = cutlass.BFloat16, cutlass.Float32
        fn = cute.compile(
            RopeFwd(),
            _fake(bf, TOK * ROW),
            _fake(bf, D),
            _fake(bf, D),
            _fake(f32, NPOS * FW),
            _fake(cutlass.Int64, TOK, 8),
            _fake(bf, TOK * NQ * D),
            _fake(bf, TOK * G * D),
            _fake(bf, TOK * G * D),
            _fake(f32, TOK * NQ, 4),
            _fake(f32, TOK * G, 4),
            _fake(f32, TOK * FW),
            _fake(bf, TOK * NQ * D),
            _fake(bf, TOK * G * D),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(t):
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def kernel_function(in0, in1, in2, in3, in4):
    assert in0.shape == (TOK, ROW) and in0.dtype == torch.bfloat16
    assert in1.shape == (D,) and in2.shape == (D,)
    assert in3.shape == (NPOS, FW) and in3.dtype == torch.float32
    assert in4.shape == (TOK,) and in4.dtype == torch.int64
    x, wq, wk, freqs = _aligned(in0), _aligned(in1), _aligned(in2), _aligned(in3)
    pos = in4.contiguous()
    dev = in0.device
    bf, f32 = torch.bfloat16, torch.float32
    q = torch.empty((TOK, NQ, D), dtype=bf, device=dev)
    k = torch.empty((TOK, G, D), dtype=bf, device=dev)
    v = torch.empty((TOK, G, D), dtype=bf, device=dev)
    rstd_q = torch.empty((TOK, NQ, 1), dtype=f32, device=dev)
    rstd_k = torch.empty((TOK, G, 1), dtype=f32, device=dev)
    f = torch.empty((TOK, 1, FW), dtype=f32, device=dev)
    q_rope = torch.empty((TOK, NQ, D), dtype=bf, device=dev)
    k_rope = torch.empty((TOK, G, D), dtype=bf, device=dev)
    _compiled(dev.index)(
        x.view(-1), wq, wk, freqs.view(-1), pos,
        q.view(-1), k.view(-1), v.view(-1), rstd_q.view(-1), rstd_k.view(-1), f.view(-1),
        q_rope.view(-1), k_rope.view(-1),
    )
    return (q, k, v, rstd_q, rstd_k, f[:, :, :D], f[:, :, D:], q_rope, k_rope)
