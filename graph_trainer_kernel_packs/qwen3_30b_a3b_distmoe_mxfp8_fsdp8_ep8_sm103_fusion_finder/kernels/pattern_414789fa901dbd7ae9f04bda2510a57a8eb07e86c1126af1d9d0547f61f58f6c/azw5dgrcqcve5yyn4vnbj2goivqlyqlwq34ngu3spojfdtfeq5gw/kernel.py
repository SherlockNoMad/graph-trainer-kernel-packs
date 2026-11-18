"""attention_norm forward (Qwen3 30B-A3B, hidden 2048): residual add + RMSNorm + MXFP8 rowwise
quantization + scale swizzle, one CuTe DSL kernel.

reference (M = 32768 rows, N = 2048):
  v2 = in0 + in1 (bf16); v5, rstd = _fused_rms_norm(v2, [2048], in2, 1e-6)
  q, _, s, _ = torchao.mxfp8_quantize(v5, rowwise=True, colwise=False, 32, 1, 'e4m3', 'rceil')
  returns (v2, v5, rstd, q, triton_mx_block_rearrange(s))

One launch (aten: add, quack RMSNorm, torchao mxfp8_quantize, triton scale swizzle). One warp per row,
ROWS rows per CTA; lane i owns the 8-column chunks at 8 i + 256 k, k = 0..7 (quack's RMSNorm layout for
N = 2048: threads_per_row 32, vecsize 8), so the sum of squares runs in quack's order. The add + RMS part
is q-writer-norm's r014 kernel (candidates/r014_4e183ee7_ah2kzcvf/q-writer-norm/t1).

Numerics (aten add / quack RMSNorm / torchao RCEIL, matched bit for bit):
  v2 = bf16(f32(in0) + f32(in1)); s = fma(x, x, s) over the lane's 64 columns of bf16 v2 in column order
  from 0; butterfly shfl.xor 16..1; rstd = rsqrt.approx.ftz(s / 2048 + 1e-6); v5 = bf16((x * rstd) * w).
  MXFP8 block (32 columns of v5 = chunk k of 4 adjacent lanes): amax = max.NaN of |x| from 0 (lane-local
  over 8 values, then shfl.xor 1, 2; max is order-independent); v = amax * 0x3B124925 (RN);
  b = cvt.rp.bf16(v & 0x7fffffff); e = (b >> 7) + round_up, round_up = !((b & 0x7f) == 0 ||
  (b & 0x7fff) < 65) && (b & 0x7f80) != 0x7f80; descale = NaN if e == 255, 2^-127 if e == 254, else
  ((254 - e) & 0xff) << 23; q = cvt.rn.satfinite.e4m3x2(x * descale) (recipe of q-writer-mxfp8, BRIEF.md).
  Scale swizzle (M % 128 == 0, 64 scale columns): byte (r, c) goes to
  (r // 128) * 128 * 64 + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 32768
N = 2048
ROWS = 4  # rows (warps) per CTA
CHUNKS = N // 256


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


def _ptr(p, loc, ip):
    return p.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


def _i32(text, args, loc, ip):
    return Int32(_asm(T.i32(), [_ir(v, loc, ip) for v in args], text, "=r" + ",r" * len(args), loc, ip))


@dsl_user_op
def ld_v4_nc(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ptr(ptr, loc, ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_ptr(ptr, loc, ip)] + [_ir(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def st_b32(ptr, v, *, loc=None, ip=None):
    _asm(None, [_ptr(ptr, loc, ip), _ir(v, loc, ip)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side_effects=True)


@dsl_user_op
def add_word(a, b, *, loc=None, ip=None):
    text = (
        "{\n.reg .b32 a0, a1, b0, b1;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
        "shl.b32 b0, $2, 16;\nand.b32 b1, $2, 0xFFFF0000;\n"
        "add.rn.f32 a0, a0, b0;\nadd.rn.f32 a1, a1, b1;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return _i32(text, (a, b), loc, ip)


@dsl_user_op
def sumsq4(s, w0, w1, w2, w3, *, loc=None, ip=None):
    """s + x0^2 + ... + x7^2 (fma chain in column order) over the bf16 pairs of 4 words."""
    text = (
        "{\n.reg .b32 a, s;\nmov.b32 s, $1;\n"
        + "".join(
            f"shl.b32 a, ${k + 2}, 16;\nfma.rn.f32 s, a, a, s;\nand.b32 a, ${k + 2}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
            for k in range(4)
        )
        + "mov.b32 $0, s;\n}"
    )
    return _i32(text, (s, w0, w1, w2, w3), loc, ip)


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def rstd_of(s, *, loc=None, ip=None):
    """rsqrt.approx.ftz(s / 2048 + 1e-6)."""
    text = "{\n.reg .b32 m;\ndiv.rn.f32 m, $1, 0f45000000;\nadd.rn.f32 m, m, 0f358637BD;\nrsqrt.approx.ftz.f32 $0, m;\n}"
    return _i32(text, (s,), loc, ip)


@dsl_user_op
def rms_word(x, g, r, *, loc=None, ip=None):
    """bf16x2 x, w -> bf16x2((x * rstd) * w)."""
    text = (
        "{\n.reg .b32 a0, a1, g0, g1;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
        "shl.b32 g0, $2, 16;\nand.b32 g1, $2, 0xFFFF0000;\n"
        "mul.rn.f32 a0, a0, $3;\nmul.rn.f32 a1, a1, $3;\n"
        "mul.rn.f32 a0, a0, g0;\nmul.rn.f32 a1, a1, g1;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return _i32(text, (x, g, r), loc, ip)


@dsl_user_op
def st_v2(ptr, w0, w1, *, loc=None, ip=None):
    _asm(
        None,
        [_ptr(ptr, loc, ip), _ir(w0, loc, ip), _ir(w1, loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def st_u8(ptr, v, *, loc=None, ip=None):
    _asm(None, [_ptr(ptr, loc, ip), _ir(v, loc, ip)], "st.global.u8 [$0], $1;", "l,r", loc, ip, side_effects=True)


@dsl_user_op
def amax8(w0, w1, w2, w3, *, loc=None, ip=None):
    """max.NaN of |x| over the 8 bf16 of 4 words, from 0 (f32 bits)."""
    text = "{\n.reg .f32 m, t;\n.reg .b32 u;\nmov.b32 m, 0;\n" + "".join(
        f"shl.b32 u, ${k + 1}, 16;\nand.b32 u, u, 0x7fffffff;\nmov.b32 t, u;\nmax.NaN.f32 m, m, t;\n"
        f"and.b32 u, ${k + 1}, 0x7fff0000;\nmov.b32 t, u;\nmax.NaN.f32 m, m, t;\n"
        for k in range(4)
    ) + "mov.b32 $0, m;\n}"
    return _i32(text, (w0, w1, w2, w3), loc, ip)


@dsl_user_op
def fmax_nan(a, b, *, loc=None, ip=None):
    text = "{\n.reg .f32 x, y;\nmov.b32 x, $1;\nmov.b32 y, $2;\nmax.NaN.f32 x, x, y;\nmov.b32 $0, x;\n}"
    return _i32(text, (a, b), loc, ip)


_SCALE = (
    "{\n.reg .f32 v;\n.reg .b16 b;\n.reg .b32 bb, ex, t, e, d;\n.reg .pred p0, p1, p2, p3, p4;\n"
    "mov.b32 v, $2;\n"
    "mul.rn.f32 v, v, 0f3B124925;\n"
    "mov.b32 t, v;\nand.b32 t, t, 0x7fffffff;\nmov.b32 v, t;\n"
    "cvt.rp.bf16.f32 b, v;\n"
    "cvt.u32.u16 bb, b;\n"
    "shr.b32 ex, bb, 7;\n"
    "and.b32 t, bb, 127;\nsetp.eq.u32 p0, t, 0;\n"
    "and.b32 t, bb, 32767;\nsetp.lt.u32 p1, t, 65;\n"
    "or.pred p0, p0, p1;\n"
    "and.b32 t, bb, 32640;\nsetp.ne.u32 p2, t, 32640;\n"
    "not.pred p0, p0;\nand.pred p2, p2, p0;\n"
    "selp.u32 t, 1, 0, p2;\n"
    "add.u32 e, ex, t;\nand.b32 e, e, 255;\n"
    "sub.u32 d, 254, e;\nand.b32 d, d, 255;\nshl.b32 d, d, 23;\n"
    "setp.eq.u32 p3, e, 254;\nsetp.eq.u32 p4, e, 255;\n"
    "selp.b32 d, 0x00400000, d, p3;\nselp.b32 d, 0x7fff0000, d, p4;\n"
    "mov.b32 $0, e;\nmov.b32 $1, d;\n}"
)


@dsl_user_op
def mx_scale(amax, *, loc=None, ip=None):
    """(e8m0 byte, descale f32 bits) of a block amax (RCEIL)."""
    res = _asm(ir.Type.parse("!llvm.struct<(i32, i32)>"), [_ir(amax, loc, ip)], _SCALE, "=r,=r,r", loc, ip)
    return _unpack(res, 2, loc, ip)


_Q8 = (
    "{\n.reg .f32 d, a0, a1, a2, a3, a4, a5, a6, a7;\n.reg .b32 t;\n.reg .b16 p0, p1, p2, p3;\n"
    "mov.b32 d, $6;\n"
    + "".join(
        f"shl.b32 t, ${k + 2}, 16;\nmov.b32 a{2 * k}, t;\nmul.rn.f32 a{2 * k}, a{2 * k}, d;\n"
        f"and.b32 t, ${k + 2}, 0xffff0000;\nmov.b32 a{2 * k + 1}, t;\nmul.rn.f32 a{2 * k + 1}, a{2 * k + 1}, d;\n"
        for k in range(4)
    )
    + "cvt.rn.satfinite.e4m3x2.f32 p0, a1, a0;\ncvt.rn.satfinite.e4m3x2.f32 p1, a3, a2;\n"
    "cvt.rn.satfinite.e4m3x2.f32 p2, a5, a4;\ncvt.rn.satfinite.e4m3x2.f32 p3, a7, a6;\n"
    "mov.b32 $0, {p0, p1};\nmov.b32 $1, {p2, p3};\n}"
)


@dsl_user_op
def quant8(w0, w1, w2, w3, d, *, loc=None, ip=None):
    """8 bf16 (4 words) times descale -> 8 e4m3 bytes (2 words, column order)."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_ir(v, loc, ip) for v in (w0, w1, w2, w3, d)],
        _Q8,
        "=r,=r,r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, loc, ip)


class AddNorm:
    @cute.jit
    def __call__(self, a, b, w, o2, o5, rstd, oq, osc, stream):
        self.kernel(a, b, w, o2, o5, rstd, oq, osc).launch(grid=[M // ROWS, 1, 1], block=[32 * ROWS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        o2: cute.Tensor,
        o5: cute.Tensor,
        rstd: cute.Tensor,
        oq: cute.Tensor,
        osc: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        row = bid * ROWS + warp
        base = row * N + lane * 8
        xa = []
        xb = []
        for k in cutlass.range_constexpr(CHUNKS):
            xa.append(ld_v4_nc(a.iterator + (base + 256 * k)))
            xb.append(ld_v4_nc(b.iterator + (base + 256 * k)))
        g = []
        for k in cutlass.range_constexpr(CHUNKS):
            g.append(ld_v4_nc(w.iterator + (lane * 8 + 256 * k)))
        v = []
        for k in cutlass.range_constexpr(CHUNKS):
            vk = [add_word(xa[k][e], xb[k][e]) for e in range(4)]
            st_v4(o2.iterator + (base + 256 * k), vk)
            v.append(vk)
        s = Int32(0)
        for k in cutlass.range_constexpr(CHUNKS):
            s = sumsq4(s, v[k][0], v[k][1], v[k][2], v[k][3])
        for off in cutlass.range_constexpr(5):
            s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
        r = rstd_of(s)
        if lane == 0:
            st_b32(rstd.iterator + row, r)
        # swizzled scale row base: (r // 128) * 8192 + (r % 32) * 16 + ((r % 128) // 32) * 4
        sbase = (row // 128) * (128 * 64) + (row % 32) * 16 + ((row % 128) // 32) * 4
        for k in cutlass.range_constexpr(CHUNKS):
            y = [rms_word(v[k][e], g[k][e], r) for e in range(4)]
            st_v4(o5.iterator + (base + 256 * k), y)
            m = amax8(y[0], y[1], y[2], y[3])
            m = fmax_nan(m, cute.arch.shuffle_sync_bfly(m, 1))
            m = fmax_nan(m, cute.arch.shuffle_sync_bfly(m, 2))
            sc = mx_scale(m)
            qw = quant8(y[0], y[1], y[2], y[3], sc[1])
            st_v2(oq.iterator + (base + 256 * k), qw[0], qw[1])
            if lane % 4 == 0:
                c = 8 * k + lane // 4  # scale column (32-column block)
                st_u8(osc.iterator + (sbase + (c // 4) * 512 + c % 4), sc[0])


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        bf = cutlass.BFloat16
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = _CACHE[dev] = cute.compile(
            AddNorm(),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(bf, N),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(Int32, M, 4),
            _fake(cutlass.Uint8, M * N, 16),
            _fake(cutlass.Uint8, M * (N // 32), 16),
            stream,
            options="--enable-tvm-ffi",
        )
    return fn


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2):
    assert in0.shape == (M, N) and in1.shape == (M, N) and in2.shape == (N,)
    assert in0.dtype == torch.bfloat16 and in1.dtype == torch.bfloat16 and in2.dtype == torch.bfloat16
    dev = in0.device
    k = _compiled(dev.index)
    o2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    o5 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((M, 1), dtype=torch.float32, device=dev)
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    sc = torch.empty((M, N // 32), dtype=torch.float8_e8m0fnu, device=dev)
    k(
        _aligned(in0).view(-1),
        _aligned(in1).view(-1),
        _aligned(in2).view(-1),
        o2.view(-1),
        o5.view(-1),
        rstd.view(torch.int32).view(-1),
        q.view(torch.uint8).view(-1),
        sc.view(torch.uint8).view(-1),
    )
    return (o2, o5, rstd, q, sc)
