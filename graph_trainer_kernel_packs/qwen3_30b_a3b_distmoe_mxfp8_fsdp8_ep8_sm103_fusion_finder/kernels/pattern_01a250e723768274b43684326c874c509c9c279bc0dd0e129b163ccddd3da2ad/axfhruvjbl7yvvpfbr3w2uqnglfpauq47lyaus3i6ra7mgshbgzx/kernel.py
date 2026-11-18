"""fp32 -> bf16 cast + dist_moe.prepare_mxfp8_weight (32x32 MXFP8, training mode) in one CuTe DSL kernel.

reference (problem.py): w = in0.to(bf16).reshape(G, R, C); q, s_fprop, s_dgrad = prepare_mxfp8_weight(w);
returns (q, s_fprop, s_dgrad, empty uint8). prepare_mxfp8_weight (dist_moe/_blockscaled.py,
quantize_block_scaled_axes with ScaleReduction.TWO_D, CUBLAS_BLOCKED, half_range_scale off for MXFP8):
one e8m0 scale per 32x32 tile of each expert's [R, C] matrix, q = rowwise e4m3 qdata [G, R, C],
s_fprop = blocked [G*R, C/32] (tile scale repeated over its 32 rows), s_dgrad = per expert blocked [C, R/32]
(tile scale repeated over its 32 columns), expert g at offset g * C * R/32. Blocked offset of (r, c) in an
[n, k] scale matrix: (r // 128) * 128 * k + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4.

Numerics: x = cvt.rn.bf16(in0) (aten _to_copy); amax = tile max of |x| (dist_moe maps NaN -> inf per element,
which gives the same byte and descale as max.NaN below); v = amax * fp32(1/448); scale byte = cvt.rp e8m0
(inf/NaN -> 0xFF); descale = (254 - byte) << 23, NaN for byte 0xFF; q = cvt.rn.satfinite.e4m3(x * descale).
This equals the torchao RCEIL recipe used below (mx_scale) for every reachable amax (byte 254 needs
amax > 448 * 2^126, beyond bf16).

CTA (bx, by): rows 128 by .. +127 (inside one expert since R % 128 == 0), columns 128 bx .. +127; warp w
owns the 32x32 tile (w // 4, w % 4); lane l owns columns 4 (l % 8) .. +3 of rows 8 (l // 8) .. +7 (one
16-byte fp32 load per row). Tile amax: lane max.NaN, then XOR shuffles over the warp. The CTA's 16 tile
scales fill exactly one 512-byte atom of s_fprop and one of s_dgrad, staged in shared memory.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(ret, args, code, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret, args, code, cons, has_side_effects=side, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def ld_nc_v2_b32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"), [_v(Int64(addr), loc, ip)],
        "ld.global.nc.L1::no_allocate.v2.b32 {$0, $1}, [$2];", "=r,=r,l", loc=loc, ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(2)]


@dsl_user_op
def st_v4_b32(addr, w, *, loc=None, ip=None):
    _asm(
        None, [_v(Int64(addr), loc, ip)] + [_v(Int32(x), loc, ip) for x in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", side=True, loc=loc, ip=ip,
    )


@dsl_user_op
def st_b32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.b32 [$0], $1;", "l,r",
         side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_u8(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip), _v(Int32(val), loc, ip)], "st.shared.u8 [$0], $1;", "r,r",
         side=True, loc=loc, ip=ip)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int32(addr), loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", side=True,
                      loc=loc, ip=ip))


@dsl_user_op
def amax_lo(acc, w, *, loc=None, ip=None):
    """max.NaN(acc, |bf16 in the low half of w|)."""
    code = "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nand.b32 t, t, 0x7fffffff;\nmax.NaN.f32 $0, $1, t;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(acc), loc, ip), _v(Int32(w), loc, ip)], code, "=f,f,r", loc=loc, ip=ip))


@dsl_user_op
def amax_hi(acc, w, *, loc=None, ip=None):
    """max.NaN(acc, |bf16 in the high half of w|)."""
    code = "{\n.reg .b32 t;\nand.b32 t, $2, 0x7fff0000;\nmax.NaN.f32 $0, $1, t;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(acc), loc, ip), _v(Int32(w), loc, ip)], code, "=f,f,r", loc=loc, ip=ip))


_SCALE = (
    "{\n.reg .f32 v;\n.reg .b16 b;\n.reg .b32 bb, ex, t, e, d;\n.reg .pred p0, p1, p2, p3, p4;\n"
    "mul.rn.f32 v, $2, 0f3B124925;\n"
    "and.b32 t, v, 0x7fffffff;\nmov.b32 v, t;\n"
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
    """(e8m0 byte, descale fp32) for a block amax (RCEIL)."""
    res = _asm(ir.Type.parse("!llvm.struct<(i32, f32)>"), [_v(Float32(amax), loc, ip)], _SCALE, "=r,=f,f",
               loc=loc, ip=ip)
    return Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)), Float32(
        llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip))


def _q4_code(halves):
    # $1..$4: words holding the 4 bf16 values; halves[k] = 0 (low) / 1 (high); $5: descale.
    lines = ["{\n.reg .b32 t0, t1, t2, t3;\n.reg .f32 a0, a1, a2, a3;\n.reg .b16 lo, hi;\n"]
    for k, h in enumerate(halves):
        if h == 0:
            lines.append(f"shl.b32 t{k}, ${k + 1}, 16;\n")
        else:
            lines.append(f"and.b32 t{k}, ${k + 1}, 0xffff0000;\n")
        lines.append(f"mov.b32 a{k}, t{k};\nmul.rn.f32 a{k}, a{k}, $5;\n")
    lines.append("cvt.rn.satfinite.e4m3x2.f32 lo, a1, a0;\ncvt.rn.satfinite.e4m3x2.f32 hi, a3, a2;\n")
    lines.append("mov.b32 $0, {lo, hi};\n}")
    return "".join(lines)


def _make_q4(halves):
    code = _q4_code(halves)

    @dsl_user_op
    def q4(w0, w1, w2, w3, d, *, loc=None, ip=None):
        args = [_v(Int32(w), loc, ip) for w in (w0, w1, w2, w3)] + [_v(Float32(d), loc, ip)]
        return Int32(_asm(T.i32(), args, code, "=r,r,r,r,r,f", loc=loc, ip=ip))

    return q4


_Q4_LO = _make_q4((0, 0, 0, 0))
_Q4_HI = _make_q4((1, 1, 1, 1))


@dsl_user_op
def ld_nc_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"), [_v(Int64(addr), loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def cvt_bf16x2(a, b, *, loc=None, ip=None):
    """bf16x2 word: low half = bf16_rn(f32 bits a), high half = bf16_rn(f32 bits b)."""
    code = "{\n.reg .f32 fa, fb;\nmov.b32 fa, $1;\nmov.b32 fb, $2;\ncvt.rn.bf16x2.f32 $0, fb, fa;\n}"
    return Int32(_asm(T.i32(), [_v(Int32(a), loc, ip), _v(Int32(b), loc, ip)], code, "=r,r,r", loc=loc, ip=ip))


@dsl_user_op
def fmax_nan(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], "max.NaN.f32 $0, $1, $2;",
                        "=f,f,f", loc=loc, ip=ip))


@dsl_user_op
def st_u8(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.u8 [$0], $1;", "l,r",
         side=True, loc=loc, ip=ip)


_Q4_MIX = _make_q4((0, 1, 0, 1))
THREADS = 512


class Mx2D:
    def __init__(self, G, R, C):
        self.G = G
        self.R = R
        self.C = C

    @cute.jit
    def __call__(self, x, q, s1, s0, stream):
        self.kernel(x, q, s1, s0).launch(
            grid=[self.C // 128, self.G * self.R // 128, 1], block=[THREADS, 1, 1], smem=1024, stream=stream
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, q: cute.Tensor, s1: cute.Tensor, s0: cute.Tensor):
        R = self.R
        C = self.C
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        ssc = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(1024), 16)
        sb1 = Int32(ssc.iterator.toint())
        sb0 = sb1 + 512
        lane = tid % 32
        w = tid // 32
        wr = w // 4
        wc = w % 4
        cg = lane % 8
        rg = lane // 8
        row0 = by * 128 + wr * 32 + rg * 8
        col0 = bx * 128 + wc * 32 + cg * 4
        off0 = Int64(row0) * C + Int64(col0)
        src = x.iterator.toint() + off0 * 4
        dst = q.iterator.toint() + off0
        raw = []
        for i in cutlass.range_constexpr(8):
            raw.append(ld_nc_v4_b32(src + i * C * 4))
        bw = []
        for i in cutlass.range_constexpr(8):
            bw.append([cvt_bf16x2(raw[i][0], raw[i][1]), cvt_bf16x2(raw[i][2], raw[i][3])])
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(2):
                amax = amax_lo(amax, bw[i][k])
                amax = amax_hi(amax, bw[i][k])
        for o in (1, 2, 4, 8, 16):
            amax = fmax_nan(amax, cute.arch.shuffle_sync_bfly(amax, o))
        e, d = mx_scale(amax)
        for i in cutlass.range_constexpr(8):
            st_b32(dst + i * C, _Q4_MIX(bw[i][0], bw[i][0], bw[i][1], bw[i][1], d))
        # s_fprop atom: rows (wr * 32 + lane) of the CTA's 128, scale column wc.
        sts_u8(sb1 + lane * 16 + wr * 4 + wc, e)
        # s_dgrad atom: columns (wc * 32 + lane) of the CTA's 128, local row block wr.
        sts_u8(sb0 + lane * 16 + wc * 4 + wr, e)
        cute.arch.sync_threads()
        if tid < 128:
            sv = lds_b32(sb1 + tid * 4)
            st_b32(s1.iterator.toint() + Int64(by) * (128 * (C // 32)) + Int64(bx) * 512 + Int64(tid) * 4, sv)
        if tid >= 128:
            if tid < 256:
                t = tid - 128
                g = (by * 128) // R
                lrb4 = ((by * 128) % R) // 128
                sv0 = lds_b32(sb0 + t * 4)
                st_b32(s0.iterator.toint() + Int64(g) * (C * (R // 32)) + Int64(bx) * (128 * (R // 32))
                       + Int64(lrb4) * 512 + Int64(t) * 4, sv0)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(G, R, C, dev):
    key = (G, R, C, dev)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            Mx2D(G, R, C),
            _fake(cutlass.Uint8, G * R * C * 4),
            _fake(cutlass.Uint8, G * R * C),
            _fake(cutlass.Uint8, G * R * C // 32),
            _fake(cutlass.Uint8, G * R * C // 32),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    return fn


def _prepare(in0, G, R, C):
    assert in0.dtype == torch.float32 and in0.is_contiguous() and in0.data_ptr() % 16 == 0
    assert R % 128 == 0 and C % 128 == 0
    dev = in0.device
    q = torch.empty((G, R, C), dtype=torch.float8_e4m3fn, device=dev)
    s1 = torch.empty((G * R, C // 32), dtype=torch.float8_e8m0fnu, device=dev)
    s0 = torch.empty((G * C, R // 32), dtype=torch.float8_e8m0fnu, device=dev)
    _compiled(G, R, C, dev.index)(
        in0.view(torch.uint8).view(-1),
        q.view(torch.uint8).view(-1),
        s1.view(torch.uint8).view(-1),
        s0.view(torch.uint8).view(-1),
    )
    return (q, s1, s0, torch.empty((0,), dtype=torch.uint8, device=dev))


def kernel_function(in0):
    G = in0.shape[0]
    R = in0.shape[1] * in0.shape[2]
    C = in0.shape[3]
    return _prepare(in0, G, R, C)
