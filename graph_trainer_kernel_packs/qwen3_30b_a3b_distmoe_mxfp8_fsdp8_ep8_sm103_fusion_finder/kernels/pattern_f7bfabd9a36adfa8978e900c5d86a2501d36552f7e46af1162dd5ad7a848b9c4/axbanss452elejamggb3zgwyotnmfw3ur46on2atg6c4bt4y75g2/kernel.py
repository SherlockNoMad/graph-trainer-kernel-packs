"""MXFP8 rowwise quantization + scale swizzle in one CuTe DSL kernel.

reference (problem.py):
  v1 = alias(in0); v2 = in0.reshape(32768, -1)
  q, _, s, _ = torchao.mxfp8_quantize(v2, rowwise=True, colwise=False, 32, 1, 'e4m3', 'rceil')
  return v1, v2, q, torchao.triton_mx_block_rearrange(s)
x = v2: [M, N] bf16 row-major. q: [M, N] e4m3 row-major. s: [M, N/32] e8m0 (one scale per 32 consecutive
elements of a row), rearranged into the cuBLAS block-scaled layout [M, N/32]:
  out[(r // 128) * 128 * (N/32) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4] = s[r, c].

Numerics (torchao _C_mxfp8 mxfp8_quantize_kernel<bf16, e4m3, 1, 32, RCEIL>, read from its SASS/PTX; same math
as the colwise kernel): amax = max.NaN of |x| over the 32 values; v = amax * 0x3B124925 (fp32 1/448, RN);
b = cvt.rp.bf16(v & 0x7fffffff); e = (b >> 7) + round_up, round_up = !((b & 0x7f) == 0 || (b & 0x7fff) < 65)
&& (b & 0x7f80) != 0x7f80; scale byte = e; descale = NaN if e == 255, 2^-127 if e == 254, else
((254 - e) & 0xff) << 23; q = cvt.rn.satfinite.e4m3(x * descale).

CTA (bx, by): rows 128 by .. +127, columns 128 bx .. +127 (4 scale columns -> one 512-byte swizzled block).
Lane l of warp w owns 8 consecutive columns 8 (l % 16) .. +7 of rows 128 by + (128 / warps) w + 2 i + l // 16: one 16-byte load per row; the 4 lanes of a 32-element block combine their amax with XOR
shuffles 1, 2 (max.NaN, order-independent). Scales are staged in shared memory and stored as 4-byte words.
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
def st_v2_b32(addr, w0, w1, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(w0), loc, ip), _v(Int32(w1), loc, ip)],
         "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def fmax_nan(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], "max.NaN.f32 $0, $1, $2;",
                        "=f,f,f", loc=loc, ip=ip))


_Q4_MIX = _make_q4((0, 1, 0, 1))
THREADS = 256  # warps split the CTA's 128 rows
ITERS = 128 // (THREADS // 32) // 2  # 2 rows per warp per iteration


class MxRowwise:
    def __init__(self, M, N):
        self.M = M
        self.N = N

    @cute.jit
    def __call__(self, x, q, s, stream):
        self.kernel(x, q, s).launch(
            grid=[self.N // 128, self.M // 128, 1], block=[THREADS, 1, 1], smem=512, stream=stream
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, q: cute.Tensor, s: cute.Tensor):
        M = self.M
        N = self.N
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        ssc = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(512), 16)
        sbase = Int32(ssc.iterator.toint())
        lane = tid % 32
        w = tid // 32
        cl = (lane % 16) * 8
        rr = lane // 16
        rpw = 128 // (THREADS // 32)
        row0 = by * 128 + w * rpw + rr
        off0 = Int64(row0) * N + Int64(bx * 128 + cl)
        src = x.iterator.toint() + off0 * 2
        dst = q.iterator.toint() + off0
        ws = []
        for i in cutlass.range_constexpr(ITERS):
            ws.append(ld_nc_v4_b32(src + i * 2 * N * 2))
        for i in cutlass.range_constexpr(ITERS):
            v = ws[i]
            amax = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                amax = amax_lo(amax, v[k])
                amax = amax_hi(amax, v[k])
            amax = fmax_nan(amax, cute.arch.shuffle_sync_bfly(amax, 1))
            amax = fmax_nan(amax, cute.arch.shuffle_sync_bfly(amax, 2))
            e, d = mx_scale(amax)
            p0 = _Q4_MIX(v[0], v[0], v[1], v[1], d)
            p1 = _Q4_MIX(v[2], v[2], v[3], v[3], d)
            st_v2_b32(dst + i * 2 * N, p0, p1)
            if lane % 4 == 0:
                lr = w * rpw + i * 2 + rr  # row within the CTA's 128
                sts_u8(sbase + (lr % 32) * 16 + (lr // 32) * 4 + (lane % 16) // 4, e)
        cute.arch.sync_threads()
        if tid < 128:
            sv = lds_b32(sbase + tid * 4)
            st_b32(s.iterator.toint() + Int64(by) * (128 * (N // 32)) + Int64(bx) * 512 + Int64(tid) * 4, sv)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(M, N, dev):
    key = (M, N, dev)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            MxRowwise(M, N),
            _fake(cutlass.Uint8, M * N * 2),
            _fake(cutlass.Uint8, M * N),
            _fake(cutlass.Uint8, M * N // 32),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    return fn


def kernel_function(in0):
    v1 = torch.ops.aten.alias.default(in0)
    v2 = torch.ops.aten.reshape.default(in0, [32768, -1])
    M, N = v2.shape
    assert M % 128 == 0 and N % 128 == 0 and v2.dtype == torch.bfloat16 and v2.is_contiguous()
    assert v2.data_ptr() % 16 == 0
    dev = v2.device
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((M, N // 32), dtype=torch.float8_e8m0fnu, device=dev)
    _compiled(M, N, dev.index)(
        v2.view(torch.uint8).view(-1),
        q.view(torch.uint8).view(-1),
        s.view(torch.uint8).view(-1),
    )
    return (v1, v2, q, s)
