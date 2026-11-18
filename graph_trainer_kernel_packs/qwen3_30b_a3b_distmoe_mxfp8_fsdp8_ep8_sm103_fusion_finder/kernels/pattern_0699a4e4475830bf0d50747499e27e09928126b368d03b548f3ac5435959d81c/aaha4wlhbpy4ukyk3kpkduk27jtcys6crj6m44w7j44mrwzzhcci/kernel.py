"""MXFP8 colwise quantization + scale swizzle in one CuTe DSL kernel.

reference (problem.py):
  _, q, _, s = torchao.mxfp8_quantize(x, rowwise=False, colwise=True, 1, 32, 'e4m3', 'rceil')
  return q, torchao.triton_mx_block_rearrange(s)
x: [M, N] bf16 row-major. q: [M, N] e4m3 column-major (stride (1, M)). s: [N, M/32] e8m0 (one scale per
32-row block of each column), rearranged into the cuBLAS block-scaled layout [N, M/32]:
  out[(c // 128) * 128 * (M/32) + (rb // 4) * 512 + (c % 32) * 16 + ((c % 128) // 32) * 4 + rb % 4] = s[c, rb].

Numerics (torchao _C_mxfp8 mxfp8_quantize_kernel<bf16, e4m3, 32, 1, RCEIL>, read from its sm_100 SASS):
  amax = max.NaN of |x| over the 32 values (NaN propagates);
  v = amax * 0x3B124925 (fp32 1/448, RN, no FTZ); b = cvt.rp.bf16(v & 0x7fffffff);
  e = (b >> 7) + round_up, round_up = !((b & 0x7f) == 0 || (b & 0x7fff) < 65) && (b & 0x7f80) != 0x7f80;
  scale byte = e; descale = NaN if e == 255, 2^-127 (0x00400000) if e == 254, else ((254 - e) & 0xff) << 23;
  q = cvt.rn.satfinite.e4m3(x * descale).

CTA (bx, by): columns 128 bx .. +127, rows 128 by .. +127 (4 row blocks, one per warp). Lane l owns
columns 4 l .. 4 l + 3 of its warp's 32 rows (32 loads of 8 B), writes each column's 32 output bytes
(2 x 16 B) and its 4 scale bytes into the CTA's 512-byte swizzled scale block in shared memory, which the
CTA then stores with one 4-byte store per thread.
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
def ld_nc_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.L1::no_allocate.b32 $0, [$1];", "=r,l",
                      loc=loc, ip=ip))


CPT = 4  # columns per lane
WARPS_C = 128 // (32 * CPT)  # warps across the CTA's 128 columns
THREADS = 128 * WARPS_C
RT = 1  # 128-row tiles per CTA (the next tile's loads are issued before the current tile is processed)


@dsl_user_op
def sts_v4_b32(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip)] + [_v(Int32(x), loc, ip) for x in w],
         "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def lds_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"), [_v(Int32(addr), loc, ip)],
               "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", side=True, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


class MxColwise:
    def __init__(self, M, N):
        self.M = M
        self.N = N

    @cute.jit
    def __call__(self, x, q, s, stream):
        self.kernel(x, q, s).launch(
            grid=[self.N // 128, self.M // (128 * RT), 1], block=[THREADS, 1, 1],
            smem=512 * RT + 128 * 128 * RT + 64, stream=stream
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, q: cute.Tensor, s: cute.Tensor):
        M = self.M
        N = self.N
        P = 128 * RT  # bytes per column of the staged output tile
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sq = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(128 * P), 128)
        ssc = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(512 * RT), 16)
        qbase = Int32(sq.iterator.toint())
        sbase = Int32(ssc.iterator.toint())
        lane = tid % 32
        w = tid // 32
        rbl = w % 4  # row block within a 128-row tile
        cw = w // 4  # column group
        cl0 = cw * (32 * CPT) + lane * CPT  # first column within the CTA's 128
        c0 = bx * 128 + cl0
        xbase = x.iterator.toint()

        def load(t):
            r0 = ((by * RT + t) * 4 + rbl) * 32
            src = xbase + (Int64(r0) * N + Int64(c0)) * 2
            ws = []
            for i in cutlass.range_constexpr(32):
                if cutlass.const_expr(CPT == 2):
                    ws.append([ld_nc_b32(src + i * N * 2)])
                if cutlass.const_expr(CPT == 4):
                    ws.append(ld_nc_v2_b32(src + i * N * 2))
            return ws

        cur = load(0)
        for t in cutlass.range_constexpr(RT):
            nxt = cur
            if cutlass.const_expr(t + 1 < RT):
                nxt = load(t + 1)
            for j in cutlass.range_constexpr(CPT):
                words = [cur[i][j // 2] for i in range(32)]
                amax = Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    if cutlass.const_expr(j % 2 == 0):
                        amax = amax_lo(amax, words[i])
                    if cutlass.const_expr(j % 2 == 1):
                        amax = amax_hi(amax, words[i])
                e, d = mx_scale(amax)
                q4 = _Q4_LO if j % 2 == 0 else _Q4_HI
                packed = []
                for k in cutlass.range_constexpr(8):
                    packed.append(q4(words[4 * k], words[4 * k + 1], words[4 * k + 2], words[4 * k + 3], d))
                cl = cl0 + j
                sw = (cl // CPT) % 8
                ch = t * 8 + rbl * 2
                sts_v4_b32(qbase + cl * P + ((ch ^ sw) * 16), packed[0:4])
                sts_v4_b32(qbase + cl * P + (((ch + 1) ^ sw) * 16), packed[4:8])
                sts_u8(sbase + t * 512 + (cl % 32) * 16 + (cl // 32) * 4 + rbl, e)
            cur = nxt
        cute.arch.sync_threads()
        NCH = 8 * RT  # 16-byte chunks per column
        qg = q.iterator.toint() + Int64(bx * 128) * M + Int64(by * RT * 128)
        for it in cutlass.range_constexpr(128 * NCH // THREADS):
            idx = it * THREADS + tid
            col = idx // NCH
            ch = idx % NCH
            val = lds_v4_b32(qbase + col * P + ((ch ^ ((col // CPT) % 8)) * 16))
            st_v4_b32(qg + Int64(col) * M + Int64(ch * 16), val)
        NW = 512 * RT // 4
        for k in cutlass.range_constexpr((NW + THREADS - 1) // THREADS):
            o = k * THREADS + tid
            if o < NW:
                sv = lds_b32(sbase + o * 4)
                st_b32(s.iterator.toint() + Int64(bx) * (128 * (M // 32)) + Int64(by) * (512 * RT) + Int64(o * 4), sv)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(M, N, dev):
    key = (M, N, dev)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            MxColwise(M, N),
            _fake(cutlass.Uint8, M * N * 2),
            _fake(cutlass.Uint8, M * N),
            _fake(cutlass.Uint8, M * N // 32),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    return fn


def kernel_function(in0):
    M, N = in0.shape
    assert M % (128 * RT) == 0 and N % 128 == 0 and in0.dtype == torch.bfloat16
    x = in0.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    dev = x.device
    q = torch.empty_strided((M, N), (1, M), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((N, M // 32), dtype=torch.float8_e8m0fnu, device=dev)
    _compiled(M, N, dev.index)(
        x.view(torch.uint8).view(-1),
        q.T.view(torch.uint8).view(-1),
        s.view(torch.uint8).view(-1),
    )
    return (q, s)
