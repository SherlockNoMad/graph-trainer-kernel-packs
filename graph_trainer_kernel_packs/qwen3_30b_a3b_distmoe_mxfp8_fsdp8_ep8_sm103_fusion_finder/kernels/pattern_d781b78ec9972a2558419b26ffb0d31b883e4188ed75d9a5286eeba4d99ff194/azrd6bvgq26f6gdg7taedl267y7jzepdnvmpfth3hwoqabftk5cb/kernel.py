"""attention_norm backward (Qwen3 30B-A3B, hidden 2048): RMSNorm backward + residual add, CuTe DSL.

reference (M = 32768 rows, N = 2048):
  dx, dw = _fused_rms_norm_backward(dy=in1, x=in2, [2048], rstd=in0, w=in3, [True, True])
  returns (dw, in4 + dx)

Two launches (aten: quack RMSNormBackward + reduce_rows + the add kernel):
  K1, quack RMSNormBackward's schedule for N = 2048 bf16 on sm_10x (RmsNormBwdConfig: 128 threads,
  threads_per_row 128, cluster 1; grid G = min(M, get_sm_count(N)) = min(M, 4 * num_SMs) since sm_103 is
  not on the tuned launch path): CTA bid handles rows bid, bid + G, ... in order; thread t owns columns
  [8t, 8t + 8) and [1024 + 8t, 1024 + 8t + 8). Per row: dx, then out = bf16(in4 + bf16(dx)) written
  directly (no dx round trip). Thread 0 streams the CTA's rows of dy, x and in4 into a STAGES-deep shared
  memory ring with 4 KiB cp.async.bulk copies (mbarrier complete_tx); a stage is refilled right after the
  row's block barrier, when every thread has read it.
  K2, torch/_native/cutedsl/reduce.py reduce_rows(partial, dw, cols=8, threads=256) order.

Numerics (quack RMSNormBackward / reduce_rows, matched bit for bit):
  w is pinned first as bf16(f32(w) + 0.0) (quack's no-op round-trip: -0 -> +0);
  xh = x * rstd, wdy = dy * w; thread dot s = fma(xh, wdy, s) over its 16 columns in column order from 0;
  warp butterfly shfl.xor 16..1; block_reduce: lane 0 of warp k stores s_k, lanes 0..3 reload s_lane,
  other lanes 0.0, butterfly 16..1 again; m = s / 2048; dx = bf16(fma(-xh, m, wdy) * rstd).
  dw partial per thread: acc = fma(dy, xh, acc) (contracted at N = 2048, unlike N = 256; checked bit for bit
  against quack's partial buffer) over the CTA's rows in order, written per CTA (tiler rows = 1, no in-CTA
  combine); reduce_rows: thread sums rows tidx / 8, + 32, ... from 0,
  shfl.xor 8, 16, warp 0 adds warps 1..7 in order; dw = bf16(acc).
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
THREADS = 128
RED_COLS = 8
STAGES = 3  # smem ring depth (rows in flight per CTA)
NIN = 3  # bulk-copied row inputs: dy, x, res
RED_THREADS = 256


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
def ld_b32_nc(ptr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_ptr(ptr, loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc, ip, side_effects=True))


@dsl_user_op
def ld_b32(ptr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_ptr(ptr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side_effects=True))


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
def st_b16_of_f32(ptr, v, *, loc=None, ip=None):
    """*ptr = bf16(v)."""
    _asm(
        None,
        [_ptr(ptr, loc, ip), _ir(v, loc, ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def sel(p, a, b, *, loc=None, ip=None):
    """a if p != 0 else b."""
    return _i32("{\n.reg .pred q;\nsetp.ne.s32 q, $1, 0;\nselp.b32 $0, $2, $3, q;\n}", (p, a, b), loc, ip)


@dsl_user_op
def sel_lt(a, b, x, y, *, loc=None, ip=None):
    """x if a < b else y."""
    return _i32("{\n.reg .pred q;\nsetp.lt.s32 q, $1, $2;\nselp.b32 $0, $3, $4, q;\n}", (a, b, x, y), loc, ip)


@dsl_user_op
def pin_word(w, *, loc=None, ip=None):
    """bf16x2(f32(w) + 0.0) per half: quack's no-op fp32 round-trip of the weight (turns -0 into +0)."""
    text = (
        "{\n.reg .b32 a0, a1;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
        "add.rn.f32 a0, a0, 0f00000000;\nadd.rn.f32 a1, a1, 0f00000000;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return _i32(text, (w,), loc, ip)


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def fmul(a, b, *, loc=None, ip=None):
    return _i32("mul.rn.f32 $0, $1, $2;", (a, b), loc, ip)


_LO = "shl.b32 {0}, {1}, 16;\n"
_HI = "and.b32 {0}, {1}, 0xFFFF0000;\n"


@dsl_user_op
def dot_word(s, dy, x, g, r, *, loc=None, ip=None):
    """s = fma(x1 * r, dy1 * g1, fma(x0 * r, dy0 * g0, s)) (f32 bits)."""
    text = "{\n.reg .b32 a, b, c, acc;\nmov.b32 acc, $1;\n"
    for half in (_LO, _HI):
        text += half.format("a", "$3") + "mul.rn.f32 a, a, $5;\n"
        text += half.format("b", "$2") + half.format("c", "$4") + "mul.rn.f32 b, b, c;\n"
        text += "fma.rn.f32 acc, a, b, acc;\n"
    text += "mov.b32 $0, acc;\n}"
    return _i32(text, (s, dy, x, g, r), loc, ip)


@dsl_user_op
def dx_word(dy, x, g, r, m, res, *, loc=None, ip=None):
    """bf16(res + bf16(fma(-(x * r), m, dy * g) * r)) per half."""
    text = "{\n.reg .b32 a, b, c, t, o0, o1;\n"
    for k, half in enumerate((_LO, _HI)):
        text += half.format("a", "$2") + "mul.rn.f32 a, a, $4;\n"
        text += half.format("b", "$1") + half.format("c", "$3") + "mul.rn.f32 b, b, c;\n"
        text += "neg.f32 a, a;\nfma.rn.f32 b, a, $5, b;\n"
        text += f"mul.rn.f32 o{k}, b, $4;\n"
    text += "cvt.rn.bf16x2.f32 t, o1, o0;\n"
    text += _LO.format("a", "t") + _HI.format("b", "t")
    text += _LO.format("c", "$6") + "add.rn.f32 a, c, a;\n"
    text += _HI.format("c", "$6") + "add.rn.f32 b, c, b;\ncvt.rn.bf16x2.f32 $0, b, a;\n}"
    return _i32(text, (dy, x, g, r, m, res), loc, ip)


@dsl_user_op
def dw_word(a0, a1, dy, x, r, valid, *, loc=None, ip=None):
    """(fma(dy0, x0 * r, a0), fma(dy1, x1 * r, a1)) if valid else (a0, a1)."""
    text = "{\n.reg .b32 a, b, t, c0, c1;\n.reg .pred q;\nmov.b32 c0, $2;\nmov.b32 c1, $3;\n"
    for k, half in enumerate((_LO, _HI)):
        text += half.format("a", "$5") + "mul.rn.f32 a, a, $6;\n" + half.format("b", "$4")
        text += f"fma.rn.f32 c{k}, b, a, c{k};\n"
    text += "setp.ne.s32 q, $7, 0;\nselp.b32 $0, c0, $2, q;\nselp.b32 $1, c1, $3, q;\n}"
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_ir(v, loc, ip) for v in (a0, a1, dy, x, r, valid)],
        text,
        "=r,=r,r,r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, loc, ip)



@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ir(addr, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_ir(addr, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side_effects=True))


@dsl_user_op
def sts_b32(addr, v, *, loc=None, ip=None):
    _asm(None, [_ir(addr, loc, ip), _ir(v, loc, ip)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side_effects=True)


@dsl_user_op
def expect_tx(mbar, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_ir(mbar, loc, ip), _ir(nbytes, loc, ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def bulk_g2s(dst, src, mbar, *, loc=None, ip=None):
    """cp.async.bulk of one 4 KiB row (src: gmem pointer, dst / mbar: smem addresses)."""
    _asm(
        None,
        [_ir(dst, loc, ip), _ptr(src, loc, ip), _ir(mbar, loc, ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], 4096, [$2];",
        "r,l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def fence_proxy_async(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip, side_effects=True)


class NormBwd:
    @cute.jit
    def __call__(self, rstd, dy, x, w, res, out, partial, grid: cutlass.Constexpr, stream):
        self.kernel(rstd, dy, x, w, res, out, partial, grid).launch(
            grid=[grid, 1, 1], block=[THREADS, 1, 1], min_blocks_per_mp=4, stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        rstd: cute.Tensor,
        dy: cute.Tensor,
        x: cute.Tensor,
        w: cute.Tensor,
        res: cute.Tensor,
        out: cute.Tensor,
        partial: cute.Tensor,
        grid: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        sdata = smem.allocate_tensor(Int32, cute.make_layout(STAGES * NIN * 1024), 128)
        red = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        nfull = M // grid
        nrows_max = (M + grid - 1) // grid
        nlast = M - nfull * grid  # CTAs bid < nlast have nfull + 1 rows
        rst = smem.allocate_tensor(Int32, cute.make_layout(nrows_max), 16)
        mbar = smem.allocate_array(cutlass.Int64, num_elems=STAGES)
        sd = Int32(sdata.iterator.toint())
        mb = Int32(mbar.toint())
        zero = Int32(0)
        nrows = sel_lt(bid, Int32(nlast), Int32(nfull + 1), Int32(nfull))
        if tid == 0:
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(mbar + s, 1)
            cute.arch.mbarrier_init_fence()
        if tid < nrows:
            rst[tid] = ld_b32_nc(rstd.iterator + (bid + tid * grid))
        c0 = tid * 8
        c1 = tid * 8 + N // 2
        g = ld_v4_nc(w.iterator + c0) + ld_v4_nc(w.iterator + c1)
        g = [pin_word(v) for v in g]  # quack: w = bf16(f32(w) + 0.0) before wdy = dy * w
        cute.arch.sync_threads()
        if tid == 0:
            for s in cutlass.range_constexpr(STAGES):
                if s < nrows:
                    row = bid + s * grid
                    expect_tx(mb + s * 8, Int32(NIN * 4096))
                    bulk_g2s(sd + (s * NIN + 0) * 4096, dy.iterator + row * N, mb + s * 8)
                    bulk_g2s(sd + (s * NIN + 1) * 4096, x.iterator + row * N, mb + s * 8)
                    bulk_g2s(sd + (s * NIN + 2) * 4096, res.iterator + row * N, mb + s * 8)
        dwf = cute.make_rmem_tensor(cute.make_layout(16), Int32)
        for e in cutlass.range_constexpr(16):
            dwf[e] = zero
        for j in cutlass.range(nrows):
            st = j % STAGES
            cute.arch.mbarrier_wait(mbar + st, (j // STAGES) % 2)
            sb = sd + st * (NIN * 4096) + tid * 16
            d = lds_v4(sb) + lds_v4(sb + 2048)
            xx = lds_v4(sb + 4096) + lds_v4(sb + 4096 + 2048)
            rr = lds_v4(sb + 8192) + lds_v4(sb + 8192 + 2048)
            r = rst[j]
            row = bid + j * grid
            s = zero
            for e in cutlass.range_constexpr(8):
                s = dot_word(s, d[e], xx[e], g[e], r)
            for off in cutlass.range_constexpr(5):
                s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
            slot = (j % 2) * 4
            if lane == 0:
                red[slot + warp] = s
            cute.arch.sync_threads()
            if tid == 0:
                jn = j + STAGES
                if jn < nrows:
                    fence_proxy_async()
                    rown = bid + jn * grid
                    sbn = sd + st * (NIN * 4096)
                    expect_tx(mb + st * 8, Int32(NIN * 4096))
                    bulk_g2s(sbn, dy.iterator + rown * N, mb + st * 8)
                    bulk_g2s(sbn + 4096, x.iterator + rown * N, mb + st * 8)
                    bulk_g2s(sbn + 8192, res.iterator + rown * N, mb + st * 8)
            v = zero
            if lane < 4:
                v = red[slot + lane]
            for off in cutlass.range_constexpr(5):
                v = fadd(v, cute.arch.shuffle_sync_bfly(v, 16 >> off))
            m = fmul(v, Int32(0x3A000000))  # / 2048, exact
            o = [dx_word(d[e], xx[e], g[e], r, m, rr[e]) for e in range(8)]
            base = row * N
            st_v4(out.iterator + (base + c0), o[0:4])
            st_v4(out.iterator + (base + c1), o[4:8])
            for e in cutlass.range_constexpr(8):
                d0, d1 = dw_word(dwf[2 * e], dwf[2 * e + 1], d[e], xx[e], r, Int32(1))
                dwf[2 * e] = d0
                dwf[2 * e + 1] = d1
        dw = [dwf[e] for e in range(16)]
        pbase = bid * N
        st_v4(partial.iterator + (pbase + c0), dw[0:4])
        st_v4(partial.iterator + (pbase + c0 + 4), dw[4:8])
        st_v4(partial.iterator + (pbase + c1), dw[8:12])
        st_v4(partial.iterator + (pbase + c1 + 4), dw[12:16])


class ReduceRows:
    @cute.jit
    def __call__(self, partial, out, grid: cutlass.Constexpr, stream):
        self.kernel(partial, out, grid).launch(grid=[N // RED_COLS, 1, 1], block=[RED_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, partial: cute.Tensor, out: cute.Tensor, grid: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        col = bidx * RED_COLS + tidx % RED_COLS
        acc = Int32(0)
        r0 = tidx // RED_COLS
        step = RED_THREADS // RED_COLS
        vals = []
        for k in cutlass.range_constexpr((grid + step - 1) // step):
            if cutlass.const_expr((k + 1) * step <= grid):
                vals.append(ld_b32_nc(partial.iterator + ((r0 + k * step) * N + col)))
            else:  # last, partial round of rows: rows past grid add nothing
                ok = sel_lt(r0 + k * step, Int32(grid), Int32(1), Int32(0))
                rk = sel_lt(r0 + k * step, Int32(grid), r0 + k * step, r0)
                vals.append(sel(ok, ld_b32_nc(partial.iterator + (rk * N + col)), Int32(-2147483648)))
        for k in cutlass.range_constexpr(len(vals)):
            acc = fadd(acc, vals[k])  # -0.0 for missing rows: acc + -0.0 == acc
        for i in cutlass.range_constexpr(2):
            acc = fadd(acc, cute.arch.shuffle_sync_bfly(acc, (1 << i) * RED_COLS))
        smem = cutlass.utils.SmemAllocator()
        sums = smem.allocate_tensor(Int32, cute.make_layout(RED_THREADS // 32 * RED_COLS), 16)
        if lane < RED_COLS:
            sums[warp * RED_COLS + lane] = acc
        cute.arch.sync_threads()
        if tidx < RED_COLS:
            for i in cutlass.range_constexpr(1, RED_THREADS // 32):
                acc = fadd(acc, sums[i * RED_COLS + tidx])
            st_b16_of_f32(out.iterator + (bidx * RED_COLS + tidx), acc)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev, grid):
    key = (dev, grid)
    fns = _CACHE.get(key)
    if fns is None:
        bf = cutlass.BFloat16
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = "--enable-tvm-ffi"
        k1 = cute.compile(
            NormBwd(),
            _fake(Int32, M, 4),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(bf, N),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(Int32, grid * N),
            grid,
            stream,
            options=opts,
        )
        k2 = cute.compile(ReduceRows(), _fake(Int32, grid * N), _fake(bf, N, 2), grid, stream, options=opts)
        fns = _CACHE[key] = (k1, k2)
    return fns


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2, in3, in4):
    dev = in1.device
    # quack's backward grid: min(M, get_sm_count(N)) = min(M, 4 * SMs) for 1024 < N <= 2048 on sm_10x
    grid = min(M, 4 * torch.cuda.get_device_properties(dev).multi_processor_count)
    k1, k2 = _compiled(dev.index, grid)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    partial = torch.empty((grid, N), dtype=torch.float32, device=dev)
    k1(
        in0.contiguous().view(torch.int32).view(-1),
        _aligned(in1).view(-1),
        _aligned(in2).view(-1),
        _aligned(in3).view(-1),
        _aligned(in4).view(-1),
        out.view(-1),
        partial.view(torch.int32).view(-1),
    )
    dw = torch.empty(N, dtype=torch.bfloat16, device=dev)
    k2(partial.view(torch.int32).view(-1), dw)
    return (dw, out)
