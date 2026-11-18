"""ffn_norm backward (DeepSeek-V3 debug model): MoE combine-gradient index_put, two adds, RMSNorm backward
and the residual add, CuTe DSL.

reference (T = 16384, N = 49152 = 3 T routed rows, D = 256):
  v3 = zeros(T, D).index_put_([in1], in0, accumulate=True); v7 = bf16(bf16(in2 + v3) + in3)
  dx, dw = _fused_rms_norm_backward(v7, in4, [256], in5, in6); returns (f32(dw), in7 + dx, (in7 + dx).t())

Three launches (plus a 64 KiB memset of the per-token counters):
  K1, one thread per routed row n: slot = atomicAdd(count[in1[n]], 1); rows[t, slot] = n.
  K2, quack RMSNormBackward's schedule (grid G = min(T, 16 * num_SMs), 4 warps, one row per warp per
  4-row tile, tiles bid, bid + G, ...): per row, sorted index_put accumulation, the adds, dx, the residual
  add; per-warp dw partials combined ((w0 + w1) + w2) + w3 into partial[bid].
  K3, torch/_native/cutedsl/reduce.py reduce_rows(partial, dw, cols=4, threads=256) order, then f32(bf16(.)).

Numerics (checked bit for bit against aten):
  index_put accumulate (indexing_backward_kernel) rounds to bf16 after every duplicate (ascending n),
  starting from +0.
  quack RMSNormBackward (N = 256 bf16 on SM100+: 128 threads, threads_per_row 32, cluster 1, lane i owns
  columns [8i, 8i + 8)): xh = x * rstd, wdy = dy * w; lane dot s = fma(xh, wdy, s) in column order from 0,
  butterfly shfl.xor 16..1; m = s / 256; dx = bf16(fma(-xh, m, wdy) * rstd); dw partial per warp
  acc + dy * xh (not contracted) over its rows in tile order, then ((w0 + w1) + w2) + w3 per CTA;
  reduce_rows: thread sum over rows tidx / 4, + 64, ... from 0, shfl.xor 4, 8, 16, then warp 0 adds warps
  1..7 in order; dw = bf16(acc), returned as f32.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384
NR = 49152
D = 256
MAXK = 4
TOK_PER_CTA = 4
THREADS = 128
RED_COLS = 4
RED_THREADS = 256
# FMA contraction of quack's float ops, found against the reference: lane dot, dx, dw accumulation
FMA_DOT = True
FMA_DX = True
FMA_DW = False


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
def ld_v4(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ptr(ptr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_v4_nc(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ptr(ptr, loc, ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_b32(ptr, *, loc=None, ip=None):
    """Plain (coherent) load: reads what K1 wrote."""
    return Int32(_asm(T.i32(), [_ptr(ptr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side_effects=True))


@dsl_user_op
def ld_b32_nc(ptr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_ptr(ptr, loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc, ip))


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
def atom_inc(ptr, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_ptr(ptr, loc, ip)], "atom.global.add.u32 $0, [$1], 1;", "=r,l", loc, ip, side_effects=True)
    )


@dsl_user_op
def sort4(a, b, c, d, *, loc=None, ip=None):
    text = (
        "{\n.reg .b32 a, b, c, d, t;\n"
        "mov.b32 a, $4;\nmov.b32 b, $5;\nmov.b32 c, $6;\nmov.b32 d, $7;\n"
        "min.s32 t, a, b;\nmax.s32 b, a, b;\nmov.b32 a, t;\n"
        "min.s32 t, c, d;\nmax.s32 d, c, d;\nmov.b32 c, t;\n"
        "min.s32 t, a, c;\nmax.s32 c, a, c;\nmov.b32 a, t;\n"
        "min.s32 t, b, d;\nmax.s32 d, b, d;\nmov.b32 b, t;\n"
        "min.s32 t, b, c;\nmax.s32 c, b, c;\nmov.b32 b, t;\n"
        "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}"
    )
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ir(v, loc, ip) for v in (a, b, c, d)],
        text,
        "=r,=r,=r,=r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def sel_slot(p, k, c, sentinel, *, loc=None, ip=None):
    """p if k < c else sentinel."""
    text = "{\n.reg .pred q;\nsetp.lt.s32 q, $2, $3;\nselp.b32 $0, $1, $4, q;\n}"
    return _i32(text, (p, k, c, sentinel), loc, ip)


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
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def acc_plain(w, x, valid, *, loc=None, ip=None):
    """w (bf16x2) + x (bf16x2), rounded to bf16, if valid; else w."""
    text = (
        "{\n.reg .b32 a0, a1, c0, c1, n;\n.reg .pred q;\n"
        "shl.b32 a0, $2, 16;\nand.b32 a1, $2, 0xFFFF0000;\n"
        "shl.b32 c0, $1, 16;\nand.b32 c1, $1, 0xFFFF0000;\n"
        "add.rn.f32 c0, c0, a0;\nadd.rn.f32 c1, c1, a1;\n"
        "cvt.rn.bf16x2.f32 n, c1, c0;\n"
        "setp.ne.s32 q, $3, 0;\nselp.b32 $0, n, $1, q;\n}"
    )
    return _i32(text, (w, x, valid), loc, ip)


def _mac(dst, a, b, fma):
    return f"fma.rn.f32 {dst}, {a}, {b}, {dst};\n" if fma else f"mul.rn.f32 t, {a}, {b};\nadd.rn.f32 {dst}, {dst}, t;\n"


@dsl_user_op
def dot_word(s, dy, x, g, r, *, loc=None, ip=None):
    """s + (x0 * r) * (dy0 * g0) + (x1 * r) * (dy1 * g1), in that order (f32 bits)."""
    text = "{\n.reg .b32 a, b, c, t, acc;\nmov.b32 acc, $1;\n"
    for half in ("shl.b32 {0}, {1}, 16;\n", "and.b32 {0}, {1}, 0xFFFF0000;\n"):
        text += half.format("a", "$3") + "mul.rn.f32 a, a, $5;\n"
        text += half.format("b", "$2") + half.format("c", "$4") + "mul.rn.f32 b, b, c;\n"
        text += _mac("acc", "a", "b", FMA_DOT)
    text += "mov.b32 $0, acc;\n}"
    return _i32(text, (s, dy, x, g, r), loc, ip)


@dsl_user_op
def dx_word(dy, x, g, r, m, res, *, loc=None, ip=None):
    """bf16(res + bf16((dy * g - (x * r) * m) * r)) per half."""
    text = "{\n.reg .b32 a, b, c, t, o0, o1;\n"
    for k, half in enumerate(("shl.b32 {0}, {1}, 16;\n", "and.b32 {0}, {1}, 0xFFFF0000;\n")):
        o = f"o{k}"
        text += half.format("a", "$2") + "mul.rn.f32 a, a, $4;\n"
        text += half.format("b", "$1") + half.format("c", "$3") + "mul.rn.f32 b, b, c;\n"
        if FMA_DX:
            text += "neg.f32 a, a;\nfma.rn.f32 b, a, $5, b;\n"
        else:
            text += "mul.rn.f32 t, a, $5;\nsub.rn.f32 b, b, t;\n"
        text += f"mul.rn.f32 {o}, b, $4;\n"
    text += "cvt.rn.bf16x2.f32 t, o1, o0;\n"
    text += "shl.b32 a, t, 16;\nand.b32 b, t, 0xFFFF0000;\nshl.b32 c, $6, 16;\nadd.rn.f32 a, c, a;\n"
    text += "and.b32 c, $6, 0xFFFF0000;\nadd.rn.f32 b, c, b;\ncvt.rn.bf16x2.f32 $0, b, a;\n}"
    return _i32(text, (dy, x, g, r, m, res), loc, ip)


@dsl_user_op
def dw_word(a0, a1, dy, x, r, valid, *, loc=None, ip=None):
    """(a0 + dy0 * (x0 * r), a1 + dy1 * (x1 * r)) if valid else (a0, a1)."""
    text = "{\n.reg .b32 a, b, t, c0, c1;\n.reg .pred q;\nmov.b32 c0, $2;\nmov.b32 c1, $3;\n"
    for k, half in enumerate(("shl.b32 {0}, {1}, 16;\n", "and.b32 {0}, {1}, 0xFFFF0000;\n")):
        text += half.format("a", "$5") + "mul.rn.f32 a, a, $6;\n" + half.format("b", "$4")
        text += _mac(f"c{k}", "b", "a", FMA_DW)
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
def div256(s, *, loc=None, ip=None):
    return _i32("div.rn.f32 $0, $1, 0f43800000;", (s,), loc, ip)


@dsl_user_op
def bf16_round_f32(s, *, loc=None, ip=None):
    """f32(bf16(s))."""
    text = "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nmov.b32 $0, {0, h};\n}"
    return _i32(text, (s,), loc, ip)


class BuildRows:
    @cute.jit
    def __call__(self, idx, count, rows, stream):
        self.kernel(idx, count, rows).launch(grid=[NR // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, idx: cute.Tensor, count: cute.Tensor, rows: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        n = bid * THREADS + tid
        t = ld_b32_nc(idx.iterator + n * 2)  # low word of the int64 token index
        slot = atom_inc(count.iterator + t)
        if slot < MAXK:
            st_b32(rows.iterator + (t * MAXK + slot), n)


class NormBwd:
    @cute.jit
    def __call__(self, in0, in2, in3, in4, in5, in6, in7, count, rows, out16, partial, grid: cutlass.Constexpr, stream):
        self.kernel(in0, in2, in3, in4, in5, in6, in7, count, rows, out16, partial, grid).launch(
            grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        in0: cute.Tensor,
        in2: cute.Tensor,
        in3: cute.Tensor,
        in4: cute.Tensor,
        in5: cute.Tensor,
        in6: cute.Tensor,
        in7: cute.Tensor,
        count: cute.Tensor,
        rows: cute.Tensor,
        out16: cute.Tensor,
        partial: cute.Tensor,
        grid: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        sdw = smem.allocate_tensor(Int32, cute.make_layout(4 * D), 16)
        g = ld_v4_nc(in6.iterator + lane * 8)
        big = Int32(2147483647)
        zero = Int32(0)
        dw = [zero, zero, zero, zero, zero, zero, zero, zero]
        ntiles = NT // TOK_PER_CTA
        for it in cutlass.range_constexpr((ntiles + grid - 1) // grid):
            tile = bid + it * grid
            valid_row = sel_slot(Int32(1), tile, Int32(ntiles), zero)
            t = sel_slot(tile, tile, Int32(ntiles), bid) * TOK_PER_CTA + warp  # clamp to a valid row
            col = t * D + lane * 8
            c = ld_b32(count.iterator + t)
            pr = ld_v4(rows.iterator + t * MAXK)
            p = sort4(
                sel_slot(pr[0], Int32(0), c, big),
                sel_slot(pr[1], Int32(1), c, big),
                sel_slot(pr[2], Int32(2), c, big),
                sel_slot(pr[3], Int32(3), c, big),
            )
            xs = []
            for k in cutlass.range_constexpr(MAXK):
                n = sel_slot(p[k], Int32(k), c, p[0])
                xs.append(ld_v4_nc(in0.iterator + (n * D + lane * 8)))
            a2 = ld_v4_nc(in2.iterator + col)
            a3 = ld_v4_nc(in3.iterator + col)
            xx = ld_v4_nc(in4.iterator + col)
            r = ld_b32_nc(in5.iterator + t)
            a7 = ld_v4_nc(in7.iterator + col)
            w = [zero, zero, zero, zero]
            for k in cutlass.range_constexpr(MAXK):
                valid = sel_slot(Int32(1), Int32(k), c, zero)
                w = [acc_plain(w[e], xs[k][e], valid) for e in range(4)]
            dy = [add_word(add_word(a2[e], w[e]), a3[e]) for e in range(4)]
            s = zero
            for e in cutlass.range_constexpr(4):
                s = dot_word(s, dy[e], xx[e], g[e], r)
            for off in cutlass.range_constexpr(5):
                s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
            m = div256(s)
            o = [dx_word(dy[e], xx[e], g[e], r, m, a7[e]) for e in range(4)]
            if valid_row != zero:
                st_v4(out16.iterator + col, o)
            for e in cutlass.range_constexpr(4):
                d0, d1 = dw_word(dw[2 * e], dw[2 * e + 1], dy[e], xx[e], r, valid_row)
                dw[2 * e] = d0
                dw[2 * e + 1] = d1
        # per-CTA combine ((w0 + w1) + w2) + w3, as quack's in-block dw reduction
        if warp > 0:
            for e in cutlass.range_constexpr(8):
                sdw[warp * D + lane * 8 + e] = dw[e]
        cute.arch.sync_threads()
        if warp == 0:
            outv = []
            for e in cutlass.range_constexpr(8):
                v = dw[e]
                for i in cutlass.range_constexpr(1, 4):
                    v = fadd(v, sdw[i * D + lane * 8 + e])
                outv.append(v)
            st_v4(partial.iterator + (bid * D + lane * 8), outv[0:4])
            st_v4(partial.iterator + (bid * D + lane * 8 + 4), outv[4:8])


class ReduceRows:
    @cute.jit
    def __call__(self, partial, out, grid: cutlass.Constexpr, stream):
        self.kernel(partial, out, grid).launch(grid=[D // RED_COLS, 1, 1], block=[RED_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, partial: cute.Tensor, out: cute.Tensor, grid: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        col = bidx * RED_COLS + tidx % RED_COLS
        acc = Int32(0)
        for row in cutlass.range(tidx // RED_COLS, grid, RED_THREADS // RED_COLS):
            acc = fadd(acc, ld_b32(partial.iterator + (row * D + col)))
        for i in cutlass.range_constexpr(3):
            acc = fadd(acc, cute.arch.shuffle_sync_bfly(acc, (1 << i) * RED_COLS))
        smem = cutlass.utils.SmemAllocator()
        sums = smem.allocate_tensor(Int32, cute.make_layout(RED_THREADS // 32 * RED_COLS), 16)
        if lane < RED_COLS:
            sums[warp * RED_COLS + lane] = acc
        cute.arch.sync_threads()
        if tidx < RED_COLS:
            for i in cutlass.range_constexpr(1, RED_THREADS // 32):
                acc = fadd(acc, sums[i * RED_COLS + tidx])
            st_b32(out.iterator + (bidx * RED_COLS + tidx), bf16_round_f32(acc))


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
        k1 = cute.compile(BuildRows(), _fake(Int32, NR * 2, 8), _fake(Int32, NT), _fake(Int32, NT * MAXK), stream, options=opts)
        k2 = cute.compile(
            NormBwd(),
            _fake(bf, NR * D),
            _fake(bf, NT * D),
            _fake(bf, NT * D),
            _fake(bf, NT * D),
            _fake(cutlass.Float32, NT, 4),
            _fake(bf, D),
            _fake(bf, NT * D),
            _fake(Int32, NT),
            _fake(Int32, NT * MAXK),
            _fake(bf, NT * D),
            _fake(Int32, grid * D),
            grid,
            stream,
            options=opts,
        )
        k3 = cute.compile(ReduceRows(), _fake(Int32, grid * D), _fake(Int32, D), grid, stream, options=opts)
        fns = _CACHE[key] = (k1, k2, k3)
    return fns


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7):
    dev = in0.device
    # quack's backward grid: min(M, get_sm_count(N)) = min(M, 16 * SMs) for N <= 256 on SM100+
    grid = min(NT, 16 * torch.cuda.get_device_properties(dev).multi_processor_count)
    k1, k2, k3 = _compiled(dev.index, grid)
    count = torch.zeros(NT, dtype=torch.int32, device=dev)
    rows = torch.empty(NT * MAXK, dtype=torch.int32, device=dev)
    k1(in1.contiguous().view(torch.int32), count, rows)
    out16 = torch.empty((NT, D), dtype=torch.bfloat16, device=dev)
    partial = torch.empty((grid, D), dtype=torch.float32, device=dev)
    k2(
        _aligned(in0).view(-1),
        _aligned(in2).view(-1),
        _aligned(in3).view(-1),
        _aligned(in4).view(-1),
        in5.contiguous().view(-1),
        _aligned(in6).view(-1),
        _aligned(in7).view(-1),
        count,
        rows,
        out16.view(-1),
        partial.view(torch.int32).view(-1),
    )
    dw = torch.empty(D, dtype=torch.float32, device=dev)
    k3(partial.view(torch.int32).view(-1), dw.view(torch.int32))
    return (dw, out16, out16.t())
