"""Attention RoPE backward + q/k RMSNorm backward + qkv-gradient cat (Qwen3 30B-A3B), CuTe DSL; the
MXFP8 quantization stays the torchao op.

reference (T = 32768 tokens, Hq = 32, Hk = 4, D = 128):
  g = f32(grad) (in1: q [T, 32, 128], in0: k [T, 4, 128]); sin = in2, cos = in3 ([T, 1, 128] f32 views)
  dy[c] = bf16(((c < 64 ? 0 + g[c + 64] * sin[c + 64] : -(g[c - 64] * sin[c - 64]) + 0)) + g[c] * cos[c])
  dx, dw = _fused_rms_norm_backward(dy, x, [128], rstd, w)   (q: in8, in7, in9; k: in5, in4, in6)
  v45 = cat([dxq.view(T, 4, 8, 128), dxk.view(T, 4, 1, 128), in10.view(T, 4, 1, 128)], 2).view(T, 5120)
  mxfp8_quantize(v45) and the two scale rearranges (torchao ops, unchanged).

Launches: K1 (one launch, two block roles) writes v45 and the per-CTA dw partials, K2 reduces both
partials, then the torchao ops.
  K1 q role (blocks [0, G)) and k role (blocks [G, 2G)), G = min(rows, 16 * num_SMs): each role follows
  torch's RMSNorm backward schedule (quack RMSNormBackward for N = 128 bf16 on sm_103: 128 threads,
  16 threads per row, 8-row tiles, tiles bid, bid + G, ...; thread t owns row t // 16 of the tile and
  columns [8 (t % 16), +8)). The RoPE partner columns c ^ 64 come from lane ^ 8 (shfl.xor 8).
  The k role also copies in10 into v45.
  K2: torch/_native/cutedsl/reduce.py reduce_rows(partial, dw, cols=4, threads=256) order.

Numerics (each aten op of the reference reproduced with explicit .rn PTX; element pairs use sm_100 f32x2
ops, which round each lane as the scalar op):
  RoPE: mul.rn, neg, add.rn with +0 (slice_backward's zeros; turns -0 into +0), add.rn, cvt.rn.bf16.
  RMSNorm backward (quack): w = bf16(f32(w) + 0) (quack's register pin: -0 -> +0); xh = x * rstd, wdy = dy * w; s = fma(xh, wdy, s) in column
  order from 0; shfl.xor 8, 4, 2, 1; m = s / 128; dx = bf16(fma(-xh, m, wdy) * rstd);
  dw: per thread fma(dy, xh, acc) over its rows in tile order; per CTA rows 0 + 1 + ... + 7
  in order into partial[bid]; reduce_rows; dw = bf16(acc).
"""

import cutlass
import cutlass.cute as cute
import torch
import torchao.prototype.mx_formats.kernels  # noqa: F401  (registers torch.ops.torchao)
from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 32768
HQ = 32
HK = 4
D = 128
ROW = 5120  # v45 row: 4 groups x (8 q heads, 1 k head, 1 v head) x 128
GROUP = 1280
FST = 256  # row stride (f32 elements) of the sin / cos views
THREADS = 128
TPR = 16  # threads per row
TILE = THREADS // TPR  # rows per tile
RED_COLS = 4
RED_THREADS = 256
NQT = NT * HQ // TILE  # q tiles
NKT = NT * HK // TILE  # k tiles


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


def _ir64(v, loc, ip):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


def _i32(text, args, loc, ip):
    return Int32(_asm(T.i32(), [_ir(v, loc, ip) for v in args], text, "=r" + ",r" * len(args), loc, ip))


@dsl_user_op
def ld_v4_nc(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ir64(addr, loc, ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_v4_stream(addr, *, loc=None, ip=None):
    """Read-once data: do not keep it in L1."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_ir64(addr, loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_b32_nc(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_ir64(addr, loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    """Coherent load: reads what an earlier kernel wrote."""
    return Int32(_asm(T.i32(), [_ir64(addr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side_effects=True))


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_ir64(addr, loc, ip)] + [_ir(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def st_b16_bf16(addr, v, *, loc=None, ip=None):
    """*addr = bf16(v) (v: f32 bits)."""
    _asm(
        None,
        [_ir64(addr, loc, ip), _ir(v, loc, ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def w_plus0(w, *, loc=None, ip=None):
    """bf16x2(f32(w) + 0.0) per half: quack pins the weight with this round trip, so -0 becomes +0."""
    text = (
        "{\n.reg .b32 a, b;\nshl.b32 a, $1, 16;\nand.b32 b, $1, 0xFFFF0000;\n"
        "add.rn.f32 a, a, 0f00000000;\nadd.rn.f32 b, b, 0f00000000;\ncvt.rn.bf16x2.f32 $0, b, a;\n}"
    )
    return _i32(text, (w,), loc, ip)


@dsl_user_op
def rope_word2(gp, go, s0, s1, c0, c1, neg, *, loc=None, ip=None):
    """bf16x2 of ((neg ? -(gp * s) : gp * s) + 0) + go * c per half (gp, go: bf16x2; s, c: f32), f32x2 ops."""
    text = (
        "{\n.reg .b32 a0, a1, b0, b1;\n.reg .b64 A, B, S, C, Z;\n.reg .pred q;\nsetp.ne.s32 q, $7, 0;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\nshl.b32 b0, $2, 16;\nand.b32 b1, $2, 0xFFFF0000;\n"
        "mov.b64 A, {a0, a1};\nmov.b64 B, {b0, b1};\nmov.b64 S, {$3, $4};\nmov.b64 C, {$5, $6};\n"
        "mov.b64 Z, 0;\n"
        "mul.rn.f32x2 A, A, S;\n@q xor.b64 A, A, 0x8000000080000000;\nadd.rn.f32x2 A, A, Z;\n"
        "mul.rn.f32x2 B, B, C;\nadd.rn.f32x2 A, A, B;\nmov.b64 {a0, a1}, A;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return _i32(text, (gp, go, s0, s1, c0, c1, neg), loc, ip)


@dsl_user_op
def dot_dw_word(s, a0, a1, dy, x, g, r, *, loc=None, ip=None):
    """s = fma(xh1, wdy1, fma(xh0, wdy0, s)); (a0, a1) = fma(dy, xh, (a0, a1)); xh = x * r, wdy = dy * g."""
    text = (
        "{\n.reg .b32 x0, x1, d0, d1, g0, g1, h0, h1, w0, w1, acc;\n.reg .b64 X, R, DY, G, XH, WDY, AC;\n"
        "shl.b32 x0, $6, 16;\nand.b32 x1, $6, 0xFFFF0000;\nshl.b32 d0, $5, 16;\nand.b32 d1, $5, 0xFFFF0000;\n"
        "shl.b32 g0, $7, 16;\nand.b32 g1, $7, 0xFFFF0000;\n"
        "mov.b64 X, {x0, x1};\nmov.b64 R, {$8, $8};\nmov.b64 DY, {d0, d1};\nmov.b64 G, {g0, g1};\n"
        "mul.rn.f32x2 XH, X, R;\nmul.rn.f32x2 WDY, DY, G;\n"
        "mov.b64 {h0, h1}, XH;\nmov.b64 {w0, w1}, WDY;\n"
        "mov.b32 acc, $3;\nfma.rn.f32 acc, h0, w0, acc;\nfma.rn.f32 acc, h1, w1, acc;\nmov.b32 $0, acc;\n"
        "mov.b64 AC, {$4, $9};\nfma.rn.f32x2 AC, DY, XH, AC;\nmov.b64 {$1, $2}, AC;\n}"
    )
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32)>"),
        [_ir(v, loc, ip) for v in (s, a0, dy, x, g, r, a1)],
        text,
        "=r,=r,=r,r,r,r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 3, loc, ip)


@dsl_user_op
def dx_word2(dy, x, g, r, m, *, loc=None, ip=None):
    """bf16(fma(xh, -m, wdy) * r) per half (the same bits as fma(-xh, m, wdy)), f32x2 ops."""
    text = (
        "{\n.reg .b32 x0, x1, d0, d1, g0, g1, nm, o0, o1;\n.reg .b64 X, R, DY, G, M, T;\n"
        "shl.b32 x0, $2, 16;\nand.b32 x1, $2, 0xFFFF0000;\nshl.b32 d0, $1, 16;\nand.b32 d1, $1, 0xFFFF0000;\n"
        "shl.b32 g0, $3, 16;\nand.b32 g1, $3, 0xFFFF0000;\nxor.b32 nm, $5, 0x80000000;\n"
        "mov.b64 X, {x0, x1};\nmov.b64 R, {$4, $4};\nmov.b64 DY, {d0, d1};\nmov.b64 G, {g0, g1};\nmov.b64 M, {nm, nm};\n"
        "mul.rn.f32x2 X, X, R;\nmul.rn.f32x2 DY, DY, G;\nfma.rn.f32x2 T, X, M, DY;\nmul.rn.f32x2 T, T, R;\n"
        "mov.b64 {o0, o1}, T;\ncvt.rn.bf16x2.f32 $0, o1, o0;\n}"
    )
    return _i32(text, (dy, x, g, r, m), loc, ip)


@dsl_user_op
def div128(s, *, loc=None, ip=None):
    return _i32("div.rn.f32 $0, $1, 0f43000000;", (s,), loc, ip)


class NormRope:
    @cute.jit
    def __call__(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, out, pq, pk, grid: cutlass.Constexpr, stream):
        self.kernel(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, out, pq, pk, grid).launch(
            grid=[2 * grid, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        in0: cute.Tensor,
        in1: cute.Tensor,
        in2: cute.Tensor,
        in3: cute.Tensor,
        in4: cute.Tensor,
        in5: cute.Tensor,
        in6: cute.Tensor,
        in7: cute.Tensor,
        in8: cute.Tensor,
        in9: cute.Tensor,
        in10: cute.Tensor,
        out: cute.Tensor,
        pq: cute.Tensor,
        pk: cute.Tensor,
        grid: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        r_in_tile = tid // TPR
        lane16 = tid % TPR
        col = lane16 * 8
        is_k = bid >= grid
        zero = Int32(0)
        # role parameters (q: blocks [0, G), k: blocks [G, 2G))
        b = bid
        ntiles = Int32(NQT)
        shift = Int32(5)
        hmask = Int32(HQ - 1)
        hs = Int32(3)
        hm = Int32(7)
        koff = Int32(0)
        a_g = in1.iterator.toint()
        a_x = in8.iterator.toint()
        a_r = in7.iterator.toint()
        a_w = in9.iterator.toint()
        a_p = pq.iterator.toint()
        if is_k:
            b = bid - grid
            ntiles = Int32(NKT)
            shift = Int32(2)
            hmask = Int32(HK - 1)
            hs = Int32(0)
            hm = Int32(0)
            koff = Int32(8 * D)
            a_g = in0.iterator.toint()
            a_x = in5.iterator.toint()
            a_r = in4.iterator.toint()
            a_w = in6.iterator.toint()
            a_p = pk.iterator.toint()
        a_sin = in2.iterator.toint()
        a_cos = in3.iterator.toint()
        a_v = in10.iterator.toint()
        a_out = out.iterator.toint()
        neg = Int32(0)
        if col >= 64:
            neg = Int32(1)
        w = [w_plus0(v) for v in ld_v4_nc(a_w + Int64(col * 2))]
        acc = cute.make_rmem_tensor(cute.make_layout(8), Int32)
        for e in cutlass.range_constexpr(8):
            acc[e] = zero
        for tile in cutlass.range(b, ntiles, grid):
            row = tile * TILE + r_in_tile
            t = row >> shift
            h = row & hmask
            go = ld_v4_stream(a_g + Int64(row) * (D * 2) + Int64(col * 2))
            xx = ld_v4_stream(a_x + Int64(row) * (D * 2) + Int64(col * 2))
            r = ld_b32_nc(a_r + Int64(row) * 4)
            fo = Int64(t) * (FST * 4)
            cs0 = ld_v4_nc(a_cos + fo + Int64(col * 4))
            cs1 = ld_v4_nc(a_cos + fo + Int64(col * 4 + 16))
            sn0 = ld_v4_nc(a_sin + fo + Int64((col ^ 64) * 4))
            sn1 = ld_v4_nc(a_sin + fo + Int64((col ^ 64) * 4 + 16))
            cs = cs0 + cs1
            sn = sn0 + sn1
            dy = []
            for e in cutlass.range_constexpr(4):
                gp = cute.arch.shuffle_sync_bfly(go[e], 8)
                dy.append(rope_word2(gp, go[e], sn[2 * e], sn[2 * e + 1], cs[2 * e], cs[2 * e + 1], neg))
            s = zero
            for e in cutlass.range_constexpr(4):
                s, d0, d1 = dot_dw_word(s, acc[2 * e], acc[2 * e + 1], dy[e], xx[e], w[e], r)
                acc[2 * e] = d0
                acc[2 * e + 1] = d1
            for off in cutlass.range_constexpr(4):
                s = fadd(s, cute.arch.shuffle_sync_bfly(s, 8 >> off))
            m = div128(s)
            o = [dx_word2(dy[e], xx[e], w[e], r, m) for e in range(4)]
            oo = Int64(t) * (ROW * 2) + Int64(((h >> hs) * GROUP + (h & hm) * D + koff + col) * 2)
            st_v4(a_out + oo, o)
            if is_k:
                vv = ld_v4_stream(a_v + Int64(row) * (D * 2) + Int64(col * 2))
                st_v4(a_out + oo + Int64(D * 2), vv)
        # per-CTA combine: row 0 + row 1 + ... + row 7, as quack's in-block dw reduction
        smem = cutlass.utils.SmemAllocator()
        sdw = smem.allocate_tensor(Int32, cute.make_layout(TILE * D), 16)
        if r_in_tile > 0:
            for e in cutlass.range_constexpr(8):
                sdw[r_in_tile * D + col + e] = acc[e]
        cute.arch.sync_threads()
        if r_in_tile == 0:
            outv = []
            for e in cutlass.range_constexpr(8):
                v = acc[e]
                for i in cutlass.range_constexpr(1, TILE):
                    v = fadd(v, sdw[i * D + col + e])
                outv.append(v)
            pa = a_p + Int64(b) * (D * 4) + Int64(col * 4)
            st_v4(pa, outv[0:4])
            st_v4(pa + Int64(16), outv[4:8])


class ReduceRows:
    @cute.jit
    def __call__(self, pq, pk, dq, dk, grid: cutlass.Constexpr, stream):
        self.kernel(pq, pk, dq, dk, grid).launch(grid=[2 * D // RED_COLS, 1, 1], block=[RED_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pq: cute.Tensor, pk: cute.Tensor, dq: cute.Tensor, dk: cute.Tensor, grid: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        b = bidx
        a_p = pq.iterator.toint()
        a_o = dq.iterator.toint()
        if bidx >= D // RED_COLS:
            b = bidx - D // RED_COLS
            a_p = pk.iterator.toint()
            a_o = dk.iterator.toint()
        col = b * RED_COLS + tidx % RED_COLS
        acc = Int32(0)
        for row in cutlass.range(tidx // RED_COLS, grid, RED_THREADS // RED_COLS):
            acc = fadd(acc, ld_b32(a_p + Int64(row * D + col) * 4))
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
            st_b16_bf16(a_o + Int64(col * 2), acc)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev, grid):
    key = (dev, grid)
    fns = _CACHE.get(key)
    if fns is None:
        bf = cutlass.BFloat16
        f32 = cutlass.Float32
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = "--enable-tvm-ffi"
        k1 = cute.compile(
            NormRope(),
            _fake(bf, NT * HK * D),
            _fake(bf, NT * HQ * D),
            _fake(f32, 1),
            _fake(f32, 1),
            _fake(f32, NT * HK, 4),
            _fake(bf, NT * HK * D),
            _fake(bf, D),
            _fake(f32, NT * HQ, 4),
            _fake(bf, NT * HQ * D),
            _fake(bf, D),
            _fake(bf, NT * HK * D),
            _fake(bf, NT * ROW),
            _fake(Int32, grid * D),
            _fake(Int32, grid * D),
            grid,
            stream,
            options=opts,
        )
        k2 = cute.compile(
            ReduceRows(), _fake(Int32, grid * D), _fake(Int32, grid * D), _fake(bf, D, 4), _fake(bf, D, 4), grid, stream, options=opts
        )
        fns = _CACHE[key] = (k1, k2)
    return fns


def _check(x, shape, stride):
    assert tuple(x.shape) == shape and tuple(x.stride()) == stride and x.data_ptr() % 16 == 0, (x.shape, x.stride())


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
    dev = in0.device
    _check(in0, (NT, HK, D), (HK * D, D, 1))
    _check(in1, (NT, HQ, D), (HQ * D, D, 1))
    _check(in2, (NT, 1, D), (FST, FST, 1))
    _check(in3, (NT, 1, D), (FST, FST, 1))
    _check(in5, (NT, HK, D), (HK * D, D, 1))
    _check(in8, (NT, HQ, D), (HQ * D, D, 1))
    _check(in10, (NT, HK, D), (HK * D, D, 1))
    assert in4.is_contiguous() and in7.is_contiguous() and in6.is_contiguous() and in9.is_contiguous()
    # torch's RMSNorm backward grid on this GPU: min(M, quack get_sm_count(128)) = min(M, 16 * SMs)
    # (sm_103 is not on rmsnorm_launch's tuned path); M >= 131072 here, so both roles use 16 * SMs.
    grid = 16 * torch.cuda.get_device_properties(dev).multi_processor_count
    k1, k2 = _compiled(dev.index, grid)
    v45 = torch.empty((NT, ROW), dtype=torch.bfloat16, device=dev)
    pq = torch.empty((grid, D), dtype=torch.float32, device=dev)
    pk = torch.empty((grid, D), dtype=torch.float32, device=dev)
    k1(
        in0.view(-1),
        in1.view(-1),
        in2.as_strided((1,), (1,)),
        in3.as_strided((1,), (1,)),
        in4.view(-1),
        in5.view(-1),
        in6.view(-1),
        in7.view(-1),
        in8.view(-1),
        in9.view(-1),
        in10.view(-1),
        v45.view(-1),
        pq.view(torch.int32).view(-1),
        pk.view(torch.int32).view(-1),
    )
    dwq = torch.empty(D, dtype=torch.bfloat16, device=dev)
    dwk = torch.empty(D, dtype=torch.bfloat16, device=dev)
    k2(pq.view(torch.int32).view(-1), pk.view(torch.int32).view(-1), dwq, dwk)
    v46 = torch.ops.torchao.mxfp8_quantize.default(v45, True, True, 32, 32, "e4m3", "rceil")
    v51 = torch.ops.torchao.triton_mx_block_rearrange.default(v46[2])
    v52 = torch.ops.torchao.triton_mx_block_rearrange.default(v46[3])
    return (dwk, dwq, v46[0], v51, v52, v46[1].t())
