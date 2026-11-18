"""MoE router + aux loss + token sort of the DeepSeek-V3 debug model (8 experts, top-3, 16384 tokens), two CuTe DSL kernels.

reference (problem.py) per token row r of in0 [16384, 8] fp32:
  s = sigmoid(in0[r]); v5[r] = topk(s + in1, 3, sorted=False) indices; w = s[v5] / (sum + 1e-20)
  v15[r] = onehot(v5[r]) & ~in2[r]; in3 += sum_r v15; aux loss (v28 sum over rows of s / L1(s) * ~in4) -> in6
  stable sort of the 49152 expert ids, v52 = in7 rows in sorted order, v54 = w in sorted order,
  v56 = token of each sorted slot, v43 = per-expert counts, v53 = cumsum(v43).

Kernel 1 (16 CTAs x 1024 threads, one row per thread): sigmoid, top-k, weights, v5, v15, a per-token record
(3 weights + packed expert ids), per-64-row-chunk expert histograms, per-CTA masked counts, and the per-CTA
partial of v28. CTA c owns rows 1024k + 64c + y (k < 16, y < 64): exactly the rows aten's dim-0 sum kernel
gives its CTA c, so the partial is reduced in aten's order (4 accumulators over k, then a tree over y).
Kernel 2 (128 + 1 CTAs x 256 threads): each of the first 128 CTAs owns 128 consecutive tokens; it computes the
stable-sort position of its 384 (token, slot) pairs (expert offset + histogram prefix + in-CTA rank), writes
v54 and v56 there and copies each token's in7 row to its 3 positions. The last CTA finishes the reductions
(v28 across the 16 partials in aten's order, the aux loss, in3/in6 updates, v43, v53).

Numerics (bitwise vs aten):
  sigmoid = 1 / fma(2^j, ex2(f), 1) with libdevice expf's range reduction (checked on all 2^32 fp32 inputs);
  topk(sorted=False) = aten sbtopk order: keys > kth in index order, then keys == kth in index order;
  sum over the 3 weights = (w0 + w2) + w1 (aten reduce, block.x = 2); L1 over 8 = tree (0+4, 2+6), (1+5, 3+7);
  every float op is an explicit .rn PTX instruction (no contraction).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
NE = 8  # experts
TOPK = 3
H = 256  # hidden (bf16 row = 512 B)
NS = NT * TOPK  # sorted slots
K1_CTAS = 128
K1_THREADS = 1024
NCHUNK = NT // 64  # 64-row histogram chunks
K2_TOK = 128  # tokens per scatter CTA
K2_CTAS = NT // K2_TOK
K2_THREADS = 256


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


def _warp_sum_u64(v):
    for off in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, off)
    return v


def _tree8(x):
    """aten block_x_reduce over 8 lanes, offsets 4, 2, 1 (ident 0 added first)."""
    z = [add_rn(Float32(0.0), x[i]) for i in range(8)]
    a = [add_rn(z[i], z[i + 4]) for i in range(4)]
    b = [add_rn(a[i], a[i + 2]) for i in range(2)]
    return add_rn(b[0], b[1])


@dsl_user_op
def ld_v4_b32_cached(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


def _bytes_sum(a, b):
    return a + b


class RouterSort:
    @cute.jit
    def __call__(self, in0, in1, in2, in3, in4, in5, in6, in7, v5, v15, v43, v52, v53, v54, v56, scratch, stream):
        self.k1(in0, in1, in2, in4, v5, v15, scratch).launch(
            grid=[K1_CTAS, 1, 1], block=[K1_THREADS, 1, 1], max_number_threads=[K1_THREADS, 1, 1], smem=4096 + 256 + 512 + 1024 + 64, stream=stream
        )
        self.k2(in3, in5, in6, in7, v43, v52, v53, v54, v56, scratch).launch(
            grid=[K2_CTAS + 1, 1, 1], block=[K2_THREADS, 1, 1], smem=4096, use_pdl=True, stream=stream
        )

    @cute.kernel
    def k1(self, in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor, in4: cute.Tensor, v5: cute.Tensor,
           v15: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s27 = smem.allocate_tensor(Float32, cute.make_layout(1024), 16)  # [k 16][yq 8][e 8]
        sred = smem.allocate_tensor(Float32, cute.make_layout(64), 16)
        shU = smem.allocate_tensor(Int32, cute.make_layout(128), 16)  # [k][e], odd warp of each chunk
        shM = smem.allocate_tensor(Int32, cute.make_layout(256), 16)  # [warp][e]
        cute.arch.griddepcontrol_launch_dependents()

        # 8 lanes per row (lane e = expert e). CTA (c, h) owns rows 1024 k + 64 c + 8 yq + h: aten's dim-0 sum
        # CTA c, y = 8 yq + h.
        c = bid // 8
        h = bid % 8
        e = tid % 8
        slot = tid // 8
        k = slot // 8
        yq = slot % 8
        lane = tid % 32
        wid = tid // 32
        gbase = lane - e
        r = k * 1024 + c * 64 + yq * 8 + h
        r64 = Int64(r)
        xe = ld_f32(_addr(in0) + r64 * 32 + Int64(e) * 4)
        be = ld_f32(_addr(in1) + Int64(e) * 4)
        m2 = ld_u8(_addr(in2) + r64)
        m4 = ld_u8(_addr(in4) + r64)

        se = sigmoid_f32(xe)
        comp = (Uint64(topk_key(add_rn(se, be))) << Uint64(3)) | Uint64(Int64(7 - e))
        m1 = comp
        for off in (1, 2, 4):
            o = cute.arch.shuffle_sync_bfly(m1, off)
            m1 = _sel(o > m1, o, m1)
        m2c = _sel(comp < m1, comp, Uint64(0))
        for off in (1, 2, 4):
            o = cute.arch.shuffle_sync_bfly(m2c, off)
            m2c = _sel(o > m2c, o, m2c)
        m3c = _sel(comp < m2c, comp, Uint64(0))
        for off in (1, 2, 4):
            o = cute.arch.shuffle_sync_bfly(m3c, off)
            m3c = _sel(o > m3c, o, m3c)
        ia = Int32(Uint64(7) - (m1 & Uint64(7)))
        ib = Int32(Uint64(7) - (m2c & Uint64(7)))
        ic = Int32(Uint64(7) - (m3c & Uint64(7)))
        kb = m2c >> Uint64(3)
        kc = m3c >> Uint64(3)
        # aten order: keys > kth (index order), then keys == kth (index order)
        swap = (kb > kc) & (ib < ia)
        i0 = _sel(swap, ib, ia)
        i1 = _sel(swap, ia, ib)
        i2 = ic
        g0 = cute.arch.shuffle_sync(se, gbase + i0)
        g1 = cute.arch.shuffle_sync(se, gbase + i1)
        g2 = cute.arch.shuffle_sync(se, gbase + i2)
        den = add_rn(add_rn(add_rn(g0, g2), g1), Float32(1e-20))
        gj = _sel(e == 0, g0, _sel(e == 1, g1, g2))
        ij = _sel(e == 0, i0, _sel(e == 1, i1, i2))
        wj = div_rn(gj, den)
        pk = i0 | (i1 << Int32(8)) | (i2 << Int32(16))
        if e < 3:
            st_b64(_addr(v5) + r64 * 24 + Int64(e) * 8, Int64(ij))
        if e < 4:
            st_b32(_addr(scratch) + r64 * 16 + Int64(e) * 4, _sel(e < 3, f2i(wj), pk))
        sel = _sel((e == i0) | (e == i1) | (e == i2), Int32(1), Int32(0))
        selm = sel & _sel(m2 == Int32(0), Int32(1), Int32(0))
        st_u8(_addr(v15) + r64 * 8 + Int64(e), selm)

        t = abs_f32(se)
        for off in (4, 2, 1):
            t = add_rn(t, cute.arch.shuffle_sync_bfly(t, off))
        l1 = clamp_min_f32(t, Float32(1e-12))
        nm4 = _sel(m4 == Int32(0), Float32(1.0), Float32(0.0))
        s27[tid] = mul_rn(div_rn(se, l1), nm4)

        base = _addr(scratch)
        hist_base = base + NT * 16
        histm_base = hist_base + NT // 8 * 8
        part_base = histm_base + K1_CTAS * 8
        # expert counts over the warp's 4 rows (lanes e, e + 8, e + 16, e + 24)
        cu = sel
        cm = selm
        for off in (8, 16):
            cu = cu + cute.arch.shuffle_sync_bfly(cu, off)
            cm = cm + cute.arch.shuffle_sync_bfly(cm, off)
        if lane < 8:
            shM[wid * 8 + e] = cm
            if wid % 2 == 1:
                shU[(wid // 2) * 8 + e] = cu
        cute.arch.sync_threads()
        if lane < 8:
            if wid % 2 == 0:
                # this CTA's 8 rows of chunk (k, c): one byte per expert
                st_u8(hist_base + Int64((k * 16 + c) * 8 + h) * 8 + Int64(e), cu + shU[(wid // 2) * 8 + e])
        if tid < 8:
            tm = Int32(0)
            for ww in cutlass.range_constexpr(32):
                tm = tm + shM[ww * 8 + tid]
            st_u8(histm_base + Int64(bid) * 8 + Int64(tid), tm)

        # v28 partial: per y = 8 yq + h, 4 accumulators over k; then aten's tree levels 32, 16, 8 (yq + 4, 2, 1)
        yy = tid // 8
        ee = tid % 8
        v = Float32(0.0)
        if tid < 64:
            vl = []
            for i in cutlass.range_constexpr(4):
                acc = add_rn(Float32(0.0), s27[(i * 8 + yy) * 8 + ee])
                for kk in (i + 4, i + 8, i + 12):
                    acc = add_rn(acc, s27[(kk * 8 + yy) * 8 + ee])
                vl.append(acc)
            v = add_rn(add_rn(add_rn(vl[0], vl[1]), vl[2]), vl[3])
            sred[yy * 8 + ee] = v
        for off in (4, 2, 1):
            cute.arch.sync_threads()
            if yy < off:
                v = add_rn(v, sred[(yy + off) * 8 + ee])
                sred[yy * 8 + ee] = v
        if tid < 8:
            st_f32(part_base + Int64((c * 8 + h) * 8 + tid) * 4, v)

    @cute.kernel
    def k2(self, in3: cute.Tensor, in5: cute.Tensor, in6: cute.Tensor, in7: cute.Tensor, v43: cute.Tensor,
           v52: cute.Tensor, v53: cute.Tensor, v54: cute.Tensor, v56: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sred = smem.allocate_tensor(Int32, cute.make_layout(64), 16)  # [warp][expert]
        sbase = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        stot = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        sscan = smem.allocate_tensor(Int64, cute.make_layout(8), 16)
        spos = smem.allocate_tensor(Int32, cute.make_layout(K2_TOK * 4), 16)
        sf = smem.allocate_tensor(Float32, cute.make_layout(8), 16)

        base = _addr(scratch)
        hist_base = base + NT * 16
        histm_base = hist_base + NT // 8 * 8
        part_base = histm_base + K1_CTAS * 8
        lane = tid % 32
        wid = tid // 32

        # prefetch this CTA's in7 rows (independent of kernel 1): warp w holds tokens 16 w .. 16 w + 15
        bidc = _sel(bid < K2_CTAS, bid, Int32(0))
        src0 = _addr(in7) + Int64(bidc * K2_TOK + wid * 16) * (H * 2) + Int64(lane) * 16
        rows = [ld_v4_b32(src0 + u * (H * 2)) for u in range(16)]
        cute.arch.griddepcontrol_wait()

        # per-expert prefix over the 64-row chunks before this CTA's tokens, and totals
        hc = Int64(0)
        for hh in cutlass.range_constexpr(4):
            hq = ld_v4_b32_cached(hist_base + Int64(tid) * 64 + hh * 16)
            hc = hc + (Int64(hq[0]) & Int64(0xFFFFFFFF)) + (Int64(hq[1]) << Int64(32))
            hc = hc + (Int64(hq[2]) & Int64(0xFFFFFFFF)) + (Int64(hq[3]) << Int64(32))
        first_chunk = bid * (K2_TOK // 64)
        before = tid < first_chunk
        for e in cutlass.range_constexpr(NE):
            cnt = Int32(hc >> Int64(8 * e)) & Int32(255)
            pk = (cnt << Int32(16)) + _sel(before, cnt, Int32(0))
            for off in (16, 8, 4, 2, 1):
                pk = pk + cute.arch.shuffle_sync_bfly(pk, off)
            if lane == 0:
                sred[wid * 8 + e] = pk
        cute.arch.sync_threads()
        if tid < 8:
            tot = Int32(0)
            for ww in cutlass.range_constexpr(8):
                tot = tot + sred[ww * 8 + tid]
            sbase[tid] = tot & Int32(0xFFFF)
            stot[tid] = tot >> Int32(16)
        cute.arch.sync_threads()

        if bid == K2_CTAS:
            # finalize: counts, cumsum, in3, aux loss -> in6. Every global load is issued before its first use.
            x3 = [ldnv_f32(_addr(in3) + 4 * e) for e in range(NE)]
            n5 = ldnv_b64(_addr(in5))
            x6 = ldnv_f32(_addr(in6))
            hm = Int64(0)
            if tid < K1_CTAS:
                hm = ld_b64(histm_base + Int64(tid) * 8)
            for e in cutlass.range_constexpr(NE):
                cnt = Int32(hm >> Int64(8 * e)) & Int32(255)
                for off in (16, 8, 4, 2, 1):
                    cnt = cnt + cute.arch.shuffle_sync_bfly(cnt, off)
                if lane == 0:
                    sred[wid * 8 + e] = cnt
            if tid < 8:
                # v28[e]: per aten CTA c, levels 4, 2, 1 over h; then the last-block combine over c
                pp = [ldnv_f32(part_base + (i * 8) * 4 + Int64(tid) * 4) for i in range(128)]
                pc = []
                for cc in cutlass.range_constexpr(16):
                    p = pp[cc * 8:cc * 8 + 8]
                    t4 = [add_rn(p[i], p[i + 4]) for i in range(4)]
                    t2 = [add_rn(t4[i], t4[i + 2]) for i in range(2)]
                    pc.append(add_rn(t2[0], t2[1]))
                fq = [add_rn(add_rn(add_rn(Float32(0.0), pc[i]), Float32(0.0)), Float32(0.0)) for i in range(16)]
                for off in (8, 4, 2, 1):
                    fq = [add_rn(fq[i], fq[i + off]) for i in range(off)]
                sf[tid] = fq[0]
            cute.arch.sync_threads()
            if tid == 0:
                run = Int32(0)
                tl = [stot[e] for e in range(NE)]
                for e in cutlass.range_constexpr(NE):
                    run = run + tl[e]
                    st_b64(_addr(v43) + 8 * e, Int64(tl[e]))
                    st_b32(_addr(v53) + 4 * e, run)
                cm = []
                for e in cutlass.range_constexpr(NE):
                    cm.append(sred[e] + sred[8 + e] + sred[16 + e] + sred[24 + e])
                v20 = [cvt_f32_s32(cm[e]) for e in range(NE)]
                for e in cutlass.range_constexpr(NE):
                    st_f32(_addr(in3) + 4 * e, add_rn(x3[e], v20[e]))
                v28 = [sf[e] for e in range(NE)]
                v29 = _tree8([abs_f32(v20[e]) for e in range(NE)])
                v30 = clamp_min_f32(v29, Float32(1e-12))
                v34 = [mul_rn(mul_rn(div_rn(v20[e], v30), Float32(8.0)), v28[e]) for e in range(NE)]
                v35 = _tree8(v34)
                n5 = _sel(n5 < Int64(1), Int64(1), n5)
                v38 = div_rn(Float32(1.0), cvt_f32_s64(n5))
                v40 = mul_rn(v35, mul_rn(v38, Float32(1.0)))
                st_f32(_addr(in6), add_rn(x6, v40))
        else:
            if tid < 8:
                off = Int32(0)
                for e in cutlass.range_constexpr(NE):
                    off = off + _sel(Int32(e) < tid, stot[e], Int32(0))
                sbase[tid] = sbase[tid] + off
            # in-CTA stable rank: exclusive scan of per-token one-hot bytes
            tok = bid * K2_TOK + tid % K2_TOK
            rec = ld_v4_b32_cached(base + Int64(tok) * 16)
            pe = rec[3]
            ex = [pe & Int32(255), (pe >> Int32(8)) & Int32(255), (pe >> Int32(16)) & Int32(255)]
            selm = (Int32(1) << ex[0]) | (Int32(1) << ex[1]) | (Int32(1) << ex[2])
            ind = Int64(_spread4(selm & Int32(15))) | (Int64(_spread4((selm >> Int32(4)) & Int32(15))) << Int64(32))
            inc = ind
            for off in (1, 2, 4, 8, 16):
                up = cute.arch.shuffle_sync_up(inc, off)
                inc = _sel(lane >= off, inc + up, inc)
            if lane == 31:
                sscan[wid] = inc
            cute.arch.sync_threads()
            if tid < K2_TOK:
                pre = Int64(0)
                for ww in cutlass.range_constexpr(K2_TOK // 32 - 1):
                    pre = pre + _sel(Int32(ww) < wid, sscan[ww], Int64(0))
                excl = inc - ind + pre
                for j in cutlass.range_constexpr(TOPK):
                    e = ex[j]
                    rank = Int32(excl >> Int64(e * 8)) & Int32(255)
                    pos = sbase[e] + rank
                    st_b32(_addr(v54) + Int64(pos) * 4, rec[j])
                    st_b64(_addr(v56) + Int64(pos) * 8, Int64(tok))
                    spos[tid * 4 + j] = pos
            cute.arch.sync_threads()
            dst0 = _addr(v52) + Int64(lane) * 16
            for u in cutlass.range_constexpr(16):
                t = wid * 16 + u
                for j in cutlass.range_constexpr(TOPK):
                    st_v4_b32(dst0 + Int64(spos[t * 4 + j]) * (H * 2), rows[u])


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


SCRATCH_BYTES = NT * 16 + NT // 8 * 8 + K1_CTAS * 8 + K1_CTAS * 8 * 4


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            RouterSort(),
            _fake(cutlass.Float32, NT * NE),  # in0
            _fake(cutlass.Float32, NE, 4),  # in1
            _fake(cutlass.Uint8, NT, 1),  # in2
            _fake(cutlass.Float32, NE, 4),  # in3
            _fake(cutlass.Uint8, NT, 1),  # in4
            _fake(cutlass.Int64, 1, 8),  # in5
            _fake(cutlass.Float32, 1, 4),  # in6
            _fake(cutlass.BFloat16, NT * H),  # in7
            _fake(cutlass.Int64, NT * TOPK, 8),  # v5
            _fake(cutlass.Uint8, NT * NE, 8),  # v15
            _fake(cutlass.Int64, NE, 8),  # v43
            _fake(cutlass.BFloat16, NS * H),  # v52
            _fake(cutlass.Int32, NE, 4),  # v53
            _fake(cutlass.Float32, NS, 4),  # v54
            _fake(cutlass.Int64, NS, 8),  # v56
            _fake(cutlass.Uint8, SCRATCH_BYTES),  # scratch
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _uninit(shape, stride, dtype, dev):
    """A tensor on new uninitialized storage. Unlike torch.empty, this launches no fill kernel under
    torch.use_deterministic_algorithms(True) (fill_uninitialized_memory): every byte is written below."""
    n = 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    storage = torch.UntypedStorage(n * dtype.itemsize, device=dev)
    return torch.empty(0, dtype=dtype, device=dev).set_(storage, 0, shape, stride)


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7):
    dev = in0.device
    x = in0.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    rows = in7.contiguous()
    if rows.data_ptr() % 16:
        rows = rows.clone()
    v5 = _uninit((NT, TOPK), (TOPK, 1), torch.int64, dev)
    v15 = _uninit((NT, NE), (NE, 1), torch.uint8, dev)
    v43 = _uninit((NE,), (1,), torch.int64, dev)
    v52 = _uninit((NS, H), (H, 1), torch.bfloat16, dev)
    v53 = _uninit((NE,), (1,), torch.int32, dev)
    v54 = _uninit((NS, 1), (1, 1), torch.float32, dev)
    v56 = _uninit((NS,), (1,), torch.int64, dev)
    scratch = _uninit((SCRATCH_BYTES,), (1,), torch.uint8, dev)
    _compiled(dev.index)(
        x.view(-1),
        in1.contiguous(),
        in2.reshape(-1).view(torch.uint8),
        in3,
        in4.reshape(-1).view(torch.uint8),
        in5.reshape(1),
        in6.reshape(1),
        rows.view(-1),
        v5.view(-1),
        v15.view(-1),
        v43,
        v52.view(-1),
        v53,
        v54.view(-1),
        v56,
        scratch,
    )
    return (v5, v15.view(torch.bool), v43, v52, v53, v54, v56.view(NS, 1).expand(NS, H))
