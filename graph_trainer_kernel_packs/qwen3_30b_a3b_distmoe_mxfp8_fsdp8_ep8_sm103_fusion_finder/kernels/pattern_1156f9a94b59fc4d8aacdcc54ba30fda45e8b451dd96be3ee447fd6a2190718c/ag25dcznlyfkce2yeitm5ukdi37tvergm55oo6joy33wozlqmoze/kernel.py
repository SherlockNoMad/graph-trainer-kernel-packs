"""MoE router forward of Qwen3 30B-A3B (32768 tokens, 128 experts, 8 slots), two CuTe DSL kernels.

reference (problem.py), per token row r of in0 [32768, 128] fp32:
  v1 = softmax(in0[r]) (v2 = alias); v5[r, k] = (8 r + k) % 128 (fixed routing, no topk); v6 = v1[r, v5[r]]
  v8 = sum_8(v6) + 1e-20; v32 = in4[r] ? 0 : (v6 / v8) * 1; v31 = in4[r] ? -1 : v5
  v15 = onehot(v5[r]) & ~in1[r]; in2 += v16 = sum_r v15 (int64); v21 = sum_r float(v15)
  v22 = L1(v1[r]); v24 = expand(clamp_min(v22, 1e-12)); v27 = ~in3; v29 = sum_r (v1 / v24) * v27

Kernel 1 (2048 CTAs x 4 warps): CTA t owns rows t + 2048 n (n < 16), exactly the rows of thread t of aten's
dim-0 sum kernel (grid.y 512, block 32 x 4, 4 outputs per thread, input_idx = y + 4 * blockIdx.y). Warp i does rows
t + 2048 (i + 4 m), m < 4: softmax, L1 norm, the per-row outputs, and the column partial A_i of v28 in aten's
order (0 + rows in m order); the CTA writes V(t) = ((A0 + A1) + A2) + A3 and its unmasked-row count to scratch.
The 8-slot gather of the 4 rows is batched through shared memory (lane = row lane // 8, slot lane % 8).
It also writes v5/v31 for rows 16 t .. 16 t + 15.
Kernel 2 (32 CTAs x 4 columns, programmatic dependent launch): S(c) = (V(4c) + V(4c+2)) + (V(4c+1) + V(4c+3))
(aten's block_y reduce), W_y = 0 + S(y) + S(y + 4) + ... (128 terms, aten's last-CTA global reduce),
v29 = (W0 + W2) + (W1 + W3); counts -> v21 and in2 += count.

Numerics (bitwise vs the aten kernels on this build; every float op is explicit .rn PTX):
  softmax (softmax_warp_forward<float, 7>): lane x owns columns x + 32 it; max is order-free here (NaN gives all
    NaN either way); e = expf(x - max) with libdevice's sequence (e = ex2.approx(f) * 2^n); sum from 0 in it
    order (fma(ex2, 2^n, sum); identical to sum + e since 2^n * ex2 is exact unless denormal, and a denormal
    term cannot change a sum that reaches 1), XOR shuffle 16..1;
    v1 = e / sum (NaN if sum == 0).
  L1 norm (reduce_kernel NormOneOps, vectorized input, block.x 32): lane x: a_i = 0 + |v1[4x + i]|,
    ((a0 + a1) + a2) + a3, shfl_down 16..1.  clamp_min keeps NaN.
  sum_8 of v6 (block.x 8): ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)), a_k = 0 + v6[k]; + 1e-20f.
  v21 and v16 are integer counts (exact in float in any order: <= 32768).
Structural assumptions: none beyond the contract (v5 is the reference's arange % 128 routing, not data).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 32768
NE = 128
TOPK = 8
NV = 2048  # aten dim-0 sum threads t (rows t + 2048 n)
NC = 512  # aten dim-0 sum CTAs c (threads 4c .. 4c + 3)
K1_CTAS = NV
K1_WARPS = 4
K1_THREADS = K1_WARPS * 32
K2_CTAS = NE // 4
K2_THREADS = 128


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


def _binf(op):
    @dsl_user_op
    def f(a, b, *, loc=None, ip=None):
        return Float32(
            _asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], f"{op} $0, $1, $2;", "=f,f,f",
                 loc=loc, ip=ip)
        )

    return f


add_rn = _binf("add.rn.f32")
sub_rn = _binf("sub.rn.f32")
mul_rn = _binf("mul.rn.f32")
div_rn = _binf("div.rn.f32")
max_f32 = _binf("max.f32")


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Float32(c), loc, ip)],
                        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", loc=loc, ip=ip))


@dsl_user_op
def abs_f32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], "abs.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def clamp_min_f32(a, lo, *, loc=None, ip=None):
    """aten clamp_min: NaN stays NaN, else max(a, lo)."""
    code = "{\n.reg .pred p;\n.reg .f32 m;\nsetp.nan.f32 p, $1, $1;\nmax.f32 m, $1, $2;\nselp.f32 $0, $1, m, p;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(lo), loc, ip)], code, "=f,f,f", loc=loc, ip=ip))


_EXPF = (
    "{\n.reg .f32 t, j, f;\n.reg .b32 jb;\n"
    "fma.rn.sat.f32 t, $2, 0f3BBB989D, 0f3F000000;\n"
    "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
    "sub.rn.f32 f, 0f4B40007F, j;\n"
    "fma.rn.f32 f, $2, 0f3FB8AA3B, f;\n"
    "fma.rn.f32 f, $2, 0f32A57060, f;\n"
    "mov.b32 jb, j;\nshl.b32 jb, jb, 23;\nmov.b32 $1, jb;\n"
    "ex2.approx.ftz.f32 $0, f;\n}"
)


@dsl_user_op
def expf_parts(x, *, loc=None, ip=None):
    """libdevice expf(x) = ex2 * scale: returns (ex2, scale)."""
    res = _asm(ir.Type.parse("!llvm.struct<(f32, f32)>"), [_v(Float32(x), loc, ip)], _EXPF, "=f,=f,f",
               loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_u8(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ldcg_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"), [_v(Int64(addr), loc, ip)],
               "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", side=True, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ldcg_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.cg.b32 $0, [$1];", "=r,l", side=True,
                      loc=loc, ip=ip))


@dsl_user_op
def ld_f32_rw(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def st_f32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Float32(val), loc, ip)], "st.global.f32 [$0], $1;", "l,f", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_b32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.b32 [$0], $1;", "l,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_u8(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(val), loc, ip)], "st.global.u8 [$0], $1;", "l,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_b64(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int64(val), loc, ip)], "st.global.b64 [$0], $1;", "l,l", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_v4_f32(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
         "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_v4_b32(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip)] + [_v(Int32(x), loc, ip) for x in w],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip), _v(Float32(a), loc, ip)], "st.shared.f32 [$0], $1;", "r,f", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip), _v(Int32(a), loc, ip)], "st.shared.b32 [$0], $1;", "r,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def sts_v4(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
         "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f", side=True, loc=loc, ip=ip)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(addr), loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int32(addr), loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", side=True,
                      loc=loc, ip=ip))


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"), [_v(Int32(addr), loc, ip)],
               "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", side=True, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def cvt_f32_s32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(a), loc, ip)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def i2f(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(a), loc, ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


def _sel(c, a, b):
    return type(a)(cutlass.select_(c, a, b))


def _addr(t):
    return t.iterator.toint()


class RouterFwd:
    @cute.jit
    def __call__(self, in0, in1, in2, in3, in4, v1, v5, v6, v8, v21, v22, v23, v27, v29, v31, v32, scratch, stream):
        self.k1(in0, in1, in3, in4, v1, v5, v6, v8, v22, v23, v27, v31, v32, scratch).launch(
            grid=[K1_CTAS, 1, 1], block=[K1_THREADS, 1, 1], smem=K1_WARPS * (NE + 32) * 4 + 64, stream=stream
        )
        self.k2(in2, v21, v29, scratch).launch(
            grid=[K2_CTAS, 1, 1], block=[K2_THREADS, 1, 1], smem=NC * 16 + 256, use_pdl=True, stream=stream
        )

    @cute.kernel
    def k1(self, in0: cute.Tensor, in1: cute.Tensor, in3: cute.Tensor, in4: cute.Tensor, v1: cute.Tensor,
           v5: cute.Tensor, v6: cute.Tensor, v8: cute.Tensor, v22: cute.Tensor, v23: cute.Tensor, v27: cute.Tensor,
           v31: cute.Tensor, v32: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(Float32, cute.make_layout(K1_WARPS * NE), 16)
        sG = smem.allocate_tensor(Float32, cute.make_layout(K1_WARPS * 32), 16)
        sC = smem.allocate_tensor(Int32, cute.make_layout(K1_WARPS), 16)
        cute.arch.griddepcontrol_launch_dependents()
        lane = tid % 32
        i = tid // 32
        sw = sA.iterator.toint() + Int32(i * NE * 4)
        sg = sG.iterator.toint() + Int32(i * 32 * 4)
        rows = [Int64(t + 2048 * (i + 4 * m)) for m in range(4)]
        # loads: layout B (lane owns columns lane + 32 it)
        xs = []
        for m in cutlass.range_constexpr(4):
            xs.append([ld_f32(_addr(in0) + rows[m] * (NE * 4) + Int64(lane * 4 + 128 * it)) for it in range(4)])
        mk1 = [ld_u8(_addr(in1) + rows[m]) for m in range(4)]
        mk3 = [ld_u8(_addr(in3) + rows[m]) for m in range(4)]
        mk4 = [ld_u8(_addr(in4) + rows[m]) for m in range(4)]

        # v5 / v31 for rows 16 t .. 16 t + 15 (coalesced): thread q writes slot q % 8 of row 16 t + q // 8
        r5 = t * 16 + tid // 8
        k5 = tid % 8
        e5 = (r5 * 8 + k5) % NE
        m5 = ld_u8(_addr(in4) + Int64(r5))
        off5 = Int64(r5) * (TOPK * 8) + Int64(k5 * 8)
        st_b64(_addr(v5) + off5, Int64(e5))
        st_b64(_addr(v31) + off5, _sel(m5 != Int32(0), Int64(-1), Int64(e5)))

        gq = t % 16  # expert group of every row of this CTA: (8 r) % 128 = 8 (r % 16)
        acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
        n22 = []
        n23 = []
        for m in cutlass.range_constexpr(4):
            r = rows[m]
            x = xs[m]
            mx = max_f32(max_f32(x[0], x[1]), max_f32(x[2], x[3]))
            for off in (16, 8, 4, 2, 1):
                mx = max_f32(mx, cute.arch.shuffle_sync_bfly(mx, off))
            ee = []
            s = Float32(0.0)
            for it in cutlass.range_constexpr(4):
                a, b = expf_parts(sub_rn(x[it], mx))
                ee.append(mul_rn(a, b))
                s = fma_rn(a, b, s)
            for off in (16, 8, 4, 2, 1):
                s = add_rn(s, cute.arch.shuffle_sync_bfly(s, off))
            nan = i2f(Int32(0x7FC00000))
            for it in cutlass.range_constexpr(4):
                sts_f32(sw + (lane + 32 * it) * 4, _sel(s == Float32(0.0), nan, div_rn(ee[it], s)))
            cute.arch.sync_warp()
            p = lds_v4(sw + lane * 16)
            cute.arch.sync_warp()
            st_v4_f32(_addr(v1) + r * (NE * 4) + Int64(lane * 16), p)
            if lane // 2 == gq:
                sts_v4(sg + Int32(m * 32) + (lane % 2) * 16, p)
            # L1 norm
            n = add_rn(add_rn(add_rn(add_rn(Float32(0.0), abs_f32(p[0])), add_rn(Float32(0.0), abs_f32(p[1]))),
                              add_rn(Float32(0.0), abs_f32(p[2]))), add_rn(Float32(0.0), abs_f32(p[3])))
            for off in (16, 8, 4, 2, 1):
                n = add_rn(n, cute.arch.shuffle_sync_down(n, off))
            v22r = cute.arch.shuffle_sync(n, 0)
            v23r = clamp_min_f32(v22r, Float32(1e-12))
            n22.append(v22r)
            n23.append(v23r)
            f3 = _sel(mk3[m] != Int32(0), Float32(0.0), Float32(1.0))
            for q in cutlass.range_constexpr(4):
                acc[q] = add_rn(acc[q], mul_rn(div_rn(p[q], v23r), f3))
        cnt = Int32(4)
        for m in cutlass.range_constexpr(4):
            cnt = cnt - _sel(mk1[m] != Int32(0), Int32(1), Int32(0))
        cute.arch.sync_warp()
        # per-row scalars: lane m < 4 stores row m
        lm = lane % 4
        rl = Int64(t + 2048 * i) + Int64(8192) * Int64(lm)
        s22 = _sel(lm == 0, n22[0], _sel(lm == 1, n22[1], _sel(lm == 2, n22[2], n22[3])))
        s23 = _sel(lm == 0, n23[0], _sel(lm == 1, n23[1], _sel(lm == 2, n23[2], n23[3])))
        s27 = _sel(lm == 0, mk3[0], _sel(lm == 1, mk3[1], _sel(lm == 2, mk3[2], mk3[3])))
        if lane < 4:
            st_f32(_addr(v22) + rl * 4, s22)
            st_f32(_addr(v23) + rl * 4, s23)
            st_u8(_addr(v27) + rl, _sel(s27 != Int32(0), Int32(0), Int32(1)))
        # gather v6 (lane = row lane // 8, slot lane % 8) and the weights
        g8 = lane // 8
        k8 = lane % 8
        rg = Int64(t + 2048 * i) + Int64(8192) * Int64(g8)
        v6v = lds_f32(sg + lane * 4)
        t8 = add_rn(Float32(0.0), v6v)
        for off in (4, 2, 1):
            t8 = add_rn(t8, cute.arch.shuffle_sync_down(t8, off))
        v8r = add_rn(cute.arch.shuffle_sync(t8, lane - k8), Float32(1e-20))
        m4g = _sel(g8 == 0, mk4[0], _sel(g8 == 1, mk4[1], _sel(g8 == 2, mk4[2], mk4[3])))
        v32v = _sel(m4g != Int32(0), Float32(0.0), mul_rn(div_rn(v6v, v8r), Float32(1.0)))
        o6 = rg * (TOPK * 4) + Int64(k8 * 4)
        st_f32(_addr(v6) + o6, v6v)
        st_f32(_addr(v32) + o6, v32v)
        if k8 == 0:
            st_f32(_addr(v8) + rg * 4, v8r)

        # column partial V(t) = ((A0 + A1) + A2) + A3
        sts_v4(sw + lane * 16, acc)
        if lane == 0:
            sts_b32(sC.iterator.toint() + Int32(i * 4), cnt)
        cute.arch.sync_threads()
        base = sA.iterator.toint() + tid * 4
        aa = [lds_f32(base + Int32(ii * NE * 4)) for ii in range(4)]
        st_f32(_addr(scratch) + Int64(t * NE + tid) * 4, add_rn(add_rn(add_rn(aa[0], aa[1]), aa[2]), aa[3]))
        if tid == 0:
            cb = sC.iterator.toint()
            st_b32(_addr(scratch) + Int64(NV * NE * 4) + Int64(t) * 4,
                   lds_b32(cb) + lds_b32(cb + 4) + lds_b32(cb + 8) + lds_b32(cb + 12))

    @cute.kernel
    def k2(self, in2: cute.Tensor, v21: cute.Tensor, v29: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sS = smem.allocate_tensor(Float32, cute.make_layout(NC * 4), 16)
        sW = smem.allocate_tensor(Float32, cute.make_layout(16), 16)
        sN = smem.allocate_tensor(Int32, cute.make_layout(4), 16)
        cute.arch.griddepcontrol_wait()
        lane = tid % 32
        # S(c) = (V(4c) + V(4c+2)) + (V(4c+1) + V(4c+3)), c = tid + 128 q (aten's per-CTA block_y reduce)
        vl = []
        for q in cutlass.range_constexpr(NC // K2_THREADS):
            c = tid + q * K2_THREADS
            vl.append([ldcg_v4_f32(_addr(scratch) + Int64((4 * c + jj) * NE + bid * 4) * 4) for jj in range(4)])
        cs = Int32(0)
        if tid < 32:
            # unmasked rows of expert group g = bid // 2: CTAs t = g mod 16
            g = bid // 2
            cl = [ldcg_b32(_addr(scratch) + Int64(NV * NE * 4) + Int64(g + 16 * (lane + 32 * q)) * 4)
                  for q in range(4)]
            cs = (cl[0] + cl[1]) + (cl[2] + cl[3])
            for off in (16, 8, 4, 2, 1):
                cs = cs + cute.arch.shuffle_sync_bfly(cs, off)
        for q in cutlass.range_constexpr(NC // K2_THREADS):
            c = tid + q * K2_THREADS
            v = vl[q]
            for jc in cutlass.range_constexpr(4):
                sS[c * 4 + jc] = add_rn(add_rn(v[0][jc], v[2][jc]), add_rn(v[1][jc], v[3][jc]))
        if tid == 0:
            sN[0] = cs
        cute.arch.sync_threads()
        if tid < 16:
            y = tid // 4
            jc = tid % 4
            wsum = Float32(0.0)
            for k in cutlass.range_constexpr(NC // 4):
                wsum = add_rn(wsum, sS[(y + 4 * k) * 4 + jc])
            sW[tid] = wsum
        cute.arch.sync_threads()
        if tid < 4:
            res = add_rn(add_rn(sW[tid], sW[tid + 8]), add_rn(sW[tid + 4], sW[tid + 12]))
            e = bid * 4 + tid
            st_f32(_addr(v29) + Int64(e) * 4, res)
            fc = cvt_f32_s32(sN[0])
            st_f32(_addr(v21) + Int64(e) * 4, fc)
            ea = _addr(in2) + Int64(e) * 4
            st_f32(ea, add_rn(ld_f32_rw(ea), fc))


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            RouterFwd(),
            _fake(cutlass.Float32, NT * NE),  # in0
            _fake(cutlass.Uint8, NT, 4),  # in1
            _fake(cutlass.Float32, NE),  # in2
            _fake(cutlass.Uint8, NT, 4),  # in3
            _fake(cutlass.Uint8, NT, 4),  # in4
            _fake(cutlass.Float32, NT * NE),  # v1
            _fake(cutlass.Int64, NT * TOPK),  # v5
            _fake(cutlass.Float32, NT * TOPK),  # v6
            _fake(cutlass.Float32, NT),  # v8
            _fake(cutlass.Float32, NE),  # v21
            _fake(cutlass.Float32, NT),  # v22
            _fake(cutlass.Float32, NT),  # v23
            _fake(cutlass.Uint8, NT),  # v27
            _fake(cutlass.Float32, NE),  # v29
            _fake(cutlass.Int64, NT * TOPK),  # v31
            _fake(cutlass.Float32, NT * TOPK),  # v32
            _fake(cutlass.Float32, NV * NE + NV),  # scratch: V(t) partials, row counts
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _flat(t, align=16):
    t = t.contiguous()
    if t.data_ptr() % align:
        t = t.clone()
    return t.view(-1)


def kernel_function(in0, in1, in2, in3, in4):
    dev = in0.device
    f32 = torch.float32
    v1 = torch.empty((NT, NE), dtype=f32, device=dev)
    v5 = torch.empty((NT, TOPK), dtype=torch.int64, device=dev)
    v6 = torch.empty((NT, TOPK), dtype=f32, device=dev)
    v8 = torch.empty((NT, 1), dtype=f32, device=dev)
    v21 = torch.empty((NE,), dtype=f32, device=dev)
    v22 = torch.empty((NT, 1), dtype=f32, device=dev)
    v23 = torch.empty((NT, 1), dtype=f32, device=dev)
    v27 = torch.empty((NT, 1), dtype=torch.bool, device=dev)
    v29 = torch.empty((NE,), dtype=f32, device=dev)
    v31 = torch.empty((NT, TOPK), dtype=torch.int64, device=dev)
    v32 = torch.empty((NT, TOPK), dtype=f32, device=dev)
    scratch = torch.empty((NV * NE + NV,), dtype=f32, device=dev)
    assert in2.is_contiguous() and in2.data_ptr() % 4 == 0  # written in place
    _compiled(dev.index)(
        _flat(in0),
        _flat(in1.view(torch.uint8), 4),
        in2,
        _flat(in3.view(torch.uint8), 4),
        _flat(in4.view(torch.uint8), 4),
        v1.view(-1),
        v5.view(-1),
        v6.view(-1),
        v8.view(-1),
        v21,
        v22.view(-1),
        v23.view(-1),
        v27.view(-1).view(torch.uint8),
        v29,
        v31.view(-1),
        v32.view(-1),
        scratch,
    )
    return (v1, v1.view(NT, NE), v5, v6, v8, v21, v22, v23.expand(NT, NE), v27, v29, v31, v32)
