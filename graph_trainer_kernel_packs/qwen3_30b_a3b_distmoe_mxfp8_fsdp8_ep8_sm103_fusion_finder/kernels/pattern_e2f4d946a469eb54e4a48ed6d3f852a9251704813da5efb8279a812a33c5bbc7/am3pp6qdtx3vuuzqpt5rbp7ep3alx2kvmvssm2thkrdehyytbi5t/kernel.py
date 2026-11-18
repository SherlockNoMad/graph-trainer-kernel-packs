"""MoE router gate backward of Qwen3 30B-A3B (32768 tokens, 128 experts, top-8), one CuTe DSL kernel.

reference (problem.py), per token row r (p = in6[r], y = in12[r]: the softmax output, d = in7[r, 0]):
  v13[e] = ((1 * in3) * in4[e]) * float(in5[r]);  v19 = -v13 * ((p / d) / d);  v20 = v13 / d
  v21 = sum_128(v19);  v25 = (in8[r] > 1e-12) ? v21 : 0;  v28 = v20 + sgn(p) * v25
  v29[k] = (in1[r] ? 0 : in0[r, k]) * 1;  v35 = -v29 * ((in9 / in10) / in10);  v36 = v29 / in10
  v39[k] = v36[k] + sum_8(v35);  v42 = zeros(128).scatter_add(in11[r], v39);  v43 = v28 + v42
  v46 = _softmax_backward_data(v43, y);  out0[r] = [bf16(v46), bf16(r1), bf16(r2)] with
  r1 = v46 - f32(bf16(v46)), r2 = r1 - f32(bf16(r1)); out1..3 are its column-block transposes.

One warp per 4 consecutive rows (8192 warps, 8 per CTA). Lane x owns columns 4x..4x+3 of each row
(float4 loads/stores) and, for the 8-slot part, slot x % 8 of row x / 8.

Numerics (bitwise vs the aten kernels the reference launches on this build; all float ops are explicit
.rn PTX, so nothing is contracted unless written as an fma):
  sum_128 (reduce_kernel, vectorized input, block.x = 32): lane x holds a_i = 0 + v[4x + i];
    ((a0 + a1) + a2) + a3, then shfl_down 16, 8, 4, 2, 1 (lane 0's value).
  sum_8 (reduce_kernel, block.x = 8, 4 rows per warp like here): lane k holds 0 + v[k]; shfl_down 4, 2, 1.
  scatter_add under deterministic mode = index_put(accumulate) on the flattened index, sort-based
    (indexing_backward_kernel_stride_1): per expert 0 + ((0 + v_k1) + v_k2 ...) over the matching slots in
    ascending k; reproduced with a per-warp shared-memory row zero-filled, then 8 sequential steps.
    Duplicate experts within a row are handled exactly.
  _softmax_backward_data (softmax_warp_backward<float, 7>, after aten's tmp = grad * output): lane x sums
    (((0 + t[x]) + t[x+32]) + t[x+64]) + t[x+96], XOR shuffle 16..1; out = fma(-y, sum, t) (nvcc contracts
    tmp - y * sum).  The [4x, 4x+3] -> [x + 32j] regrouping goes through shared memory.
  gt(in8, 1e-12) compares in float32; sgn(p) = (0 < p) - (p < 0); bf16 = cvt.rn.bf16 (aten's __float2bfloat16).

Structural assumptions (from the contract): in7 has stride (1, 0) (the clamped L1 norm expanded over the 128
experts), so only in7[:, 0] is read; in11 values are in [0, 128) (aten device-asserts otherwise). When in6 and
in12 are the same tensor (they are in training: both the router softmax output) it is read once.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 32768  # tokens
NE = 128  # experts
TOPK = 8
RPW = 4  # rows per warp
WARPS = 8
THREADS = WARPS * 32
CTAS = NT // (RPW * WARPS)
SMEM_FLOATS = WARPS * RPW * NE
SMBWD_FMA = True


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


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Float32(c), loc, ip)],
                        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", loc=loc, ip=ip))


@dsl_user_op
def neg_f32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], "neg.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def sgn_f32(a, *, loc=None, ip=None):
    """aten sign: float((0 < a) - (a < 0)); NaN and +-0 -> +0."""
    code = (
        "{\n.reg .pred p, q;\n.reg .f32 t;\nsetp.gt.f32 p, $1, 0f00000000;\nsetp.lt.f32 q, $1, 0f00000000;\n"
        "selp.f32 t, 0fBF800000, 0f00000000, q;\nselp.f32 $0, 0f3F800000, t, p;\n}"
    )
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], code, "=f,f", loc=loc, ip=ip))


@dsl_user_op
def gt_sel(n, a, *, loc=None, ip=None):
    """where(n > float(1e-12), a, 0.0)."""
    code = "{\n.reg .pred p;\nsetp.gt.f32 p, $1, 0f2B8CBCCC;\nselp.f32 $0, $2, 0f00000000, p;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(n), loc, ip), _v(Float32(a), loc, ip)], code, "=f,f,f", loc=loc, ip=ip))


def _ldv(n, kind, space="global.nc"):
    c = "f" if kind == "f32" else "r"
    regs = ", ".join(f"${i}" for i in range(n))

    @dsl_user_op
    def f(addr, *, loc=None, ip=None):
        ty = T.f32() if kind == "f32" else T.i32()
        res = _asm(
            ir.Type.parse("!llvm.struct<(" + ", ".join([("f32" if kind == "f32" else "i32")] * n) + ")>"),
            [_v(Int64(addr), loc, ip)],
            f"ld.{space}.v{n}.{'f32' if kind == 'f32' else 'b32'} {{{regs}}}, [${n}];",
            ",".join([f"={c}"] * n) + ",l",
            loc=loc,
            ip=ip,
        )
        cls = Float32 if kind == "f32" else Int32
        return [cls(llvm.extractvalue(ty, res, [k], loc=loc, ip=ip)) for k in range(n)]

    return f


ld_v4_f32 = _ldv(4, "f32")


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_u8(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def st_v2_b32(addr, a, b, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int32(a), loc, ip), _v(Int32(b), loc, ip)],
         "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_v4(addr, w, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
         "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm(None, [_v(Int32(addr), loc, ip), _v(Float32(a), loc, ip)], "st.shared.f32 [$0], $1;", "r,f", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(addr), loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"), [_v(Int32(addr), loc, ip)],
               "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", side=True, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    """Two floats -> packed bf16 (round to nearest even), lo in the low half."""
    return Int32(_asm(T.i32(), [_v(Float32(hi), loc, ip), _v(Float32(lo), loc, ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;",
                      "=r,f,f", loc=loc, ip=ip))


@dsl_user_op
def i2f(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(a), loc, ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


def _sel(c, a, b):
    return type(a)(cutlass.select_(c, a, b))


def _addr(t):
    return t.iterator.toint()


class RouterGateBwd:
    def __init__(self, alias_y):
        self.alias_y = alias_y

    @cute.jit
    def __call__(self, in0, in1, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, out0, stream):
        self.kern(in0, in1, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, out0).launch(
            grid=[CTAS, 1, 1], block=[THREADS, 1, 1], smem=SMEM_FLOATS * 4 + 64, stream=stream
        )

    @cute.kernel
    def kern(self, in0: cute.Tensor, in1: cute.Tensor, in3: cute.Tensor, in4: cute.Tensor, in5: cute.Tensor,
             in6: cute.Tensor, in7: cute.Tensor, in8: cute.Tensor, in9: cute.Tensor, in10: cute.Tensor,
             in11: cute.Tensor, in12: cute.Tensor, out0: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(Float32, cute.make_layout(SMEM_FLOATS), 16)
        lane = tid % 32
        wid = tid // 32
        sw = sbuf.iterator.toint() + Int32(wid * (RPW * NE * 4))
        rb = Int64((bid * WARPS + wid) * RPW)

        # ---- loads ----
        # 8-slot part: lane = (row u8, slot k8)
        u8 = lane // 8
        k8 = lane % 8
        rr = rb + Int64(u8)
        a0 = ld_f32(_addr(in0) + rr * (TOPK * 4) + Int64(k8) * 4)
        a9 = ld_f32(_addr(in9) + rr * (TOPK * 4) + Int64(k8) * 4)
        ex = ld_b32(_addr(in11) + rr * (TOPK * 8) + Int64(k8) * 8)
        m1 = ld_u8(_addr(in1) + rr)
        s10 = ld_f32(_addr(in10) + rr * 4)
        # main part: lane owns columns 4 lane .. 4 lane + 3
        col = Int64(lane) * 16
        p = []
        y = []
        for u in cutlass.range_constexpr(RPW):
            p.append(ld_v4_f32(_addr(in6) + (rb + Int64(u)) * (NE * 4) + col))
        if cutlass.const_expr(not self.alias_y):
            for u in cutlass.range_constexpr(RPW):
                y.append(ld_v4_f32(_addr(in12) + (rb + Int64(u)) * (NE * 4) + col))
        dv = ld_v4_f32(_addr(in7) + rb * 4)
        n8 = ld_v4_f32(_addr(in8) + rb * 4)
        m5 = ld_b32(_addr(in5) + rb)
        w4 = ld_v4_f32(_addr(in4) + col)
        v6 = mul_rn(Float32(1.0), ld_f32(_addr(in3)))

        # zero-fill this warp's scatter rows
        z = Float32(0.0)
        for u in cutlass.range_constexpr(RPW):
            sts_v4(sw + Int32(u * NE * 4) + lane * 16, [z, z, z, z])

        # ---- 8-slot part ----
        v29 = mul_rn(_sel(m1 != Int32(0), Float32(0.0), a0), Float32(1.0))
        v35 = mul_rn(neg_f32(v29), div_rn(div_rn(a9, s10), s10))
        v36 = div_rn(v29, s10)
        t8 = add_rn(Float32(0.0), v35)
        for off in (4, 2, 1):
            t8 = add_rn(t8, cute.arch.shuffle_sync_down(t8, off))
        v37 = cute.arch.shuffle_sync(t8, lane - k8)
        v39 = add_rn(v36, v37)
        cute.arch.sync_warp()
        # scatter_add: sequential over slots, one row per active lane
        sa = sw + (Int32(u8 * NE) + ex) * Int32(4)
        for k in cutlass.range_constexpr(TOPK):
            if k8 == k:
                sts_f32(sa, add_rn(lds_f32(sa), v39))
            cute.arch.sync_warp()

        # ---- main part ----
        v9 = [mul_rn(v6, w4[i]) for i in range(4)]
        tmp = []
        for u in cutlass.range_constexpr(RPW):
            d = dv[u]
            f5 = _sel(((m5 >> Int32(8 * u)) & Int32(255)) != Int32(0), Float32(1.0), Float32(0.0))
            v13 = [mul_rn(v9[i], f5) for i in range(4)]
            v19 = [mul_rn(neg_f32(v13[i]), div_rn(div_rn(p[u][i], d), d)) for i in range(4)]
            v20 = [div_rn(v13[i], d) for i in range(4)]
            a = [add_rn(Float32(0.0), v19[i]) for i in range(4)]
            s = add_rn(add_rn(add_rn(a[0], a[1]), a[2]), a[3])
            for off in (16, 8, 4, 2, 1):
                s = add_rn(s, cute.arch.shuffle_sync_down(s, off))
            v21 = cute.arch.shuffle_sync(s, 0)
            v25 = gt_sel(n8[u], v21)
            g = lds_v4(sw + Int32(u * NE * 4) + lane * 16)
            yy = p[u] if cutlass.const_expr(self.alias_y) else y[u]
            t = []
            for i in cutlass.range_constexpr(4):
                v28 = add_rn(v20[i], mul_rn(sgn_f32(p[u][i]), v25))
                v43 = add_rn(v28, add_rn(Float32(0.0), g[i]))
                t.append(mul_rn(v43, yy[i]))
            tmp.append(t)
            sts_v4(sw + Int32(u * NE * 4) + lane * 16, t)
        cute.arch.sync_warp()
        for u in cutlass.range_constexpr(RPW):
            yy = p[u] if cutlass.const_expr(self.alias_y) else y[u]
            s = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                s = add_rn(s, lds_f32(sw + Int32((u * NE + j * 32) * 4) + lane * 4))
            for off in (16, 8, 4, 2, 1):
                s = add_rn(s, cute.arch.shuffle_sync_bfly(s, off))
            h0 = []
            h1 = []
            h2 = []
            for i in cutlass.range_constexpr(4):
                if cutlass.const_expr(SMBWD_FMA):
                    gi = fma_rn(neg_f32(yy[i]), s, tmp[u][i])
                else:
                    gi = sub_rn(tmp[u][i], mul_rn(yy[i], s))
                b0 = bf16x2(gi, gi)
                r1 = sub_rn(gi, i2f(b0 << Int32(16)))
                b1 = bf16x2(r1, r1)
                r2 = sub_rn(r1, i2f(b1 << Int32(16)))
                h0.append(gi)
                h1.append(r1)
                h2.append(r2)
            ob = _addr(out0) + (rb + Int64(u)) * (3 * NE * 2) + Int64(lane) * 8
            st_v2_b32(ob, bf16x2(h0[0], h0[1]), bf16x2(h0[2], h0[3]))
            st_v2_b32(ob + NE * 2, bf16x2(h1[0], h1[1]), bf16x2(h1[2], h1[3]))
            st_v2_b32(ob + NE * 4, bf16x2(h2[0], h2[1]), bf16x2(h2[2], h2[3]))


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev, alias_y):
    key = (dev, alias_y)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            RouterGateBwd(alias_y),
            _fake(cutlass.Float32, NT * TOPK),  # in0
            _fake(cutlass.Uint8, NT, 4),  # in1
            _fake(cutlass.Float32, 1, 4),  # in3
            _fake(cutlass.Float32, NE),  # in4
            _fake(cutlass.Uint8, NT, 4),  # in5
            _fake(cutlass.Float32, NT * NE),  # in6
            _fake(cutlass.Float32, NT),  # in7[:, 0]
            _fake(cutlass.Float32, NT),  # in8
            _fake(cutlass.Float32, NT * TOPK),  # in9
            _fake(cutlass.Float32, NT, 4),  # in10
            _fake(cutlass.Int64, NT * TOPK, 8),  # in11
            _fake(cutlass.Float32, NT * NE),  # in12
            _fake(cutlass.BFloat16, NT * 3 * NE),  # out0
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    return fn


def _flat(t, align=16):
    t = t.contiguous()
    if t.data_ptr() % align:
        t = t.clone()
    return t.view(-1)


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
    dev = in0.device
    out0 = torch.empty((NT, 3 * NE), dtype=torch.bfloat16, device=dev)
    alias_y = in6.data_ptr() == in12.data_ptr() and in6.stride() == in12.stride()
    _compiled(dev.index, alias_y)(
        _flat(in0),
        _flat(in1.view(torch.uint8), 4),
        _flat(in3, 4),
        _flat(in4),
        _flat(in5.view(torch.uint8), 4),
        _flat(in6),
        _flat(in7[:, 0]),
        _flat(in8),
        _flat(in9),
        _flat(in10, 4),
        _flat(in11, 8),
        _flat(in12),
        out0.view(-1),
    )
    return (out0, out0[:, 0:NE].t(), out0[:, NE:2 * NE].t(), out0[:, 2 * NE:].t())
