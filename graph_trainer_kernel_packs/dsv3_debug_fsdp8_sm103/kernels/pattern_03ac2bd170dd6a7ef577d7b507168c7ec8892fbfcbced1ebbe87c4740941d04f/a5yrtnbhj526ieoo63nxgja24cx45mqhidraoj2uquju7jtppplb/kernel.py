"""MoE router gate backward of the DeepSeek-V3 debug model (16384 tokens, top-3 of 8 experts), two CuTe DSL kernels.

reference (problem.py):
  v4[i] = sum_256(in0[i] * in1[i])              (49152 sorted slots; f32 * bf16 -> f32)
  v7 = zeros.index_put([in2], v4, accumulate)   (in2: the sort permutation, unique)
  v17 = v7 / in4 + sum_3(-v7 * in3)             (per token, 3 slots)
  v20 = zeros.scatter_add(-1, in5, v17); v24 = sigmoid_backward(in6 + v20, in7)
  out0 = [bf16(v24), bf16(r1), bf16(r2)] with r1 = v24 - f32(bf16(v24)), r2 = r1 - f32(bf16(r1)); out1..3 its column views.

Kernel 1: one warp per slot row, in aten's reduce order for the 256-wide sum (vectorized input: lane x holds
elements 4x + i and 128 + 4x + i in 4 accumulators, combined 0+1+2+3, then shfl_down 16, 8, 4, 2, 1), and the
index_put write v7[in2[i]] = 0 + (0 + v4[i]) (aten's deterministic sort-based index_put with unique indices).
Kernel 2 (programmatic dependent launch): one thread per token for the rest. The 3-sum is (a + c) + b (aten
reduce, block.x = 2). scatter_add under deterministic mode is index_put(accumulate) on the flattened index:
per expert 0 + ((0 + v_j1) + v_j2 ...) over matching slots j in order, which is also exact with duplicates.
All float ops are explicit .rn PTX.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


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


NT = 16384
NS = NT * 3
H = 256
K1_ROWS_PER_WARP = 4
K1_THREADS = 256
K1_CTAS = NS // (K1_THREADS // 32 * K1_ROWS_PER_WARP)
K2_THREADS = 256
K2_CTAS = NT // K2_THREADS

sub_rn = _binf("sub.rn.f32")


@dsl_user_op
def neg_f32(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], "neg.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def ld_v2_b32(addr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(2)]


@dsl_user_op
def ld_nc_b64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.nc.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_nc_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    """Two floats -> packed bf16 (round to nearest even), lo in the low half."""
    return Int32(_asm(T.i32(), [_v(Float32(hi), loc, ip), _v(Float32(lo), loc, ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;",
                      "=r,f,f", loc=loc, ip=ip))


def _bf_lo(w):
    return i2f(w << Int32(16))


def _bf_hi(w):
    return i2f(w & Int32(-65536))


class RouterGateBwd:
    @cute.jit
    def __call__(self, in0, in1, in2, in3, in4, in5, in6, in7, out0, scratch, stream):
        self.k1(in0, in1, in2, scratch).launch(grid=[K1_CTAS, 1, 1], block=[K1_THREADS, 1, 1], stream=stream)
        self.k2(in3, in4, in5, in6, in7, out0, scratch).launch(
            grid=[K2_CTAS, 1, 1], block=[K2_THREADS, 1, 1], use_pdl=True, stream=stream
        )

    @cute.kernel
    def k1(self, in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        lane = tid % 32
        wid = tid // 32
        loads = []
        for u in cutlass.range_constexpr(K1_ROWS_PER_WARP):
            row = Int64((bid * (K1_THREADS // 32) + wid) * K1_ROWS_PER_WARP + u)
            a0 = _addr(in0) + row * (H * 4) + Int64(lane) * 16
            b0 = _addr(in1) + row * (H * 2) + Int64(lane) * 8
            loads.append((ld_v4_f32(a0), ld_v4_f32(a0 + 512), ld_v2_b32(b0), ld_v2_b32(b0 + 256), ld_nc_b64(_addr(in2) + row * 8)))
        for x0, x1, y0, y1, dst in loads:
            f0 = [_bf_lo(y0[0]), _bf_hi(y0[0]), _bf_lo(y0[1]), _bf_hi(y0[1])]
            f1 = [_bf_lo(y1[0]), _bf_hi(y1[0]), _bf_lo(y1[1]), _bf_hi(y1[1])]
            p = [add_rn(add_rn(Float32(0.0), mul_rn(x0[i], f0[i])), mul_rn(x1[i], f1[i])) for i in range(4)]
            q = add_rn(add_rn(add_rn(p[0], p[1]), p[2]), p[3])
            for off in (16, 8, 4, 2, 1):
                q = add_rn(q, cute.arch.shuffle_sync_down(q, off))
            if lane == 0:
                st_f32(_addr(scratch) + dst * 4, add_rn(Float32(0.0), add_rn(Float32(0.0), q)))

    @cute.kernel
    def k2(self, in3: cute.Tensor, in4: cute.Tensor, in5: cute.Tensor, in6: cute.Tensor, in7: cute.Tensor,
           out0: cute.Tensor, scratch: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        t = Int64(bid * K2_THREADS + tid)
        x3 = [ld_nc_f32(_addr(in3) + t * 12 + 4 * j) for j in range(3)]
        x4 = ld_nc_f32(_addr(in4) + t * 4)
        x5 = [ld_nc_b64(_addr(in5) + t * 24 + 8 * j) for j in range(3)]
        x6 = ld_v4_f32(_addr(in6) + t * 32) + ld_v4_f32(_addr(in6) + t * 32 + 16)
        x7 = ld_v4_f32(_addr(in7) + t * 32) + ld_v4_f32(_addr(in7) + t * 32 + 16)
        cute.arch.griddepcontrol_wait()
        v9 = [ld_f32(_addr(scratch) + t * 12 + 4 * j) for j in range(3)]
        v12 = [mul_rn(neg_f32(v9[j]), x3[j]) for j in range(3)]
        v15 = add_rn(add_rn(v12[0], v12[2]), v12[1])
        v17 = [add_rn(div_rn(v9[j], x4), v15) for j in range(3)]
        h = [[], [], []]
        for e in cutlass.range_constexpr(8):
            g = Float32(0.0)
            for j in cutlass.range_constexpr(3):
                g = _sel(x5[j] == Int64(e), add_rn(g, v17[j]), g)
            v22 = add_rn(x6[e], add_rn(Float32(0.0), g))
            v24 = mul_rn(mul_rn(v22, sub_rn(Float32(1.0), x7[e])), x7[e])
            h[0].append(v24)
        for e in cutlass.range_constexpr(8):
            v24 = h[0][e]
            r1 = sub_rn(v24, _bf_lo(bf16x2(v24, v24)))
            r2 = sub_rn(r1, _bf_lo(bf16x2(r1, r1)))
            h[1].append(r1)
            h[2].append(r2)
        for s in cutlass.range_constexpr(3):
            w = [bf16x2(h[s][2 * q], h[s][2 * q + 1]) for q in range(4)]
            st_v4_b32(_addr(out0) + t * 48 + 16 * s, w)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            RouterGateBwd(),
            _fake(cutlass.Float32, NS * H),  # in0
            _fake(cutlass.BFloat16, NS * H),  # in1
            _fake(cutlass.Int64, NS, 8),  # in2
            _fake(cutlass.Float32, NT * 3, 4),  # in3
            _fake(cutlass.Float32, NT, 4),  # in4
            _fake(cutlass.Int64, NT * 3, 8),  # in5
            _fake(cutlass.Float32, NT * 8),  # in6
            _fake(cutlass.Float32, NT * 8),  # in7
            _fake(cutlass.BFloat16, NT * 24),  # out0
            _fake(cutlass.Float32, NS, 4),  # scratch (v7)
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned16(t):
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def kernel_function(in0, in1, in2, in3, in4, in5, in6, in7):
    dev = in0.device
    out0 = torch.empty((NT, 24), dtype=torch.bfloat16, device=dev)
    scratch = torch.empty((NS,), dtype=torch.float32, device=dev)
    _compiled(dev.index)(
        _aligned16(in0).view(-1),
        _aligned16(in1).view(-1),
        in2.contiguous(),
        in3.contiguous().view(-1),
        in4.contiguous().view(-1),
        in5.contiguous().view(-1),
        _aligned16(in6).view(-1),
        _aligned16(in7).view(-1),
        out0.view(-1),
        scratch,
    )
    return (out0, out0[:, 0:8].t(), out0[:, 8:16].t(), out0[:, 16:24].t())
