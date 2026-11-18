"""ffn_norm forward (Qwen3 30B-A3B, hidden 2048): residual add + RMSNorm in one CuTe DSL kernel.

reference (M = 32768 rows, N = 2048):
  v2 = in0 + in1 (bf16); v5, rstd = _fused_rms_norm(v2, [2048], in2, 1e-6); returns (v2, v5, rstd)

One launch (aten: the add kernel + quack RMSNorm). One warp per row, ROWS rows per CTA; lane i owns the
8-column chunks at 8 i + 256 k, k = 0..7 (quack's RMSNorm layout for N = 2048: threads_per_row 32,
vecsize 8), so the sum of squares runs in quack's order.

Numerics (aten add / quack RMSNorm, matched bit for bit):
  v2 = bf16(f32(in0) + f32(in1)); s = fma(x, x, s) over the lane's 64 columns of bf16 v2 in column order
  from 0; butterfly shfl.xor 16..1; rstd = rsqrt.approx.ftz(s / 2048 + 1e-6); v5 = bf16((x * rstd) * w).
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
ROWS = 4  # rows (warps) per CTA
CHUNKS = N // 256


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
def sumsq4(s, w0, w1, w2, w3, *, loc=None, ip=None):
    """s + x0^2 + ... + x7^2 (fma chain in column order) over the bf16 pairs of 4 words."""
    text = (
        "{\n.reg .b32 a, s;\nmov.b32 s, $1;\n"
        + "".join(
            f"shl.b32 a, ${k + 2}, 16;\nfma.rn.f32 s, a, a, s;\nand.b32 a, ${k + 2}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
            for k in range(4)
        )
        + "mov.b32 $0, s;\n}"
    )
    return _i32(text, (s, w0, w1, w2, w3), loc, ip)


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def rstd_of(s, *, loc=None, ip=None):
    """rsqrt.approx.ftz(s / 2048 + 1e-6)."""
    text = "{\n.reg .b32 m;\ndiv.rn.f32 m, $1, 0f45000000;\nadd.rn.f32 m, m, 0f358637BD;\nrsqrt.approx.ftz.f32 $0, m;\n}"
    return _i32(text, (s,), loc, ip)


@dsl_user_op
def rms_word(x, g, r, *, loc=None, ip=None):
    """bf16x2 x, w -> bf16x2((x * rstd) * w)."""
    text = (
        "{\n.reg .b32 a0, a1, g0, g1;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
        "shl.b32 g0, $2, 16;\nand.b32 g1, $2, 0xFFFF0000;\n"
        "mul.rn.f32 a0, a0, $3;\nmul.rn.f32 a1, a1, $3;\n"
        "mul.rn.f32 a0, a0, g0;\nmul.rn.f32 a1, a1, g1;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return _i32(text, (x, g, r), loc, ip)


class AddNorm:
    @cute.jit
    def __call__(self, a, b, w, o2, o5, rstd, stream):
        self.kernel(a, b, w, o2, o5, rstd).launch(grid=[M // ROWS, 1, 1], block=[32 * ROWS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        o2: cute.Tensor,
        o5: cute.Tensor,
        rstd: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        row = bid * ROWS + warp
        base = row * N + lane * 8
        xa = []
        xb = []
        for k in cutlass.range_constexpr(CHUNKS):
            xa.append(ld_v4_nc(a.iterator + (base + 256 * k)))
            xb.append(ld_v4_nc(b.iterator + (base + 256 * k)))
        g = []
        for k in cutlass.range_constexpr(CHUNKS):
            g.append(ld_v4_nc(w.iterator + (lane * 8 + 256 * k)))
        v = []
        for k in cutlass.range_constexpr(CHUNKS):
            vk = [add_word(xa[k][e], xb[k][e]) for e in range(4)]
            st_v4(o2.iterator + (base + 256 * k), vk)
            v.append(vk)
        s = Int32(0)
        for k in cutlass.range_constexpr(CHUNKS):
            s = sumsq4(s, v[k][0], v[k][1], v[k][2], v[k][3])
        for off in cutlass.range_constexpr(5):
            s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
        r = rstd_of(s)
        for k in cutlass.range_constexpr(CHUNKS):
            st_v4(o5.iterator + (base + 256 * k), [rms_word(v[k][e], g[k][e], r) for e in range(4)])
        if lane == 0:
            st_b32(rstd.iterator + row, r)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        bf = cutlass.BFloat16
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = _CACHE[dev] = cute.compile(
            AddNorm(),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(bf, N),
            _fake(bf, M * N),
            _fake(bf, M * N),
            _fake(Int32, M, 4),
            stream,
            options="--enable-tvm-ffi",
        )
    return fn


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2):
    dev = in0.device
    k = _compiled(dev.index)
    o2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    o5 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((M, 1), dtype=torch.float32, device=dev)
    k(
        _aligned(in0).view(-1),
        _aligned(in1).view(-1),
        _aligned(in2).view(-1),
        o2.view(-1),
        o5.view(-1),
        rstd.view(torch.int32).view(-1),
    )
    return (o2, o5, rstd)
