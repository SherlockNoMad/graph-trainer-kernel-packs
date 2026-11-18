"""MoE combine + residual adds + RMSNorm (DeepSeek-V3 debug model, routed_experts forward), CuTe DSL.

reference (T = 16384 tokens, N = 49152 = 3 T routed rows, D = 256):
  v3 = bf16(f32(in0) * in1)                         routed outputs times router scores ([N, D])
  v6 = deterministic_scatter_add(in2, in3, v3)      out = in2; out[in3[n, 0]] += v3[n]
  v10 = bf16(in5 + bf16(v6 + in4)); v13 = _fused_rms_norm(v10, [256], in6, 1e-6)[0]
  returns (v10, v13)

Two launches (plus a 64 KiB memset of the per-token counters):
  K1, one thread per routed row n: slot = atomicAdd(count[in3[n, 0]], 1); rows[t, slot] = n.
  K2, one warp per token, lane i owns columns [8i, 8i + 8): sorts the token's rows ascending (so the
  atomic slot order does not matter), accumulates, adds, RMS-normalizes and writes v10 and v13.

Numerics (checked bit for bit against aten):
  scatter_add under deterministic mode is index_put_(accumulate=True) (_scatter_via_index_put), which
  sorts the indices stably (cub radix sort of (index, arange)) and runs indexing_backward_kernel: per
  duplicate in ascending n, weight = f32(out[t]) + f32(v3[n]) and out[t] = bf16(weight), so the
  destination is rounded to bf16 after every row.
  _fused_rms_norm dispatches to torch's vendored quack RMSNorm: for N = 256 bf16, threads_per_row 32,
  vecsize 8, lane i owns x[8i:8i+8]; lane sum (((0 + x0^2) + x1^2) + ...) in column order, butterfly
  shfl.xor 16, 8, 4, 2, 1, rstd = rsqrt.approx.ftz(s / 256 + eps), y = bf16((x * rstd) * w).

Assumes every token has at most MAXK routed rows (top-k routing without drops gives exactly k = 3).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
NR = 49152  # routed rows
D = 256
MAXK = 4  # row slots per token
TOK_PER_CTA = 4
THREADS = 128


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


_ACC = (
    "{\n.reg .b32 a0, a1, s, c0, c1, n;\n.reg .pred q;\n"
    "shl.b32 a0, $2, 16;\nand.b32 a1, $2, 0xFFFF0000;\n"
    "mul.rn.f32 a0, a0, $3;\nmul.rn.f32 a1, a1, $3;\n"
    "cvt.rn.bf16x2.f32 s, a1, a0;\n"
    "shl.b32 a0, s, 16;\nand.b32 a1, s, 0xFFFF0000;\n"
    "shl.b32 c0, $1, 16;\nand.b32 c1, $1, 0xFFFF0000;\n"
    "add.rn.f32 c0, c0, a0;\nadd.rn.f32 c1, c1, a1;\n"
    "cvt.rn.bf16x2.f32 n, c1, c0;\n"
    "setp.ne.s32 q, $4, 0;\n"
    "selp.b32 $0, n, $1, q;\n}"
)


@dsl_user_op
def acc_word(w, x, sc, valid, *, loc=None, ip=None):
    """w (bf16x2 destination) += bf16(f32(x) * sc), rounded to bf16, if valid; else w."""
    return _i32(_ACC, (w, x, sc, valid), loc, ip)


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
def sumsq8(w0, w1, w2, w3, *, loc=None, ip=None):
    text = (
        "{\n.reg .b32 a, s;\nmov.b32 s, 0f00000000;\n"
        + "".join(
            f"shl.b32 a, ${k + 1}, 16;\nfma.rn.f32 s, a, a, s;\nand.b32 a, ${k + 1}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
            for k in range(4)
        )
        + "mov.b32 $0, s;\n}"
    )
    return _i32(text, (w0, w1, w2, w3), loc, ip)


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return _i32("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def rstd_of(s, *, loc=None, ip=None):
    text = "{\n.reg .b32 m;\ndiv.rn.f32 m, $1, 0f43800000;\nadd.rn.f32 m, m, 0f358637BD;\nrsqrt.approx.ftz.f32 $0, m;\n}"
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


class Combine:
    @cute.jit
    def __call__(self, in0, in1, in2, in4, in5, in6, count, rows, out10, out13, stream):
        self.kernel(in0, in1, in2, in4, in5, in6, count, rows, out10, out13).launch(
            grid=[NT // TOK_PER_CTA, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        in0: cute.Tensor,
        in1: cute.Tensor,
        in2: cute.Tensor,
        in4: cute.Tensor,
        in5: cute.Tensor,
        in6: cute.Tensor,
        count: cute.Tensor,
        rows: cute.Tensor,
        out10: cute.Tensor,
        out13: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        t = bid * TOK_PER_CTA + tid // 32
        lane = tid % 32
        col = t * D + lane * 8
        c = ld_b32(count.iterator + t)
        pr = ld_v4(rows.iterator + t * MAXK)
        big = Int32(2147483647)
        p = sort4(
            sel_slot(pr[0], Int32(0), c, big),
            sel_slot(pr[1], Int32(1), c, big),
            sel_slot(pr[2], Int32(2), c, big),
            sel_slot(pr[3], Int32(3), c, big),
        )
        w = ld_v4_nc(in2.iterator + col)
        xs = []
        scs = []
        for k in cutlass.range_constexpr(MAXK):
            n = sel_slot(p[k], Int32(k), c, p[0])  # unused slots reread row p[0], result discarded
            xs.append(ld_v4_nc(in0.iterator + (n * D + lane * 8)))
            scs.append(ld_b32_nc(in1.iterator + n))
        a4 = ld_v4_nc(in4.iterator + col)
        a5 = ld_v4_nc(in5.iterator + col)
        g = ld_v4_nc(in6.iterator + lane * 8)
        for k in cutlass.range_constexpr(MAXK):
            valid = sel_slot(Int32(1), Int32(k), c, Int32(0))
            w = [acc_word(w[e], xs[k][e], scs[k], valid) for e in range(4)]
        v10 = [add_word(a5[e], add_word(w[e], a4[e])) for e in range(4)]
        st_v4(out10.iterator + col, v10)
        s = sumsq8(v10[0], v10[1], v10[2], v10[3])
        for off in cutlass.range_constexpr(5):
            s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
        r = rstd_of(s)
        st_v4(out13.iterator + col, [rms_word(v10[e], g[e], r) for e in range(4)])


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fns = _CACHE.get(dev)
    if fns is None:
        bf = cutlass.BFloat16
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        k1 = cute.compile(
            BuildRows(),
            _fake(Int32, NR * 2, 8),
            _fake(Int32, NT, 16),
            _fake(Int32, NT * MAXK, 16),
            stream,
            options="--enable-tvm-ffi",
        )
        k2 = cute.compile(
            Combine(),
            _fake(bf, NR * D),
            _fake(cutlass.Float32, NR, 4),
            _fake(bf, NT * D),
            _fake(bf, NT * D),
            _fake(bf, NT * D),
            _fake(bf, D),
            _fake(Int32, NT, 16),
            _fake(Int32, NT * MAXK, 16),
            _fake(bf, NT * D),
            _fake(bf, NT * D),
            stream,
            options="--enable-tvm-ffi",
        )
        fns = _CACHE[dev] = (k1, k2)
    return fns


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2, in3, in4, in5, in6):
    dev = in0.device
    k1, k2 = _compiled(dev.index)
    idx = in3.select(1, 0).contiguous()  # [N] int64 (in3 is expanded over D)
    count = torch.zeros(NT, dtype=torch.int32, device=dev)
    rows = torch.empty(NT * MAXK, dtype=torch.int32, device=dev)
    k1(idx.view(torch.int32), count, rows)
    out10 = torch.empty((NT, D), dtype=torch.bfloat16, device=dev)
    out13 = torch.empty((NT, D), dtype=torch.bfloat16, device=dev)
    k2(
        _aligned(in0).view(-1),
        in1.contiguous().view(-1),
        _aligned(in2).view(-1),
        _aligned(in4).view(-1),
        _aligned(in5).view(-1),
        _aligned(in6).view(-1),
        count,
        rows,
        out10.view(-1),
        out13.view(-1),
    )
    return (out10, out13)
