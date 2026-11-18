"""Forward of one chunk of Qwen3's chunked cross-entropy loss (4096 tokens, vocab 151936), two CuTe DSL kernels.

reference (problem.py; x = in0 bf16 logits [4096, V], tgt = in1 int64 [4096], in2 int64 token count,
in3 f32 running loss sum):
  lp = log_softmax(f32(x), 1)                (returned twice: lp and alias(lp), same storage)
  loss, tw = nll_loss_forward(lp, tgt, None, sum reduction, ignore_index -100)
  acc = in3 + loss / in2
  r006 returns (lp, alias(lp), tw, acc, ones_like(loss)); r005 (the last chunk) returns
  (lp, alias(lp), tw, ones_like(loss), clone(acc)).  _VARIANT selects the output order.

Numerics are those of the verified 8-chunk kernel of the old region r004 (q-writer-loss/t1, verified by
q-verifier-r004), applied to one chunk.

Kernel 1 (one CTA of 1024 threads per row, 4096 CTAs) reproduces aten's cunn_SoftMaxForward<ILP=4, float>
(block 1024, input 16 B aligned so shift = 0) bit for bit; physical thread v is aten thread v:
  aten thread v owns elements 4(v + 1024k) + j (k < 37, j < 4, in that order) and the tail element
  151552 + v (v < 384).
  The row is read from global memory once and stays on chip: steps k < 10 in registers (8 B loads),
  steps 10..36 (columns [40960, 151552), contiguous) in shared memory by 4 TMA bulk copies on an
  mbarrier (216 KB, so one CTA per SM).
  max: per-thread fmaxf from -FLT_MAX (order free), block max (the sign of a zero max is unobservable:
  a zero max means a row sum >= 2 when zeros of both signs are present, so out = d - logsum either way).
  sum: acc = fma(ex2(f), 2^j, acc) with libdevice expf's range reduction of (x - max) (nvcc contracts
  expf's final multiply into the add: FFMA in aten's SASS), elements in aten's order into one accumulator;
  the exp pieces of element pairs run as f32x2 FFMA2 / FADD2 (each lane rounds like the scalar op; no
  .ftz; fma.rn.sat has no f32x2 form and stays scalar). aten's BlockReduce order: shfl_down 16, 8, 4, 2, 1
  in each warp, warp sums to smem, warp 0 shfl_down 16..1 over the 32 warp sums.
  out = (x - max) - logf(sum) (FADD2 pairs), logf = libdevice __nv_logf PTX (copied from nvcc 13 output);
  output stores are L2 evict_first.
  Thread 0 writes lp[row, tgt[row]] (the same expression) to a scratch vector (rows with tgt == -100
  are skipped; kernel 2 never uses them).
Kernel 2 (one CTA, 256 threads = aten nll_loss_threads(4096)): aten nll_loss_forward_reduce_cuda_kernel_2d:
  thread i sums acc -= lp_t over rows i, i+256, ... (skipping -100), shared-memory tree 128..1; then
  loss / f32(in2) (div.rn), in3 + that (add.rn), tw, ones.
Labels other than -100 must be in [0, V) (aten device-asserts that; this kernel does not check it).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 4096
V = 151936
THREADS = 512
FULL = 37  # 16 B steps per physical thread
TAIL_BASE = 4 * 1024 * FULL  # 151552
TAIL = V - TAIL_BASE  # 384 aten threads -> 192 physical threads
assert TAIL_BASE + TAIL == V and TAIL % 2 == 0
IGNORE_INDEX = -100
_VARIANT = "r005"  # output order: "r006" or "r005"
NLL_THREADS = 256
NLL_ITERS = ROWS // NLL_THREADS


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


def _struct(n, ty):
    return ir.Type.parse("!llvm.struct<(" + ", ".join([ty] * n) + ")>")


def _unpack(res, n, cls, ty, loc, ip):
    return [cls(llvm.extractvalue(ty, res, [k], loc=loc, ip=ip)) for k in range(n)]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    res = _asm(
        _struct(4, "i32"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_u16(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.nc.u16 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(addr), loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.nc.s64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def st_f32(addr, x, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Float32(x), loc, ip)], "st.global.f32 [$0], $1;", "l,f", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_v4_f32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
        "st.global.cs.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_v2_f32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
        "st.global.cs.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        side=True,
        loc=loc,
        ip=ip,
    )


def _binf(op):
    @dsl_user_op
    def f(a, b, *, loc=None, ip=None):
        return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], f"{op} $0, $1, $2;", "=f,f,f", loc=loc, ip=ip))

    return f


add_rn = _binf("add.rn.f32")
sub_rn = _binf("sub.rn.f32")
max_f32 = _binf("max.f32")
div_rn = _binf("div.rn.f32")


@dsl_user_op
def max_combine(a, b, *, loc=None, ip=None):
    """aten Max<float>::combine: a < b ? b : a."""
    code = "{\n.reg .pred p;\nsetp.lt.f32 p, $1, $2;\nselp.f32 $0, $2, $1, p;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], code, "=f,f,f", loc=loc, ip=ip))


@dsl_user_op
def bf16_lo_f32(w, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(w), loc, ip)], "shl.b32 $0, $1, 16;", "=f,r", loc=loc, ip=ip))


def _unpack_code(src):
    return f"shl.b32 xl, {src}, 16;\nand.b32 xh, {src}, 0xFFFF0000;\n"


_MAX8 = (
    "{\n.reg .b32 xl, xh;\n"
    + "".join(_unpack_code(f"${2 + w}") + "max.f32 $0, $0, xl;\nmax.f32 $0, $0, xh;\n" for w in range(4))
    + "}"
)


@dsl_user_op
def max8(m, w, *, loc=None, ip=None):
    """fmaxf of m with the 8 bf16 of w (4 words)."""
    return Float32(
        _asm(T.f32(), [_v(Float32(m), loc, ip)] + [_v(Int32(x), loc, ip) for x in w], _MAX8.replace("$0, $0", "$0, $0"), "=f,0,r,r,r,r", loc=loc, ip=ip)
    )


_MAX2 = "{\n.reg .b32 xl, xh;\n" + _unpack_code("$2") + "max.f32 $0, $0, xl;\nmax.f32 $0, $0, xh;\n}"


@dsl_user_op
def max2(m, w, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(m), loc, ip), _v(Int32(w), loc, ip)], _MAX2, "=f,0,r", loc=loc, ip=ip))


def _sumexp_elem(x, acc, mx):
    """acc = fma(ex2(f), 2^j, acc) for expf(x - max) (aten SumExpFloat as compiled)."""
    return (
        f"sub.rn.f32 d, {x}, {mx};\n"
        "fma.rn.sat.f32 t, d, 0f3BBB989D, 0f3F000000;\n"
        "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
        "sub.rn.f32 f, 0f4B40007F, j;\n"
        "fma.rn.f32 f, d, 0f3FB8AA3B, f;\n"
        "fma.rn.f32 f, d, 0f32A57060, f;\n"
        "mov.b32 jb, j;\nshl.b32 jb, jb, 23;\nmov.b32 sc, jb;\n"
        "ex2.approx.ftz.f32 e, f;\n"
        f"fma.rn.f32 {acc}, e, sc, {acc};\n"
    )


_SUMEXP_REGS = "{\n.reg .b32 xl, xh, jb;\n.reg .f32 d, t, j, f, sc, e;\n"

# $0 = a0, $1 = a1 (tied to $2, $3), words $4..$7, max $8
_SUMEXP8 = (
    _SUMEXP_REGS
    + "".join(
        _unpack_code(f"${4 + w}") + _sumexp_elem("xl", f"${w // 2}", "$8") + _sumexp_elem("xh", f"${w // 2}", "$8")
        for w in range(4)
    )
    + "}"
)


@dsl_user_op
def sumexp8(a0, a1, w, mx, *, loc=None, ip=None):
    res = _asm(
        _struct(2, "f32"),
        [_v(Float32(a0), loc, ip), _v(Float32(a1), loc, ip)] + [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip)],
        _SUMEXP8,
        "=f,=f,0,1,r,r,r,r,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


# $0 = a0, $1 = a1 (tied), word $4 (lo -> a0, hi -> a1), max $5
_SUMEXP2 = _SUMEXP_REGS + _unpack_code("$4") + _sumexp_elem("xl", "$0", "$5") + _sumexp_elem("xh", "$1", "$5") + "}"


@dsl_user_op
def sumexp2(a0, a1, w, mx, *, loc=None, ip=None):
    res = _asm(
        _struct(2, "f32"),
        [_v(Float32(a0), loc, ip), _v(Float32(a1), loc, ip), _v(Int32(w), loc, ip), _v(Float32(mx), loc, ip)],
        _SUMEXP2,
        "=f,=f,0,1,r,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


def _epi_code(nw):
    # outputs $0..$(2nw-1); words $(2nw)..; max, logsum after them
    mx, ls = f"${3 * nw}", f"${3 * nw + 1}"
    code = "{\n.reg .b32 xl, xh;\n.reg .f32 d;\n"
    for w in range(nw):
        code += _unpack_code(f"${2 * nw + w}")
        code += f"sub.rn.f32 d, xl, {mx};\nsub.rn.f32 ${2 * w}, d, {ls};\n"
        code += f"sub.rn.f32 d, xh, {mx};\nsub.rn.f32 ${2 * w + 1}, d, {ls};\n"
    return code + "}"


_EPI8 = _epi_code(4)
_EPI2 = _epi_code(1)


@dsl_user_op
def epi(w, mx, ls, *, loc=None, ip=None):
    nw = len(w)
    res = _asm(
        _struct(2 * nw, "f32"),
        [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip), _v(Float32(ls), loc, ip)],
        _EPI8 if nw == 4 else _EPI2,
        ",".join(["=f"] * (2 * nw) + ["r"] * nw + ["f", "f"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2 * nw, Float32, T.f32(), loc, ip)


# libdevice __nv_logf (non-FTZ) as nvcc 13 emits it
_LOGF = (
    "{\n.reg .pred p1, p2, p3;\n.reg .b32 r2, r3, r4, r5;\n.reg .f32 f2, f3, f4, f5, f6, f7, f8, f9, f17, f18, f19, f20, f21;\n"
    "setp.lt.f32 p1, $1, 0f00800000;\n"
    "mul.rn.f32 f2, $1, 0f4B000000;\n"
    "selp.f32 f3, f2, $1, p1;\n"
    "selp.f32 f4, 0fC1B80000, 0f00000000, p1;\n"
    "mov.b32 r2, f3;\n"
    "add.s32 r3, r2, -1059760811;\n"
    "and.b32 r4, r3, -8388608;\n"
    "sub.s32 r5, r2, r4;\n"
    "mov.b32 f5, r5;\n"
    "cvt.rn.f32.s32 f6, r4;\n"
    "fma.rn.f32 f7, f6, 0f34000000, f4;\n"
    "add.rn.f32 f8, f5, 0fBF800000;\n"
    "fma.rn.f32 f9, f8, 0fBE055027, 0f3E1039F6;\n"
    "fma.rn.f32 f9, f9, f8, 0fBDF8CDCC;\n"
    "fma.rn.f32 f9, f9, f8, 0f3E0F2955;\n"
    "fma.rn.f32 f9, f9, f8, 0fBE2AD8B9;\n"
    "fma.rn.f32 f9, f9, f8, 0f3E4CED0B;\n"
    "fma.rn.f32 f9, f9, f8, 0fBE7FFF22;\n"
    "fma.rn.f32 f9, f9, f8, 0f3EAAAA78;\n"
    "fma.rn.f32 f9, f9, f8, 0fBF000000;\n"
    "mul.rn.f32 f17, f8, f9;\n"
    "fma.rn.f32 f18, f17, f8, f8;\n"
    "fma.rn.f32 f19, f7, 0f3F317218, f18;\n"
    "setp.gt.u32 p2, r2, 2139095039;\n"
    "fma.rn.f32 f20, f3, 0f7F800000, 0f7F800000;\n"
    "selp.f32 f21, f20, f19, p2;\n"
    "setp.eq.f32 p3, f3, 0f00000000;\n"
    "selp.f32 $0, 0fFF800000, f21, p3;\n}"
)


@dsl_user_op
def logf_(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(x), loc, ip)], _LOGF, "=f,f", loc=loc, ip=ip))


@dsl_user_op
def nll_step(acc, w, t, x, *, loc=None, ip=None):
    """t != -100: acc - x, w + 1."""
    code = (
        "{\n.reg .pred p;\n.reg .f32 a, b;\n"
        f"setp.ne.s64 p, $6, {IGNORE_INDEX};\n"
        "sub.rn.f32 a, $2, $5;\nadd.rn.f32 b, $3, 0f3F800000;\n"
        "selp.f32 $0, a, $2, p;\nselp.f32 $1, b, $3, p;\n}"
    )
    res = _asm(
        _struct(2, "f32"),
        [_v(Float32(acc), loc, ip), _v(Float32(w), loc, ip), _v(Int32(0), loc, ip), _v(Float32(x), loc, ip), _v(Int64(t), loc, ip)],
        code.replace("$5", "$5").replace("$6", "$6"),
        "=f,=f,f,f,r,f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


@dsl_user_op
def cvt_f32_s64(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int64(a), loc, ip)], "cvt.rn.f32.s64 $0, $1;", "=f,l", loc=loc, ip=ip))


# Packed (f32x2) forms of the same IEEE ops: FFMA2 / FADD2 round each lane like the scalar instruction
# (no .ftz; .rn / .rm as in the scalar code). fma.rn.sat has no f32x2 form and stays scalar.
_PK_REGS = (
    "{\n.reg .b32 xp, xq, dp, dq, tp, tq, jp, jq, fp, fq, ep, eq, c0, c1;\n"
    ".reg .b64 X, D, TT, J, F, E, SC, A, MX, K252, KMAG, K7F, KL2E, KLO;\n"
)


def _pk_consts(mx):
    return (
        f"mov.b64 MX, {{{mx}, {mx}}};\n"
        "mov.b32 c0, 0f437C0000;\nmov.b64 K252, {c0, c0};\n"
        "mov.b32 c0, 0f4B400001;\nmov.b64 KMAG, {c0, c0};\n"
        "mov.b32 c0, 0f4B40007F;\nmov.b64 K7F, {c0, c0};\n"
        "mov.b32 c0, 0f3FB8AA3B;\nmov.b64 KL2E, {c0, c0};\n"
        "mov.b32 c0, 0f32A57060;\nmov.b64 KLO, {c0, c0};\n"
    )


def _pk_pair(xp_code, xq_code):
    """A = fma(ex2(f), 2^j, A) for the element pair (xp -> a0, xq -> a1)."""
    return (
        xp_code + xq_code
        + "mov.b64 X, {xp, xq};\n"
        "sub.rn.f32x2 D, X, MX;\n"
        "mov.b64 {dp, dq}, D;\n"
        "fma.rn.sat.f32 tp, dp, 0f3BBB989D, 0f3F000000;\n"
        "fma.rn.sat.f32 tq, dq, 0f3BBB989D, 0f3F000000;\n"
        "mov.b64 TT, {tp, tq};\n"
        "fma.rm.f32x2 J, TT, K252, KMAG;\n"
        "sub.rn.f32x2 F, K7F, J;\n"
        "fma.rn.f32x2 F, D, KL2E, F;\n"
        "fma.rn.f32x2 F, D, KLO, F;\n"
        "mov.b64 {jp, jq}, J;\nshl.b32 jp, jp, 23;\nshl.b32 jq, jq, 23;\nmov.b64 SC, {jp, jq};\n"
        "mov.b64 {fp, fq}, F;\n"
        "ex2.approx.ftz.f32 ep, fp;\nex2.approx.ftz.f32 eq, fq;\n"
        "mov.b64 E, {ep, eq};\n"
        "fma.rn.f32x2 A, E, SC, A;\n"
    )


def _lo(dst, src):
    return f"shl.b32 {dst}, {src}, 16;\n"


def _hi(dst, src):
    return f"and.b32 {dst}, {src}, 0xFFFF0000;\n"


# $0 = a0, $1 = a1 (tied to $2, $3), words $4..$7 (words 0, 1 -> a0; 2, 3 -> a1), max $8
_SUMEXP8P = (
    _PK_REGS + _pk_consts("$8") + "mov.b64 A, {$2, $3};\n"
    + _pk_pair(_lo("xp", "$4"), _lo("xq", "$6"))
    + _pk_pair(_hi("xp", "$4"), _hi("xq", "$6"))
    + _pk_pair(_lo("xp", "$5"), _lo("xq", "$7"))
    + _pk_pair(_hi("xp", "$5"), _hi("xq", "$7"))
    + "mov.b64 {$0, $1}, A;\n}"
)


@dsl_user_op
def sumexp8p(a0, a1, w, mx, *, loc=None, ip=None):
    res = _asm(
        _struct(2, "f32"),
        [_v(Float32(a0), loc, ip), _v(Float32(a1), loc, ip)] + [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip)],
        _SUMEXP8P,
        "=f,=f,0,1,r,r,r,r,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


def _epip_code(nw):
    # outputs $0..$(2nw-1); words $(2nw)..; max, logsum after them
    mx, ls = f"${3 * nw}", f"${3 * nw + 1}"
    code = (
        "{\n.reg .b32 xl, xh;\n.reg .b64 X, D, O, MX, LS;\n"
        f"mov.b64 MX, {{{mx}, {mx}}};\nmov.b64 LS, {{{ls}, {ls}}};\n"
    )
    for w in range(nw):
        code += _unpack_code(f"${2 * nw + w}")
        code += "mov.b64 X, {xl, xh};\nsub.rn.f32x2 D, X, MX;\nsub.rn.f32x2 O, D, LS;\n"
        code += f"mov.b64 {{${2 * w}, ${2 * w + 1}}}, O;\n"
    return code + "}"


_EPI8P = _epip_code(4)


@dsl_user_op
def epi8p(w, mx, ls, *, loc=None, ip=None):
    res = _asm(
        _struct(8, "f32"),
        [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip), _v(Float32(ls), loc, ip)],
        _EPI8P,
        ",".join(["=f"] * 8 + ["r"] * 4 + ["f", "f"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 8, Float32, T.f32(), loc, ip)


@dsl_user_op
def ld_v2(addr, *, loc=None, ip=None):
    res = _asm(_struct(2, "i32"), [_v(Int64(addr), loc, ip)], "ld.global.nc.L1::no_allocate.v2.b32 {$0, $1}, [$2];", "=r,=r,l", loc=loc, ip=ip)
    return _unpack(res, 2, Int32, T.i32(), loc, ip)


@dsl_user_op
def ld_shared_v2(addr, *, loc=None, ip=None):
    res = _asm(_struct(2, "i32"), [_v(Int32(addr), loc, ip)], "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r", side=True, loc=loc, ip=ip)
    return _unpack(res, 2, Int32, T.i32(), loc, ip)


@dsl_user_op
def mbar_arm(mbar, *, loc=None, ip=None):
    """Thread 0: init the mbarrier (count 1) and arm it for SMEM_BYTES_DATA bytes of bulk copies."""
    code = (
        "mbarrier.init.shared::cta.b64 [$0], 1;\n"
        "fence.mbarrier_init.release.cluster;\n"
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {SMEM_BYTES_DATA};\n"
    )
    _asm(None, [_v(Int32(mbar), loc, ip)], code, "r", side=True, loc=loc, ip=ip)


@dsl_user_op
def bulk_g2s(sdst, gsrc, mbar, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(sdst), loc, ip), _v(Int64(gsrc), loc, ip), _v(Int32(mbar), loc, ip)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {COPY_BYTES}, [$2];",
        "r,l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait0(mbar, *, loc=None, ip=None):
    code = (
        "{\n.reg .pred p;\nWAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], 0;\n"
        "@!p bra WAIT_${:uid};\n}"
    )
    _asm(None, [_v(Int32(mbar), loc, ip)], code, "r", side=True, loc=loc, ip=ip)


@dsl_user_op
def prefetch_l2_bulk(gaddr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(gaddr), loc, ip), _v(Int32(nbytes), loc, ip)],
        "{\n.reg .b64 pol;\ncreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, pol;\n}",
        "l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_v4_f32_ef(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Float32(x), loc, ip) for x in w],
        "{\n.reg .b64 pol;\ncreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
        "st.global.L1::no_allocate.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, pol;\n}",
        "l,f,f,f,f",
        side=True,
        loc=loc,
        ip=ip,
    )


# one accumulator, 4 elements (words $2, $3) in order; exp pieces computed pairwise (j0, j1), (j2, j3)
def _pk_pair1(xp_code, xq_code):
    return (
        xp_code + xq_code
        + "mov.b64 X, {xp, xq};\n"
        "sub.rn.f32x2 D, X, MX;\n"
        "mov.b64 {dp, dq}, D;\n"
        "fma.rn.sat.f32 tp, dp, 0f3BBB989D, 0f3F000000;\n"
        "fma.rn.sat.f32 tq, dq, 0f3BBB989D, 0f3F000000;\n"
        "mov.b64 TT, {tp, tq};\n"
        "fma.rm.f32x2 J, TT, K252, KMAG;\n"
        "sub.rn.f32x2 F, K7F, J;\n"
        "fma.rn.f32x2 F, D, KL2E, F;\n"
        "fma.rn.f32x2 F, D, KLO, F;\n"
        "mov.b64 {jp, jq}, J;\nshl.b32 jp, jp, 23;\nshl.b32 jq, jq, 23;\n"
        "mov.b64 {fp, fq}, F;\n"
        "ex2.approx.ftz.f32 ep, fp;\nex2.approx.ftz.f32 eq, fq;\n"
        "fma.rn.f32 $0, ep, jp, $0;\n"
        "fma.rn.f32 $0, eq, jq, $0;\n"
    )


# $0 = acc (tied to $1), words $2, $3, max $4
_SUMEXP4P = (
    _PK_REGS + _pk_consts("$4")
    + _pk_pair1(_lo("xp", "$2"), _hi("xq", "$2"))
    + _pk_pair1(_lo("xp", "$3"), _hi("xq", "$3"))
    + "}"
)


@dsl_user_op
def sumexp4p(a, w, mx, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip)] + [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip)],
            _SUMEXP4P,
            "=f,0,r,r,f",
            loc=loc,
            ip=ip,
        )
    )


# tail element (bf16 in the low half of $1) into acc $0 (tied $2), max $3
_SUMEXP1 = _SUMEXP_REGS + "shl.b32 xl, $1, 16;\n" + _sumexp_elem("xl", "$0", "$3") + "}"


@dsl_user_op
def sumexp1(a, w, mx, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(w), loc, ip), _v(Float32(a), loc, ip), _v(Float32(mx), loc, ip)], _SUMEXP1, "=f,r,0,f", loc=loc, ip=ip))


_EPI4P = _epip_code(2)


@dsl_user_op
def epi4p(w, mx, ls, *, loc=None, ip=None):
    res = _asm(
        _struct(4, "f32"),
        [_v(Int32(x), loc, ip) for x in w] + [_v(Float32(mx), loc, ip), _v(Float32(ls), loc, ip)],
        _EPI4P,
        ",".join(["=f"] * 4 + ["r"] * 2 + ["f", "f"]),
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 4, Float32, T.f32(), loc, ip)


@dsl_user_op
def epi1(w, mx, ls, *, loc=None, ip=None):
    code = "{\n.reg .b32 xl;\n.reg .f32 d;\nshl.b32 xl, $1, 16;\nsub.rn.f32 d, xl, $2;\nsub.rn.f32 $0, d, $3;\n}"
    return Float32(_asm(T.f32(), [_v(Int32(w), loc, ip), _v(Float32(mx), loc, ip), _v(Float32(ls), loc, ip)], code, "=f,r,f,f", loc=loc, ip=ip))


@dsl_user_op
def max4(m, w, *, loc=None, ip=None):
    """fmaxf of m with the 4 bf16 of w (2 words)."""
    code = "{\n.reg .b32 xl, xh;\n" + "".join(_unpack_code(f"${2 + k}") + "max.f32 $0, $0, xl;\nmax.f32 $0, $0, xh;\n" for k in range(2)) + "}"
    return Float32(_asm(T.f32(), [_v(Float32(m), loc, ip)] + [_v(Int32(x), loc, ip) for x in w], code, "=f,0,r,r", loc=loc, ip=ip))


@dsl_user_op
def max1(m, w, *, loc=None, ip=None):
    code = "{\n.reg .b32 xl;\nshl.b32 xl, $2, 16;\nmax.f32 $0, $0, xl;\n}"
    return Float32(_asm(T.f32(), [_v(Float32(m), loc, ip), _v(Int32(w), loc, ip)], code, "=f,0,r", loc=loc, ip=ip))


THREADS1 = 1024  # = aten's block: physical thread v is aten thread v
REG_STEPS = 10  # 8 B steps per thread held in registers (aten steps 0..9)
SMEM_STEPS = FULL - REG_STEPS  # aten steps 10..36: columns [40960, 151552), one contiguous range, by TMA
SMEM_BYTES_DATA = SMEM_STEPS * THREADS1 * 8  # 221184
NCOPY = 4
COPY_BYTES = SMEM_BYTES_DATA // NCOPY
assert COPY_BYTES * NCOPY == SMEM_BYTES_DATA and COPY_BYTES % 16 == 0
SMEM_BYTES = SMEM_BYTES_DATA + 1024
ROW_BYTES = V * 2
PF_CHUNK = ROW_BYTES // 16  # next-row L2 prefetch: 16 pieces of 18992 B
assert PF_CHUNK * 16 == ROW_BYTES and PF_CHUNK % 16 == 0
PREFETCH = False


class LogSoftmaxRows:
    """One CTA of 1024 threads per row (aten's thread layout); the whole row stays on chip: aten steps 0..9
    in registers, steps 10..36 in shared memory (TMA bulk loads, so one CTA per SM). PREFETCH (off: measured
    slower) prefetches row + resident_ctas into L2 (evict_last)."""

    def __init__(self, resident):
        self.resident = resident

    @cute.jit
    def __call__(self, x, o, tg, scratch, stream):
        self.kernel(x, o, tg, scratch).launch(grid=[ROWS, 1, 1], block=[THREADS1, 1, 1], smem=SMEM_BYTES, stream=stream)

    @cute.kernel
    def kernel(self, x, o, tg, scratch):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        xa = x.iterator.toint() + Int64(row) * ROW_BYTES
        oa = o.iterator.toint() + Int64(row) * (V * 4)
        ta = tg.iterator.toint() + Int64(row) * 8

        smem = cutlass.utils.SmemAllocator()
        scache = smem.allocate_tensor(Int32, cute.make_layout(SMEM_BYTES_DATA // 4), 128)
        smax = smem.allocate_tensor(Float32, cute.make_layout(32), 16)
        ssum = smem.allocate_tensor(Float32, cute.make_layout(32), 16)
        sbc = smem.allocate_tensor(Float32, cute.make_layout(4), 16)
        smb = smem.allocate_tensor(Int64, cute.make_layout(2), 16)
        sdata = Int32(scache.iterator.toint())
        mbar = Int32(smb.iterator.toint())
        sbase = sdata + tid * 8

        if tid == 0:
            mbar_arm(mbar)
            for c in cutlass.range_constexpr(NCOPY):
                bulk_g2s(sdata + c * COPY_BYTES, xa + (REG_STEPS * THREADS1 * 8 + c * COPY_BYTES), mbar)
        regs = []
        for k in cutlass.range_constexpr(REG_STEPS):
            regs.append(ld_v2(xa + Int64(tid + k * THREADS1) * 8))
        tw = ld_u16(xa + (TAIL_BASE * 2) + Int64(tid % TAIL) * 2)
        if cutlass.const_expr(PREFETCH):
            if lane == 0 and warp < 16:
                if row + self.resident < ROWS:
                    prefetch_l2_bulk(xa + Int64(self.resident) * ROW_BYTES + warp * PF_CHUNK, PF_CHUNK)

        # pass 1: max
        m = Float32(-3.40282346638528859812e38)
        for k in cutlass.range_constexpr(REG_STEPS):
            m = max4(m, regs[k])
        if tid < TAIL:
            m = max1(m, tw)
        cute.arch.sync_threads()  # mbarrier initialized before anyone waits on it
        mbar_wait0(mbar)
        for k in cutlass.range(SMEM_STEPS, unroll=3):
            m = max4(m, ld_shared_v2(sbase + k * (THREADS1 * 8)))
        for off in cutlass.range_constexpr(5):
            m = max_combine(m, cute.arch.shuffle_sync_bfly(m, 16 >> off))
        if lane == 0:
            smax[warp] = m
        cute.arch.sync_threads()
        mx = smax[0]
        for w in cutlass.range_constexpr(1, 32):
            mx = max_combine(mx, smax[w])

        # pass 2: sum of exp in aten's order (steps 0..36, then the tail element), aten's BlockReduce
        a = Float32(0.0)
        for k in cutlass.range_constexpr(REG_STEPS):
            a = sumexp4p(a, regs[k], mx)
        for k in cutlass.range(SMEM_STEPS, unroll=3):
            a = sumexp4p(a, ld_shared_v2(sbase + k * (THREADS1 * 8)), mx)
        if tid < TAIL:
            a = sumexp1(a, tw, mx)
        for off in cutlass.range_constexpr(5):
            a = add_rn(a, cute.arch.shuffle_sync_down(a, 16 >> off))
        if lane == 0:
            ssum[warp] = a
        cute.arch.sync_threads()
        if warp == 0:
            s = ssum[lane]
            for off in cutlass.range_constexpr(5):
                s = add_rn(s, cute.arch.shuffle_sync_down(s, 16 >> off))
            if lane == 0:
                sbc[0] = s
        cute.arch.sync_threads()
        ls = logf_(sbc[0])

        if tid == 0:
            t = ld_s64(ta)
            if t != IGNORE_INDEX:
                xt = bf16_lo_f32(ld_u16(xa + t * 2))
                st_f32(scratch.iterator.toint() + Int64(row) * 4, sub_rn(sub_rn(xt, mx), ls))

        # pass 3: write log-probs
        for k in cutlass.range_constexpr(REG_STEPS):
            st_v4_f32_ef(oa + Int64(tid + k * THREADS1) * 16, epi4p(regs[k], mx, ls))
        for k in cutlass.range(SMEM_STEPS, unroll=3):
            st_v4_f32_ef(oa + Int64(tid + (REG_STEPS + k) * THREADS1) * 16, epi4p(ld_shared_v2(sbase + k * (THREADS1 * 8)), mx, ls))
        if tid < TAIL:
            st_f32(oa + (TAIL_BASE * 4) + Int64(tid) * 4, epi1(tw, mx, ls))


class NllReduce:
    @cute.jit
    def __call__(self, scratch, tg, ndiv, init, tw, ones, loss, stream):
        self.kernel(scratch, tg, ndiv, init, tw, ones, loss).launch(
            grid=[1, 1, 1], block=[NLL_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, scratch, tg, ndiv, init, tw, ones, loss):
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        sx = smem.allocate_tensor(Float32, cute.make_layout(NLL_THREADS), 16)
        sw = smem.allocate_tensor(Float32, cute.make_layout(NLL_THREADS), 16)
        acc = Float32(0.0)
        wt = Float32(0.0)
        for k in cutlass.range_constexpr(NLL_ITERS):
            i = tid + k * NLL_THREADS
            t = ld_s64(tg.iterator.toint() + Int64(i) * 8)
            xv = ld_f32(scratch.iterator.toint() + Int64(i) * 4)
            acc, wt = nll_step(acc, wt, t, xv)
        sx[tid] = acc
        sw[tid] = wt
        cute.arch.sync_threads()
        for si in cutlass.range_constexpr(8):
            stride = (NLL_THREADS // 2) >> si
            if tid < stride:
                sx[tid] = add_rn(sx[tid], sx[tid + stride])
                sw[tid] = add_rn(sw[tid], sw[tid + stride])
            cute.arch.sync_threads()
        if tid == 0:
            nd = cvt_f32_s64(ld_s64(ndiv.iterator.toint()))
            r = add_rn(ld_f32(init.iterator.toint()), div_rn(sx[0], nd))
            st_f32(tw.iterator.toint(), sw[0])
            st_f32(ones.iterator.toint(), Float32(1.0))
            st_f32(loss.iterator.toint(), r)


_CACHE = {}


def _fake(dtype, n, align):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fns = _CACHE.get(dev)
    if fns is None:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        k1 = cute.compile(
            LogSoftmaxRows(torch.cuda.get_device_properties(dev).multi_processor_count),
            _fake(cutlass.BFloat16, ROWS * V, 16),
            _fake(cutlass.Float32, ROWS * V, 16),
            _fake(cutlass.Int64, ROWS, 8),
            _fake(cutlass.Float32, ROWS, 16),
            stream,
            options="--enable-tvm-ffi",
        )
        k2 = cute.compile(
            NllReduce(),
            _fake(cutlass.Float32, ROWS, 16),
            _fake(cutlass.Int64, ROWS, 8),
            _fake(cutlass.Int64, 1, 8),
            _fake(cutlass.Float32, 1, 4),
            _fake(cutlass.Float32, 1, 4),
            _fake(cutlass.Float32, 1, 4),
            _fake(cutlass.Float32, 1, 4),
            stream,
            options="--enable-tvm-ffi",
        )
        fns = (k1, k2)
        _CACHE[dev] = fns
    return fns


def kernel_function(in0, in1, in2, in3):
    assert in0.shape == (ROWS, V) and in0.is_contiguous() and in0.data_ptr() % 16 == 0
    assert in1.shape == (ROWS,) and in1.is_contiguous()
    dev = in0.device
    k1, k2 = _compiled(dev.index)
    lp = torch.empty((ROWS, V), dtype=torch.float32, device=dev)
    scratch = torch.empty(ROWS, dtype=torch.float32, device=dev)
    k1(in0.view(-1), lp.view(-1), in1, scratch)
    tw = torch.empty((), dtype=torch.float32, device=dev)
    ones = torch.empty((), dtype=torch.float32, device=dev)
    acc = torch.empty((), dtype=torch.float32, device=dev)
    k2(scratch, in1, in2.view(1), in3.view(1), tw.view(1), ones.view(1), acc.view(1))
    if _VARIANT == "r006":
        return (lp, torch.ops.aten.alias.default(lp), tw, acc, ones)
    return (lp, torch.ops.aten.alias.default(lp), tw, ones, acc)
