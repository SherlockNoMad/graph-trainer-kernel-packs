"""MLA RoPE recompute (DeepSeek-V3 debug model, attention.rope backward/recompute), one CuTe DSL kernel.

reference, per token t (pos = in3[t], f = in2[pos] (32 complex64)):
  q = in0[t] (16 heads x 192: 128 nope | 64 rope), kv = in1[t] (512 kv | 64 k_pe), kvb = in4[t] (16 x 256)
  out1[t, h] = [q[h, :128] | bf16(complex(q[h, 128:]) * f)]          (v39, [1,16,T,192] view)
  out2[t, h] = [kvb[h, :128] | bf16(complex(k_pe) * f)]              (v41, k_pe rope broadcast over heads)
  out4, out5 = _fused_rms_norm(kv, [512], in5, 1e-6)                 (v46 bf16, v49 rstd f32)
  out6 = out7 = conj(f)                                              (two clones)
  out0 = in1[:, :512] and out3 = in4 v-half views (aliases, no data movement).

One CTA (128 threads) per token; each thread moves 2 nope vectors of q and of k (16 B each), computes one
16 B rope vector of q and of k; warp 0 writes the conj vectors and warp 1 does the RMS norm of the row.

Numerics (checked bit for bit against aten):
  complex64 mul (c10::complex operator*=, nvcc contracts): re = fma(a, c, -(b*d)), im = fma(b, c, a*d).
  bf16 rounding: cvt.rn.
  conj (clone of _conj, vectorized_elementwise_kernel conj_kernel_cuda): imag * -1.0 (mul.rn.f32): every
  NaN becomes 0x7FFFFFFF, +-0 / inf / denormals flip sign (probe: tools/conj_nan_probe2.py).
  _fused_rms_norm dispatches (torch/_native/ops/norm/rmsnorm_impl.py) to torch's vendored quack RMSNorm
  (torch/_vendor/quack/rmsnorm.py); for N=512 bf16 its config is num_threads 128, threads_per_row 32,
  cluster_n 1, vecsize 8: one warp per row, lane i owns x[8i:8i+8] and x[256+8i:256+8i+8]. Lane sum
  (((0 + x_0^2) + x_1^2) + ...) over those 16 elements in that order (TensorSSA.reduce -> ordered
  vector.reduction), butterfly shfl.xor 16, 8, 4, 2, 1, rstd = rsqrt.approx.ftz(s / 512 + eps),
  y = bf16((x * rstd) * w).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
NH = 16
QW = 192  # q / out head width
KVW = 576  # in1 row
KBW = 256  # in4 head width
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


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


@dsl_user_op
def ld_v4(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_v4_cached(ptr, *, loc=None, ip=None):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, loc, ip)


@dsl_user_op
def ld_b64_lo(ptr, *, loc=None, ip=None):
    """Low 32 bits of an int64 (positions are < 2^31)."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 2, loc, ip)[0]


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [_ir(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def st_b32(ptr, v, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), _ir(v, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


_ROPE = (
    "{\n"
    ".reg .b32 a, b, t, re, im;\n"
    "shl.b32 a, $1, 16;\n"
    "and.b32 b, $1, 0xFFFF0000;\n"
    "mul.rn.f32 t, b, $3;\n"
    "neg.f32 t, t;\n"
    "fma.rn.f32 re, a, $2, t;\n"
    "mul.rn.f32 t, a, $3;\n"
    "fma.rn.f32 im, b, $2, t;\n"
    "cvt.rn.bf16x2.f32 $0, im, re;\n"
    "}"
)


@dsl_user_op
def rope_word(x, c, d, *, loc=None, ip=None):
    """bf16x2 (re, im) times complex64 (c, d) -> bf16x2, aten complex64 mul numerics."""
    return Int32(
        _asm(T.i32(), [_ir(v, loc, ip) for v in (x, c, d)], _ROPE, "=r,r,r,r", loc, ip)
    )


_SUMSQ = (
    "{\n.reg .b32 a, s;\nmov.b32 s, 0f00000000;\n"
    + "".join(
        f"shl.b32 a, ${k + 1}, 16;\nfma.rn.f32 s, a, a, s;\nand.b32 a, ${k + 1}, 0xFFFF0000;\nfma.rn.f32 s, a, a, s;\n"
        for k in range(8)
    )
    + "mov.b32 $0, s;\n}"
)


@dsl_user_op
def sumsq16(w, *, loc=None, ip=None):
    """(((0 + x0^2) + x1^2) + ...) + x15^2 of 16 bf16 (8 words, element order), f32 bits. Squares of
    bf16 are exact in f32, so fma == mul + add."""
    return Int32(_asm(T.i32(), [_ir(v, loc, ip) for v in w], _SUMSQ, "=r" + ",r" * 8, loc, ip))


@dsl_user_op
def neg_f32(x, *, loc=None, ip=None):
    """-x as aten's vectorized conj_kernel_cuda computes it: a float multiply by -1 (canonical NaN
    0x7FFFFFFF for every NaN; +-0, inf and denormals flip sign exactly), not a sign-bit flip."""
    return Int32(_asm(T.i32(), [_ir(x, loc, ip)], "mul.rn.f32 $0, $1, 0fBF800000;", "=r,r", loc, ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_ir(a, loc, ip), _ir(b, loc, ip)], "add.rn.f32 $0, $1, $2;", "=r,r,r", loc, ip)
    )


@dsl_user_op
def rstd_of(s, *, loc=None, ip=None):
    text = (
        "{\n.reg .b32 m;\n"
        "div.rn.f32 m, $1, 0f44000000;\n"
        "add.rn.f32 m, m, 0f358637BD;\n"
        "rsqrt.approx.ftz.f32 $0, m;\n}"
    )
    return Int32(_asm(T.i32(), [_ir(s, loc, ip)], text, "=r,r", loc, ip))


@dsl_user_op
def rms_word(x, g, r, *, loc=None, ip=None):
    """bf16x2 x, gamma -> bf16x2(gamma * (rstd * x))."""
    text = (
        "{\n.reg .b32 a0, a1, g0, g1;\n"
        "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
        "shl.b32 g0, $2, 16;\nand.b32 g1, $2, 0xFFFF0000;\n"
        "mul.rn.f32 a0, $3, a0;\nmul.rn.f32 a1, $3, a1;\n"
        "mul.rn.f32 a0, g0, a0;\nmul.rn.f32 a1, g1, a1;\n"
        "cvt.rn.bf16x2.f32 $0, a1, a0;\n}"
    )
    return Int32(_asm(T.i32(), [_ir(v, loc, ip) for v in (x, g, r)], text, "=r,r,r,r", loc, ip))


class RopeRecompute:
    @cute.jit
    def __call__(self, in0, in1, in2f, in3, in4, in5, out_q, out_k, out_rms, out_rstd, out_c0, out_c1, stream):
        self.kernel(in0, in1, in2f, in3, in4, in5, out_q, out_k, out_rms, out_rstd, out_c0, out_c1).launch(
            grid=[NT, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        in0: cute.Tensor,
        in1: cute.Tensor,
        in2f: cute.Tensor,
        in3: cute.Tensor,
        in4: cute.Tensor,
        in5: cute.Tensor,
        out_q: cute.Tensor,
        out_k: cute.Tensor,
        out_rms: cute.Tensor,
        out_rstd: cute.Tensor,
        out_c0: cute.Tensor,
        out_c1: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()

        pos = ld_b64_lo(in3.iterator + t)
        fbase = pos * 64
        qrow = t * (NH * QW)
        # rope vector: head h, complex group j (4 complex = 8 bf16)
        rh = tid // 8
        rj = tid % 8
        f0 = ld_v4_cached(in2f.iterator + (fbase + rj * 8))
        f1 = ld_v4_cached(in2f.iterator + (fbase + rj * 8 + 4))
        qr = ld_v4(in0.iterator + (qrow + rh * QW + 128 + rj * 8))
        kr = ld_v4_cached(in1.iterator + (t * KVW + 512 + rj * 8))
        # nope copies
        copies = []
        for i in cutlass.range_constexpr(2):
            c = tid + i * THREADS
            ch = c // 16
            cw = c % 16
            qo = qrow + ch * QW + cw * 8
            qv = ld_v4(in0.iterator + qo)
            kv = ld_v4(in4.iterator + (t * (NH * KBW) + ch * KBW + cw * 8))
            copies.append((qo, qv, kv))
        # rms norm inputs

        for qo, qv, kv in copies:
            st_v4(out_q.iterator + qo, qv)
            st_v4(out_k.iterator + qo, kv)
        fr = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]]
        ro = qrow + rh * QW + 128 + rj * 8
        st_v4(out_q.iterator + ro, [rope_word(qr[k], fr[2 * k], fr[2 * k + 1]) for k in range(4)])
        st_v4(out_k.iterator + ro, [rope_word(kr[k], fr[2 * k], fr[2 * k + 1]) for k in range(4)])

        # conj(f), twice
        if tid < 32:
            cj = tid % 16
            cv = ld_v4_cached(in2f.iterator + (fbase + cj * 4))
            cw4 = [cv[0], neg_f32(cv[1]), cv[2], neg_f32(cv[3])]
            if tid < 16:
                st_v4(out_c0.iterator + (t * 64 + cj * 4), cw4)
            else:
                st_v4(out_c1.iterator + (t * 64 + cj * 4), cw4)

        # RMS norm of in1[t, :512] on warp 1, quack RMSNorm order (one warp per row, threads_per_row 32,
        # vecsize 8): lane i owns x[8i:8i+8] and x[256+8i:256+8i+8].
        if tid >= 32 and tid < 64:
            lane = tid - 32
            xa = ld_v4_cached(in1.iterator + (t * KVW + lane * 8))
            xb = ld_v4_cached(in1.iterator + (t * KVW + 256 + lane * 8))
            ga = ld_v4_cached(in5.iterator + lane * 8)
            gb = ld_v4_cached(in5.iterator + (256 + lane * 8))
            s = sumsq16([xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]])
            for off in cutlass.range_constexpr(5):
                s = fadd(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
            r = rstd_of(s)
            ya = []
            yb = []
            for k in cutlass.range_constexpr(4):
                ya.append(rms_word(xa[k], ga[k], r))
                yb.append(rms_word(xb[k], gb[k], r))
            st_v4(out_rms.iterator + (t * 512 + lane * 8), ya)
            st_v4(out_rms.iterator + (t * 512 + 256 + lane * 8), yb)
            if lane == 0:
                st_b32(out_rstd.iterator + t, r)

_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        bf = cutlass.BFloat16
        fn = cute.compile(
            RopeRecompute(),
            _fake(bf, NT * NH * QW),
            _fake(bf, NT * KVW),
            _fake(cutlass.Float32, 2048 * 64),
            _fake(cutlass.Int64, NT, 8),
            _fake(bf, NT * NH * KBW),
            _fake(bf, 512),
            _fake(bf, NT * NH * QW),
            _fake(bf, NT * NH * QW),
            _fake(bf, NT * 512),
            _fake(cutlass.Float32, NT, 4),
            _fake(cutlass.Float32, NT * 64),
            _fake(cutlass.Float32, NT * 64),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2, in3, in4, in5):
    dev = in0.device
    x0 = _aligned(in0)
    x1 = _aligned(in1)
    f = _aligned(torch.view_as_real(in2.contiguous()))
    p = in3.contiguous()
    x4 = _aligned(in4)
    g = _aligned(in5)
    out_q = torch.empty((NT, NH, QW), dtype=torch.bfloat16, device=dev)
    out_k = torch.empty((NT, NH, QW), dtype=torch.bfloat16, device=dev)
    out_rms = torch.empty((NT, 512), dtype=torch.bfloat16, device=dev)
    out_rstd = torch.empty((NT, 1), dtype=torch.float32, device=dev)
    c0 = torch.empty((NT, 1, 32), dtype=torch.complex64, device=dev)
    c1 = torch.empty((NT, 1, 32), dtype=torch.complex64, device=dev)
    _compiled(dev.index)(
        x0.view(-1),
        x1.view(-1),
        f.view(-1),
        p.view(-1),
        x4.view(-1),
        g.view(-1),
        out_q.view(-1),
        out_k.view(-1),
        out_rms.view(-1),
        out_rstd.view(-1),
        torch.view_as_real(c0).view(-1),
        torch.view_as_real(c1).view(-1),
    )
    v5 = in1[:, :512]
    v43 = in4.view(NT, NH, KBW)[:, :, 128:].transpose(0, 1).unsqueeze(0)
    return (
        v5,
        out_q.transpose(0, 1).unsqueeze(0),
        out_k.transpose(0, 1).unsqueeze(0),
        v43,
        out_rms,
        out_rstd,
        c0,
        c1,
    )
