"""RoPE backward of the MLA k_pe with the head sum, and the kv-gradient cat (DeepSeek-V3 debug model).

reference, per token t: g = in1[0, :, t] (16 heads x 192: 128 nope | 64 rope), v = in0[0, :, t] (16 x 128),
f = in2[t] (32 complex64):
  out0[t, h] = [g[h, :128] | v[h]]                                  ([16384, 4096]), out1 = out0.t()
  out2[t] = bf16(complex(f32(bf16(sum_h g[h, 128:]))) * f)          ([16384, 64])
in0/in1 views are physically [T, 16, 128] / [T, 16, 192].

One CTA (128 threads) per token; each thread copies 4 vectors (16 B) of out0. Warp 0 also does the head
sum: lane i * 8 + j loads column vector j (8 elements) of heads i, i + 4, i + 8, i + 12.

Numerics (checked bit for bit against aten):
  bf16 sum over 16 heads (aten Reduce.cuh thread_reduce, vt0 = 4 accumulators, f32): acc_i =
  (((0 + x_i) + x_{i+4}) + x_{i+8}) + x_{i+12}, sum = ((acc_0 + acc_1) + acc_2) + acc_3, cvt.rn to bf16.
  complex64 mul as aten (c10::complex operator*=, nvcc contracts): re = fma(a, c, -(b*d)),
  im = fma(b, c, a*d); bf16 rounding cvt.rn.
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
IW = 192  # in1 head width (128 | 64)
VW = 128  # in0 head width
OW = 256  # out head width
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


_ACC = (
    "{\n"
    ".reg .b32 lo, hi, x;\n"
    "mov.b32 lo, 0f00000000;\n"
    "mov.b32 hi, 0f00000000;\n"
    + "".join(
        f"shl.b32 x, ${k + 2}, 16;\nadd.rn.f32 lo, lo, x;\nand.b32 x, ${k + 2}, 0xFFFF0000;\nadd.rn.f32 hi, hi, x;\n"
        for k in range(4)
    )
    + "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}"
)


@dsl_user_op
def acc_word(w0, w1, w2, w3, *, loc=None, ip=None):
    """bf16x2 words of 4 heads -> f32 ((((0 + x0) + x1) + x2) + x3) of the low and high halves."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_ir(v, loc, ip) for v in (w0, w1, w2, w3)],
        _ACC,
        "=r,=r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, loc, ip)


@dsl_user_op
def sum4_bf16x2(lo, hi, *, loc=None, ip=None):
    """lo, hi: 4 f32 accumulators each -> bf16x2(((a0 + a1) + a2) + a3) of both."""
    text = (
        "{\n.reg .b32 l, h;\n"
        "add.rn.f32 l, $1, $2;\nadd.rn.f32 l, l, $3;\nadd.rn.f32 l, l, $4;\n"
        "add.rn.f32 h, $5, $6;\nadd.rn.f32 h, h, $7;\nadd.rn.f32 h, h, $8;\n"
        "cvt.rn.bf16x2.f32 $0, h, l;\n}"
    )
    return Int32(_asm(T.i32(), [_ir(v, loc, ip) for v in list(lo) + list(hi)], text, "=r" + ",r" * 8, loc, ip))


class RopeSumBwd:
    @cute.jit
    def __call__(self, in0, in1, in2f, out0, out2, stream):
        self.kernel(in0, in1, in2f, out0, out2).launch(grid=[NT, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, in0: cute.Tensor, in1: cute.Tensor, in2f: cute.Tensor, out0: cute.Tensor, out2: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        copies = []
        for k in cutlass.range_constexpr(2):
            c = tid + k * THREADS
            h = c // 16
            w = c % 16
            copies.append((t * (NH * OW) + h * OW + w * 8, ld_v4(in1.iterator + (t * (NH * IW) + h * IW + w * 8))))
        for k in cutlass.range_constexpr(2):
            c = tid + k * THREADS
            h = c // 16
            w = c % 16
            copies.append((t * (NH * OW) + h * OW + 128 + w * 8, ld_v4(in0.iterator + (t * (NH * VW) + h * VW + w * 8))))
        for o, v in copies:
            st_v4(out0.iterator + o, v)

        if tid < 32:
            ai = tid // 8  # accumulator (head mod 4)
            j = tid % 8  # column vector of the 64 rope elements
            xs = []
            for m in cutlass.range_constexpr(4):
                xs.append(ld_v4(in1.iterator + (t * (NH * IW) + (ai + 4 * m) * IW + 128 + j * 8)))
            # acc[2k], acc[2k + 1]: low / high element of word k for accumulator ai
            acc = []
            for k in cutlass.range_constexpr(4):
                acc.extend(acc_word(xs[0][k], xs[1][k], xs[2][k], xs[3][k]))
            # parts[i]: accumulator i's values, from lane j + 8 i
            parts = [acc]
            for di in cutlass.range_constexpr(3):
                sh = []
                for e in cutlass.range_constexpr(8):
                    sh.append(cute.arch.shuffle_sync_down(acc[e], 8 * (di + 1)))
                parts.append(sh)
            if tid < 8:
                f0 = ld_v4(in2f.iterator + (t * 64 + j * 8))
                f1 = ld_v4(in2f.iterator + (t * 64 + j * 8 + 4))
                res = []
                for k in cutlass.range_constexpr(4):
                    s = sum4_bf16x2(
                        [parts[0][2 * k], parts[1][2 * k], parts[2][2 * k], parts[3][2 * k]],
                        [parts[0][2 * k + 1], parts[1][2 * k + 1], parts[2][2 * k + 1], parts[3][2 * k + 1]],
                    )
                    fc = f0 if k < 2 else f1
                    res.append(rope_word(s, fc[(2 * k) % 4], fc[(2 * k) % 4 + 1]))
                st_v4(out2.iterator + (t * 64 + j * 8), res)

_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        bf = cutlass.BFloat16
        fn = cute.compile(
            RopeSumBwd(),
            _fake(bf, NT * NH * VW),
            _fake(bf, NT * NH * IW),
            _fake(cutlass.Float32, NT * 64),
            _fake(bf, NT * NH * OW),
            _fake(bf, NT * 64),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(x, a=16):
    x = x.contiguous()
    return x if x.data_ptr() % a == 0 else x.clone()


def kernel_function(in0, in1, in2):
    v = _aligned(in0.squeeze(0).transpose(0, 1))  # [T, 16, 128], a view for the training layout
    g = _aligned(in1.squeeze(0).transpose(0, 1))  # [T, 16, 192]
    f = _aligned(torch.view_as_real(in2.contiguous()))
    out0 = torch.empty((NT, NH * OW), dtype=torch.bfloat16, device=in0.device)
    out2 = torch.empty((NT, 64), dtype=torch.bfloat16, device=in0.device)
    _compiled(in0.device.index)(v.view(-1), g.view(-1), f.view(-1), out0.view(-1), out2.view(-1))
    return (out0, out0.t(), out2)
