"""MLA attention RoPE forward with the kv RMS norm (DeepSeek-V3 debug model), one CuTe DSL kernel.

reference (bf16 in/out):
  q = in0.view(T, 16, 192): out36 = [q_nope | rope(q_pe)] (T x 16 x 192 contiguous, returned as the
  transposed view (1, 16, T, 192)); out34 = rope(in1[:, 512:]) (T x 64, returned expanded to 16 heads);
  out31 = rms_norm(in1[:, :512], in4, 1e-6) (T x 512).
  rope(x)[2k:2k+2] = bf16(complex(x[2k], x[2k+1]) * freqs[pos, k]) in float32.

One launch, two block roles:
- RMS blocks (first RMS_BLOCKS blocks): one warp per token. The RMS norm reproduces torch's vendored
  quack RMSNorm forward (torch/_vendor/quack/rmsnorm.py, which aten._fused_rms_norm dispatches to on
  this build for N=512 bf16: config num_threads=128, threads_per_row=32, cluster_n=1, vecsize=8):
  the same tiled copy and per-thread fragment ((8,1),1,2), the same TensorSSA reduce of x*x, the same
  warp butterfly, rstd = rsqrt(sum / 512 + eps) (fastmath), y = bf16((x * rstd) * w). Lanes 0..7 also
  apply RoPE to the token's k_pe (16 B each).
- Q blocks: each thread moves 16-byte vectors of in0 to out36; vectors in the q_pe part of a head
  (columns 128..191) get RoPE.
aten's complex64 mul (c10::complex operator*, nvcc contraction), checked against aten:
  re = fma(a, c, -(b * d)), im = fma(b, c, a * d)   (x = a + bi, f = c + di)
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TOK = 16384
HEADS = 16
QD = 192  # per-head q width: 128 nope + 64 pe
KVW = 576  # in1 row: 512 kv + 64 k_pe
N = 512  # RMS width
VEC = 8  # bf16 per 16-byte vector
QVPR = HEADS * QD // VEC  # 384 vectors per in0 row
QVPH = QD // VEC  # 24 vectors per head; vectors 16..23 are q_pe
NQV = TOK * QVPR
THREADS = 128
ITEMS = 2  # q vectors per thread
Q_BLOCKS = NQV // (THREADS * ITEMS)
ROWS_PER_BLOCK = THREADS // 32
RMS_BLOCKS = TOK // ROWS_PER_BLOCK
NPOS = 2048


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _unpack(res, n, loc, ip):
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(n)]


@dsl_user_op
def ld_v4_stream(ptr, *, loc=None, ip=None):
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
def ld_freq(fptr, pos_ptr, j, *, loc=None, ip=None):
    """The 4 complex64 freqs[pos, 4j..4j+3] as 8 f32 words; pos = positions[row] (int64, low word,
    negative wraps like aten indexing)."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>"),
        [
            fptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            pos_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            _v(j, loc, ip),
        ],
        "{\n"
        ".reg .s32 p, q;\n"
        ".reg .pred n;\n"
        ".reg .u64 a;\n"
        "ld.global.nc.b32 p, [$9];\n"
        "setp.lt.s32 n, p, 0;\n"
        f"add.s32 q, p, {NPOS};\n"
        "selp.s32 p, q, p, n;\n"
        "shl.b32 p, p, 5;\n"  # pos * 32 complex
        "shl.b32 q, $10, 2;\n"
        "add.s32 p, p, q;\n"  # + 4j complex
        "mul.wide.s32 a, p, 8;\n"
        "add.u64 a, a, $8;\n"
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [a];\n"
        "ld.global.nc.v4.b32 {$4, $5, $6, $7}, [a+16];\n"
        "}",
        "=r,=r,=r,=r,=r,=r,=r,=r,l,l,r",
        loc,
        ip,
    )
    return _unpack(res, 8, loc, ip)


_ROPE2 = (
    "{\n"
    ".reg .b32 a, b, re, im, t;\n"
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
def rope_word(w, c, d, *, loc=None, ip=None):
    """One bf16x2 word (re lo, im hi) times the complex (c, d) given as f32 bits."""
    return Int32(_asm(T.i32(), [_v(w, loc, ip), _v(c, loc, ip), _v(d, loc, ip)], _ROPE2, "=r,r,r,r", loc, ip))


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [_v(v, loc, ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


def _rope_vec(x, f):
    return [rope_word(x[k], f[2 * k], f[2 * k + 1]) for k in range(4)]


class RopeFwd:
    @cute.jit
    def __call__(self, mQ, mKV, mFreq, mPos, mW, mO31, mO34, mO36, eps: Float32, stream):
        self.kernel(mQ, mKV, mFreq, mPos, mW, mO31, mO34, mO36, eps).launch(
            grid=[RMS_BLOCKS + Q_BLOCKS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mQ, mKV, mFreq, mPos, mW, mO31, mO34, mO36, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx < RMS_BLOCKS:
            # --- RMS norm, as quack's RMSNorm.kernel (no smem staging; same fragments and math) ---
            tiler_mn = (ROWS_PER_BLOCK, N)
            copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            thr_layout = cute.make_ordered_layout((ROWS_PER_BLOCK, 32), order=(1, 0))
            val_layout = cute.make_layout((1, VEC))
            tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
            mX = cute.make_tensor(mKV.iterator, cute.make_layout((TOK, N), stride=(KVW, 1)))
            mO = cute.make_tensor(mO31.iterator, cute.make_layout((TOK, N), stride=(N, 1)))
            mWe = cute.make_tensor(mW.iterator, cute.make_layout(tiler_mn, stride=(0, 1)))
            gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
            gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
            gW = cute.local_tile(mWe, tiler_mn, (0, 0))
            thr = tiled_copy.get_slice(tidx)
            tXgX = thr.partition_S(gX)
            tXgW = thr.partition_S(gW)
            tXgO = thr.partition_D(gO)
            tXrX = cute.make_rmem_tensor_like(tXgX)
            tXrW = cute.make_rmem_tensor_like(tXgW)
            tXrO = cute.make_rmem_tensor_like(tXgO)
            cute.copy(copy_atom, tXgX, tXrX)
            cute.copy(copy_atom, tXgW, tXrW)
            x = tXrX.load().to(Float32)
            sum_sq = (x * x).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
            sum_sq = cute.arch.warp_reduction(sum_sq, operator.add, threads_in_group=32)
            rstd = cute.math.rsqrt(sum_sq / N + eps, fastmath=True)
            y = x * rstd
            w = tXrW.load().to(Float32)
            y *= w
            tXrO.store(y.to(cutlass.BFloat16))
            cute.copy(copy_atom, tXrO, tXgO)
            # --- k_pe RoPE: lanes 0..7 of each warp, 16 B each ---
            lane = tidx % 32
            if lane < 8:
                row = bidx * ROWS_PER_BLOCK + tidx // 32
                xk = ld_v4_stream(mKV.iterator + (row * KVW + N + lane * VEC))
                f = ld_freq(mFreq.iterator, mPos.iterator + row, lane)
                st_v4(mO34.iterator + (row * 64 + lane * VEC), _rope_vec(xk, f))
        else:
            # --- q: copy q_nope, RoPE q_pe ---
            qb = bidx - RMS_BLOCKS
            loads = []
            for it in cutlass.range_constexpr(ITEMS):
                idx = (qb * ITEMS + it) * THREADS + tidx
                loads.append((idx, ld_v4_stream(mQ.iterator + idx * VEC)))
            for idx, xq in loads:
                row = idx // QVPR
                c = idx % QVPH
                if c >= 16:
                    f = ld_freq(mFreq.iterator, mPos.iterator + row, c - 16)
                    st_v4(mO36.iterator + idx * VEC, _rope_vec(xq, f))
                else:
                    st_v4(mO36.iterator + idx * VEC, xq)


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            RopeFwd(),
            _fake(cutlass.BFloat16, TOK * HEADS * QD),
            _fake(cutlass.BFloat16, TOK * KVW),
            _fake(cutlass.Float32, NPOS * 64),
            _fake(cutlass.Int64, TOK, 8),
            _fake(cutlass.BFloat16, N),
            _fake(cutlass.BFloat16, TOK * N),
            _fake(cutlass.BFloat16, TOK * 64),
            _fake(cutlass.BFloat16, TOK * HEADS * QD),
            Float32(1e-6),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(t):
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def kernel_function(in0, in1, in2, in3, in4):
    assert in0.shape == (TOK, HEADS * QD) and in1.shape == (TOK, KVW)
    assert in2.shape == (NPOS, 32) and in2.dtype == torch.complex64 and in3.shape == (TOK,)
    q, kv, w = _aligned(in0), _aligned(in1), _aligned(in4)
    freq = torch.view_as_real(_aligned(in2)).reshape(-1)
    pos = in3.contiguous()
    dev = in0.device
    out31 = torch.empty((TOK, N), dtype=torch.bfloat16, device=dev)
    k_pe = torch.empty((TOK, 64), dtype=torch.bfloat16, device=dev)
    q_out = torch.empty((TOK, HEADS * QD), dtype=torch.bfloat16, device=dev)
    _compiled(dev.index)(
        q.view(-1), kv.view(-1), freq, pos, w, out31.view(-1), k_pe.view(-1), q_out.view(-1), 1e-6
    )
    out34 = k_pe.unsqueeze(1).expand(-1, HEADS, -1)
    out36 = q_out.view(TOK, HEADS, QD).transpose(0, 1).unsqueeze(0)
    return (out31, out34, out36)
