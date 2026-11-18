"""MLA attention input layout (DeepSeek-V3 debug model, forward), one CuTe DSL kernel.

reference (bf16):
  kv = in0.view(T, 16, 256); k_nope = kv[..., :128]; v = kv[..., 128:]
  out0 = cat([k_nope, in1], -1) (T x 16 x 192 contiguous, returned as the view (1, 16, T, 192))
  out1 = v.transpose(0, 1).unsqueeze(0) (a view of in0, no data movement)
in1 is k_pe expanded over the 16 heads (stride (64, 0, 1)): only its T x 64 storage is read.

Each thread writes 16-byte vectors of out0: vector c of a head (24 per head) comes from in0
(c < 16, streamed once) or from k_pe (c >= 16, read by all 16 heads, kept cacheable).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TOK = 16384
HEADS = 16
KVD = 256  # per-head in0 width: 128 k_nope + 128 v
OD = 192  # per-head out width: 128 k_nope + 64 k_pe
PE = 64
VEC = 8  # bf16 per 16-byte vector
OVPR = HEADS * OD // VEC  # 384 output vectors per token
OVPH = OD // VEC  # 24 vectors per head; 0..15 k_nope, 16..23 k_pe
NOV = TOK * OVPR
THREADS = 128
ITEMS = 2  # output vectors per thread
BLOCKS = NOV // (THREADS * ITEMS)


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


def _ld4(text, ptr, loc, ip):
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        text,
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_v4_stream(ptr, *, loc=None, ip=None):
    return _ld4("ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];", ptr, loc, ip)


@dsl_user_op
def ld_v4_cached(ptr, *, loc=None, ip=None):
    return _ld4("ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", ptr, loc, ip)


@dsl_user_op
def st_v4(ptr, w, *, loc=None, ip=None):
    _asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)] + [Int32(v).ir_value(loc=loc, ip=ip) for v in w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


class KvLayout:
    @cute.jit
    def __call__(self, mKV, mPE, mOut, stream):
        self.kernel(mKV, mPE, mOut).launch(grid=[BLOCKS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mKV, mPE, mOut):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        for it in cutlass.range_constexpr(ITEMS):
            idx = (bidx * ITEMS + it) * THREADS + tidx
            row = idx // OVPR
            rem = idx - row * OVPR
            h = rem // OVPH
            c = rem - h * OVPH
            if c < 16:
                x = ld_v4_stream(mKV.iterator + (row * (HEADS * KVD) + h * KVD + c * VEC))
                st_v4(mOut.iterator + idx * VEC, x)
            else:
                x = ld_v4_cached(mPE.iterator + (row * PE + (c - 16) * VEC))
                st_v4(mOut.iterator + idx * VEC, x)


_CACHE = {}


def _fake(n):
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, (n,), stride=(1,), assumed_align=16)


def _compiled(dev):
    fn = _CACHE.get(dev)
    if fn is None:
        fn = cute.compile(
            KvLayout(),
            _fake(TOK * HEADS * KVD),
            _fake(TOK * PE),
            _fake(TOK * HEADS * OD),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev] = fn
    return fn


def _aligned(t):
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def kernel_function(in0, in1):
    assert in0.shape == (TOK, HEADS * KVD) and in1.shape == (TOK, HEADS, PE)
    kv = _aligned(in0)
    pe = _aligned(in1[:, 0])
    out = torch.empty((TOK, HEADS * OD), dtype=torch.bfloat16, device=in0.device)
    _compiled(in0.device.index)(kv.view(-1), pe.view(-1), out.view(-1))
    out0 = out.view(TOK, HEADS, OD).transpose(0, 1).unsqueeze(0)
    out1 = in0.view(TOK, HEADS, KVD)[:, :, KVD // 2 :].transpose(0, 1).unsqueeze(0)
    return (out0, out1)
