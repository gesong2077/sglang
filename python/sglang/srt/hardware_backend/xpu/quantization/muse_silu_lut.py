"""Candidate: exact BF16-domain FP32 SiLU factor lookup, per-device 256 KiB.

Only prescaled contiguous BF16 [M,19968] is supported. No host weight pool.
Table is created with the same Triton expression as the deployed kernel.
Numeric equivalence is a test requirement, not assumed from this construction.
"""
import logging
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _make_table(T, B:tl.constexpr):
    bits = tl.program_id(0)*B + tl.arange(0,B)
    gate = bits.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)
    factor = gate/(1.0+libdevice.exp(-gate))
    tl.store(T+bits,factor)


@triton.jit
def _lookup_quant(C,T,Q,S,H:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B);valid=col<H
    gate=tl.load(C+row*2*H+col,valid,0)
    up=tl.load(C+row*2*H+col+H,valid,0).to(tl.float32)
    bits=gate.to(tl.uint16,bitcast=True).to(tl.int32)
    factor=tl.load(T+bits)
    value=(factor*up).to(tl.bfloat16).to(tl.float32)
    value=tl.where(valid,value,0)
    scale=tl.maximum(tl.max(tl.abs(value)),1.e-10)/127.
    quant=libdevice.rint(value/scale).to(tl.int8)
    tl.store(Q+row*H+col,quant,valid);tl.store(S+row,scale)


def make_table(device):
    table=torch.empty(65536,dtype=torch.float32,device=device)
    _make_table[(256,)](table,256,num_warps=4,grf_mode='128')
    return table


def forward(c,table,warps=32,grf=128):
    if c.dtype!=torch.bfloat16 or c.ndim!=2 or c.shape[1]!=19968 or not c.is_contiguous():
        raise ValueError('Only contiguous prescaled BF16 Muse gate/up')
    if table.shape!=(65536,) or table.dtype!=torch.float32 or table.device!=c.device:
        raise ValueError('Matching FP32 table required')
    m=c.shape[0];h=9984
    q=torch.empty((m,h),device=c.device,dtype=torch.int8)
    s=torch.empty(m,device=c.device,dtype=torch.float32)
    _lookup_quant[(m,)](c,table,q,s,h,16384,num_warps=warps,grf_mode=str(grf))
    return q,s


_tables = {}


def quantize(c):
    # Each stream owns its immutable table. Initialization and consumption are
    # ordered on that stream without cross-stream waits in the layer hot path.
    # No decode use; the caller guards the verified prescaled prefill shape.
    stream = torch.xpu.current_stream(c.device)
    key = (c.device.index, stream.sycl_queue)
    table = _tables.get(key)
    if table is None:
        table = make_table(c.device)
        _tables[key] = table
        logging.getLogger(__name__).info(
            "Muse prescaled BF16 SiLU lookup initialized: device=%s table_bytes=262144 warps=8",
            c.device,
        )
    return forward(c,table,warps=8,grf=128)
