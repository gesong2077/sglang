"""One existing oneCCL collective for exact gate INT8 bytes and FP32 scales.

The producing gate quantizer writes a contiguous [q rows, scale rows] block
per destination, aligned to128bytes. Each received q/scales view remains
contiguous for oneDNN; <=127 zero padding bytes per destination.
No additional quantization, transport compression or native IPC implementation.
"""
from dataclasses import dataclass
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice
from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import W8A8Activation,dynamic_w8a8_linear

@dataclass(frozen=True)
class PackedGate:
    storage: torch.Tensor
    m: int
    k: int
    half: int
    alltoall: bool

@triton.jit
def _gate_packed(C,XS,WS,A,P,M:tl.constexpr,H:tl.constexpr,HALF:tl.constexpr,B:tl.constexpr,SPAN:tl.constexpr,PRESCALED:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B);valid=(col<H)&(row<M)
    gate=tl.load(C+row*H+col,valid,0).to(tl.float32)
    if not PRESCALED:
        gate=gate*tl.load(XS+row,row<M,0)*tl.load(WS+col,col<H,0)
    gate=gate.to(tl.bfloat16).to(tl.float32)
    sigmoid=tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    value=tl.load(A+row*H+col,valid,0).to(tl.float32)*sigmoid
    value=value.to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(value)),1.0e-10)/127.0
    q=libdevice.rint(value/scale).to(tl.int8)
    block=row//HALF;local=row%HALF;base=block*SPAN
    tl.store(P+base+local*H+col,q,col<H)
    sp=(P+base+HALF*H).to(tl.pointer_type(tl.float32))
    tl.store(sp+local,tl.where(row<M,scale,0.0))
    if SPAN>HALF*(H+4):
        if local==HALF-1:
            tail=tl.arange(0,128)
            tl.store(P+base+HALF*(H+4)+tail,0,tail<SPAN-HALF*(H+4))

def quantize(accum,scales,weight_scale,attention,*,prescaled=True):
    m,k=attention.shape;alltoall=m>=2048;half=(m+1)//2 if alltoall else m
    blocks=2 if alltoall else 1
    span=triton.cdiv(half*(k+4),128)*128
    raw=torch.empty(blocks*span,dtype=torch.int8,device=attention.device)
    _gate_packed[(blocks*half,)](accum,scales,weight_scale,attention,raw,m,k,half,triton.next_power_of_2(k),span,prescaled,num_warps=16,grf_mode='128')
    return PackedGate(raw,m,k,half,alltoall)

def unpack(storage,block,half,k):
    span=triton.cdiv(half*(k+4),128)*128
    base=block*span
    q=storage[base:base+half*k].view(half,k)
    scales=storage[base+half*k:base+half*(k+4)].view(torch.float32)
    return q,scales

def projection(attn,gated,group,rank):
    m,k,half=gated.m,gated.k,gated.half;local_m=(m+1)//2
    begin=rank*local_m;end=min((rank+1)*local_m,m);valid=end-begin
    received=torch.empty_like(gated.storage) if gated.alltoall else torch.empty(gated.storage.numel()*2,dtype=torch.int8,device=gated.storage.device)
    if gated.alltoall:dist.all_to_all_single(received.view(torch.int32),gated.storage.view(torch.int32),group=group)
    else:dist.all_gather_single(received.view(torch.int32),gated.storage.view(torch.int32),group=group)
    own=(attn.o_proj.weight,attn.o_proj.weight_scale);peer=(attn._xpu_peer_o_weight,attn._xpu_peer_o_scale)
    parts=(own,peer) if rank==0 else (peer,own);outputs=[]
    for source,(weight,scale) in enumerate(parts):
        q,s=unpack(received,source,half,k)
        start=0 if gated.alltoall else begin
        x=W8A8Activation(q[start:start+valid],s[start:start+valid],(valid,k))
        outputs.append(dynamic_w8a8_linear(x,weight,scale))
    result=outputs[0]+outputs[1]
    if valid!=local_m:
        padded=torch.empty((local_m,6656),dtype=result.dtype,device=result.device)
        padded[:valid].copy_(result);padded[valid:].zero_();result=padded
    return result
