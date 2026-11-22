"""Exact softcap/argmax on TP2 shards, with an existing oneCCL gather.

Finite ties and unsanitized NaNs follow torch.argmax's lowest-index rule.
Tokens fit exactly in FP32. Two local reductions do not alter logit arithmetic.
"""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _partial(X,P,VALID:tl.constexpr,OFFSET:tl.constexpr,CAP:tl.constexpr,SANITIZE:tl.constexpr,B:tl.constexpr):
    block=tl.program_id(0);cols=block*B+tl.arange(0,B)
    x=tl.load(X+cols,cols<VALID,0).to(tl.float32)
    x=x/CAP
    x=libdevice.tanh(x)
    x=x*CAP
    # With finite cap, infinities become +/-CAP. Only NaNs need special
    # handling. +inf is a selection sentinel when sanitization is disabled.
    x=tl.where(x!=x,-1.0e30 if SANITIZE else float('inf'),x)
    x=tl.where(cols<VALID,x,-float('inf'))
    value=tl.max(x,axis=0)
    index=tl.min(tl.where((x==value)&(cols<VALID),cols+OFFSET,2147483647),axis=0)
    tl.store(P+block*2,value);tl.store(P+block*2+1,index.to(tl.float32))

@triton.jit
def _merge(P,O,N:tl.constexpr,B:tl.constexpr,TOKEN_ONLY:tl.constexpr):
    ids=tl.arange(0,B);v=tl.load(P+ids*2,ids<N,-float('inf'))
    ix=tl.load(P+ids*2+1,ids<N,float('inf'))
    best=tl.max(v,axis=0);index=tl.min(tl.where(v==best,ix,float('inf')),axis=0)
    if TOKEN_ONLY:tl.store(O,index.to(tl.int64))
    else:
        tl.store(O,best);tl.store(O+1,index)

def local_pair(logits,rank,vocab=202048,cap=20.0,sanitize=False):
    assert logits.shape==(1,101056) and logits.is_contiguous()
    assert logits.dtype in [torch.bfloat16,torch.float32]
    valid=min(101056,vocab-rank*101056);blocks=triton.cdiv(valid,1024)
    partial=torch.empty((blocks,2),dtype=torch.float32,device=logits.device)
    pair=torch.empty((1,2),dtype=torch.float32,device=logits.device)
    _partial[(blocks,)](logits,partial,valid,rank*101056,cap,sanitize,1024,num_warps=4)
    _merge[(1,)](partial,pair,blocks,triton.next_power_of_2(blocks),False,num_warps=4)
    return pair

def distributed_argmax(logits,rank,group,vocab=202048,cap=20.0,sanitize=False):
    pair=local_pair(logits,rank,vocab,cap,sanitize)
    gathered=torch.empty((2,2),dtype=torch.float32,device=logits.device)
    dist.all_gather_single(gathered,pair,group=group)
    out=torch.empty((1,),dtype=torch.int64,device=logits.device)
    _merge[(1,)](gathered,out,2,2,True,num_warps=4)
    return out

def gather_full(logits,group):
    """Original BF16 gather, vocab crop, FP32 cast and softcap fallback."""
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
    out=torch.empty((2,101056),device=logits.device,dtype=logits.dtype)
    dist.all_gather_single(out,logits,group=group)
    result=out.view(1,-1)[:,:202048].float()
    softcap_inplace_logits(result,20.0)
    return result

def eligible(runner,batch):
    info=batch.sampling_info
    return (
        batch.forward_mode.is_decode() and info.is_all_greedy
        and not batch.return_logprob
        and runner.sampling_observer is None and runner.watermark_state is None
        and not info.grammars and info.grammar_mask is None
        and not info.has_custom_logit_processor and not info.has_watermark_candidates
        and info.logit_bias is None
        and info.acc_additive_penalties is None and info.acc_scaling_penalties is None
        and not (info.penalizer_orchestrator and info.penalizer_orchestrator.is_required)
        and info.sampling_mask_batch_indices is None
        and info.sampling_support_logprobs_capture_indices is None
    )

_observations={}
def shadow_check(logits,token,rank,group,used):
    """Explicit diagnostic only: sync and write comparisons; never timed."""
    directory=os.environ.get('SGLANG_XPU_MUSE_COMPACT_GREEDY_SHADOW')
    if not directory:return
    from sglang.srt.utils.async_probe import sanitize_nan_logits
    full=gather_full(logits,group);sanitize_nan_logits(full)
    torch.testing.assert_close(token,full.argmax(-1),rtol=0,atol=0)
    counts=_observations.setdefault(rank,dict(eligible=0,fallback=0,exact=0))
    counts['eligible' if used else 'fallback']+=1;counts['exact']+=1
    root=Path(directory);root.mkdir(exist_ok=True)
    (root/f'rank{rank}.json').write_text(json.dumps(counts,indent=2)+'\n')

class Model(torch.nn.Module):
    """Local wrapper; full two-rank correctness is in greedy_bench.py."""
    def forward(self,logits):return local_pair(logits,0)
