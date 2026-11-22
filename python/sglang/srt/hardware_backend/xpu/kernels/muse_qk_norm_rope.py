"""Experimental non-mutating Muse Q/K RMSNorm + cached GPT-J/NeoX RoPE."""
import ctypes as C
import os
import torch
_lib=C.CDLL(os.environ['SGLANG_XPU_MUSE_QK_NORM_ROPE_LIB'])
_lib.qknr_run.argtypes=[C.c_void_p]*8+[C.c_int]*3+[C.c_float];_lib.qknr_run.restype=C.c_int
_lib.qknr_error.restype=C.c_char_p
def forward(q,k,weight,eps,positions=None,cache=None,*,is_neox=False):
 m=q.shape[0]
 if q.shape!=(m,2048) or k.shape!=(m,128) or any(x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.device!=q.device for x in [q,k]) or weight.shape!=(128,) or weight.dtype not in [torch.bfloat16,torch.float32] or not weight.is_contiguous() or weight.device!=q.device:raise ValueError('Muse TP2 BF16 contiguous Q/K required')
 rope=cache is not None
 if rope and (cache.dtype!=torch.float32 or cache.ndim!=2 or cache.shape[1]!=128 or not cache.is_contiguous() or positions.dtype!=torch.int64 or positions.shape!=(m,) or not positions.is_contiguous() or cache.device!=q.device or positions.device!=q.device):raise ValueError('Contiguous FP32 cache and INT64 positions required')
 oq=torch.empty_like(q);ok=torch.empty_like(k)
 if _lib.qknr_run(torch.xpu.current_stream(q.device).sycl_queue,*[x.data_ptr() if x is not None else 0 for x in [q,k,weight,cache,positions,oq,ok]],m,int(weight.dtype==torch.float32),(2 if is_neox else 1) if rope else 0,eps):raise RuntimeError(_lib.qknr_error().decode())
 return oq,ok
