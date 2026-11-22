"""Exact BF16 TP sum -> native RMSNorm -> BF16 residual, on the current queue."""
import ctypes as C
import os
import torch
_lib=C.CDLL(os.environ['SGLANG_XPU_MUSE_PEER_NORM_LIB'])
_lib.mpn_run.argtypes=[C.c_void_p]*6+[C.c_int]*2+[C.c_float];_lib.mpn_run.restype=C.c_int
_lib.mpn_error.restype=C.c_char_p

def fused_peer_norm(x,peer,residual,norm):
 m,k=x.shape
 if k!=6656 or peer.shape!=x.shape or residual.shape!=x.shape or norm.weight.numel()!=k or any(t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.device!=x.device for t in [x,peer,residual,norm.weight]):raise ValueError('BF16 contiguous6656 rows required')
 y=torch.empty_like(x)
 if _lib.mpn_run(torch.xpu.current_stream(x.device).sycl_queue,*[t.data_ptr() for t in [x,peer,residual,norm.weight,y]],m,k,norm.variance_epsilon):raise RuntimeError(_lib.mpn_error().decode())
 return y
