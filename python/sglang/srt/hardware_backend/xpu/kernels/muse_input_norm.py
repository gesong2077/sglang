"""Exact native RMSNorm output plus its existing dynamic INT8 gate activation."""
import ctypes as C
import os
import torch
from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import W8A8Activation
_lib=C.CDLL(os.environ['SGLANG_XPU_MUSE_INPUT_NORM_LIB'])
_lib.miq_run.argtypes=[C.c_void_p]*6+[C.c_int]*2+[C.c_float]
_lib.miq_run.restype=C.c_int
_lib.miq_error.restype=C.c_char_p

def input_norm_quant(x,norm):
    m,k=x.shape
    w=norm.weight
    if k!=6656 or m<1 or any(t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.device!=x.device for t in [x,w]) or w.numel()!=k:
        raise ValueError('Input norm fusion requires contiguous BF16 width6656')
    y=torch.empty_like(x);q=torch.empty_like(x,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
    if _lib.miq_run(torch.xpu.current_stream(x.device).sycl_queue,*[t.data_ptr() for t in [x,w,y,q,s]],m,k,norm.variance_epsilon):
        raise RuntimeError(_lib.miq_error().decode())
    return y,W8A8Activation(q,s,(m,k))
