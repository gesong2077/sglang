// Exact native RMSNorm BF16 materialization followed by cached GPT-J/NeoX RoPE.
// Uses pinned Apache-2.0 sgl-kernel-xpu RMSNorm helpers; unique kernel identity.
#include "/home/rorschach/sglang/python/sglang/srt/hardware_backend/xpu/kernels/native_norm_templates.h"
#include <string>
namespace at::native::xpu {
template<typename W,int STYLE>struct MuseQKNormRopeKernel {
 static constexpr int V=std::is_same_v<W,float>?4:8;
 using BF=c10::BFloat16;using Base=RMSNormNoRstdForward<BF,W>;
 using Scratch=std::conditional_t<std::is_same_v<W,float> && STYLE==2,sycl::local_accessor<float,1>,int>;
 Scratch scratch;
 BF*q,*k,*oq,*ok;W*weight;float*cache;int64_t*positions;int m;float eps;
 [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
  uint32_t idx=item.get_group(0),lane=item.get_local_id(0);bool isq=idx<uint32_t(m*16);
  uint32_t row=isq?idx:idx-m*16,token=isq?row/16:row;
  BF*x=(isq?q:k)+row*128,*y=(isq?oq:ok)+row*128;
  Base norm(x,y,weight,eps);NormFastParams cfg{128,128/V,128,128};
  using Vec=aligned_vector_loop<BF,V>;using WV=aligned_vector_loop<W,V>;
  Vec reg[1];float sum=0;
  norm.template reduce_combine<V,1,true,Vec,uint32_t>(item,cfg,0,sum,reg);
  float inv=norm.reduce_project(item,cfg,sum);WV w=*reinterpret_cast<WV*>(weight+lane*V);Vec out;
  #pragma unroll
  for(int v=0;v<V;++v)out[v]=Base::apply_weight(float(reg[0][v]),inv,float(w[v]));
  if constexpr(STYLE!=0){
   float*cs=cache+positions[token]*128;
   if constexpr(STYLE==1){
    #pragma unroll
    for(int v=0;v<V;v+=2){
     int h=(lane*V+v)/2;float c=cs[h],s=cs[64+h],a=float(out[v]),b=float(out[v+1]);
     out[v]=BF(a*c-b*s);out[v+1]=BF(a*s+b*c);
    }
   }else{
    float other[V];
    if constexpr(std::is_same_v<W,float>){
     #pragma unroll
     for(int v=0;v<V;++v)scratch[lane*V+v]=float(out[v]);
     item.barrier(sycl::access::fence_space::local_space);
     #pragma unroll
     for(int v=0;v<V;++v)other[v]=scratch[(lane*V+v)^64];
    }else{
     #pragma unroll
     for(int v=0;v<V;++v)other[v]=sycl::permute_group_by_xor(item.get_sub_group(),float(out[v]),8);
    }
    #pragma unroll
    for(int v=0;v<V;++v){
     int col=lane*V+v,h=col%64;float c=cs[h],s=cs[64+h],a=float(out[v]),b=other[v];
     // Cache native RoPE contracts the first product; make this explicit
     // across half-split lanes to avoid contracting the other product.
     out[v]=BF(col<64 ? sycl::fma(a,c,-(b*s)) : sycl::fma(b,s,a*c));
    }
   }
  }
  *reinterpret_cast<Vec*>(y+lane*V)=out;
 }
};
template<typename W,int STYLE>void muse_qknr_launch(void*queue,void*q,void*k,void*w,void*cache,void*pos,void*oq,void*ok,int m,float eps){
 using BF=c10::BFloat16;
 using Fn=MuseQKNormRopeKernel<W,STYLE>;
 constexpr int wg=std::is_same_v<W,float>?32:16;
 auto&sq=*static_cast<sycl::queue*>(queue);
 if constexpr(std::is_same_v<W,float> && STYLE==2){
  sq.submit([&](sycl::handler&h){
   sycl::local_accessor<float,1> scratch(sycl::range<1>(128),h);
   Fn fn{scratch,(BF*)q,(BF*)k,(BF*)oq,(BF*)ok,(W*)w,(float*)cache,(int64_t*)pos,m,eps};
   h.parallel_for(sycl::nd_range<1>(m*17*wg,wg),fn);
  });
 }else{
  Fn fn{0,(BF*)q,(BF*)k,(BF*)oq,(BF*)ok,(W*)w,(float*)cache,(int64_t*)pos,m,eps};
  sycl_kernel_submit(sycl::range<1>(m*17*wg),sycl::range<1>(wg),sq,fn);
 }
}
}
static thread_local std::string qknr_last_error;
extern "C" const char*qknr_error(){return qknr_last_error.c_str();}
extern "C" int qknr_run(void*queue,void*q,void*k,void*w,void*cache,void*pos,void*oq,void*ok,int m,int wfloat,int rope,float eps){
 try{if(m<1)throw std::runtime_error("Invalid rows");using namespace at::native::xpu;
 #define RUN(W,R) muse_qknr_launch<W,R>(queue,q,k,w,cache,pos,oq,ok,m,eps)
 if(wfloat){if(rope==2){RUN(float,2);}else if(rope==1){RUN(float,1);}else{RUN(float,0);}}
 else{if(rope==2){RUN(c10::BFloat16,2);}else if(rope==1){RUN(c10::BFloat16,1);}else{RUN(c10::BFloat16,0);}}
 #undef RUN
 return 0;}catch(const std::exception&e){qknr_last_error=e.what();return -1;}
}
