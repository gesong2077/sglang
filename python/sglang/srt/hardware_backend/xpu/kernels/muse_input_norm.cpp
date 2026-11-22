// Exact sgl-kernel-xpu RMS reduction and BF16 conversion, plus existing INT8 quant.
// Reuses the Apache-2.0 pinned norm templates; no changes to Q/K/V values.
#include "native_norm_templates.h"
#include <string>
namespace at::native::xpu {
class MuseInputNormQuant : public RMSNormNoRstdForward<c10::BFloat16,c10::BFloat16> {
 using BF=c10::BFloat16;
 using Base=RMSNormNoRstdForward<BF,BF>;
 public:
 int8_t *quant;float *scales;
 MuseInputNormQuant(BF*x,BF*y,BF*w,float eps,int8_t*q,float*s):Base(x,y,w,eps),quant(q),scales(s){}
 template<int V,int I,bool CACHE,typename Vec,typename WV,typename Index,typename Cfg>
 void update(sycl::nd_item<1> item,const Cfg&cfg,Index xo,Index yo,float inv,const Vec(&reg)[CACHE?I:1])const{
  static_assert(CACHE,"Only register-cached Muse width supported");
  Vec values[I];float amax=0;Index lid=item.get_local_id(0);
  #pragma unroll
  for(int it=0;it<I;++it){
   Index col=(Index(it)*cfg.workgroup_size+lid)*V;
   if(col<cfg.Plane){
    Vec xv=reg[it],out;WV wv=*reinterpret_cast<WV*>(this->gamma_data+col);
    #pragma unroll
    for(int v=0;v<V;++v){out[v]=Base::apply_weight(float(xv[v]),inv,float(wv[v]));amax=sycl::fmax(amax,sycl::fabs(float(out[v])));}
    *reinterpret_cast<Vec*>(this->Y_data+yo+col)=out;values[it]=out;
   }
  }
  amax=sycl::reduce_over_group(item.get_group(),amax,sycl::maximum<float>());
  float scale=sycl::fmax(amax,1.e-10f)*(1.f/127.f);
  if(lid==0)scales[item.get_group(0)]=scale;
  #pragma unroll
  for(int it=0;it<I;++it){
   Index col=(Index(it)*cfg.workgroup_size+lid)*V;
   if(col<cfg.Plane){
    #pragma unroll
    for(int v=0;v<V;++v)quant[yo+col+v]=int8_t(sycl::rint(float(values[it][v])*sycl::native::recip(scale)));
   }
  }
 }
};
void launch(void*queue,void*x,void*w,void*y,void*q,void*s,int m,int k,float eps){
 using BF=c10::BFloat16;NormConfig cfg(NormConfig::FastRowTag{},m,k,2,8,k);
 MuseInputNormQuant norm((BF*)x,(BF*)y,(BF*)w,eps,(int8_t*)q,(float*)s);
 auto launch=[&](auto count){
  constexpr int I=decltype(count)::value;
  using Fn=RMSNormNoRstdKernelFunctor<BF,BF,8,I,MuseInputNormQuant,true,uint32_t,NormFastParams>;
  NormFastParams p{k,cfg.workgroup_size,k,k};
  sycl_kernel_submit(sycl::range<1>(m*cfg.workgroup_size),sycl::range<1>(cfg.workgroup_size),*static_cast<sycl::queue*>(queue),Fn(norm,p));
 };
 int count=(k+cfg.workgroup_size*8-1)/(cfg.workgroup_size*8);
 switch(count){case 1:launch(std::integral_constant<int,1>{});break;case 2:launch(std::integral_constant<int,2>{});break;case 4:launch(std::integral_constant<int,4>{});break;case 7:launch(std::integral_constant<int,7>{});break;case 8:launch(std::integral_constant<int,8>{});break;default:throw std::runtime_error("Unsupported cached iteration count");}
}
}
static thread_local std::string error;
extern "C" const char* miq_error(){return error.c_str();}
extern "C" int miq_run(void*queue,void*x,void*w,void*y,void*q,void*s,int m,int k,float eps){
 try{if(k!=6656||m<1)throw std::runtime_error("Only Muse width6656 supported");at::native::xpu::launch(queue,x,w,y,q,s,m,k,eps);return 0;}catch(const std::exception&e){error=e.what();return -1;}
}
