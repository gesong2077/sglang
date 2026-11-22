// Original pinned sgl-kernel-xpu RMS reduction, with the two BF16 TP partials
// added and rounded before normalization. No changes to scale or precision.
#include "native_norm_templates.h"
#include <string>
namespace at::native::xpu {
class MusePeerNorm : public RMSNormNoRstdForward<c10::BFloat16,c10::BFloat16> {
 using BF=c10::BFloat16;using Base=RMSNormNoRstdForward<BF,BF>;
 public:
 BF *peer,*residual;
 MusePeerNorm(BF*x,BF*p,BF*r,BF*w,BF*y,float eps):Base(x,y,w,eps),peer(p),residual(r){}
 template<int V,int I,bool CACHE,typename Vec,typename Index,typename Cfg>
 void reduce_combine(sycl::nd_item<1> item,const Cfg&cfg,Index xo,float&sum,Vec(&reg)[CACHE?I:1]) const {
  static_assert(CACHE);Index lid=item.get_local_id(0);
  #pragma unroll
  for(int it=0;it<I;++it){
   Index col=(Index(it)*cfg.workgroup_size+lid)*V;
   if(col<cfg.Plane){
    Vec a=*reinterpret_cast<Vec*>(this->X_data+xo+col),b=*reinterpret_cast<Vec*>(peer+xo+col);
    #pragma unroll
    for(int v=0;v<V;++v)a[v]=BF(float(a[v])+float(b[v]));
    reg[it]=a;
    #pragma unroll
    for(int v=0;v<V;++v){float f=float(a[v]);sum+=f*f;}
   }
  }
 }
 template<int V,int I,bool CACHE,typename Vec,typename WV,typename Index,typename Cfg>
 void update(sycl::nd_item<1> item,const Cfg&cfg,Index xo,Index yo,float inv,const Vec(&reg)[CACHE?I:1]) const {
  static_assert(CACHE);Index lid=item.get_local_id(0);
  #pragma unroll
  for(int it=0;it<I;++it){
   Index col=(Index(it)*cfg.workgroup_size+lid)*V;
   if(col<cfg.Plane){
    Vec r=*reinterpret_cast<Vec*>(residual+xo+col),y;WV w=*reinterpret_cast<WV*>(this->gamma_data+col);
    #pragma unroll
    for(int v=0;v<V;++v){BF z=Base::apply_weight(float(reg[it][v]),inv,float(w[v]));y[v]=BF(float(z)+float(r[v]));}
    *reinterpret_cast<Vec*>(this->Y_data+yo+col)=y;
   }
  }
 }
};
void muse_peer_launch(void*queue,void*x,void*p,void*r,void*w,void*y,int m,int k,float eps){
 using BF=c10::BFloat16;NormConfig cfg(NormConfig::FastRowTag{},m,k,2,8,k);
 MusePeerNorm norm((BF*)x,(BF*)p,(BF*)r,(BF*)w,(BF*)y,eps);
 auto launch=[&](auto count){
  constexpr int I=decltype(count)::value;
  using Fn=RMSNormNoRstdKernelFunctor<BF,BF,8,I,MusePeerNorm,true,uint32_t,NormFastParams>;
  NormFastParams params{k,cfg.workgroup_size,k,k};
  sycl_kernel_submit(sycl::range<1>(m*cfg.workgroup_size),sycl::range<1>(cfg.workgroup_size),*static_cast<sycl::queue*>(queue),Fn(norm,params));
 };
 int count=(k+cfg.workgroup_size*8-1)/(cfg.workgroup_size*8);
 switch(count){case 1:launch(std::integral_constant<int,1>{});break;case 2:launch(std::integral_constant<int,2>{});break;case 4:launch(std::integral_constant<int,4>{});break;case 7:launch(std::integral_constant<int,7>{});break;case 8:launch(std::integral_constant<int,8>{});break;default:throw std::runtime_error("Unsupported norm cache count");}
}
}
static thread_local std::string mpn_last_error;
extern "C" const char* mpn_error(){return mpn_last_error.c_str();}
extern "C" int mpn_run(void*q,void*x,void*p,void*r,void*w,void*y,int m,int k,float eps){
 try{if(k!=6656||m<1)throw std::runtime_error("Only Muse6656 verified");at::native::xpu::muse_peer_launch(q,x,p,r,w,y,m,k,eps);return 0;}
 catch(const std::exception&e){mpn_last_error=e.what();return -1;}
}
