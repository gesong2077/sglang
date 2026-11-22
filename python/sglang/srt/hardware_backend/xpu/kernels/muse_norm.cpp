// Uses the pinned sgl-kernel-xpu v0.5.0 RMS reduction/order and BF16 conversion.
// Original implementation: Apache-2.0, https://github.com/sgl-project/sgl-kernel-xpu
#include "native_norm_templates.h"
#include <string>

namespace at::native::xpu {
template<int MODE>
class MuseNorm : public RMSNormNoRstdForward<c10::BFloat16,c10::BFloat16> {
  using BF=c10::BFloat16;
  using Base=RMSNormNoRstdForward<BF,BF>;
 public:
  BF *residual,*second_weight,*norm_output;
  int8_t *quant;
  float *scales;
  float second_eps;
  MuseNorm(BF*x,BF*y,BF*w,float eps,BF*r,BF*w2,BF*o,int8_t*q,float*s,float eps2)
    :Base(x,y,w,eps),residual(r),second_weight(w2),norm_output(o),quant(q),scales(s),second_eps(eps2){}

  template<int V,int I,bool CACHE,typename Vec,typename WV,typename Index,typename Cfg>
  void update(sycl::nd_item<1> item,const Cfg&cfg,Index xo,Index yo,float inv,
              const Vec(&reg)[CACHE?I:1]) const {
    static_assert(CACHE,"Only verified, register-cached Muse width supported");
    Vec sums[I];
    float variance=0;
    const Index lid=item.get_local_id(0);
    #pragma unroll
    for(int it=0;it<I;++it){
      Index col=(Index(it)*cfg.workgroup_size+lid)*V;
      if(col<cfg.Plane){
        Vec xv=reg[it],rv=*reinterpret_cast<Vec*>(residual+xo+col),yv;
        WV wv=*reinterpret_cast<WV*>(this->gamma_data+col);
        #pragma unroll
        for(int v=0;v<V;++v){
          BF normalized=Base::apply_weight(float(xv[v]),inv,float(wv[v]));
          yv[v]=BF(float(normalized)+float(rv[v]));
          if constexpr(MODE>=2){float f=float(yv[v]);variance+=f*f;}
        }
        *reinterpret_cast<Vec*>(this->Y_data+yo+col)=yv;
        sums[it]=yv;
      }
    }
    if constexpr(MODE>=2){
      variance=sycl::reduce_over_group(item.get_group(),variance,sycl::plus<float>());
      float inv2=Numerics<float>::rsqrt(variance/float(cfg.Plane)+second_eps);
      float amax=0;
      #pragma unroll
      for(int it=0;it<I;++it){
        Index col=(Index(it)*cfg.workgroup_size+lid)*V;
        if(col<cfg.Plane){
          WV w2=*reinterpret_cast<WV*>(second_weight+col);
          Vec out;
          #pragma unroll
          for(int v=0;v<V;++v){
            out[v]=Base::apply_weight(float(sums[it][v]),inv2,float(w2[v]));
            if constexpr(MODE==3)amax=sycl::fmax(amax,sycl::fabs(float(out[v])));
          }
          if constexpr(MODE==2)*reinterpret_cast<Vec*>(norm_output+yo+col)=out;
          sums[it]=out;
        }
      }
      if constexpr(MODE==3){
        amax=sycl::reduce_over_group(item.get_group(),amax,sycl::maximum<float>());
        float scale=sycl::fmax(amax,1.0e-10f)*(1.0f/127.0f);
        if(lid==0)scales[item.get_group(0)]=scale;
        #pragma unroll
        for(int it=0;it<I;++it){
          Index col=(Index(it)*cfg.workgroup_size+lid)*V;
          if(col<cfg.Plane){
            #pragma unroll
            for(int v=0;v<V;++v)quant[yo+col+v]=int8_t(sycl::rint(float(sums[it][v])*sycl::native::recip(scale)));
          }
        }
      }
    }
  }
};

template<int MODE>void muse_norm_launch(void* queue,void*x,void*r,void*w,void*w2,void*y,void*z,void*q,void*s,int m,int k,float eps,float eps2){
  using BF=c10::BFloat16;
  NormConfig cfg(NormConfig::FastRowTag{},m,k,2,8,k);
  MuseNorm<MODE> norm((BF*)x,(BF*)y,(BF*)w,eps,(BF*)r,(BF*)w2,(BF*)z,(int8_t*)q,(float*)s,eps2);
  auto launch=[&](auto count){
    constexpr int I=decltype(count)::value;
    using Fn=RMSNormNoRstdKernelFunctor<BF,BF,8,I,MuseNorm<MODE>,true,uint32_t,NormFastParams>;
    NormFastParams p{k,cfg.workgroup_size,k,k};
    sycl_kernel_submit(sycl::range<1>(m*cfg.workgroup_size),sycl::range<1>(cfg.workgroup_size),
                      *static_cast<sycl::queue*>(queue),Fn(norm,p));
  };
  int count=(k+cfg.workgroup_size*8-1)/(cfg.workgroup_size*8);
  switch(count){
    case 1:launch(std::integral_constant<int,1>{});break;
    case 2:launch(std::integral_constant<int,2>{});break;
    case 4:launch(std::integral_constant<int,4>{});break;
    case 7:launch(std::integral_constant<int,7>{});break;
    case 8:launch(std::integral_constant<int,8>{});break;
    default:throw std::runtime_error("Unsupported cached iteration count");
  }
}
}
static thread_local std::string mn_last_error;
extern "C" const char* mn_error(){return mn_last_error.c_str();}
extern "C" int mn_run(void*queue,void*x,void*r,void*w,void*w2,void*y,void*z,void*q,void*s,int m,int k,int mode,float eps,float eps2){
  try{
    if(k!=6656||m<1)throw std::runtime_error("Only Muse width6656 is verified");
    using namespace at::native::xpu;
    #define RUN(M) muse_norm_launch<M>(queue,x,r,w,w2,y,z,q,s,m,k,eps,eps2)
    if(mode==1){RUN(1);}else if(mode==2){RUN(2);}else if(mode==3){RUN(3);}else{throw std::runtime_error("Bad mode");}
    #undef RUN
    return 0;
  }catch(const std::exception&e){mn_last_error=e.what();return -1;}
}
