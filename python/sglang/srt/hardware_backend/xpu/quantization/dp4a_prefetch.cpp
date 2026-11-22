// Exact signed INT8 accumulation; prefetch only within each loaded weight row.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <string>
namespace es=sycl::ext::intel::esimd;
static thread_local std::string error;
template<int P,int L,bool L1>void run(sycl::queue&q,const int8_t*x,const int8_t*w,int32_t*out,int k,int n){
 q.parallel_for(sycl::nd_range<1>(((n+L-1)/L)*L,L),[=](sycl::nd_item<1>it)SYCL_ESIMD_KERNEL{
  int row=it.get_global_linear_id();if(row>=n)return;
  es::simd<int32_t,64>sum=0;
  for(int col=0;col<k;col+=256){
   if constexpr(P>0){
    if(col+P*256<k){
     if constexpr(L1)es::prefetch<int32_t,64>(reinterpret_cast<const int32_t*>(w+row*k+col+P*256),es::properties{es::cache_hint_L1<es::cache_hint::cached>,es::cache_hint_L2<es::cache_hint::cached>});
     else es::prefetch<int32_t,64>(reinterpret_cast<const int32_t*>(w+row*k+col+P*256),es::properties{es::cache_hint_L1<es::cache_hint::uncached>,es::cache_hint_L2<es::cache_hint::cached>});
    }
   }
   es::simd<int8_t,256>a=es::block_load<int8_t,256>(x+col),b=es::block_load<int8_t,256>(w+row*k+col);
   es::simd<int32_t,64>ai=a.template bit_cast_view<int32_t>(),bi=b.template bit_cast_view<int32_t>();
   sum=es::dp4a<int32_t>(sum,ai,bi);
  }
  out[row]=es::reduce<int32_t>(sum,std::plus<>());
 });
}
extern "C" const char* pf_error(){return error.c_str();}
extern "C" int pf_run(void*queue,void*x,void*w,void*out,int k,int n,int p,int l,int l1){
 try{
  if(k%256||k<256||n<1)throw std::runtime_error("Unsupported shape");
  auto&q=*static_cast<sycl::queue*>(queue);auto a=static_cast<int8_t*>(x),b=static_cast<int8_t*>(w);auto c=static_cast<int32_t*>(out);
  #define CFG(P,L,C) if(p==P&&l==L&&l1==C){run<P,L,C>(q,a,b,c,k,n);return 0;}
  CFG(0,16,0) CFG(0,32,0) CFG(1,16,1) CFG(1,32,1) CFG(2,16,1) CFG(2,32,1) CFG(4,16,1) CFG(4,32,1)
  CFG(1,16,0) CFG(1,32,0) CFG(2,16,0) CFG(2,32,0) CFG(4,16,0) CFG(4,32,0)
  #undef CFG
  throw std::runtime_error("Unsupported config");
 }catch(const std::exception&e){error=e.what();return -1;}
}
