// Experimental exact signed INT8 dot products on the current PyTorch queue.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <string>
namespace es = sycl::ext::intel::esimd;
static thread_local std::string error;

template<int B, int L> void run(sycl::queue &q, const int8_t *x,
    const int8_t *w, int32_t *out, int k, int n) {
  q.parallel_for(sycl::nd_range<1>(((n+L-1)/L)*L, L),
    [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      int row = it.get_global_linear_id();
      if (row >= n) return;
      es::simd<int32_t,B/4> sum = 0;
      for (int col=0;col<k;col+=B) {
        es::simd<int8_t,B> av = es::block_load<int8_t,B>(x+col);
        es::simd<int8_t,B> bv = es::block_load<int8_t,B>(w+row*k+col);
        es::simd<int32_t,B/4> ai=av.template bit_cast_view<int32_t>();
        es::simd<int32_t,B/4> bi=bv.template bit_cast_view<int32_t>();
        // ESIMD DP4A takes accumulator first, then the two packed operands.
        sum=es::dp4a<int32_t>(sum,ai,bi);
      }
      out[row]=es::reduce<int32_t>(sum,std::plus<>());
    });
}
extern "C" const char* dg_error(){return error.c_str();}
extern "C" int dg_run(void *queue,void *x,void *w,void *out,int k,int n,int b,int l){
  try {
    if(k%b)throw std::runtime_error("K must be divisible by block");
    auto &q=*static_cast<sycl::queue*>(queue);
    auto a=static_cast<int8_t*>(x),weights=static_cast<int8_t*>(w);
    auto c=static_cast<int32_t*>(out);
    #define CASE(B,L) if(b==B && l==L){run<B,L>(q,a,weights,c,k,n);return 0;}
    CASE(128,16) CASE(128,32) CASE(128,64)
    CASE(256,16) CASE(256,32) CASE(256,64)
    #undef CASE
    throw std::runtime_error("Unsupported config");
  }catch(const std::exception&e){error=e.what();return -1;}
}
