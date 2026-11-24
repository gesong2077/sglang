// Experimental direct oneDNN W8A8; preserves per-row activation scales.
// A host-only bridge uses the current PyTorch SYCL queue, never a second queue.
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace dnnl;
static thread_local std::string last_error;
struct Plan {
  engine eng;
  memory::desc a,b,c,xs,ws;
  matmul::primitive_desc pd;
  matmul op;
  int mode;
  Plan(sycl::queue &q, int m, int k, int n, bool bt, int md, bool pair=false):
    eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
    a({m,k},memory::data_type::s8,memory::dims{k,1}),
    b({k,n},memory::data_type::s8,bt?memory::dims{1,k}:memory::dims{n,1}),
    c({m,n},md==0?memory::data_type::s32:memory::data_type::bf16,memory::dims{n,1}),
    xs({m,1},memory::data_type::f32,memory::dims{1,1}),
    ws({1,n},memory::data_type::f32,memory::dims{n,1}),mode(md) {
      if (pair) {
        using dtype = memory::data_type;
        using tag = memory::format_tag;
        a = memory::desc({2,m,k},dtype::s8,tag::abc);
        b = memory::desc({2,k,n},dtype::s8,tag::acb);
        c = memory::desc({2,m,n},dtype::bf16,tag::abc);
        xs = memory::desc({2,m,1},dtype::f32,tag::abc);
        ws = memory::desc({2,1,n},dtype::f32,tag::abc);
      }
      primitive_attr attr;
      attr.set_scratchpad_mode(scratchpad_mode::user);
      if(mode==1) {
        attr.set_scales_mask(DNNL_ARG_SRC,1);
        attr.set_scales_mask(DNNL_ARG_WEIGHTS,2);
      } else if(mode==2) {
        post_ops post;
        post.append_binary(algorithm::binary_mul,xs);
        post.append_binary(algorithm::binary_mul,ws);
        attr.set_post_ops(post);
      }
      pd=matmul::primitive_desc(eng,a,b,c,attr);
      op=matmul(pd);
  }
};
extern "C" {
const char* ow_error(){return last_error.c_str();}
void* ow_create(void* queue,int m,int k,int n,int bt,int mode) {
  try{return new Plan(*static_cast<sycl::queue*>(queue),m,k,n,bt,mode);}
  catch(const std::exception& e){last_error=e.what();return nullptr;}
}
void* ow_create_pair(void* queue,int m,int k,int n) {
  try{return new Plan(*static_cast<sycl::queue*>(queue),m,k,n,true,2,true);}
  catch(const std::exception& e){last_error=e.what();return nullptr;}
}
size_t ow_scratch(void* ptr){return static_cast<Plan*>(ptr)->pd.scratchpad_desc().get_size();}
const char* ow_impl(void* ptr){return static_cast<Plan*>(ptr)->pd.impl_info_str();}
void ow_destroy(void* ptr){delete static_cast<Plan*>(ptr);}
int ow_run(void* ptr,void* queue,void* a,void* b,void* xs,void* ws,void* c,void* scratch) {
  try {
    auto &p=*static_cast<Plan*>(ptr);
    auto &q=*static_cast<sycl::queue*>(queue);
    auto s=sycl_interop::make_stream(p.eng,q);
    auto mem=[&](const memory::desc &d,void* data){return sycl_interop::make_memory(d,p.eng,sycl_interop::memory_kind::usm,data);};
    std::unordered_map<int,memory> args{{DNNL_ARG_SRC,mem(p.a,a)},{DNNL_ARG_WEIGHTS,mem(p.b,b)},
      {DNNL_ARG_DST,mem(p.c,c)},{DNNL_ARG_SCRATCHPAD,mem(p.pd.scratchpad_desc(),scratch)}};
    if(p.mode==1){args.emplace(DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC,mem(p.xs,xs));args.emplace(DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,mem(p.ws,ws));}
    if(p.mode==2){args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)|DNNL_ARG_SRC_1,mem(p.xs,xs));args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(1)|DNNL_ARG_SRC_1,mem(p.ws,ws));}
    sycl_interop::execute(p.op,s,args);
    return 0;
  }catch(const std::exception& e){last_error=e.what();return -1;}
}
}
