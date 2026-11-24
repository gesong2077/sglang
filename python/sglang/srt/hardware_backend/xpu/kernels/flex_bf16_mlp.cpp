// SPDX-License-Identifier: Apache-2.0
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <string>
#include <sycl/sycl.hpp>

namespace {
using namespace dnnl;
using dtype = memory::data_type;
using tag = memory::format_tag;
thread_local std::string last_error;

struct Plan {
    engine device;
    memory::desc input, canonical, output;
    matmul::primitive_desc descriptor;
    matmul product;
    reorder weight_reorder;

    explicit Plan(sycl::queue &queue)
        : device(sycl_interop::make_engine(queue.get_device(), queue.get_context())),
          input({4096, 4096}, dtype::bf16, tag::ab), canonical({4096, 12288}, dtype::bf16, tag::ab),
          output({4096, 12288}, dtype::bf16, tag::ab) {
        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);
        descriptor = matmul::primitive_desc(
            device, input, memory::desc({4096, 12288}, dtype::bf16, tag::any), output, attr);
        product = matmul(descriptor);
        weight_reorder =
            reorder(reorder::primitive_desc(device, canonical, device, descriptor.weights_desc()));
    }
    memory tensor(const memory::desc &desc, void *address) {
        return sycl_interop::make_memory(desc, device, sycl_interop::memory_kind::usm, address);
    }
};
} // namespace
extern "C" {
const char *flex_mlp_error() { return last_error.c_str(); }

void *flex_mlp_create(void *queue) {
    try {
        return new Plan(*static_cast<sycl::queue *>(queue));
    } catch (const std::exception &e) {
        last_error = e.what();
        return nullptr;
    }
}
void flex_mlp_destroy(void *plan) { delete static_cast<Plan *>(plan); }

size_t flex_mlp_packed_bytes(void *handle) {
    return static_cast<Plan *>(handle)->descriptor.weights_desc().get_size();
}
size_t flex_mlp_scratch_bytes(void *handle) {
    return static_cast<Plan *>(handle)->descriptor.scratchpad_desc().get_size();
}
int flex_mlp_pack(void *handle, void *queue, void *weight, void *packed) {
    try {
        auto &plan = *static_cast<Plan *>(handle);
        auto stream = sycl_interop::make_stream(plan.device, *static_cast<sycl::queue *>(queue));
        sycl_interop::execute(plan.weight_reorder, stream,
                              {{DNNL_ARG_FROM, plan.tensor(plan.canonical, weight)},
                               {DNNL_ARG_TO, plan.tensor(plan.descriptor.weights_desc(), packed)}});
        return 0;
    } catch (const std::exception &e) {
        last_error = e.what();
        return -1;
    }
}
int flex_mlp_run(void *handle, void *queue, void *input, void *packed, void *output,
                 void *scratch) {
    try {
        auto &plan = *static_cast<Plan *>(handle);
        auto stream = sycl_interop::make_stream(plan.device, *static_cast<sycl::queue *>(queue));
        sycl_interop::execute(
            plan.product, stream,
            {{DNNL_ARG_SRC, plan.tensor(plan.input, input)},
             {DNNL_ARG_WEIGHTS, plan.tensor(plan.descriptor.weights_desc(), packed)},
             {DNNL_ARG_DST, plan.tensor(plan.output, output)},
             {DNNL_ARG_SCRATCHPAD, plan.tensor(plan.descriptor.scratchpad_desc(), scratch)}});
        return 0;
    } catch (const std::exception &e) {
        last_error = e.what();
        return -1;
    }
}
}
