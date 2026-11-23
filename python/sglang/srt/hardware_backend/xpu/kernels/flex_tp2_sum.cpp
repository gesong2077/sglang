// SPDX-License-Identifier: Apache-2.0
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/math/imf_rounding_math.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using Import = ze_result_t (*)(ze_driver_handle_t, void*, size_t);
using Release = ze_result_t (*)(ze_driver_handle_t, void*);
using Check = ze_result_t (*)(ze_driver_handle_t, void*, void**);
thread_local std::string last_error;

template <typename F>
F lookup(ze_driver_handle_t driver, const char* name) {
    void* fn = nullptr;
    if (zeDriverGetExtensionFunctionAddress(driver, name, &fn) != ZE_RESULT_SUCCESS)
        return nullptr;
    return reinterpret_cast<F>(fn);
}

// The first four slots hold two generations of two rank-local messages. The
// final page is control memory, with one host-written counter per cache line.
struct Context {
    sycl::queue tx;
    uint16_t *host, *peer;
    uint64_t* flags;
    int rank;
    size_t capacity;
    uint64_t sequence = 0;
    sycl::event consumer;
    ze_driver_handle_t driver;
    Release release;

    Context(sycl::queue& q, void* data, void* scratch, int r, size_t bytes)
        : tx(q.get_context(), q.get_device(), sycl::property::queue::in_order{}),
          host(static_cast<uint16_t*>(data)), peer(static_cast<uint16_t*>(scratch)),
          flags(reinterpret_cast<uint64_t*>(host + 4 * (bytes / 2))),
          rank(r), capacity(bytes / 2) {
        if (!q.is_in_order() || bytes == 0 || bytes % 4096 || r < 0 || r > 1)
            throw std::invalid_argument("invalid queue, rank, or capacity");
        driver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            q.get_device().get_platform());
        auto import = lookup<Import>(driver, "zexDriverImportExternalPointer");
        release = lookup<Release>(driver, "zexDriverReleaseImportedPointer");
        auto check = lookup<Check>(driver, "zexDriverGetHostPointerBaseAddress");
        if (!import || !release || !check ||
            import(driver, data, 4 * bytes) != ZE_RESULT_SUCCESS)
            throw std::runtime_error("cannot register the shared staging region");
        void* base = nullptr;
        if (check(driver, data, &base) != ZE_RESULT_SUCCESS || base != data) {
            release(driver, data);
            throw std::runtime_error("driver did not register the complete staging region");
        }
    }

    ~Context() {
        // Keep the Python-owned host mapping and XPU scratch alive through DMA.
        tx.wait();
        if (sequence) consumer.wait();
        release(driver, host);
    }

    void run(sycl::queue& q, uint16_t* x, size_t count, bool truncate) {
        if (!count || count > capacity || !q.is_in_order() ||
            q.get_device() != tx.get_device() || q.get_context() != tx.get_context())
            throw std::invalid_argument("unsupported queue or payload");

        // Include the actual caller stream's producers. On a stream change, also
        // wait for the last consumer of our reusable device scratch.
        auto ready = sequence ? q.ext_oneapi_submit_barrier({consumer})
                              : q.ext_oneapi_submit_barrier();
        ++sequence;
        auto* bank = host + (sequence & 1) * 2 * capacity;
        tx.memcpy(bank + rank * capacity, x, count * 2, ready).wait_and_throw();
        __atomic_store_n(flags + 8 * rank, sequence, __ATOMIC_RELEASE);
        // An advance to generation g implies that our read of g-1 finished.
        // Waiting for the peer's g before returning therefore protects its
        // read of g-1 from our overwrite two generations later.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (__atomic_load_n(flags + 8 * (1-rank), __ATOMIC_ACQUIRE) < sequence) {
            for (int i = 0; i < 64; ++i) _mm_pause();
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("timed out waiting for peer TP2 DMA");
            std::this_thread::yield();
        }
        // Only the DMA engine accesses the registered mapping. A GPU compute
        // load from another process's imported mapping is not coherent on Flex.
        q.memcpy(peer, bank + (1 - rank) * capacity, count * 2);
        auto* other = peer;
        consumer = q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            size_t i = id[0];
            float a = sycl::bit_cast<float>(uint32_t(x[i]) << 16);
            float b = sycl::bit_cast<float>(uint32_t(other[i]) << 16);
            uint32_t bits = sycl::bit_cast<uint32_t>(sycl::ext::intel::math::fadd_rn(a,b));
            if (!truncate) bits += 0x7fffU + ((bits >> 16) & 1U);
            x[i] = uint16_t(bits >> 16);
        });
    }
};
}  // namespace

extern "C" const char* flex_tp2_error() { return last_error.c_str(); }

extern "C" void* flex_tp2_create(void* queue, void* data, void* scratch, int rank,
                                 size_t bytes) {
    try {
        return new Context(*static_cast<sycl::queue*>(queue), data, scratch, rank, bytes);
    } catch (const std::exception& exc) {
        last_error = exc.what();
        return nullptr;
    }
}

extern "C" int flex_tp2_sum(void* context, void* queue, void* input,
                            size_t count, int truncate) {
    try {
        static_cast<Context*>(context)->run(*static_cast<sycl::queue*>(queue),
                                           static_cast<uint16_t*>(input), count, truncate);
        return 0;
    } catch (const std::exception& exc) {
        last_error = exc.what();
        return -1;
    }
}

extern "C" void flex_tp2_destroy(void* context) {
    delete static_cast<Context*>(context);
}
