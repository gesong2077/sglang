// SPDX-License-Identifier: Apache-2.0
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/math/imf_rounding_math.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
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

void require_ze(ze_result_t status, const char* api) {
    if (status != ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(api) + " returned Level Zero status " + std::to_string(status));
}

// Two generations of GPU-local input from the other rank. Each process maps
// the other card's DMA-BUF on its own device, and only pushes to that mapping.
struct PeerInbox {
    using Backend = sycl::backend;
    ze_context_handle_t ctx;
    ze_device_handle_t device;
    const size_t bytes;
    uint16_t *local = nullptr, *remote = nullptr;
    int export_fd = -1;

    PeerInbox(sycl::queue& q, size_t size)
        : ctx(sycl::get_native<Backend::ext_oneapi_level_zero>(q.get_context())),
          device(sycl::get_native<Backend::ext_oneapi_level_zero>(q.get_device())), bytes(size) {}
    void allocate() {
        ze_external_memory_export_desc_t ext{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC};
        ext.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
        ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &ext};
        void* ptr = nullptr;
        require_ze(zeMemAllocDevice(ctx, &desc, bytes, 65536, device, &ptr), "allocate DMA-BUF inbox");
        local = static_cast<uint16_t*>(ptr);
        ze_external_memory_export_fd_t fd{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD};
        fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
        ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, &fd};
        require_ze(zeMemGetAllocProperties(ctx,local,&props,nullptr),"export DMA-BUF");
        export_fd = fd.fd;
        if (export_fd < 0) throw std::runtime_error("invalid DMA-BUF export");
    }
    void open(int fd) {
        ze_external_memory_import_fd_t ext{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD};
        ext.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
        ext.fd = fd;
        ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &ext};
        void* ptr = nullptr;
        require_ze(zeMemAllocDevice(ctx, &desc, bytes, 65536, device, &ptr), "import peer DMA-BUF");
        remote = static_cast<uint16_t*>(ptr);
        size_t mapped = 0;
        require_ze(zeMemGetAddressRange(ctx,remote,nullptr,&mapped),"check imported DMA-BUF");
        if (mapped < bytes) throw std::runtime_error("peer DMA-BUF is smaller than its two banks");
    }
    ~PeerInbox() {
        if (remote) zeMemFree(ctx,remote);
        // NEO tracks the exported handle on the allocation. Freeing local closes it.
        if (local) zeMemFree(ctx,local);
    }
};

// The first four slots hold two generations of two rank-local messages. The
// final page is control memory, with one host-written counter per cache line.
struct Context {
    sycl::queue owner;
    uint16_t *host, *peer;
    uint64_t* flags;
    int rank;
    size_t capacity;
    uint64_t sequence = 0;
    sycl::event consumer;
    ze_driver_handle_t driver;
    Release release;
    std::unique_ptr<PeerInbox> inbox;
    sycl::queue last_queue;

    Context(sycl::queue& q, void* data, void* scratch, int r, size_t bytes)
        : owner(q),
          host(static_cast<uint16_t*>(data)), peer(static_cast<uint16_t*>(scratch)),
          flags(reinterpret_cast<uint64_t*>(host + 4 * (bytes / 2))),
          rank(r), capacity(bytes / 2), last_queue(q) {
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
        if (sequence) consumer.wait();
        release(driver, host);
    }

    void run(sycl::queue& q, uint16_t* x, size_t count, bool truncate, bool direct = false) {
        if (!count || count > capacity || !q.is_in_order() ||
            q.get_device() != owner.get_device() || q.get_context() != owner.get_context())
            throw std::invalid_argument("unsupported queue or payload");

        // Submit the copy itself in order on the actual caller's queue. A
        // barrier that explicitly waits for the previous consumer alone is not
        // a completion event for later producers on this queue. Another queue
        // waiting on such a barrier can publish the old contents of x.
        const size_t parity = (sequence + 1) & 1;
        auto* bank = host + parity * 2 * capacity;
        if (direct && (!inbox || !inbox->remote || count != capacity))
            throw std::invalid_argument("invalid DMA-BUF sum geometry or peer mapping");
        auto* dest = direct ? inbox->remote + parity * capacity : bank + rank * capacity;
        // Reuse a bank only after its previous consumer, including stream switches.
        // Submitting the copy itself on q keeps writes produced after that consumer ordered.
        if (sequence && last_queue != q) consumer.wait_and_throw();
        q.memcpy(dest, x, count * 2).wait_and_throw();
        ++sequence;
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
        auto* other = peer;
        if (direct) {
            other = inbox->local + parity * capacity;
        } else {
            q.memcpy(other, bank + (1-rank)*capacity, count * 2);
        }
        consumer = q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            size_t i = id[0];
            float a = sycl::bit_cast<float>(uint32_t(x[i]) << 16);
            float b = sycl::bit_cast<float>(uint32_t(other[i]) << 16);
            uint32_t bits = sycl::bit_cast<uint32_t>(sycl::ext::intel::math::fadd_rn(a,b));
            if (!truncate) bits += 0x7fffU + ((bits >> 16) & 1U);
            x[i] = uint16_t(bits >> 16);
        });
        last_queue=q;
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

extern "C" int flex_tp2_peer_export(void* handle, int* fd) {
    try {
        auto& context = *static_cast<Context*>(handle);
        context.inbox = std::make_unique<PeerInbox>(context.owner, 4 * context.capacity);
        context.inbox->allocate();
        *fd = context.inbox->export_fd;
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}
extern "C" int flex_tp2_peer_open(void* handle, int fd) {
    try {
        auto& context = *static_cast<Context*>(handle);
        context.inbox->open(fd);
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}
extern "C" void flex_tp2_peer_close(void* handle) {
    static_cast<Context*>(handle)->inbox.reset();
}
extern "C" int flex_tp2_peer_sum(void* handle, void* queue, void* input, size_t count, int truncate) {
    try {
        static_cast<Context*>(handle)->run(*static_cast<sycl::queue*>(queue),
                                           static_cast<uint16_t*>(input),count,truncate,true);
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}
