// SPDX-License-Identifier: Apache-2.0
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/esimd.hpp>
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

// Lossless wire layout for the 4096 x 4096 BF16 residual. Sign and mantissa
// use one byte. Neighbouring exponents often fit a 16-value interval, so one
// nibble and the maximum exponent per 128 values recover the original bits.
// Groups with broader ranges or special values also store their high nibbles.
namespace es = sycl::ext::intel::esimd;
constexpr size_t plane_elements = 4096 * 4096;
constexpr size_t plane_groups = plane_elements / 128;

void encode_peer_planes(sycl::queue& q, const uint16_t* input, uint8_t* dest, uint8_t* tags) {
    q.parallel_for(sycl::range<1>(plane_groups), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const size_t offset = id[0] * 128;
        const auto bits = es::block_load<uint16_t, 128>(input + offset);
        es::simd<uint16_t, 128> exponents = (bits >> 7) & 255;
        const auto high = es::hmax<uint16_t>(exponents);
        const bool raw = high == 255 || high - es::hmin<uint16_t>(exponents) > 15;
        auto store_nibbles = [](uint8_t* out, es::simd<uint16_t, 128> n) SYCL_ESIMD_FUNCTION {
            es::simd<uint16_t, 64> pairs = (n.select<64, 2>(0).read() & 15)
                | ((n.select<64, 2>(1).read() & 15) << 4);
            es::block_store(out, es::convert<uint8_t>(pairs));
        };
        es::block_store(dest + offset, es::convert<uint8_t>((bits & 127) | ((bits >> 8) & 128)));
        store_nibbles(dest + plane_elements + offset / 2, exponents);
        if (raw) store_nibbles(dest + 3 * plane_elements / 2 + offset / 2, exponents >> 4);
        tags[id[0]] = raw ? 255 : static_cast<uint8_t>(high);
    });
}

ESIMD_INLINE es::simd<uint16_t, 64> decode_peer_plane(
    const uint8_t* src, const uint8_t* tags, size_t offset) SYCL_ESIMD_FUNCTION {
    auto unpack = [](const uint8_t* addr) SYCL_ESIMD_FUNCTION {
        const auto n = es::block_load<uint8_t, 32>(addr);
        es::simd<uint8_t, 64> out;
        out.select<32, 2>(0) = n & 15;
        out.select<32, 2>(1) = n >> 4;
        return out;
    };
    const auto low = unpack(src + plane_elements + offset / 2);
    const uint8_t tag = tags[offset / 128];
    es::simd<uint8_t, 64> exponents;
    if (tag == 255) exponents = (unpack(src + 3 * plane_elements / 2 + offset / 2) << 4) | low;
    else exponents = tag - ((tag - low) & 15);
    const auto sm = es::convert<uint16_t>(es::block_load<uint8_t, 64>(src + offset));
    return ((sm & 128) << 8) | (es::convert<uint16_t>(exponents) << 7) | (sm & 127);
}

// Both operands reside in local VRAM after the posted peer write. Block
// loads/stores let one SIMD invocation reduce a whole aligned cache line pair.
template <bool Gated, bool Packed = false>
sycl::event add_peer_blocks(sycl::queue& q, uint16_t* x, const uint16_t* other,
                            size_t count, bool truncate, const uint16_t* residual = nullptr,
                            const uint16_t* gate = nullptr, const uint8_t* tags = nullptr) {
    namespace es = sycl::ext::intel::esimd;
    constexpr int width = 64;
    return q.parallel_for(sycl::range<1>(count / width), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const size_t offset = id[0] * width;
        const auto a = es::block_load<uint16_t, width>(x + offset);
        es::simd<uint16_t, width> b;
        if constexpr (Packed) b = decode_peer_plane(reinterpret_cast<const uint8_t*>(other), tags, offset);
        else b = es::block_load<uint16_t, width>(other + offset);
        es::simd<uint32_t, width> a_bits = es::convert<uint32_t>(a) << 16;
        es::simd<uint32_t, width> b_bits = es::convert<uint32_t>(b) << 16;
        es::simd<float, width> sum = a_bits.bit_cast_view<float>() + b_bits.bit_cast_view<float>();
        es::simd<uint32_t, width> result = sum.bit_cast_view<uint32_t>();
        const auto magnitude = result & 0x7fffffffU;
        // Match the existing fadd_rn implementation for NaNs and signed zero
        // when cancelling opposite BF16 subnormals (or adding signed zeros).
        const auto nan = magnitude > 0x7f800000U;
        const auto zero = (magnitude == 0) & ((a & 0x7fffU) < 0x80);
        if ((nan | zero).any()) {
            result.merge(0x7fc00000U, nan);
            es::simd<uint32_t, width> sign = a_bits & 0x80000000U;
            sign.merge(b_bits & 0x80000000U, (a & 0x7fffU) == 0);
            result.merge(sign, zero);
        }
        if (!truncate) result += 0x7fffU + ((result >> 16) & 1U);
        es::simd<uint16_t, width> reduced = es::convert<uint16_t>(result >> 16);
        if constexpr (Gated) {
            auto as_f32 = [](es::simd<uint16_t, width> bits) SYCL_ESIMD_FUNCTION {
                es::simd<uint32_t, width> wide = es::convert<uint32_t>(bits) << 16;
                return es::simd<float, width>(wide.template bit_cast_view<float>());
            };
            auto round_bf16 = [](es::simd<float, width> value) SYCL_ESIMD_FUNCTION {
                es::simd<uint32_t, width> bits = value.template bit_cast_view<uint32_t>();
                // Triton preserves the quiet NaN payload in the high 16 bits.
                bits += 0x7fffU + ((bits >> 16) & 1U);
                return es::convert<uint16_t>(bits >> 16);
            };
            const auto gate_bits = es::block_load<uint16_t, width>(gate + offset % 4096);
            const auto residual_bits = es::block_load<uint16_t, width>(residual + offset);
            es::simd<float, width> product = as_f32(reduced) * as_f32(gate_bits);
            auto update_product = round_bf16(product);
            // Triton multiplies update * gate with the update as the first NaN source.
            update_product.merge(reduced, (reduced & 0x7fffU) > 0x7f80U);
            es::simd<float, width> out = as_f32(residual_bits) + as_f32(update_product);
            reduced = round_bf16(out);
        }
        es::block_store(x + offset, reduced);
    });
}

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

    void publish(sycl::queue& q, void* dest, const void* source, size_t bytes) {
        // Reuse a bank only after its previous consumer, including stream switches.
        // Submitting the copy itself on q keeps writes produced after that consumer ordered.
        if (sequence && last_queue != q) consumer.wait_and_throw();
        q.memcpy(dest, source, bytes).wait_and_throw();
        signal_peer();
    }

    void signal_peer() {
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
    }

    void exchange(sycl::queue& q, const void* input, void* output, size_t bytes) {
        if (!input || !output || !bytes || bytes > capacity * sizeof(uint16_t) ||
            !inbox || !inbox->local || !inbox->remote || !q.is_in_order() ||
            q.get_context() != owner.get_context() || q.get_device() != owner.get_device())
            throw std::invalid_argument("unsupported Flex peer byte exchange");
        const size_t offset = ((sequence + 1) & 1) * capacity;
        publish(q, inbox->remote + offset, input, bytes);
        consumer = q.memcpy(output, inbox->local + offset, bytes);
        last_queue = q;
    }

    void run(sycl::queue& q, uint16_t* x, size_t count, bool truncate, bool direct = false,
             const uint16_t* residual = nullptr, const uint16_t* gate = nullptr) {
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
        if (direct && residual && count == plane_elements && inbox->bytes > 4 * capacity) {
            if (sequence && last_queue != q) consumer.wait_and_throw();
            const size_t tag_offset = 4 * capacity + parity * plane_groups;
            auto* tags = reinterpret_cast<uint8_t*>(peer);
            encode_peer_planes(q, x, reinterpret_cast<uint8_t*>(dest), tags);
            // The in-order queue completes both the direct peer writes and this
            // last tag copy before either rank can consume the encoded bank.
            q.memcpy(reinterpret_cast<uint8_t*>(inbox->remote) + tag_offset, tags, plane_groups)
                .wait_and_throw();
            signal_peer();
            consumer = add_peer_blocks<true, true>(
                q, x, inbox->local + parity * capacity, count, truncate, residual, gate,
                reinterpret_cast<const uint8_t*>(inbox->local) + tag_offset);
            last_queue = q;
            return;
        }
        publish(q, dest, x, count * sizeof(uint16_t));
        // Only the DMA engine accesses the registered mapping. A GPU compute
        // load from another process's imported mapping is not coherent on Flex.
        auto* other = peer;
        if (direct) {
            other = inbox->local + parity * capacity;
        } else {
            q.memcpy(other, bank + (1-rank)*capacity, count * 2);
        }
        if (direct && count == 16 * 1024 * 1024 &&
            (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(other)) % 128 == 0) {
            consumer = residual
                ? add_peer_blocks<true>(q, x, other, count, truncate, residual, gate)
                : add_peer_blocks<false>(q, x, other, count, truncate);
            last_queue = q;
            return;
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

static int export_inbox(void* handle, int* fd, size_t extra_bytes) {
    try {
        auto& context = *static_cast<Context*>(handle);
        context.inbox = std::make_unique<PeerInbox>(context.owner, 4 * context.capacity + extra_bytes);
        context.inbox->allocate();
        *fd = context.inbox->export_fd;
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}
extern "C" int flex_tp2_peer_export(void* handle, int* fd) {
    return export_inbox(handle, fd, 0);
}
extern "C" int flex_tp2_peer_export_planes(void* handle, int* fd) {
    return export_inbox(handle, fd, 2 * plane_groups);
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

extern "C" int flex_tp2_peer_sum_gated(void* handle, void* queue, void* input,
                                       size_t count, int truncate,
                                       const void* residual, const void* gate) {
    try {
        if (!handle || !queue || !input || !residual || !gate || count != 16 * 1024 * 1024 ||
            (reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(residual) |
             reinterpret_cast<uintptr_t>(gate)) % 128 != 0) {
            throw std::invalid_argument("invalid gated sum geometry");
        }
        static_cast<Context*>(handle)->run(
            *static_cast<sycl::queue*>(queue), static_cast<uint16_t*>(input), count,
            truncate, true, static_cast<const uint16_t*>(residual),
            static_cast<const uint16_t*>(gate));
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}

extern "C" int flex_tp2_peer_exchange(void* handle, void* queue, const void* input,
                                     void* output, size_t bytes) {
    try {
        if (!handle || !queue) throw std::invalid_argument("missing Flex coordinator or queue");
        static_cast<Context*>(handle)->exchange(*static_cast<sycl::queue*>(queue), input, output, bytes);
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}

extern "C" int flex_tp2_local_update(void* handle, void* queue, uint16_t* first,
                                    const uint16_t* second, const uint16_t* residual,
                                    const uint16_t* gate, int truncate) {
    try {
        if (!handle || !queue || !first || !second || !residual || !gate)
            throw std::invalid_argument("missing Flex local update argument");
        auto& ctx = *static_cast<Context*>(handle);
        auto& q = *static_cast<sycl::queue*>(queue);
        const auto a = reinterpret_cast<uintptr_t>(first);
        const auto b = reinterpret_cast<uintptr_t>(second);
        const auto r = reinterpret_cast<uintptr_t>(residual);
        const auto g = reinterpret_cast<uintptr_t>(gate);
        constexpr size_t elements = 2048 * 4096;
        constexpr size_t bytes = elements * sizeof(uint16_t);
        if (!q.is_in_order() || q.get_context() != ctx.owner.get_context() ||
            q.get_device() != ctx.owner.get_device() || ((a | b | r | g) % 128) ||
            (a != b && a < b + bytes && b < a + bytes) ||
            (a != r && a < r + bytes && r < a + bytes) ||
            (a < g + 4096 * sizeof(uint16_t) && g < a + bytes))
            throw std::invalid_argument("unsupported Flex local update queue, alignment or overlap");
        add_peer_blocks<true>(q, first, second, elements, truncate, residual, gate);
        return 0;
    } catch (const std::exception& exc) { last_error = exc.what(); return -1; }
}
