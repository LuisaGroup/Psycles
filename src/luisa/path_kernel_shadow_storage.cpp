#include "path_kernel_shadow_storage.h"

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] auto runtime_storage(
    const luisa::compute::SOA<ShadowIntersectionBatchCall> &storage,
    luisa::compute::Expr<std::uint32_t> capacity) noexcept {
    return luisa::compute::Expr<
        luisa::compute::SOA<ShadowIntersectionBatchCall>>{
        luisa::compute::Expr<
            luisa::compute::Buffer<std::uint32_t>>{storage.buffer()},
        luisa::compute::UInt{0u}, capacity, luisa::compute::UInt{0u}};
}

}// namespace

ShadowIntersectionBatchStorage::ShadowIntersectionBatchStorage(
    luisa::compute::Device &device, std::uint32_t capacity) noexcept
    : _batches{device.create_soa<ShadowIntersectionBatchCall>(capacity)},
      _capacity{capacity} {
    LUISA_ASSERT(capacity != 0u,
                 "Shadow-intersection storage capacity must be positive.");
}

std::uint32_t ShadowIntersectionBatchStorage::capacity() const noexcept {
    return _capacity;
}

luisa::compute::Var<ShadowIntersectionCall>
ShadowIntersectionBatchStorage::read(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> hit_index,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
    const auto storage = runtime_storage(_batches, runtime_capacity);
    return storage.hits[hit_index].read(invocation);
}

void ShadowIntersectionBatchStorage::write(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> hit_index,
    const luisa::compute::Var<ShadowIntersectionCall> &hit,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
    const auto storage = runtime_storage(_batches, runtime_capacity);
    storage.hits[hit_index].write(invocation, hit);
}

}// namespace psycles::luisa_backend::detail
