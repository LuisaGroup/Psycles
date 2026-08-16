#include "path_kernel_shadow_storage.h"

#include <psycles/luisa/surface_ray.h>

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

luisa::compute::Var<ShadowIntersectionBatchCall>
ShadowIntersectionBatchStorage::materialize(
    luisa::compute::Expr<std::uint32_t> invocation,
    const luisa::compute::Var<ShadowIntersectionSummaryCall> &summary,
    luisa::compute::Expr<float> miss_distance,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
    luisa::compute::Var<ShadowIntersectionBatchCall> batch;
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
        auto &hit = batch->hits[static_cast<luisa::uint>(index)];
        hit->instance = surface_ray::invalid_primitive;
        hit->primitive = surface_ray::invalid_primitive;
        hit->hit_type =
            static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
        hit->distance = miss_distance;
        hit->barycentric = luisa::compute::make_float2(0.0f);
        $if(static_cast<std::uint32_t>(index) < summary->count) {
            hit = read(invocation, static_cast<std::uint32_t>(index),
                       runtime_capacity);
        };
    }
    batch->count = summary->count;
    batch->total = summary->total;
    batch->blocked = summary->blocked;
    return batch;
}

}// namespace psycles::luisa_backend::detail
