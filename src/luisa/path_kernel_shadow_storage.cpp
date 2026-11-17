#include "path_kernel_shadow_storage.h"

#include <psycles/luisa/surface_ray.h>

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {

ShadowIntersectionBatchStorage::ShadowIntersectionBatchStorage(
    luisa::compute::Device &device, std::uint32_t capacity) noexcept
    : _records{device.create_buffer<luisa::uint2>(
          static_cast<std::size_t>(total_uint2_slots) *
          static_cast<std::size_t>(capacity))},
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
  using namespace luisa::compute;
  const UInt base = (hit_index * words_per_hit) * runtime_capacity + invocation;
  const auto w0 = _records->read(base);
  const auto w1 = _records->read(base + runtime_capacity);
  const auto w2 = _records->read(base + runtime_capacity * 2u);
  Var<ShadowIntersectionCall> hit;
  hit->instance = w0.x;
  hit->primitive = w0.y;
  hit->distance = as<float>(w1.x);
  hit->hit_type = w1.y;
  hit->barycentric = make_float2(as<float>(w2.x), as<float>(w2.y));
  return hit;
}

luisa::compute::UInt2 ShadowIntersectionBatchStorage::read_identity(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> hit_index,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  return _records->read((hit_index * words_per_hit) * runtime_capacity +
                        invocation);
}

luisa::compute::Float ShadowIntersectionBatchStorage::read_distance(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> hit_index,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  return luisa::compute::as<float>(
      _records
          ->read((hit_index * words_per_hit + 1u) * runtime_capacity +
                 invocation)
          .x);
}

void ShadowIntersectionBatchStorage::write(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> hit_index,
    const luisa::compute::Var<ShadowIntersectionCall> &hit,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  using namespace luisa::compute;
  const UInt base = (hit_index * words_per_hit) * runtime_capacity + invocation;
  _records->write(base, make_uint2(hit->instance, hit->primitive));
  _records->write(base + runtime_capacity,
                  make_uint2(as<luisa::uint>(hit->distance), hit->hit_type));
  _records->write(base + runtime_capacity * 2u,
                  make_uint2(as<luisa::uint>(hit->barycentric.x),
                             as<luisa::uint>(hit->barycentric.y)));
}

void ShadowIntersectionBatchStorage::write_summary(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> count,
    luisa::compute::Expr<std::uint32_t> total,
    luisa::compute::Expr<std::uint32_t> blocked,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  using namespace luisa::compute;
  const UInt packed = (count & 7u) | ((blocked & 1u) << 3u);
  _records->write(summary_slot * runtime_capacity + invocation,
                  make_uint2(packed, total));
}

luisa::compute::Var<ShadowIntersectionSummaryCall>
ShadowIntersectionBatchStorage::read_summary(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  using namespace luisa::compute;
  const auto s = _records->read(summary_slot * runtime_capacity + invocation);
  Var<ShadowIntersectionSummaryCall> summary;
  summary->count = s.x & 7u;
  summary->blocked = (s.x >> 3u) & 1u;
  summary->total = s.y;
  return summary;
}

void ShadowIntersectionBatchStorage::write_batch(
    luisa::compute::Expr<std::uint32_t> invocation,
    const luisa::compute::Var<ShadowIntersectionBatchCall> &batch,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  for (auto index = std::size_t{0u}; index < shadow_intersection_batch_capacity;
       ++index) {
    write(invocation, static_cast<std::uint32_t>(index),
          batch->hits[static_cast<luisa::uint>(index)], runtime_capacity);
  }
  write_summary(invocation, batch->count, batch->total, batch->blocked,
                runtime_capacity);
}

luisa::compute::Var<ShadowIntersectionBatchCall>
ShadowIntersectionBatchStorage::read_batch(
    luisa::compute::Expr<std::uint32_t> invocation,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  luisa::compute::Var<ShadowIntersectionBatchCall> batch;
  for (auto index = std::size_t{0u}; index < shadow_intersection_batch_capacity;
       ++index) {
    batch->hits[static_cast<luisa::uint>(index)] =
        read(invocation, static_cast<std::uint32_t>(index), runtime_capacity);
  }
  const auto summary = read_summary(invocation, runtime_capacity);
  batch->count = summary->count;
  batch->total = summary->total;
  batch->blocked = summary->blocked;
  return batch;
}

void ShadowIntersectionBatchStorage::relocate(
    luisa::compute::Expr<std::uint32_t> source,
    luisa::compute::Expr<std::uint32_t> destination,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  for (std::uint32_t slot = 0u; slot < total_uint2_slots; ++slot) {
    const auto offset = slot * runtime_capacity;
    _records->write(offset + destination, _records->read(offset + source));
  }
}

luisa::compute::Var<ShadowIntersectionBatchCall>
ShadowIntersectionBatchStorage::materialize(
    luisa::compute::Expr<std::uint32_t> invocation,
    const luisa::compute::Var<ShadowIntersectionSummaryCall> &summary,
    luisa::compute::Expr<float> miss_distance,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept {
  luisa::compute::Var<ShadowIntersectionBatchCall> batch;
  for (auto index = std::size_t{0u}; index < shadow_intersection_batch_capacity;
       ++index) {
    auto &hit = batch->hits[static_cast<luisa::uint>(index)];
    hit->instance = surface_ray::invalid_primitive;
    hit->primitive = surface_ray::invalid_primitive;
    hit->hit_type = static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
    hit->distance = miss_distance;
    hit->barycentric = luisa::compute::make_float2(0.0f);
    $if(static_cast<std::uint32_t>(index) < summary->count) {
      hit =
          read(invocation, static_cast<std::uint32_t>(index), runtime_capacity);
    };
  }
  batch->count = summary->count;
  batch->total = summary->total;
  batch->blocked = summary->blocked;
  return batch;
}

} // namespace psycles::luisa_backend::detail
