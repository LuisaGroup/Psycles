#pragma once

#include "path_tracer_types.h"

#include <cstdint>

#include <luisa/dsl/soa.h>
#include <luisa/runtime/device.h>

namespace psycles::luisa_backend::detail {

// Maps one active lane of a one-dimensional launch to its transient-storage
// owner. `physical_block_size` must equal the enclosing kernel's x block size;
// then block_x * physical_block_size + thread_x is injective over all active
// physical lanes, independently of coroutine logical IDs and queue order.
[[nodiscard]] inline auto shadow_storage_invocation(
    luisa::compute::Expr<std::uint32_t> physical_block_size) noexcept {
    return luisa::compute::block_x() * physical_block_size +
           luisa::compute::thread_x();
}

// Transient, invocation-indexed storage for the four shadow intersections
// retained during one traversal. The physical continuation lane is the owner:
// all candidate callbacks for that lane are serialized by RayQuery, while
// distinct lanes address distinct SoA elements and therefore need no atomics.
class ShadowIntersectionBatchStorage {

private:
    // Flat uint2 SoA layout (13 uint2 words = 104B per invocation):
    //   slots [hit*3 + 0]: (instance, primitive)
    //   slots [hit*3 + 1]: (as<uint>(distance), hit_type)
    //   slots [hit*3 + 2]: (as<uint>(barycentric.x), as<uint>(barycentric.y))
    //   slot  [12]:        ((count & 7u) | ((blocked & 1u) << 3u), total)
    luisa::compute::Buffer<luisa::uint2> _records;
    std::uint32_t _capacity{};

public:
    static constexpr std::uint32_t words_per_hit = 3u;
    static constexpr std::uint32_t summary_slot =
        static_cast<std::uint32_t>(shadow_intersection_batch_capacity) *
        words_per_hit;
    static constexpr std::uint32_t total_uint2_slots = summary_slot + 1u;

    ShadowIntersectionBatchStorage(luisa::compute::Device &device,
                                   std::uint32_t capacity) noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return _records.size_bytes();
    }

    [[nodiscard]] luisa::compute::Var<ShadowIntersectionCall>
    read(luisa::compute::Expr<std::uint32_t> invocation,
         luisa::compute::Expr<std::uint32_t> hit_index,
         luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    [[nodiscard]] luisa::compute::UInt2
    read_identity(luisa::compute::Expr<std::uint32_t> invocation,
                  luisa::compute::Expr<std::uint32_t> hit_index,
                  luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    [[nodiscard]] luisa::compute::Float
    read_distance(luisa::compute::Expr<std::uint32_t> invocation,
                  luisa::compute::Expr<std::uint32_t> hit_index,
                  luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    void write(luisa::compute::Expr<std::uint32_t> invocation,
               luisa::compute::Expr<std::uint32_t> hit_index,
               const luisa::compute::Var<ShadowIntersectionCall> &hit,
               luisa::compute::Expr<std::uint32_t> runtime_capacity) const
        noexcept;

    void write_summary(
        luisa::compute::Expr<std::uint32_t> invocation,
        luisa::compute::Expr<std::uint32_t> count,
        luisa::compute::Expr<std::uint32_t> total,
        luisa::compute::Expr<std::uint32_t> blocked,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    [[nodiscard]] luisa::compute::Var<ShadowIntersectionSummaryCall>
    read_summary(
        luisa::compute::Expr<std::uint32_t> invocation,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    void write_batch(
        luisa::compute::Expr<std::uint32_t> invocation,
        const luisa::compute::Var<ShadowIntersectionBatchCall> &batch,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    [[nodiscard]] luisa::compute::Var<ShadowIntersectionBatchCall>
    read_batch(
        luisa::compute::Expr<std::uint32_t> invocation,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    void relocate(
        luisa::compute::Expr<std::uint32_t> source,
        luisa::compute::Expr<std::uint32_t> destination,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    // Materialization is intentionally a consumer-side operation. Calling it
    // only after the summary RayQuery callable returns keeps the four retained
    // records outside the traversal callback ABI and private live range.
    [[nodiscard]] luisa::compute::Var<ShadowIntersectionBatchCall>
    materialize(
        luisa::compute::Expr<std::uint32_t> invocation,
        const luisa::compute::Var<ShadowIntersectionSummaryCall> &summary,
        luisa::compute::Expr<float> miss_distance,
        luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;
};

}// namespace psycles::luisa_backend::detail
