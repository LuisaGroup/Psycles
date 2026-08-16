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
    luisa::compute::SOA<ShadowIntersectionBatchCall> _batches;
    std::uint32_t _capacity{};

public:
    ShadowIntersectionBatchStorage(luisa::compute::Device &device,
                                   std::uint32_t capacity) noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] luisa::compute::Var<ShadowIntersectionCall>
    read(luisa::compute::Expr<std::uint32_t> invocation,
         luisa::compute::Expr<std::uint32_t> hit_index,
         luisa::compute::Expr<std::uint32_t> runtime_capacity) const noexcept;

    void write(luisa::compute::Expr<std::uint32_t> invocation,
               luisa::compute::Expr<std::uint32_t> hit_index,
               const luisa::compute::Var<ShadowIntersectionCall> &hit,
               luisa::compute::Expr<std::uint32_t> runtime_capacity) const
        noexcept;
};

}// namespace psycles::luisa_backend::detail
