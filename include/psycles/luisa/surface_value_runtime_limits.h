#pragma once

#include <psycles/compiler/surface_execution_plan.h>

namespace psycles::luisa_backend {

// Shared compact-interpreter ABI. Host planning and device-local arrays must
// use this one capacity vector so a legal plan cannot fail only after upload.
inline constexpr compiler::SurfaceValueStorageCapacity
    compact_surface_value_storage_capacity{
        .scalar_slots = 8u,
        .vector_slots = 12u,
        .unsigned_integer_slots = 1u};

} // namespace psycles::luisa_backend
