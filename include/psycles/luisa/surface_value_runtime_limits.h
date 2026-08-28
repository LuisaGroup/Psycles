#pragma once

#include <psycles/compiler/surface_svm_program.h>

namespace psycles::luisa_backend {

// Cycles 5.2 intern/cycles/kernel/svm/types.h::SVM_STACK_SIZE. The replacement
// interpreter uses the same semantic upper bound: every computed surface
// value and closure weight is allocated in one 32-bit lane stack, while
// authored parameters remain in immutable SoA buffers.
inline constexpr std::uint32_t surface_svm_stack_lane_capacity =
    compiler::surface_svm_stack_lane_capacity;

// Component bounds are implied by the physical lane limit and exist only to
// stop a typed coloring before its lane packing can overflow. They are not
// independent, scene-facing limits.
inline constexpr compiler::SurfaceValueStorageCapacity
    compact_surface_value_storage_capacity{
        .scalar_slots = surface_svm_stack_lane_capacity,
        .vector_slots = surface_svm_stack_lane_capacity / 3u,
        .unsigned_integer_slots = surface_svm_stack_lane_capacity / 2u,
        .stack_lanes = surface_svm_stack_lane_capacity};

} // namespace psycles::luisa_backend
