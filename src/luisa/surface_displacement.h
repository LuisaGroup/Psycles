#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

struct SurfaceDisplacementInput {
  Float height;
  Float midlevel;
  Float scale;
  Float3 normal;
  Float3 normal_to_world_x;
  Float3 normal_to_world_y;
  Float3 normal_to_world_z;
};

// Exact scalar Displacement endpoints from Cycles 5.2
// svm_node_displacement. WORLD deliberately does not normalize a linked
// normal. OBJECT applies M^T to the world normal, normalizes in object space,
// and transforms the resulting displacement direction by M.
[[nodiscard]] Float3
displacement_world_inline(const SurfaceDisplacementInput &input) noexcept;

[[nodiscard]] Float3
displacement_object_inline(const SurfaceDisplacementInput &input) noexcept;

} // namespace psycles::luisa_backend::detail
