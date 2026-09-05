#pragma once

#include <psycles/core/math.h>

namespace psycles::compiler {

// Cycles' affine Transform inverse in Psycles' column-major Mat4f storage.
// This is a host scene/graph-compilation operation; it is never emitted as device
// work or used to pre-bake Blender scene data.
[[nodiscard]] Mat4f
cycles_inverse_affine_transform(const Mat4f &transform) noexcept;

} // namespace psycles::compiler
