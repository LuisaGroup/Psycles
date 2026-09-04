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

struct SurfaceVectorDisplacementInput {
  Float3 vector;
  Float midlevel;
  Float scale;
  Float3 shading_normal;
  Float3 object_tangent;
  Float tangent_sign;
  Bool tangent_attribute_found;
  Bool tangent_sign_found;
  Float3 dpdu;
  Float3 normal_to_world_x;
  Float3 normal_to_world_y;
  Float3 normal_to_world_z;
};

struct SurfaceVectorDisplacementTangent {
  Float3 object_tangent;
  Float tangent_sign;
  Bool tangent_attribute_found;
  Bool tangent_sign_found;
};

// Cycles materializes the unnamed UV tangent pair as standard attributes on
// meshes. Curves have no such attribute domain, so the Vector Displacement
// node must independently fall back to normalize(ShaderData::dPdu) and +1.
[[nodiscard]] SurfaceVectorDisplacementTangent
vector_displacement_default_tangent(Float3 object_tangent, Float tangent_sign,
                                    Bool is_curve,
                                    UInt geometry_index) noexcept;

// Exact scalar Displacement endpoints from Cycles 5.2
// svm_node_displacement. WORLD deliberately does not normalize a linked
// normal. OBJECT applies M^T to the world normal, normalizes in object space,
// and transforms the resulting displacement direction by M.
[[nodiscard]] Float3
displacement_world_inline(const SurfaceDisplacementInput &input) noexcept;

[[nodiscard]] Float3
displacement_object_inline(const SurfaceDisplacementInput &input) noexcept;

// Exact Vector Displacement endpoint from Cycles 5.2.1. `space` contains the
// NodeNormalMapSpace value (tangent=0, object=1, world=2); authored vector,
// midlevel and scale remain ordinary device values.
[[nodiscard]] Float3 vector_displacement_inline(
    const SurfaceVectorDisplacementInput &input,
    UInt space) noexcept;

} // namespace psycles::luisa_backend::detail
