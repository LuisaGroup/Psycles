#pragma once

namespace psycles::luisa_backend::detail {

struct GeometryUpload;

// Rebuild the current corner tangent attributes from the current positions,
// normals, topology, and raw UV layers. The implementation is the same
// MikkTSpace revision used by the aligned Cycles checkout.
void recompute_cycles_tangent_space(GeometryUpload &upload);

// Named ORIGINAL-base attributes are separate immutable bindings. Call this
// once after constructing the initial MikkTSpace frames and before any true
// displacement modifies the current attributes.
void initialize_cycles_undisplaced_tangent_space(GeometryUpload &upload);

}// namespace psycles::luisa_backend::detail
