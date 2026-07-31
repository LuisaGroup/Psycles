#pragma once

#include "path_kernel_builder.h"
#include "path_kernel_triangle_geometry.h"

#include <psycles/luisa/triangle_light_sampling.h>

#include <memory>

namespace psycles::luisa_backend::detail {

struct EmissiveTriangleGeometryContext {
    Var<EmissiveTriangleGpu> emitter;
    TriangleGeometryContext triangle;
    Float3 p0;
    Float3 p1;
    Float3 p2;
    Float3 geometric_normal;
    Float area;
    Bool sample_front;
    Bool sample_back;
};

struct EmissiveTriangleSegmentSample {
    EmissiveTriangleGeometryContext geometry;
    TriangleLightSample light;
    Bool valid;
};

struct EmissiveTriangleLightSample {
    EmissiveTriangleGeometryContext geometry;
    TriangleLightSample light;
    Float3 radiance;
    Float pdf;
    Bool valid;
};

struct EmissiveTrianglePdf {
    Float value;
    Bool valid;
};

// Shared host-stage component for mesh-emitter geometry, Cycles triangle
// measures, side selection, and raw emission-closure evaluation. Surface NEE,
// volume NEE, and forward-hit MIS call the same semantic boundary while Luisa
// still records each use directly into the fused path kernel.
class EmissiveTriangleComponent {

  public:
    virtual ~EmissiveTriangleComponent() noexcept =
        default;

    [[nodiscard]] virtual EmissiveTriangleSegmentSample
    from_segment(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index,
        Float3 reference,
        Float2 random) const noexcept = 0;

    [[nodiscard]] virtual EmissiveTriangleLightSample
    from_position(
        PathSampleContext &sample,
        UInt emitter_index,
        Float3 reference,
        Float2 random) const noexcept = 0;

    [[nodiscard]] virtual EmissiveTrianglePdf
    from_intersection(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt instance_index,
        UInt primitive_index,
        Float3 reference,
        Float3 light_position,
        Float3 p0,
        Float3 p1,
        Float3 p2,
        Float3 oriented_geometric_normal)
        const noexcept = 0;
};

[[nodiscard]]
std::shared_ptr<const EmissiveTriangleComponent>
make_emissive_triangle_component();

}// namespace psycles::luisa_backend::detail
