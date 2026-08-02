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

// A proposal is deliberately radiometry-free. Cycles samples and rejects the
// light geometry before evaluating the emitter shader; keeping radiance out of
// this type makes that phase boundary impossible to bypass accidentally.
struct EmissiveTriangleLightProposal {
    EmissiveTriangleGeometryContext geometry;
    TriangleLightSample light;
    Float pdf;
    Bool valid;
};

struct EmissiveTrianglePdf {
    Float value;
    Bool valid;
};

// Shared host-stage component for mesh-emitter geometry, Cycles triangle
// measures and side selection. Raw emission-closure evaluation is a separate
// operation, matching Cycles' proposal -> geometric rejection -> shader
// evaluation state machine. Surface NEE, volume NEE, and forward-hit MIS call
// the same host-stage component while Luisa records each use directly into the
// fused path kernel.
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

    [[nodiscard]] virtual EmissiveTriangleLightProposal
    from_position(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index,
        Float3 reference,
        Float2 random) const noexcept = 0;

    [[nodiscard]] virtual Float3
    evaluate_emission(
        PathSampleContext &sample,
        const EmissiveTriangleLightProposal
            &proposal) const noexcept = 0;

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
