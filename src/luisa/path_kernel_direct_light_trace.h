#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct DirectLightSampleRecord {
    UInt type;
    UInt emitter_id;
    UInt primitive;
    UInt object;
    Int light_group;
    UInt shader;
    Float pdf;
    Float selection_pdf;
    Float evaluation_factor;
    Float3 direction;
    Float3 position;
    Float3 geometric_normal;
    Float distance;
};

struct DirectLightEvaluationRecord {
    Float distance;
    Float bsdf_pdf;
    Float mis_weight;
};

// Host-stage trace policy. The enabled implementation records Luisa DSL AST
// at Cycles' semantic boundaries; the null implementation emits no shader
// statements, branches, or runtime storage accesses.
class DirectLightTraceRecorder {

  public:
    virtual ~DirectLightTraceRecorder() noexcept = default;

    // Records a proposal attempt which reached Cycles'
    // light_sample_from_position boundary but returned false. An NEE stage
    // which is disabled before that boundary must not call this method.
    virtual void record_failed_sample(
        PathBounceContext &bounce) const noexcept = 0;
    virtual void record_sample(
        PathBounceContext &bounce,
        const DirectLightSampleRecord &record)
        const noexcept = 0;
    virtual void record_evaluation(
        PathBounceContext &bounce,
        const DirectLightEvaluationRecord &record)
        const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<
    const DirectLightTraceRecorder>
make_direct_light_trace_recorder(bool enabled);

}// namespace psycles::luisa_backend::detail
