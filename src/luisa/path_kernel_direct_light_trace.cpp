#include "path_kernel_direct_light_trace.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float trace_identity(UInt value) noexcept {
    return select(
        -1.0f,
        cast<float>(value),
        value != surface_ray::invalid_primitive);
}

class NullDirectLightTraceRecorder final
    : public DirectLightTraceRecorder {

  public:
    void record_failed_sample(
        PathBounceContext &,
        const DirectLightFailedSampleRecord &)
        const noexcept override {}

    void record_sample(
        PathBounceContext &,
        const DirectLightSampleRecord &)
        const noexcept override {}

    void record_evaluation(
        PathBounceContext &,
        const DirectLightEvaluationRecord &)
        const noexcept override {}

    void record_weighted_bsdf(
        PathBounceContext &,
        Float3) const noexcept override {}

    void record_transport(
        PathBounceContext &,
        const DirectLightTransportRecord &)
        const noexcept override {}

    void record_shadow(
        PathBounceContext &,
        const DirectLightShadowRecord &)
        const noexcept override {}

    void record_contribution(
        PathBounceContext &,
        Float3) const noexcept override {}
};

class CyclesDirectLightTraceRecorder final
    : public DirectLightTraceRecorder {

  public:
    void record_failed_sample(
        PathBounceContext &bounce,
        const DirectLightFailedSampleRecord &record)
        const noexcept override {
        auto &sample = bounce.sample;
        const auto &event = bounce.path_step;
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_meta,
            make_float3(
                -1.0f,
                cast<float>(record.emitter_id),
                cast<float>(record.primitive)));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_id,
            make_float3(
                cast<float>(record.object),
                cast<float>(record.visibility_flag),
                0.0f));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_pdf,
            make_float3(
                0.0f,
                record.selection_pdf,
                0.0f));
    }

    void record_sample(
        PathBounceContext &bounce,
        const DirectLightSampleRecord &record)
        const noexcept override {
        auto &sample = bounce.sample;
        const auto &event = bounce.path_step;
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_meta,
            make_float3(
                cast<float>(record.type),
                cast<float>(record.emitter_id),
                cast<float>(record.primitive)));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_id,
            make_float3(
                cast<float>(record.object),
                cast<float>(record.light_group),
                0.0f));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_shader,
            sample.trace_uint32(record.shader));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_pdf,
            make_float3(
                record.pdf,
                record.selection_pdf,
                record.evaluation_factor));
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_d,
            record.direction);
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_p,
            record.position);
        sample.trace_write_event(
            event,
            path_trace_schema::EventSlot::light_ng,
            record.geometric_normal);
    }

    void record_evaluation(
        PathBounceContext &bounce,
        const DirectLightEvaluationRecord &record)
        const noexcept override {
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::light_eval,
            make_float3(
                record.distance,
                record.bsdf_pdf,
                record.mis_weight));
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_bsdf,
            record.bsdf);
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_diffuse,
            record.diffuse);
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_glossy,
            record.glossy);
    }

    void record_weighted_bsdf(
        PathBounceContext &bounce,
        Float3 weighted_bsdf) const noexcept override {
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_weighted_bsdf,
            weighted_bsdf);
    }

    void record_transport(
        PathBounceContext &bounce,
        const DirectLightTransportRecord &record)
        const noexcept override {
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_light_shader,
            record.light_shader);
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_unshadowed,
            record.unshadowed);
    }

    void record_shadow(
        PathBounceContext &bounce,
        const DirectLightShadowRecord &record)
        const noexcept override {
        auto &sample = bounce.sample;
        const auto &event = bounce.path_step;
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_ray_p,
            record.origin);
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_ray_d,
            record.direction);
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_ray_range,
            make_float3(
                record.minimum,
                record.maximum,
                select(0.0f, 1.0f, record.cast_shadow)));
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_source,
            make_float3(
                trace_identity(record.source_object),
                trace_identity(record.source_primitive),
                select(0.0f, 1.0f, record.skip_self)));
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_light,
            make_float3(
                trace_identity(record.light_object),
                trace_identity(record.light_primitive),
                0.0f));
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_hit_id,
            make_float3(
                select(-1.0f,
                       trace_identity(record.first_object),
                       record.first_hit),
                select(-1.0f,
                       trace_identity(record.first_primitive),
                       record.first_hit),
                select(-1.0f,
                       trace_identity(record.first_kind),
                       record.first_hit)));
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_hit_coord,
            select(make_float3(0.0f),
                   make_float3(record.first_distance,
                               record.first_barycentric),
                   record.first_hit));
        sample.trace_write_shadow_event(
            event,
            path_trace_schema::ShadowEventSlot::shadow_transmittance,
            record.transmittance);
    }

    void record_contribution(
        PathBounceContext &bounce,
        Float3 contribution) const noexcept override {
        bounce.sample.trace_write_event(
            bounce.path_step,
            path_trace_schema::EventSlot::nee_contribution,
            contribution);
    }
};

}// namespace

std::shared_ptr<const DirectLightTraceRecorder>
make_direct_light_trace_recorder(bool enabled) {
    if (enabled) {
        return std::make_shared<
            CyclesDirectLightTraceRecorder>();
    }
    return std::make_shared<
        NullDirectLightTraceRecorder>();
}

}// namespace psycles::luisa_backend::detail
