#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

[[nodiscard]] luisa::uint2 to_luisa(
    MaterialBinding binding) noexcept {
    return luisa::make_uint2(
        binding.surface_tag, binding.parameter_block);
}

[[nodiscard]] Var<SurfacePointCall> pack_surface_point(
    const SurfacePoint &point) noexcept {
    Var<SurfacePointCall> result;
    result.position = point.position;
    result.object_position = point.object_position;
    result.object_location = point.object_location;
    result.generated = point.generated;
    result.geometric_normal = point.geometric_normal;
    result.shading_normal = point.shading_normal;
    result.object_shading_normal =
        point.object_shading_normal;
    result.object_tangent = point.object_tangent;
    result.tangent_sign = point.tangent_sign;
    result.normal_to_world_x = point.normal_to_world_x;
    result.normal_to_world_y = point.normal_to_world_y;
    result.normal_to_world_z = point.normal_to_world_z;
    result.dpdu = point.dpdu;
    result.dpdv = point.dpdv;
    result.dPdx = point.dPdx;
    result.dPdy = point.dPdy;
    result.object_dPdx = point.object_dPdx;
    result.object_dPdy = point.object_dPdy;
    result.generated_dx = point.generated_dx;
    result.generated_dy = point.generated_dy;
    result.incoming = point.incoming;
    result.uv = point.uv;
    result.uv_dx = point.uv_dx;
    result.uv_dy = point.uv_dy;
    result.geometry_index = point.geometry_index;
    result.barycentric = point.barycentric;
    result.barycentric_dx = point.barycentric_dx;
    result.barycentric_dy = point.barycentric_dy;
    result.instance_id = point.instance_id;
    result.primitive_id = point.primitive_id;
    result.parameter_block = point.parameter_block;
    result.object_random = point.object_random;
    result.particle_index = point.particle_index;
    result.random_per_island = point.random_per_island;
    result.ray_visibility = point.ray_visibility;
    result.ray_events = point.ray_events;
    result.ray_depth = point.ray_depth;
    result.diffuse_depth = point.diffuse_depth;
    result.glossy_depth = point.glossy_depth;
    result.transparent_depth = point.transparent_depth;
    result.transmission_depth = point.transmission_depth;
    result.ray_length = point.ray_length;
    result.time = point.time;
    result.back_facing = select(
        0u, 1u, point.back_facing);
    return result;
}

[[nodiscard]] SurfacePoint unpack_surface_point(
    const Var<SurfacePointCall> &point) noexcept {
    return {
        .position = point.position,
        .object_position = point.object_position,
        .object_location = point.object_location,
        .generated = point.generated,
        .geometric_normal = point.geometric_normal,
        .shading_normal = point.shading_normal,
        .object_shading_normal =
            point.object_shading_normal,
        .object_tangent = point.object_tangent,
        .tangent_sign = point.tangent_sign,
        .normal_to_world_x = point.normal_to_world_x,
        .normal_to_world_y = point.normal_to_world_y,
        .normal_to_world_z = point.normal_to_world_z,
        .dpdu = point.dpdu,
        .dpdv = point.dpdv,
        .dPdx = point.dPdx,
        .dPdy = point.dPdy,
        .object_dPdx = point.object_dPdx,
        .object_dPdy = point.object_dPdy,
        .generated_dx = point.generated_dx,
        .generated_dy = point.generated_dy,
        .incoming = point.incoming,
        .uv = point.uv,
        .uv_dx = point.uv_dx,
        .uv_dy = point.uv_dy,
        .geometry_index = point.geometry_index,
        .barycentric = point.barycentric,
        .barycentric_dx = point.barycentric_dx,
        .barycentric_dy = point.barycentric_dy,
        .instance_id = point.instance_id,
        .primitive_id = point.primitive_id,
        .parameter_block = point.parameter_block,
        .object_random = point.object_random,
        .particle_index = point.particle_index,
        .random_per_island = point.random_per_island,
        .ray_visibility = point.ray_visibility,
        .ray_events = point.ray_events,
        .ray_depth = point.ray_depth,
        .diffuse_depth = point.diffuse_depth,
        .glossy_depth = point.glossy_depth,
        .transparent_depth = point.transparent_depth,
        .transmission_depth = point.transmission_depth,
        .ray_length = point.ray_length,
        .time = point.time,
        .back_facing = point.back_facing != 0u};
}

[[nodiscard]] Var<SurfaceEvaluationCall>
pack_surface_evaluation(
    const SurfaceEvaluation &evaluation) noexcept {
    Var<SurfaceEvaluationCall> result;
    result.f = evaluation.f;
    result.pdf = evaluation.pdf;
    result.diffuse_f = evaluation.diffuse_f;
    result.diffuse_pdf = evaluation.diffuse_pdf;
    result.events = evaluation.events;
    return result;
}

[[nodiscard]] SurfaceEvaluation unpack_surface_evaluation(
    const Var<SurfaceEvaluationCall> &evaluation) noexcept {
    return {
        .f = evaluation.f,
        .pdf = evaluation.pdf,
        .diffuse_f = evaluation.diffuse_f,
        .diffuse_pdf = evaluation.diffuse_pdf,
        .events = evaluation.events};
}

[[nodiscard]] Var<SurfaceSampleCall> pack_surface_sample(
    const SurfaceSample &sample) noexcept {
    Var<SurfaceSampleCall> result;
    result.f = sample.evaluation.f;
    result.pdf = sample.evaluation.pdf;
    result.diffuse_f = sample.evaluation.diffuse_f;
    result.diffuse_pdf = sample.evaluation.diffuse_pdf;
    result.events = sample.evaluation.events;
    result.wi = sample.wi;
    result.eta = sample.eta;
    result.roughness = sample.roughness;
    result.runtime_flags = sample.runtime_flags;
    result.valid = select(0u, 1u, sample.valid);
    return result;
}

[[nodiscard]] SurfaceSample unpack_surface_sample(
    const Var<SurfaceSampleCall> &sample) noexcept {
    return {
        .evaluation = {
            .f = sample.f,
            .pdf = sample.pdf,
            .diffuse_f = sample.diffuse_f,
            .diffuse_pdf = sample.diffuse_pdf,
            .events = sample.events},
        .wi = sample.wi,
        .eta = sample.eta,
        .roughness = sample.roughness,
        .runtime_flags = sample.runtime_flags,
        .valid = sample.valid != 0u};
}

[[nodiscard]] Var<SurfaceClosureTraceCall>
pack_surface_closure_trace(
    const SurfaceClosureTrace &trace) noexcept {
    Var<SurfaceClosureTraceCall> result;
    result.count = trace.count;
    result.runtime_flags = trace.runtime_flags;
    result.index = trace.index;
    result.type = trace.type;
    result.sample_weight = trace.sample_weight;
    result.weight = trace.weight;
    result.normal = trace.normal;
    result.valid = select(0u, 1u, trace.valid);
    return result;
}

[[nodiscard]] SurfaceClosureTrace
unpack_surface_closure_trace(
    const Var<SurfaceClosureTraceCall> &trace) noexcept {
    return {
        .count = trace.count,
        .runtime_flags = trace.runtime_flags,
        .index = trace.index,
        .type = trace.type,
        .sample_weight = trace.sample_weight,
        .weight = trace.weight,
        .normal = trace.normal,
        .valid = trace.valid != 0u};
}

[[nodiscard]] Var<SurfaceSampleTraceCall>
pack_surface_sample_trace(
    const SurfaceSampleTrace &trace) noexcept {
    Var<SurfaceSampleTraceCall> result;
    result.f = trace.sample.evaluation.f;
    result.pdf = trace.sample.evaluation.pdf;
    result.diffuse_f =
        trace.sample.evaluation.diffuse_f;
    result.diffuse_pdf =
        trace.sample.evaluation.diffuse_pdf;
    result.events = trace.sample.evaluation.events;
    result.wi = trace.sample.wi;
    result.eta = trace.sample.eta;
    result.roughness = trace.sample.roughness;
    result.runtime_flags =
        trace.sample.runtime_flags;
    result.valid =
        select(0u, 1u, trace.sample.valid);
    result.closure_index = trace.closure_index;
    result.closure_type = trace.closure_type;
    result.closure_sample_weight =
        trace.closure_sample_weight;
    result.selection_rescaled =
        trace.selection_rescaled;
    result.closure_weight = trace.closure_weight;
    result.closure_normal = trace.closure_normal;
    result.closure_valid =
        select(0u, 1u, trace.closure_valid);
    return result;
}

[[nodiscard]] SurfaceSampleTrace
unpack_surface_sample_trace(
    const Var<SurfaceSampleTraceCall> &trace) noexcept {
    return {
        .sample = {
            .evaluation = {
                .f = trace.f,
                .pdf = trace.pdf,
                .diffuse_f = trace.diffuse_f,
                .diffuse_pdf = trace.diffuse_pdf,
                .events = trace.events},
            .wi = trace.wi,
            .eta = trace.eta,
            .roughness = trace.roughness,
            .runtime_flags = trace.runtime_flags,
            .valid = trace.valid != 0u},
        .closure_index = trace.closure_index,
        .closure_type = trace.closure_type,
        .closure_sample_weight =
            trace.closure_sample_weight,
        .selection_rescaled =
            trace.selection_rescaled,
        .closure_weight = trace.closure_weight,
        .closure_normal = trace.closure_normal,
        .closure_valid =
            trace.closure_valid != 0u};
}

[[nodiscard]] Var<SurfaceAovCall> pack_surface_aov(
    const SurfaceAov &aov) noexcept {
    Var<SurfaceAovCall> result;
    result.albedo = aov.albedo;
    result.glossy_albedo = aov.glossy_albedo;
    result.transmission_albedo = aov.transmission_albedo;
    result.roughness = aov.roughness;
    result.normal = aov.normal;
    result.transparency = aov.transparency;
    return result;
}

[[nodiscard]] SurfaceAov unpack_surface_aov(
    const Var<SurfaceAovCall> &aov) noexcept {
    return {
        .albedo = aov.albedo,
        .glossy_albedo = aov.glossy_albedo,
        .transmission_albedo = aov.transmission_albedo,
        .roughness = aov.roughness,
        .normal = aov.normal,
        .transparency = aov.transparency};
}

[[nodiscard]] luisa::float3 to_luisa(Vec3f value) noexcept {
    return luisa::make_float3(value.x, value.y, value.z);
}

[[nodiscard]] luisa::float2 to_luisa(Vec2f value) noexcept {
    return luisa::make_float2(value.x, value.y);
}

[[nodiscard]] luisa::float4x4 to_luisa(Mat4f value) noexcept {
    const auto &e = value.elements;
    return make_float4x4(
        luisa::make_float4(e[0u], e[1u], e[2u], e[3u]),
        luisa::make_float4(e[4u], e[5u], e[6u], e[7u]),
        luisa::make_float4(e[8u], e[9u], e[10u], e[11u]),
        luisa::make_float4(e[12u], e[13u], e[14u], e[15u]));
}

[[nodiscard]] Vec3f matrix_axis(
    const Mat4f &matrix,
    std::size_t column) noexcept {
    const auto offset = column * 4u;
    return {
        matrix.elements[offset],
        matrix.elements[offset + 1u],
        matrix.elements[offset + 2u]};
}

[[nodiscard]] Vec3f matrix_translation(
    const Mat4f &matrix) noexcept {
    return {
        matrix.elements[12u],
        matrix.elements[13u],
        matrix.elements[14u]};
}

[[nodiscard]] luisa::float4 parameter_value(
    const contract::SocketValue &value) noexcept {
    using contract::SocketType;
    switch (value.type) {
        case SocketType::boolean:
            return luisa::make_float4(
                std::get<bool>(value.value) ? 1.0f : 0.0f);
        case SocketType::integer:
            return luisa::make_float4(
                static_cast<float>(
                    std::get<std::int64_t>(value.value)));
        case SocketType::unsigned_integer:
            return luisa::make_float4(
                static_cast<float>(
                    std::get<std::uint64_t>(value.value)));
        case SocketType::floating:
            return luisa::make_float4(
                std::get<float>(value.value));
        case SocketType::float2: {
            const auto v = std::get<Vec2f>(value.value);
            return luisa::make_float4(v.x, v.y, 0.0f, 0.0f);
        }
        case SocketType::float3:
        case SocketType::color:
        case SocketType::spectrum:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal: {
            const auto v = std::get<Vec3f>(value.value);
            return luisa::make_float4(v.x, v.y, v.z, 0.0f);
        }
        default:
            return luisa::make_float4(0.0f);
    }
}

[[nodiscard]] PixelWindow effective_window(
    const RenderSettings &settings) noexcept {
    if (settings.window.width != 0u &&
        settings.window.height != 0u) {
        return settings.window;
    }
    return {
        .x = 0u,
        .y = 0u,
        .width = settings.full_extent.width,
        .height = settings.full_extent.height};
}

[[nodiscard]] std::uint32_t pass_channels(
    const PassRequest &pass) noexcept {
    return std::max(pass.channels, 1u);
}

[[nodiscard]] luisa::float3 safe_divide_even_color(
    const luisa::float4 &numerator,
    const luisa::float4 &denominator) noexcept {
    auto x = denominator.x != 0.0f
                 ? numerator.x / denominator.x
                 : 0.0f;
    auto y = denominator.y != 0.0f
                 ? numerator.y / denominator.y
                 : 0.0f;
    auto z = denominator.z != 0.0f
                 ? numerator.z / denominator.z
                 : 0.0f;
    if (denominator.x == 0.0f) {
        if (denominator.y == 0.0f) {
            x = z;
            y = z;
        } else if (denominator.z == 0.0f) {
            x = y;
            z = y;
        } else {
            x = 0.5f * (y + z);
        }
    } else if (denominator.y == 0.0f) {
        if (denominator.z == 0.0f) {
            y = x;
            z = x;
        } else {
            y = 0.5f * (x + z);
        }
    } else if (denominator.z == 0.0f) {
        z = 0.5f * (x + y);
    }
    return luisa::make_float3(x, y, z);
}

[[nodiscard]] bool supported_pass(PassKind pass) noexcept {
    switch (pass) {
        case PassKind::combined:
        case PassKind::normal:
        case PassKind::albedo:
        case PassKind::glossy_color:
        case PassKind::transmission_color:
        case PassKind::emission:
        case PassKind::environment:
        case PassKind::diffuse_direct:
        case PassKind::diffuse_indirect:
        case PassKind::glossy_direct:
        case PassKind::glossy_indirect:
        case PassKind::transmission_direct:
        case PassKind::transmission_indirect:
        case PassKind::denoising_normal:
        case PassKind::denoising_albedo:
        case PassKind::sample_count:
            return true;
        default:
            return false;
    }
}

void diagnose(
    std::vector<RenderDiagnostic> &diagnostics,
    std::string message) {
    diagnostics.emplace_back(RenderDiagnostic{
        .message = std::move(message)});
}


}// namespace psycles::luisa_backend::detail
