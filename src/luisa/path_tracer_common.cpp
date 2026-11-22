#include "path_tracer_internal.h"

#include <bit>

namespace psycles::luisa_backend::detail {

[[nodiscard]] MaterialBindingGpu to_luisa(
    MaterialBinding binding) noexcept {
    return {
        .surface_tag = binding.surface_tag,
        .parameter_block = binding.parameter_block,
        .cycles_shader_index =
            binding.cycles_shader_index,
        .material_identity =
            binding.material_identity,
        .flags =
            (binding.flags &
             ~material_emission_sampling_mask) |
            material_emission_sampling_bits(
                static_cast<std::uint32_t>(
                    binding.emission_sampling)),
        .volume_sampling =
            static_cast<std::uint32_t>(
                binding.volume_sampling)};
}

[[nodiscard]] Var<ShaderEvaluationStateCall>
pack_shader_evaluation_state(
    const cycles_path_state::ShaderEvaluationState
        &state) noexcept {
    Var<ShaderEvaluationStateCall> result;
    result.ray_visibility = state.ray_visibility;
    result.ray_events = state.ray_events;
    result.ray_depth = state.ray_depth;
    result.diffuse_depth = state.diffuse_depth;
    result.glossy_depth = state.glossy_depth;
    result.transparent_depth = state.transparent_depth;
    result.transmission_depth = state.transmission_depth;
    result.portal_depth = state.portal_depth;
    return result;
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

[[nodiscard]] float scalar_parameter_value(
    const contract::SocketValue &value) noexcept {
    using contract::SocketType;
    switch (value.type) {
        case SocketType::boolean:
            return std::get<bool>(value.value) ? 1.0f : 0.0f;
        case SocketType::integer:
            return static_cast<float>(
                std::get<std::int64_t>(value.value));
        case SocketType::unsigned_integer:
            return static_cast<float>(
                std::get<std::uint64_t>(value.value));
        case SocketType::floating:
            return std::get<float>(value.value);
        default:
            std::abort();
    }
}

[[nodiscard]] luisa::float3 vector_parameter_value(
    const contract::SocketValue &value) noexcept {
    using contract::SocketType;
    switch (value.type) {
        case SocketType::float2: {
            const auto v = std::get<Vec2f>(value.value);
            return luisa::make_float3(v.x, v.y, 0.0f);
        }
        case SocketType::float3:
        case SocketType::color:
        case SocketType::spectrum:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal: {
            const auto v = std::get<Vec3f>(value.value);
            return luisa::make_float3(v.x, v.y, v.z);
        }
        default:
            std::abort();
    }
}

[[nodiscard]] luisa::float3 unsigned_parameter_value(
    const contract::SocketValue &value) noexcept {
    if (value.type != contract::SocketType::unsigned_integer) {
        std::abort();
    }
    const auto word = std::get<std::uint64_t>(value.value);
    return luisa::make_float3(
        std::bit_cast<float>(static_cast<std::uint32_t>(word)),
        std::bit_cast<float>(static_cast<std::uint32_t>(word >> 32u)),
        0.0f);
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
        case PassKind::volume_direct:
        case PassKind::volume_indirect:
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
