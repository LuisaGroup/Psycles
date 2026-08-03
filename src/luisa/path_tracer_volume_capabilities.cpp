#include "path_tracer_volume_capabilities.h"

#include <array>
#include <cstdint>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool is_spatial_source(
    compiler::ValueOperation operation) noexcept {
    using compiler::ValueOperation;
    switch (operation) {
        case ValueOperation::surface_position:
        case ValueOperation::shading_normal:
        case ValueOperation::geometric_normal:
        case ValueOperation::incoming:
        case ValueOperation::tangent:
        case ValueOperation::uv:
        case ValueOperation::generated:
        case ValueOperation::object_position:
        case ValueOperation::object_location:
        case ValueOperation::object_random:
        case ValueOperation::particle_index:
        case ValueOperation::particle_random:
        case ValueOperation::back_facing:
        case ValueOperation::pointiness:
        case ValueOperation::random_per_island:
        case ValueOperation::image_color:
        case ValueOperation::image_alpha:
        case ValueOperation::environment_color:
        case ValueOperation::environment_alpha:
        case ValueOperation::attribute_color:
        case ValueOperation::attribute_factor:
        case ValueOperation::attribute_alpha:
        case ValueOperation::normal_map:
        case ValueOperation::bump:
        case ValueOperation::noise_factor:
        case ValueOperation::noise_color:
        case ValueOperation::white_noise_value:
        case ValueOperation::white_noise_color:
        case ValueOperation::checker_color:
        case ValueOperation::checker_factor:
        case ValueOperation::brick_color:
        case ValueOperation::brick_factor:
        case ValueOperation::wave_color:
        case ValueOperation::wave_factor:
        case ValueOperation::voronoi_distance:
        case ValueOperation::voronoi_color:
        case ValueOperation::voronoi_position:
        case ValueOperation::voronoi_w:
        case ValueOperation::voronoi_radius:
        case ValueOperation::gradient:
        case ValueOperation::hosek_wilkie_sky:
        case ValueOperation::nishita_sky:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool is_light_path_source(
    compiler::ValueOperation operation) noexcept {
    using compiler::ValueOperation;
    switch (operation) {
        case ValueOperation::path_is_camera:
        case ValueOperation::path_is_shadow:
        case ValueOperation::path_is_diffuse:
        case ValueOperation::path_is_glossy:
        case ValueOperation::path_is_singular:
        case ValueOperation::path_is_reflection:
        case ValueOperation::path_is_transmission:
        case ValueOperation::path_is_volume_scatter:
        case ValueOperation::path_ray_length:
        case ValueOperation::path_ray_depth:
        case ValueOperation::path_diffuse_depth:
        case ValueOperation::path_glossy_depth:
        case ValueOperation::path_transparent_depth:
        case ValueOperation::path_transmission_depth:
            return true;
        default:
            return false;
    }
}

class HomogeneousVolumeAnalysis {

  private:
    enum class Visit : std::uint8_t {
        unseen,
        active,
        homogeneous,
        varying
    };

    const compiler::SurfaceProgram &_program;
    std::vector<Visit> _values;
    std::vector<Visit> _volumes;

    [[nodiscard]] bool value(
        compiler::ValueExpressionId id) noexcept {
        if (!id.valid()) {
            return true;
        }
        if (id.value >= _values.size()) {
            return false;
        }
        auto &visit = _values[id.value];
        if (visit == Visit::homogeneous) {
            return true;
        }
        if (visit == Visit::varying ||
            visit == Visit::active) {
            return false;
        }
        visit = Visit::active;
        const auto &instruction =
            _program.value_instructions()[id.value];
        auto homogeneous =
            !is_spatial_source(instruction.operation);
        const std::array dependencies{
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
            instruction.f,
            instruction.g,
            instruction.h,
            instruction.i,
            instruction.j};
        for (const auto dependency : dependencies) {
            const auto dependency_homogeneous =
                value(dependency);
            homogeneous =
                homogeneous &&
                dependency_homogeneous;
        }
        visit = homogeneous
                    ? Visit::homogeneous
                    : Visit::varying;
        return homogeneous;
    }

    [[nodiscard]] bool volume(
        compiler::VolumeExpressionId id) noexcept {
        if (!id.valid()) {
            return true;
        }
        if (id.value >= _volumes.size()) {
            return false;
        }
        auto &visit = _volumes[id.value];
        if (visit == Visit::homogeneous) {
            return true;
        }
        if (visit == Visit::varying ||
            visit == Visit::active) {
            return false;
        }
        visit = Visit::active;
        const auto &instruction =
            _program.volume_instructions()[id.value];
        const std::array values{
            instruction.color,
            instruction.density,
            instruction.anisotropy,
            instruction.ior,
            instruction.backscatter,
            instruction.alpha,
            instruction.diameter,
            instruction.scatter_coefficients,
            instruction.absorption_coefficients,
            instruction.absorption_color,
            instruction.emission_coefficients,
            instruction.emission_strength,
            instruction.emission_color,
            instruction.blackbody_intensity,
            instruction.blackbody_tint,
            instruction.temperature,
            instruction.factor};
        auto homogeneous = true;
        for (const auto dependency : values) {
            const auto dependency_homogeneous =
                value(dependency);
            homogeneous =
                homogeneous &&
                dependency_homogeneous;
        }
        const auto a_homogeneous =
            volume(instruction.a);
        const auto b_homogeneous =
            volume(instruction.b);
        homogeneous =
            homogeneous &&
            a_homogeneous &&
            b_homogeneous;
        visit = homogeneous
                    ? Visit::homogeneous
                    : Visit::varying;
        return homogeneous;
    }

  public:
    explicit HomogeneousVolumeAnalysis(
        const compiler::SurfaceProgram &program)
        : _program{program},
          _values(
              program.value_instructions().size(),
              Visit::unseen),
          _volumes(
              program.volume_instructions().size(),
              Visit::unseen) {}

    [[nodiscard]] bool analyze() noexcept {
        return volume(_program.volume_root());
    }
};

}// namespace

VolumeProgramCapabilities
VolumeProgramCapabilityComponent::analyze(
    const compiler::SurfaceProgram &program)
    const noexcept {
    auto has_spatial_values = false;
    auto has_light_path = false;
    for (const auto &instruction :
         program.value_instructions()) {
        has_spatial_values =
            has_spatial_values ||
            is_spatial_source(instruction.operation);
        has_light_path =
            has_light_path ||
            is_light_path_source(
                instruction.operation);
    }
    HomogeneousVolumeAnalysis homogeneous{program};
    const auto is_homogeneous =
        homogeneous.analyze();
    return {
        .has_spatial_values =
            has_spatial_values,
        .homogeneous =
            is_homogeneous,
        .has_light_path =
            has_light_path};
}

void VolumeProgramCapabilityComponent::
    merge_surface_flags(
        std::vector<std::uint32_t> &flags,
        std::uint32_t surface_tag,
        const compiler::SurfaceProgram &program) const {
    if (flags.size() <= surface_tag) {
        flags.resize(
            static_cast<std::size_t>(
                surface_tag) +
                1u,
            0u);
    }
    const auto capabilities =
        analyze(program);
    auto &surface_flags =
        flags[surface_tag];
    surface_flags |=
        capabilities.homogeneous
            ? 0u
            : volume_surface_flag_heterogeneous;
    surface_flags |=
        capabilities.has_light_path
            ? volume_surface_flag_light_path
            : 0u;
}

}// namespace psycles::luisa_backend::detail
