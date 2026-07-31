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
        case ValueOperation::random_per_island:
        case ValueOperation::image_color:
        case ValueOperation::image_alpha:
        case ValueOperation::attribute_color:
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
        case ValueOperation::gradient:
        case ValueOperation::nishita_sky:
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
            homogeneous =
                homogeneous && value(dependency);
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
            homogeneous =
                homogeneous && value(dependency);
        }
        homogeneous =
            homogeneous &&
            volume(instruction.a) &&
            volume(instruction.b);
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
    for (const auto &instruction :
         program.value_instructions()) {
        has_spatial_values =
            has_spatial_values ||
            is_spatial_source(instruction.operation);
    }
    HomogeneousVolumeAnalysis homogeneous{program};
    return {
        .has_spatial_values =
            has_spatial_values,
        .homogeneous =
            homogeneous.analyze()};
}

}// namespace psycles::luisa_backend::detail
