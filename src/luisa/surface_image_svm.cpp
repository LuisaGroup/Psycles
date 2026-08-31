#include "surface_image_svm.h"

#include "surface_image_box.h"
#include "surface_image_sampling.h"
#include "surface_math.h"

#include <array>
#include <cstdlib>

#include <luisa/dsl/sugar.h>

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/native_vector_math.h>
#include <psycles/luisa/spherical_geometry.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3
cycles_safe_normalize_direction(Float3 direction) noexcept {
    return native_vector_math::safe_normalize_nonzero(direction);
}

[[nodiscard]] Float2
cycles_direction_to_equirectangular(Float3 direction) noexcept {
    direction = cycles_safe_normalize_direction(direction);
    const auto length = sqrt(dot(direction, direction));
    Float2 uv = make_float2(0.0f);
    $if(length != 0.0f) {
        const auto azimuth =
            spherical_geometry::canonical_direction_azimuth(direction);
        uv = make_float2(
            (spherical_geometry::pi - azimuth) / spherical_geometry::two_pi,
            1.0f - acos(direction.z / length) / spherical_geometry::pi);
    };
    return uv;
}

[[nodiscard]] Float2 cycles_direction_to_mirrorball(Float3 direction) noexcept {
    direction = cycles_safe_normalize_direction(direction);
    direction.y -= 1.0f;
    const auto divisor = 2.0f * sqrt(max(-0.5f * direction.y, 0.0f));
    $if(divisor > 0.0f) { direction /= divisor; };
    return 0.5f * (direction.xz() + 1.0f);
}

template <typename Evaluate>
[[nodiscard]] Float4
dispatch_image_sampling_mode(UInt sampling_key,
                             std::span<const std::uint16_t> immediate_domain,
                             Evaluate &&evaluate) noexcept {
    Float4 sampled = make_float4(0.0f);
    luisa::compute::detail::SwitchStmtBuilder{sampling_key} % [&] {
        std::array<bool, compiler::surface_value_image_sampling_key_count>
            emitted{};
        for (const auto encoded : immediate_domain) {
            const auto raw_interpolation =
                (static_cast<std::uint32_t>(encoded) &
                 compiler::surface_value_image_interpolation_mask) >>
                compiler::surface_value_image_interpolation_shift;
            const auto interpolation =
                compiler::canonical_surface_value_image_interpolation(
                    raw_interpolation);
            const auto extension = static_cast<std::uint32_t>(encoded) &
                                   compiler::surface_value_image_extension_mask;
            const auto key = compiler::make_surface_value_image_sampling_key(
                interpolation, extension);
            if (key >= emitted.size()) {
                std::abort();
            }
            if (emitted[key]) {
                continue;
            }
            emitted[key] = true;
            luisa::compute::detail::SwitchCaseStmtBuilder{key} %
                [&, interpolation, extension] {
                    sampled = evaluate(interpolation, extension);
                };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid Image Texture SVM sampling immediate");
        };
    };
    return sampled;
}

class ShaderServicesImageBoxTextureSampler final
    : public SurfaceImageBoxTextureSampler {

  private:
    const ShaderServices &_services;
    std::uint32_t _interpolation;
    std::uint32_t _extension;

  public:
    ShaderServicesImageBoxTextureSampler(const ShaderServices &services,
                                         std::uint32_t interpolation,
                                         std::uint32_t extension) noexcept
        : _services{services}, _interpolation{interpolation},
          _extension{extension} {}

    [[nodiscard]] Float4 sample(Expr<std::uint32_t> texture_handle,
                                Float2 uv) const noexcept override {
        return _services.texture_2d(texture_handle, uv, make_float2(0.0f),
                                    make_float2(0.0f), _interpolation,
                                    _extension);
    }
};

void validate_surface_image_shape(
    SurfaceImageSvmShape shape,
    std::span<const std::uint16_t> immediate_domain) noexcept {
    if (immediate_domain.empty()) {
        std::abort();
    }
    for (const auto encoded : immediate_domain) {
        const auto projection =
            (static_cast<std::uint32_t>(encoded) &
             compiler::surface_value_image_projection_mask) >>
            compiler::surface_value_image_projection_shift;
        if (shape == SurfaceImageSvmShape::environment) {
            if (projection > 1u) {
                std::abort();
            }
        } else if ((shape == SurfaceImageSvmShape::image_box) !=
                   (projection == 1u)) {
            std::abort();
        }
    }
}

} // namespace

Float4 evaluate_surface_image_svm(
    const ShaderServices &services, const SurfacePoint &point,
    SurfaceImageSvmShape shape, UInt immediate,
    std::span<const std::uint16_t> immediate_domain, Float3 coordinate,
    UInt texture_handle, Float projection_blend) noexcept {
    validate_surface_image_shape(shape, immediate_domain);
    const auto extension =
        immediate & compiler::surface_value_image_extension_mask;
    const auto interpolation =
        (immediate & compiler::surface_value_image_interpolation_mask) >>
        compiler::surface_value_image_interpolation_shift;
    const auto projection =
        (immediate & compiler::surface_value_image_projection_mask) >>
        compiler::surface_value_image_projection_shift;
    const auto unassociate_alpha =
        (immediate & compiler::surface_value_image_unassociate_alpha_bit) != 0u;
    const auto encoded_as_srgb =
        (immediate & compiler::surface_value_image_srgb_bit) != 0u;
    const auto interpolation_family = select(
        compiler::surface_value_image_interpolation_family_count - 1u,
        interpolation,
        interpolation <
            compiler::surface_value_image_interpolation_family_count - 1u);
    const auto sampling_key =
        interpolation_family *
            compiler::surface_value_image_extension_mode_count +
        extension;
    const auto sample_uv = [&](Float2 uv) noexcept {
        // Blender UVs use a bottom-left origin while decoded host images are
        // uploaded in top-to-bottom row order.
        uv.y = 1.0f - uv.y;
        auto sampled = dispatch_image_sampling_mode(
            sampling_key, immediate_domain,
            [&](std::uint32_t static_interpolation,
                std::uint32_t static_extension) noexcept {
                return services.texture_2d(
                    texture_handle, uv, make_float2(0.0f), make_float2(0.0f),
                    static_interpolation, static_extension);
            });
        return decode_surface_image_sample(sampled, unassociate_alpha,
                                           encoded_as_srgb);
    };

    if (shape == SurfaceImageSvmShape::environment) {
        Float2 uv = cycles_direction_to_equirectangular(coordinate);
        $if(projection == 1u) {
            uv = cycles_direction_to_mirrorball(coordinate);
        };
        return sample_uv(uv);
    }

    if (shape == SurfaceImageSvmShape::image_box) {
        // Cycles transforms the current sd->N back to object space. This is
        // observably different from reading the mesh normal after an
        // automatic-bump SetNormal stage.
        const auto column_x = point.normal_to_world_x;
        const auto column_y = point.normal_to_world_y;
        const auto column_z = point.normal_to_world_z;
        const auto determinant = dot(column_x, cross(column_y, column_z));
        const auto safe_determinant =
            select(1.0f, determinant, abs(determinant) > 1.0e-20f);
        const auto signed_normal = safe_normalize(
            make_float3(dot(point.shading_normal, cross(column_y, column_z)),
                        dot(point.shading_normal, cross(column_z, column_x)),
                        dot(point.shading_normal, cross(column_x, column_y))) /
                safe_determinant,
            point.object_shading_normal);
        const SurfaceImageBoxInput box_input{
            .coordinate = coordinate,
            .signed_normal = signed_normal,
            .blend = projection_blend,
            .texture_handle = texture_handle,
            .unassociate_alpha = unassociate_alpha,
            .encoded_as_srgb = encoded_as_srgb};
        return dispatch_image_sampling_mode(
            sampling_key, immediate_domain,
            [&](std::uint32_t static_interpolation,
                std::uint32_t static_extension) noexcept {
                if (const auto provider =
                        services.surface_image_box_provider()) {
                    return provider->evaluate(box_input, static_interpolation,
                                              static_extension);
                }
                const ShaderServicesImageBoxTextureSampler sampler{
                    services, static_interpolation, static_extension};
                return evaluate_surface_image_box(box_input, sampler);
            });
    }

    Float2 uv = coordinate.xy();
    $if(projection == 2u) {
        const auto direction = (coordinate - 0.5f) * 2.0f;
        const auto length_squared = dot(direction, direction);
        Float2 spherical = make_float2(0.0f);
        $if(length_squared > 0.0f) {
            Float u = 0.0f;
            $if((direction.x != 0.0f) | (direction.y != 0.0f)) {
                u = 0.5f - atan2(direction.x, direction.y) /
                               spherical_geometry::two_pi;
            };
            const auto z = luisa::compute::clamp(
                direction.z * rsqrt(length_squared), -1.0f, 1.0f);
            spherical = make_float2(u, 1.0f - acos(z) / spherical_geometry::pi);
        };
        uv = spherical;
    }
    $elif(projection == 3u) {
        const auto direction = (coordinate - 0.5f) * 2.0f;
        const auto radial_length =
            sqrt(direction.x * direction.x + direction.y * direction.y);
        Float2 tube = make_float2(0.0f);
        $if(radial_length > 0.0f) {
            tube = make_float2((1.0f - atan2(direction.x / radial_length,
                                             direction.y / radial_length) /
                                           spherical_geometry::pi) *
                                   0.5f,
                               (direction.z + 1.0f) * 0.5f);
        };
        uv = tube;
    };
    return sample_uv(uv);
}

} // namespace psycles::luisa_backend::detail
