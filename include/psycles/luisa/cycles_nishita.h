#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_nishita.h> through the Psycles::luisa target."
#endif

#include <array>
#include <cstddef>
#include <cstdint>

#include <psycles/luisa/spherical_geometry.h>

#include <psycles/luisa/surface.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_nishita {

// Luisa DSL lowering of Blender 4.5.10's Cycles Nishita sky
// precomputation. Constants, quadrature, spectral accumulation, LUT layout,
// ground fade, and solar-disc interpolation follow the authoritative source
// at intern/sky/source/sky_nishita.cpp and intern/cycles/kernel/svm/sky.h.

inline constexpr std::uint32_t lut_width = 512u;
inline constexpr std::uint32_t lut_height = 128u;
inline constexpr std::uint32_t half_lut_width = lut_width / 2u;

inline constexpr float pi = 3.1415926535897932f;
inline constexpr float half_pi = 1.5707963267948966f;
inline constexpr float two_pi = 6.2831853071795864f;
inline constexpr float rayleigh_scale = 8.0e3f;
inline constexpr float mie_scale = 1.2e3f;
inline constexpr float mie_coeff = 2.0e-5f;
inline constexpr float mie_g = 0.76f;
inline constexpr float squared_mie_g = mie_g * mie_g;
inline constexpr float earth_radius = 6360.0e3f;
inline constexpr float atmosphere_radius = 6420.0e3f;
inline constexpr std::uint32_t integration_steps = 32u;
inline constexpr std::size_t wavelength_count = 21u;
inline constexpr float wavelength_step = 20.0f;

inline constexpr std::array<float, wavelength_count> irradiance{
    1.45756829855592995315f,
    1.56596305559738380175f,
    1.65148449067670455293f,
    1.71496242737209314555f,
    1.75797983805020541226f,
    1.78256407885924539336f,
    1.79095108475838560302f,
    1.78541550133410664714f,
    1.76815554864306845317f,
    1.74122069647250410362f,
    1.70647127164943679389f,
    1.66556087452739887134f,
    1.61993437242451854274f,
    1.57083597368892080581f,
    1.51932335059305478886f,
    1.46628494965214395407f,
    1.41245852740172450623f,
    1.35844961970384092709f,
    1.30474913844739281998f,
    1.25174963272610817455f,
    1.19975998755420620867f};

inline constexpr std::array<float, wavelength_count> rayleigh_coeff{
    0.00005424820087636473f,
    0.00004418549866505454f,
    0.00003635151910165377f,
    0.00003017929012024763f,
    0.00002526320226989157f,
    0.00002130859310621843f,
    0.00001809838025320633f,
    0.00001547057129129042f,
    0.00001330284977336850f,
    0.00001150184784075764f,
    0.00000999557429990163f,
    0.00000872799973630707f,
    0.00000765513700977967f,
    0.00000674217203751443f,
    0.00000596134125832052f,
    0.00000529034598065810f,
    0.00000471115687557433f,
    0.00000420910481110487f,
    0.00000377218381260133f,
    0.00000339051255477280f,
    0.00000305591531679811f};

inline constexpr std::array<float, wavelength_count> ozone_coeff{
    0.00000000325126849861f,
    0.00000000585395365047f,
    0.00000001977191155085f,
    0.00000007309568762914f,
    0.00000020084561514287f,
    0.00000040383958096161f,
    0.00000063551335912363f,
    0.00000096707041180970f,
    0.00000154797400424410f,
    0.00000209038647223331f,
    0.00000246128056164565f,
    0.00000273551299461512f,
    0.00000215125863128643f,
    0.00000159051840791988f,
    0.00000112356197979857f,
    0.00000073527551487574f,
    0.00000046450130357806f,
    0.00000033096079921048f,
    0.00000022512612292678f,
    0.00000014879129266490f,
    0.00000016828623364192f};

inline constexpr std::array<luisa::float3, wavelength_count> cmf_xyz{{
    {0.00136800000f, 0.00003900000f, 0.00645000100f},
    {0.01431000000f, 0.00039600000f, 0.06785001000f},
    {0.13438000000f, 0.00400000000f, 0.64560000000f},
    {0.34828000000f, 0.02300000000f, 1.74706000000f},
    {0.29080000000f, 0.06000000000f, 1.66920000000f},
    {0.09564000000f, 0.13902000000f, 0.81295010000f},
    {0.00490000000f, 0.32300000000f, 0.27200000000f},
    {0.06327000000f, 0.71000000000f, 0.07824999000f},
    {0.29040000000f, 0.95400000000f, 0.02030000000f},
    {0.59450000000f, 0.99500000000f, 0.00390000000f},
    {0.91630000000f, 0.87000000000f, 0.00165000100f},
    {1.06220000000f, 0.63100000000f, 0.00080000000f},
    {0.85444990000f, 0.38100000000f, 0.00019000000f},
    {0.44790000000f, 0.17500000000f, 0.00002000000f},
    {0.16490000000f, 0.06100000000f, 0.00000000000f},
    {0.04677000000f, 0.01700000000f, 0.00000000000f},
    {0.01135916000f, 0.00410200000f, 0.00000000000f},
    {0.00289932700f, 0.00104700000f, 0.00000000000f},
    {0.00069007860f, 0.00024920000f, 0.00000000000f},
    {0.00016615050f, 0.00006000000f, 0.00000000000f},
    {0.00004150994f, 0.00001499000f, 0.00000000000f},
}};

inline constexpr std::array<float, 8u> quadrature_nodes{
    0.006811185292f,
    0.03614807107f,
    0.09004346519f,
    0.1706680068f,
    0.2818362161f,
    0.4303406404f,
    0.6296271457f,
    0.9145252695f};

inline constexpr std::array<float, 8u> quadrature_weights{
    0.01750893642f,
    0.04135477391f,
    0.06678839063f,
    0.09507698807f,
    0.1283416365f,
    0.1707430204f,
    0.2327233347f,
    0.3562490486f};

struct SunPixels {
    Float3 bottom_xyz;
    Float3 top_xyz;
};

[[nodiscard]] inline Float3 geographical_to_direction(
    Float latitude,
    Float longitude) noexcept {
    return make_float3(
        cos(latitude) * cos(longitude),
        cos(latitude) * sin(longitude),
        sin(latitude));
}

[[nodiscard]] inline Float density_rayleigh(
    Float height) noexcept {
    return exp(-height / rayleigh_scale);
}

[[nodiscard]] inline Float density_mie(
    Float height) noexcept {
    return exp(-height / mie_scale);
}

[[nodiscard]] inline Float density_ozone(
    Float height) noexcept {
    Float density = 0.0f;
    $if ((height >= 10000.0f) & (height < 25000.0f)) {
        density = height / 15000.0f - 2.0f / 3.0f;
    };
    $if ((height >= 25000.0f) & (height < 40000.0f)) {
        density = -(height / 15000.0f - 8.0f / 3.0f);
    };
    return density;
}

[[nodiscard]] inline Float phase_rayleigh(Float mu) noexcept {
    return (3.0f / (16.0f * pi)) * (1.0f + mu * mu);
}

[[nodiscard]] inline Float phase_mie(Float mu) noexcept {
    const auto denominator = pow(
        1.0f + squared_mie_g - 2.0f * mie_g * mu,
        1.5f);
    return (3.0f * (1.0f - squared_mie_g) *
            (1.0f + mu * mu)) /
           (8.0f * pi * (2.0f + squared_mie_g) *
            denominator);
}

[[nodiscard]] inline Bool surface_intersection(
    Float3 position,
    Float3 direction) noexcept {
    const auto b = -2.0f * dot(direction, -position);
    const auto c =
        dot(position, position) -
        earth_radius * earth_radius;
    return (direction.z < 0.0f) &
           (b * b - 4.0f * c >= 0.0f);
}

[[nodiscard]] inline Float3 atmosphere_intersection(
    Float3 position,
    Float3 direction) noexcept {
    const auto b = -2.0f * dot(direction, -position);
    const auto c =
        dot(position, position) -
        atmosphere_radius * atmosphere_radius;
    const auto discriminant = max(b * b - 4.0f * c, 0.0f);
    const auto t = (-b + sqrt(discriminant)) * 0.5f;
    return position + direction * t;
}

[[nodiscard]] inline Float3 ray_optical_depth(
    Float3 origin,
    Float3 direction) noexcept {
    const auto ray_end =
        atmosphere_intersection(origin, direction);
    const auto ray_length = length(ray_end - origin);
    const auto segment = ray_length * direction;
    Float3 optical_depth = make_float3(0.0f);
    for (std::size_t i = 0u;
         i < quadrature_nodes.size();
         ++i) {
        const auto position =
            origin + quadrature_nodes[i] * segment;
        const auto height =
            length(position) - earth_radius;
        const auto density = make_float3(
            density_rayleigh(height),
            density_mie(height),
            density_ozone(height));
        optical_depth += density * quadrature_weights[i];
    }
    return optical_depth * ray_length;
}

[[nodiscard]] inline Float3 spectrum_to_xyz(
    const luisa::compute::ArrayFloat<wavelength_count>
        &spectrum) noexcept {
    Float3 xyz = make_float3(0.0f);
    for (std::size_t wavelength = 0u;
         wavelength < wavelength_count;
         ++wavelength) {
        xyz += make_float3(cmf_xyz[wavelength]) *
               spectrum[wavelength];
    }
    return xyz * wavelength_step;
}

[[nodiscard]] inline Float3 single_scattering_xyz(
    Float3 ray_direction,
    Float3 sun_direction,
    Float3 ray_origin,
    Float air_density,
    Float dust_density,
    Float ozone_density) noexcept {
    const auto ray_end =
        atmosphere_intersection(ray_origin, ray_direction);
    const auto ray_length = length(ray_end - ray_origin);
    const auto segment_length =
        ray_length / static_cast<float>(integration_steps);
    const auto segment = segment_length * ray_direction;
    Float3 optical_depth = make_float3(0.0f);
    luisa::compute::ArrayFloat<wavelength_count> spectrum{};
    for (std::size_t wavelength = 0u;
         wavelength < wavelength_count;
         ++wavelength) {
        spectrum[wavelength] = 0.0f;
    }
    const auto mu = dot(ray_direction, sun_direction);
    const auto phase_function = make_float3(
        phase_rayleigh(mu), phase_mie(mu), 0.0f);
    const auto density_scale = make_float3(
        air_density, dust_density, ozone_density);
    Float3 position = ray_origin + 0.5f * segment;

    $for (step, integration_steps) {
        static_cast<void>(step);
        const auto height =
            length(position) - earth_radius;
        const auto density =
            density_scale *
            make_float3(
                density_rayleigh(height),
                density_mie(height),
                density_ozone(height));
        optical_depth += segment_length * density;
        $if (!surface_intersection(position, sun_direction)) {
            const auto light_optical_depth =
                density_scale *
                ray_optical_depth(position, sun_direction);
            const auto total_optical_depth =
                optical_depth + light_optical_depth;
            for (std::size_t wavelength = 0u;
                 wavelength < wavelength_count;
                 ++wavelength) {
                const auto extinction_density =
                    total_optical_depth *
                    make_float3(
                        rayleigh_coeff[wavelength],
                        1.11f * mie_coeff,
                        ozone_coeff[wavelength]);
                const auto attenuation = exp(
                    -(extinction_density.x +
                      extinction_density.y +
                      extinction_density.z));
                const auto scattering_density =
                    density *
                    make_float3(
                        rayleigh_coeff[wavelength],
                        mie_coeff,
                        0.0f);
                spectrum[wavelength] +=
                    attenuation *
                    dot(phase_function, scattering_density) *
                    irradiance[wavelength] *
                    segment_length;
            }
        };
        position += segment;
    };
    return spectrum_to_xyz(spectrum);
}

[[nodiscard]] inline Float3 sky_lut_texel(
    UInt x,
    UInt y,
    Float sun_elevation,
    Float altitude,
    Float air_density,
    Float dust_density,
    Float ozone_density) noexcept {
    const auto clamped_altitude =
        clamp(altitude, 1.0f, 59999.0f);
    const auto camera_position = make_float3(
        0.0f, 0.0f, earth_radius + clamped_altitude);
    const auto sun_direction =
        geographical_to_direction(sun_elevation, 0.0f);
    constexpr auto latitude_step =
        half_pi / static_cast<float>(lut_height);
    constexpr auto longitude_step =
        two_pi / static_cast<float>(lut_width);
    constexpr auto half_latitude_step =
        latitude_step * 0.5f;
    const auto normalized_y =
        cast<float>(y) / static_cast<float>(lut_height);
    const auto latitude =
        (half_pi + half_latitude_step) *
        normalized_y * normalized_y;
    const auto longitude =
        longitude_step * cast<float>(x) - pi;
    const auto direction =
        geographical_to_direction(latitude, longitude);
    return single_scattering_xyz(
        direction,
        sun_direction,
        camera_position,
        air_density,
        dust_density,
        ozone_density);
}

template<typename Texture>
[[nodiscard]] inline Float3 sky_radiance_xyz(
    const Texture &texture,
    Float3 direction,
    Float sun_rotation) noexcept {
    const auto theta =
        acos(clamp(direction.z, -1.0f, 1.0f));
    const auto phi = atan2(direction.y, direction.x);
    const auto x =
        fract((-phi - half_pi + sun_rotation) / two_pi);
    Float3 xyz = make_float3(0.0f);
    $if (direction.z >= 0.0f) {
        const auto direction_elevation = half_pi - theta;
        const auto y = sqrt(max(
            direction_elevation / half_pi, 0.0f));
        xyz = texture.sample(make_float2(x, y)).xyz();
    }
    $else {
        $if (direction.z >= -0.4f) {
            Float fade = 1.0f + direction.z * 2.5f;
            fade = fade * fade * fade;
            xyz =
                texture.sample(make_float2(x, -0.5f)).xyz() *
                fade;
        };
    };
    return xyz;
}

[[nodiscard]] inline Float3 sun_disc_radiance_xyz(
    Float3 direction,
    Float3 sun_direction,
    Float3 pixel_bottom_xyz,
    Float3 pixel_top_xyz,
    Float sun_elevation,
    Float angular_diameter,
    Float sun_intensity) noexcept {
    const auto angle = spherical_geometry::precise_angle(
        direction, sun_direction);
    const auto radius = angular_diameter * 0.5f;
    const auto direction_elevation =
        half_pi - acos(clamp(direction.z, -1.0f, 1.0f));
    Float3 xyz = make_float3(0.0f);
    $if (sun_elevation - radius > 0.0f) {
        $if (sun_elevation + radius > 0.0f) {
            const auto y =
                ((direction_elevation - sun_elevation) /
                 angular_diameter) +
                0.5f;
            xyz =
                pixel_bottom_xyz * (1.0f - y) +
                pixel_top_xyz * y;
        };
    }
    $else {
        $if (sun_elevation + radius > 0.0f) {
            const auto y =
                direction_elevation /
                (sun_elevation + radius);
            xyz =
                pixel_bottom_xyz * (1.0f - y) +
                pixel_top_xyz * y;
        };
    };
    const auto normalized_angle = angle / radius;
    const auto limb_darkening =
        1.0f -
        0.6f *
            (1.0f -
             sqrt(max(
                 1.0f -
                     normalized_angle * normalized_angle,
                 0.0f)));
    const auto inside =
        (direction.z >= 0.0f) &
        (angular_diameter >= 0.0f) &
        (angle < radius);
    return select(
        make_float3(0.0f),
        xyz * (sun_intensity * limb_darkening),
        inside);
}

[[nodiscard]] inline Float3 sun_radiation_xyz(
    Float3 direction,
    Float altitude,
    Float air_density,
    Float dust_density,
    Float solid_angle) noexcept {
    const auto camera_position = make_float3(
        0.0f, 0.0f, earth_radius + altitude);
    const auto optical_depth =
        ray_optical_depth(camera_position, direction);
    luisa::compute::ArrayFloat<wavelength_count> spectrum{};
    for (std::size_t wavelength = 0u;
         wavelength < wavelength_count;
         ++wavelength) {
        const auto transmittance =
            rayleigh_coeff[wavelength] *
                optical_depth.x * air_density +
            1.11f * mie_coeff *
                optical_depth.y * dust_density;
        spectrum[wavelength] =
            irradiance[wavelength] *
            exp(-transmittance) /
            max(solid_angle, 1.0e-20f);
    }
    return spectrum_to_xyz(spectrum);
}

[[nodiscard]] inline SunPixels sun_pixels(
    Float sun_elevation,
    Float angular_diameter,
    Float altitude,
    Float air_density,
    Float dust_density) noexcept {
    const auto clamped_altitude =
        clamp(altitude, 1.0f, 59999.0f);
    const auto half_angular = angular_diameter * 0.5f;
    const auto solid_angle = spherical_geometry::cap_solid_angle(half_angular);
    const auto bottom =
        max(sun_elevation - half_angular, 0.0f);
    const auto top =
        max(sun_elevation + half_angular, 0.0f);
    return {
        .bottom_xyz = sun_radiation_xyz(
            geographical_to_direction(bottom, 0.0f),
            clamped_altitude,
            air_density,
            dust_density,
            solid_angle),
        .top_xyz = sun_radiation_xyz(
            geographical_to_direction(top, 0.0f),
            clamped_altitude,
            air_density,
            dust_density,
            solid_angle)};
}

}// namespace psycles::luisa_backend::cycles_nishita
