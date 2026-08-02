#include "../src/luisa/cycles_shader_identity.h"
#include "../src/luisa/path_tracer_scene_geometry.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

namespace identity =
    psycles::luisa_backend::detail::cycles_shader_identity;
using psycles::contract::RayVisibility;
using psycles::contract::all_ray_visibility;
using psycles::contract::visibility_bit;
using psycles::luisa_backend::detail::
    CyclesPrimitiveIntervalError;
using psycles::luisa_backend::detail::
    CyclesPrimitiveIntervalResolver;
using psycles::luisa_backend::detail::
    world_triangle_area;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_surface_shader_composition() {
    require(
        identity::surface(6u, false) == 0x40000006u,
        "flat surface shader identity changed");
    require(
        identity::surface(6u, true) == 0xc0000006u,
        "smooth surface shader identity changed");
    require(
        (identity::surface(0x1234u, false) &
         identity::shader_mask) == 0x1234u,
        "surface flags changed the shader-vector index");
}

void test_light_shader_composition() {
    using psycles::contract::LightType;
    require(
        identity::light_type(LightType::point) == 0u &&
            identity::light_type(LightType::distant) == 1u &&
            identity::light_type(LightType::background) == 2u &&
            identity::light_type(LightType::area) == 3u &&
            identity::light_type(LightType::spot) == 4u,
        "scene light types no longer map to Cycles' kernel ABI");

    const auto point_visibility =
        all_ray_visibility &
        ~visibility_bit(RayVisibility::camera);
    require(
        identity::analytic_light(
            5u,
            true,
            point_visibility,
            true,
            false) == 0x41000005u,
        "delta point-light shader identity changed");
    require(
        identity::analytic_light(
            5u,
            true,
            point_visibility,
            true,
            true) == 0x51000005u,
        "finite point-light MIS flag changed");
    require(
        identity::analytic_light(
            5u,
            false,
            all_ray_visibility,
            false,
            false) ==
            (identity::exclude_shadow_catcher | 5u),
        "cast-shadow or shadow-catcher composition changed");

    const auto visibility =
        all_ray_visibility &
        ~visibility_bit(RayVisibility::diffuse) &
        ~visibility_bit(RayVisibility::glossy) &
        ~visibility_bit(RayVisibility::transmission) &
        ~visibility_bit(RayVisibility::volume_scatter);
    require(
        identity::object_visibility(visibility, true) ==
            (identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter),
        "object ray visibility did not map to Cycles shader flags");

    require(
        identity::emissive_triangle(
            9u,
            true,
            visibility,
            false) ==
            (identity::surface(9u, true) |
             identity::use_mis |
             identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter |
             identity::exclude_shadow_catcher),
        "triangle-light shader identity changed");

    require(
        identity::background_light(
            3u,
            false,
            visibility &
                ~visibility_bit(RayVisibility::camera)) ==
            (3u |
             identity::use_mis |
             identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter),
        "background-light shader identity changed");
}

void test_geometry_identity() {
    CyclesPrimitiveIntervalResolver resolver;
    const auto first = resolver.resolve(3u);
    const auto explicit_gap = resolver.resolve(2u, 5u);
    const auto implicit_after_gap = resolver.resolve(4u);
    require(
        first.offset == 0u &&
            explicit_gap.offset == 5u &&
            implicit_after_gap.offset == 7u &&
            resolver.end() == 11u,
        "implicit and explicit Cycles primitive prefixes diverged");

    const auto overlap = resolver.resolve(1u, 10u);
    require(
        !overlap.offset &&
            overlap.error ==
                CyclesPrimitiveIntervalError::overlap &&
            resolver.end() == 11u,
        "overlapping primitive interval was accepted or mutated state");

    CyclesPrimitiveIntervalResolver boundary;
    const auto last = boundary.resolve(
        1u,
        std::numeric_limits<std::uint32_t>::max());
    const auto overflow = boundary.resolve(1u);
    require(
        last.offset ==
                std::numeric_limits<std::uint32_t>::max() &&
            !overflow.offset &&
            overflow.error ==
                CyclesPrimitiveIntervalError::out_of_range,
        "Cycles primitive address-space boundary changed");

    psycles::Mat4f transform;
    transform.elements[0u] = 2.0f;
    transform.elements[5u] = 3.0f;
    require(
        world_triangle_area(
            transform,
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}) == 3.0f,
        "emissive triangle area ignored instance transform");
}

}// namespace

int main() {
    test_surface_shader_composition();
    test_light_shader_composition();
    test_geometry_identity();
    return EXIT_SUCCESS;
}
