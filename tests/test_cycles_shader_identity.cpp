#include "../src/luisa/cycles_shader_identity.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

namespace identity =
    psycles::luisa_backend::detail::cycles_shader_identity;
using psycles::contract::RayVisibility;
using psycles::contract::all_ray_visibility;
using psycles::contract::visibility_bit;

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
}

}// namespace

int main() {
    test_surface_shader_composition();
    test_light_shader_composition();
    return EXIT_SUCCESS;
}
