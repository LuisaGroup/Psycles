#include "../src/luisa/path_kernel_emissive_triangle.h"
#include "../src/luisa/path_tracer_scene_upload.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;

struct LookupCase {
    std::uint32_t instance;
    std::uint32_t primitive;
    std::uint32_t emission_sampling;
    bool found;
};

constexpr std::array emitters{
    EmissiveTriangleGpu{
        .instance_index = 1u,
        .primitive_index = 2u,
        .emission_sampling = 11u},
    EmissiveTriangleGpu{
        .instance_index = 1u,
        .primitive_index = 7u,
        .emission_sampling = 17u},
    EmissiveTriangleGpu{
        .instance_index = 3u,
        .primitive_index = 0u,
        .emission_sampling = 30u},
    EmissiveTriangleGpu{
        .instance_index = 3u,
        .primitive_index = 9u,
        .emission_sampling = 39u},
    EmissiveTriangleGpu{
        .instance_index = 4u,
        .primitive_index = 1u,
        .emission_sampling = 41u},
    EmissiveTriangleGpu{
        .instance_index = 10u,
        .primitive_index = 0u,
        .emission_sampling = 100u},
    EmissiveTriangleGpu{
        .instance_index = 10u,
        .primitive_index = 12u,
        .emission_sampling = 112u}};

constexpr std::array cases{
    LookupCase{1u, 2u, 11u, true},
    LookupCase{3u, 0u, 30u, true},
    LookupCase{10u, 12u, 112u, true},
    LookupCase{0u, 99u, 0u, false},
    LookupCase{1u, 3u, 0u, false},
    LookupCase{2u, 99u, 0u, false},
    LookupCase{3u, 10u, 0u, false},
    LookupCase{11u, 0u, 0u, false}};

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    luisa::vector<EmissiveTriangleGpu> sorted_emitters;
    sorted_emitters.reserve(emitters.size());
    for (const auto &emitter : emitters) {
        sorted_emitters.emplace_back(emitter);
    }
    auto duplicate_emitters = sorted_emitters;
    duplicate_emitters[3u] = duplicate_emitters[2u];
    auto unordered_emitters = sorted_emitters;
    std::swap(unordered_emitters[2u], unordered_emitters[3u]);
    if (!emissive_triangle_keys_strictly_increasing(
            sorted_emitters) ||
        emissive_triangle_keys_strictly_increasing(
            duplicate_emitters) ||
        emissive_triangle_keys_strictly_increasing(
            unordered_emitters)) {
        std::cerr
            << "Emissive triangle host ordering invariant failed\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();
    scene->emissive_triangle_buffer =
        device.create_buffer<EmissiveTriangleGpu>(
            emitters.size());
    scene->emissive_triangle_count =
        static_cast<std::uint32_t>(emitters.size());
    auto output = device.create_buffer<luisa::uint4>(
        cases.size());
    const auto component =
        make_emissive_triangle_component();

    Kernel1D evaluate =
        [scene, component](BufferUInt4 records) noexcept {
            const auto write = [&]<std::size_t index>() noexcept {
                constexpr auto test = cases[index];
                const auto result =
                    component->find_intersection_emitter(
                        scene,
                        test.instance,
                        test.primitive);
                records.write(
                    index,
                    make_uint4(
                        result.emission_sampling,
                        select(0u, 1u, result.found),
                        test.instance,
                        test.primitive));
            };
            [&]<std::size_t... indices>(
                std::index_sequence<indices...>) noexcept {
                (write.template operator()<indices>(), ...);
            }(std::make_index_sequence<cases.size()>{});
        };

    auto shader = device.compile(
        evaluate,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    std::array<luisa::uint4, cases.size()> actual{};
    stream
        << scene->emissive_triangle_buffer.copy_from(
               luisa::span{emitters})
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    for (auto index = std::size_t{0u};
         index < cases.size(); ++index) {
        const auto expected_found =
            cases[index].found ? 1u : 0u;
        if (actual[index].x !=
                cases[index].emission_sampling ||
            actual[index].y != expected_found ||
            actual[index].z != cases[index].instance ||
            actual[index].w != cases[index].primitive) {
            std::cerr
                << "Emissive triangle ordered lookup failed on "
                << backend << " for key ("
                << cases[index].instance << ", "
                << cases[index].primitive << "): got sampling="
                << actual[index].x << ", found="
                << actual[index].y << "; expected sampling="
                << cases[index].emission_sampling << ", found="
                << expected_found << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
