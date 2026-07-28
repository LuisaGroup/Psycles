#include <psycles/adapter/blender_scene.h>
#include <psycles/contract/render.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <luisa/luisa-compute.h>

namespace {

template<typename T>
[[nodiscard]] std::optional<T> parse_unsigned(
    std::string_view text) {
    T result{};
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return result;
}

}// namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr
            << "usage: psycles_render_blender_scene "
               "<export-directory> <output.ppm> "
               "[backend=fallback] [width] [height] [samples]\n";
        return EXIT_FAILURE;
    }
    const auto bundle = std::filesystem::path{argv[1]};
    const auto output = std::filesystem::path{argv[2]};
    const auto backend_name =
        std::string_view{argc > 3 ? argv[3] : "fallback"};

    auto imported =
        psycles::adapter::load_blender_scene_bundle(bundle);
    for (const auto &diagnostic : imported.diagnostics) {
        std::cerr
            << (diagnostic.severity ==
                        psycles::adapter::
                            BlenderSceneDiagnosticSeverity::error
                    ? "error: "
                    : "warning: ")
            << diagnostic.message << '\n';
    }
    if (!imported.ok()) {
        return EXIT_FAILURE;
    }

    auto width = imported.width;
    auto height = imported.height;
    auto samples = imported.samples;
    if (argc > 4) {
        auto value = parse_unsigned<std::uint32_t>(argv[4]);
        if (!value || *value == 0u) {
            return EXIT_FAILURE;
        }
        width = *value;
    }
    if (argc > 5) {
        auto value = parse_unsigned<std::uint32_t>(argv[5]);
        if (!value || *value == 0u) {
            return EXIT_FAILURE;
        }
        height = *value;
    }
    if (argc > 6) {
        auto value = parse_unsigned<std::uint32_t>(argv[6]);
        if (!value || *value == 0u) {
            return EXIT_FAILURE;
        }
        samples = *value;
    }

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend_name);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true}};
    const auto compile_begin = std::chrono::steady_clock::now();
    auto compilation = renderer.compile_scene(*imported.scene);
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << "error: " << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }
    const auto compile_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - compile_begin)
            .count();

    const psycles::contract::RenderSettings settings{
        .full_extent = {.width = width, .height = height},
        .window = {},
        .seed = imported.seed,
        .transparent_background =
            imported.transparent_background,
        .pixel_filter = imported.pixel_filter,
        .filter_width = imported.filter_width,
        .pass_alpha_threshold =
            imported.pass_alpha_threshold,
        .integrator = imported.integrator,
        .passes = {
            {.kind = psycles::contract::PassKind::combined,
             .name = "Combined",
             .light_group = {},
             .channels = 4u},
            {.kind = psycles::contract::PassKind::normal,
             .name = "Normal",
             .light_group = {},
             .channels = 3u},
            {.kind = psycles::contract::PassKind::albedo,
             .name = "Albedo",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::glossy_color,
             .name = "GlossCol",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::
                     transmission_color,
             .name = "TransCol",
             .light_group = {},
             .channels = 3u},
            {.kind = psycles::contract::PassKind::emission,
             .name = "Emit",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::environment,
             .name = "Env",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::diffuse_direct,
             .name = "DiffDir",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::diffuse_indirect,
             .name = "DiffInd",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::glossy_direct,
             .name = "GlossDir",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::glossy_indirect,
             .name = "GlossInd",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::
                     transmission_direct,
             .name = "TransDir",
             .light_group = {},
             .channels = 3u},
            {.kind =
                 psycles::contract::PassKind::
                     transmission_indirect,
             .name = "TransInd",
             .light_group = {},
             .channels = 3u}}};
    const auto session_begin =
        std::chrono::steady_clock::now();
    auto session =
        renderer.create_session(*compilation.scene, settings);
    if (!session) {
        return EXIT_FAILURE;
    }
    const auto session_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - session_begin)
            .count();
    psycles::io::MemoryOutputSink sink;
    const auto render_begin = std::chrono::steady_clock::now();
    if (!session->render_samples(
            {.first = 0u,
             .count = samples,
             .offset = 0u,
             .total = samples},
            sink)) {
        return EXIT_FAILURE;
    }
    const auto render_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - render_begin)
            .count();
    const auto *combined =
        sink.find(psycles::contract::PassKind::combined);
    const auto *normal =
        sink.find(psycles::contract::PassKind::normal);
    const auto *albedo =
        sink.find(psycles::contract::PassKind::albedo);
    if (combined == nullptr || normal == nullptr ||
        albedo == nullptr) {
        return EXIT_FAILURE;
    }
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    if (!psycles::io::write_ppm(
            *combined,
            output,
            {.exposure_stops = -2.0f,
             .apply_aces_tonemap = true,
             .apply_srgb_transfer = true})) {
        return EXIT_FAILURE;
    }
    const auto stem = output.parent_path() / output.stem();
    const auto combined_path =
        std::filesystem::path{stem.string() + "-combined.pfm"};
    const auto normal_path =
        std::filesystem::path{stem.string() + "-normal.pfm"};
    const auto albedo_path =
        std::filesystem::path{stem.string() + "-albedo.pfm"};
    if (!psycles::io::write_pfm(*combined, combined_path) ||
        !psycles::io::write_pfm(*normal, normal_path) ||
        !psycles::io::write_pfm(*albedo, albedo_path)) {
        return EXIT_FAILURE;
    }
    constexpr std::array component_passes{
        std::pair{
            psycles::contract::PassKind::glossy_color,
            std::string_view{"glossy-color"}},
        std::pair{
            psycles::contract::PassKind::transmission_color,
            std::string_view{"transmission-color"}},
        std::pair{
            psycles::contract::PassKind::emission,
            std::string_view{"emission"}},
        std::pair{
            psycles::contract::PassKind::environment,
            std::string_view{"environment"}},
        std::pair{
            psycles::contract::PassKind::diffuse_direct,
            std::string_view{"diffuse-direct"}},
        std::pair{
            psycles::contract::PassKind::diffuse_indirect,
            std::string_view{"diffuse-indirect"}},
        std::pair{
            psycles::contract::PassKind::glossy_direct,
            std::string_view{"glossy-direct"}},
        std::pair{
            psycles::contract::PassKind::glossy_indirect,
            std::string_view{"glossy-indirect"}},
        std::pair{
            psycles::contract::PassKind::transmission_direct,
            std::string_view{"transmission-direct"}},
        std::pair{
            psycles::contract::PassKind::transmission_indirect,
            std::string_view{"transmission-indirect"}}};
    for (const auto &[kind, suffix] : component_passes) {
        const auto *pass = sink.find(kind);
        const auto path = std::filesystem::path{
            stem.string() + "-" + std::string{suffix} + ".pfm"};
        if (pass == nullptr ||
            !psycles::io::write_pfm(*pass, path)) {
            return EXIT_FAILURE;
        }
    }

    std::cout
        << "Luisa/" << backend_name << " compiled "
        << imported.scene->geometries.size() << " geometries, "
        << imported.scene->instances.size() << " instances, "
        << imported.scene->materials.size() << " materials in "
        << compile_seconds << " s\n"
        << "Luisa shader JIT completed in "
        << session_seconds << " s\n"
        << "Rendered " << width << 'x' << height << " at "
        << samples << " spp in " << render_seconds << " s: "
        << output << '\n'
        << "Linear Combined: " << combined_path << '\n'
        << "Linear Normal:   " << normal_path << '\n'
        << "Linear Albedo:   " << albedo_path << '\n';
    return EXIT_SUCCESS;
}
