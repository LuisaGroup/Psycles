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
#include <yyjson.h>

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

class MemoryPathTraceSink final
    : public psycles::luisa_backend::LuisaPathTraceSink {

private:
    std::optional<psycles::luisa_backend::LuisaPathTrace> _trace;

public:
    void write(
        const psycles::luisa_backend::LuisaPathTrace &trace) override {
        _trace = trace;
    }

    [[nodiscard]] const std::optional<
        psycles::luisa_backend::LuisaPathTrace> &
    trace() const noexcept {
        return _trace;
    }
};

[[nodiscard]] bool write_raw_path_trace(
    const psycles::luisa_backend::LuisaPathTrace &trace,
    const std::filesystem::path &path) {
    auto *document = yyjson_mut_doc_new(nullptr);
    if (document == nullptr) {
        return false;
    }
    auto *root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_obj_add_str(
        document,
        root,
        "schema",
        "psycles.cycles-path-trace-raw");
    yyjson_mut_obj_add_uint(
        document,
        root,
        "trace_schema_version",
        psycles::luisa_backend::path_trace_schema::version);
    yyjson_mut_obj_add_uint(
        document, root, "pixel_x", trace.pixel_x);
    yyjson_mut_obj_add_uint(
        document, root, "pixel_y", trace.pixel_y);
    yyjson_mut_obj_add_uint(
        document, root, "sample", trace.sample);
    auto *slots = yyjson_mut_arr(document);
    for (const auto &slot : trace.slots) {
        auto *rgba = yyjson_mut_arr(document);
        for (const auto value : slot) {
            yyjson_mut_arr_add_real(
                document,
                rgba,
                static_cast<double>(value));
        }
        yyjson_mut_arr_add_val(slots, rgba);
    }
    yyjson_mut_obj_add_val(document, root, "slots", slots);
    yyjson_write_err error{};
    const auto written = yyjson_mut_write_file(
        path.string().c_str(),
        document,
        YYJSON_WRITE_PRETTY,
        nullptr,
        &error);
    if (!written) {
        std::cerr
            << "error: could not write path trace JSON: "
            << (error.msg != nullptr ? error.msg : "unknown error")
            << '\n';
    }
    yyjson_mut_doc_free(document);
    return written;
}

}// namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr
            << "usage: psycles_render_blender_scene "
               "<export-directory> <output.ppm> "
               "[backend=fallback] [width] [height] [samples] "
               "[max-samples-per-dispatch=8] "
               "[path-trace.json] [trace-x] [trace-y] "
               "[trace-sample=0]\n";
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
    auto max_samples_per_dispatch = std::uint32_t{8u};
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
    if (argc > 7) {
        auto value = parse_unsigned<std::uint32_t>(argv[7]);
        if (!value || *value == 0u) {
            return EXIT_FAILURE;
        }
        max_samples_per_dispatch = *value;
    }
    std::optional<std::filesystem::path> path_trace_output;
    auto path_trace_x = width / 2u;
    auto path_trace_y = height / 2u;
    auto path_trace_sample = std::uint32_t{0u};
    if (argc > 8) {
        path_trace_output =
            std::filesystem::path{argv[8]};
    }
    if (argc > 9) {
        auto value = parse_unsigned<std::uint32_t>(argv[9]);
        if (!value || *value >= width) {
            return EXIT_FAILURE;
        }
        path_trace_x = *value;
    }
    if (argc > 10) {
        auto value = parse_unsigned<std::uint32_t>(argv[10]);
        if (!value || *value >= height) {
            return EXIT_FAILURE;
        }
        path_trace_y = *value;
    }
    if (argc > 11) {
        auto value = parse_unsigned<std::uint32_t>(argv[11]);
        if (!value || *value >= samples) {
            return EXIT_FAILURE;
        }
        path_trace_sample = *value;
    }
    auto path_trace_sink =
        path_trace_output
            ? std::make_shared<MemoryPathTraceSink>()
            : std::shared_ptr<MemoryPathTraceSink>{};
    std::optional<
        psycles::luisa_backend::LuisaPathTraceRequest>
        path_trace_request;
    if (path_trace_sink) {
        path_trace_request =
            psycles::luisa_backend::LuisaPathTraceRequest{
                .pixel_x = path_trace_x,
                .pixel_y = path_trace_y,
                .sample = path_trace_sample,
                .sink = path_trace_sink};
    }

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend_name);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .max_samples_per_dispatch =
             max_samples_per_dispatch,
         .path_trace = path_trace_request}};
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
             .name = "DiffCol",
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
    if (path_trace_output) {
        if (!path_trace_sink->trace()) {
            std::cerr
                << "error: requested path trace was not produced\n";
            return EXIT_FAILURE;
        }
        if (!path_trace_output->parent_path().empty()) {
            std::filesystem::create_directories(
                path_trace_output->parent_path());
        }
        if (!write_raw_path_trace(
                *path_trace_sink->trace(),
                *path_trace_output)) {
            return EXIT_FAILURE;
        }
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

#if defined(PSYCLES_WITH_OPENIMAGEIO)
    const auto exr_path =
        std::filesystem::path{stem.string() + ".exr"};
    std::string exr_error;
    if (!psycles::io::write_multilayer_exr(
            sink.images(),
            exr_path,
            "ViewLayer",
            &exr_error)) {
        std::cerr
            << "error: could not write multilayer OpenEXR: "
            << exr_error << '\n';
        return EXIT_FAILURE;
    }
#endif

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
        << "Linear Albedo:   " << albedo_path << '\n'
        << (path_trace_output
                ? "Path trace:      " +
                      path_trace_output->string() + "\n"
                : std::string{})
#if defined(PSYCLES_WITH_OPENIMAGEIO)
        << "Multilayer EXR:  " << exr_path << '\n'
#endif
        ;
    return EXIT_SUCCESS;
}
