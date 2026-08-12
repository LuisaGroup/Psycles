#include <psycles/adapter/blender_scene.h>
#include <psycles/contract/render.h>
#include <psycles/io/image.h>
#include <psycles/io/progressive_pixel_probe.h>
#include <psycles/luisa/path_tracer.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>
#include <yyjson.h>

namespace {

template<typename T>
[[nodiscard]] std::optional<T> parse_unsigned(std::string_view text) {
    T result{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return result;
}

void enforce_vulkan_native_xir_spirv(std::string_view backend_name) {
    if (backend_name != "vk" && backend_name != "vulkan") {
        return;
    }
    if (std::getenv("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV") != nullptr) {
        return;
    }
#ifdef _WIN32
    _putenv_s("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1");
#else
  setenv("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1", 1);
#endif
}

class MemoryPathTraceSink final
    : public psycles::luisa_backend::LuisaPathTraceSink {

private:
    std::optional<psycles::luisa_backend::LuisaPathTrace> _trace;

public:
  void write(const psycles::luisa_backend::LuisaPathTrace &trace) override {
        _trace = trace;
    }

  [[nodiscard]] const std::optional<psycles::luisa_backend::LuisaPathTrace> &
    trace() const noexcept {
        return _trace;
    }
};

[[nodiscard]] bool
write_raw_path_trace(const psycles::luisa_backend::LuisaPathTrace &trace,
    const std::filesystem::path &path) {
    auto *document = yyjson_mut_doc_new(nullptr);
    if (document == nullptr) {
        return false;
    }
    auto *root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
  yyjson_mut_obj_add_str(document, root, "schema",
        "psycles.cycles-path-trace-raw");
  yyjson_mut_obj_add_uint(document, root, "trace_schema_version",
        psycles::luisa_backend::path_trace_schema::version);
  yyjson_mut_obj_add_uint(document, root, "pixel_x", trace.pixel_x);
  yyjson_mut_obj_add_uint(document, root, "pixel_y", trace.pixel_y);
  yyjson_mut_obj_add_uint(document, root, "sample", trace.sample);
    auto *slots = yyjson_mut_arr(document);
    for (const auto &slot : trace.slots) {
        auto *rgba = yyjson_mut_arr(document);
        for (const auto value : slot) {
      yyjson_mut_arr_add_real(document, rgba, static_cast<double>(value));
        }
        yyjson_mut_arr_add_val(slots, rgba);
    }
    yyjson_mut_obj_add_val(document, root, "slots", slots);
    yyjson_write_err error{};
    const auto written = yyjson_mut_write_file(
      path.string().c_str(), document, YYJSON_WRITE_PRETTY, nullptr, &error);
    if (!written) {
    std::cerr << "error: could not write path trace JSON: "
              << (error.msg != nullptr ? error.msg : "unknown error") << '\n';
    }
    yyjson_mut_doc_free(document);
    return written;
}

[[nodiscard]] bool write_sample_chunk_probe(
    const std::filesystem::path &path, std::uint32_t pixel_x,
    std::uint32_t pixel_y, std::uint32_t total_samples,
    std::span<const psycles::io::ProgressiveSampleChunk> records) {
    auto *document = yyjson_mut_doc_new(nullptr);
    if (document == nullptr) {
        return false;
    }
    auto *root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
  yyjson_mut_obj_add_str(document, root, "schema",
        "psycles.sample-chunk-pixel.v1");
  yyjson_mut_obj_add_uint(document, root, "pixel_x", pixel_x);
  yyjson_mut_obj_add_uint(document, root, "pixel_y", pixel_y);
  yyjson_mut_obj_add_str(document, root, "coordinate_convention",
        "Cycles film coordinates; origin lower-left");
  yyjson_mut_obj_add_uint(document, root, "total_samples", total_samples);
  yyjson_mut_obj_add_str(document, root, "value_semantics",
        "delta(progressive_output * rendered_sample_count)");
    auto *chunk_array = yyjson_mut_arr(document);
    for (const auto &record : records) {
        auto *chunk = yyjson_mut_obj(document);
    yyjson_mut_obj_add_uint(document, chunk, "sample_first",
            record.sample_first);
    yyjson_mut_obj_add_uint(document, chunk, "sample_count",
            record.sample_count);
        auto *passes = yyjson_mut_obj(document);
        for (const auto &pass : record.passes) {
            auto *values = yyjson_mut_arr(document);
            for (const auto value : pass.values) {
        yyjson_mut_arr_add_real(document, values, value);
            }
      yyjson_mut_obj_add_val(document, passes, pass.pass.name.c_str(), values);
        }
    yyjson_mut_obj_add_val(document, chunk, "passes", passes);
        yyjson_mut_arr_add_val(chunk_array, chunk);
    }
  yyjson_mut_obj_add_val(document, root, "chunks", chunk_array);
    yyjson_write_err error{};
    const auto written = yyjson_mut_write_file(
      path.string().c_str(), document, YYJSON_WRITE_PRETTY, nullptr, &error);
    if (!written) {
    std::cerr << "error: could not write sample-chunk probe JSON: "
              << (error.msg != nullptr ? error.msg : "unknown error") << '\n';
    }
    yyjson_mut_doc_free(document);
    return written;
}

}// namespace

int main(int argc, char **argv) {
    if (argc < 3) {
    std::cerr << "usage: psycles_render_blender_scene "
               "<export-directory> <output.ppm> "
               "[backend=fallback] [width] [height] [samples] "
               "[max-samples-per-dispatch=64] "
               "[path-trace.json|-] [trace-x] [trace-y] "
               "[trace-sample=0] [sample-first=0] "
               "[sample-count=samples-sample-first] "
               "[sample-chunk-pixel.json|-] "
               "[probe-chunk-size=1] [probe-full-frame=0] "
                 "[scheduler=megakernel|megakernel-per-sample|wavefront|"
                 "wavefront-graph|wavefront-staged|persistent] "
               "[wavefront-execution-block-size=32] "
               "[persistent-workers=32768] "
               "[persistent-block-size=32] "
               "[persistent-fetch-size=1] "
                 "[staged-surface-sorting=1] "
                 "[staged-direct-light-queue=0] "
                 "[wavefront-counter-readback-batch-size=4] "
                 "[wavefront-counter-readback-pipeline-depth=2] "
                 "[wavefront-tail-megakernel-threshold=4096]\n";
        return EXIT_FAILURE;
    }
    const auto bundle = std::filesystem::path{argv[1]};
    const auto output = std::filesystem::path{argv[2]};
  const auto backend_name = std::string_view{argc > 3 ? argv[3] : "fallback"};

  auto imported = psycles::adapter::load_blender_scene_bundle(bundle);
    for (const auto &diagnostic : imported.diagnostics) {
        std::cerr
            << (diagnostic.severity ==
                    psycles::adapter::BlenderSceneDiagnosticSeverity::error
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
    auto max_samples_per_dispatch = std::uint32_t{64u};
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
    if (argc > 8 && std::string_view{argv[8]} != "-") {
    path_trace_output = std::filesystem::path{argv[8]};
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
    auto sample_first = std::uint32_t{0u};
    if (argc > 12) {
        auto value = parse_unsigned<std::uint32_t>(argv[12]);
        if (!value || *value >= samples) {
            return EXIT_FAILURE;
        }
        sample_first = *value;
    }
    auto sample_count = samples - sample_first;
    if (argc > 13) {
        auto value = parse_unsigned<std::uint32_t>(argv[13]);
        if (!value || *value == 0u ||
            static_cast<std::uint64_t>(sample_first) +
                    static_cast<std::uint64_t>(*value) >
                static_cast<std::uint64_t>(samples)) {
            return EXIT_FAILURE;
        }
        sample_count = *value;
    }
    if (path_trace_output) {
        if (argc <= 11) {
            path_trace_sample = sample_first;
        }
    const auto sample_end = static_cast<std::uint64_t>(sample_first) +
            static_cast<std::uint64_t>(sample_count);
        if (path_trace_sample < sample_first ||
        static_cast<std::uint64_t>(path_trace_sample) >= sample_end) {
            return EXIT_FAILURE;
        }
    }
    std::optional<std::filesystem::path> sample_chunk_output;
    if (argc > 14) {
        const auto output_argument = std::string_view{argv[14]};
        if (output_argument != "-") {
      sample_chunk_output = std::filesystem::path{output_argument};
            if (sample_chunk_output->empty()) {
                return EXIT_FAILURE;
            }
        }
    }
    auto probe_chunk_size = std::uint32_t{1u};
    if (argc > 15) {
        auto value = parse_unsigned<std::uint32_t>(argv[15]);
        if (!value || *value == 0u || *value > sample_count) {
            return EXIT_FAILURE;
        }
        probe_chunk_size = *value;
    }
    auto probe_full_frame = false;
    if (argc > 16) {
        auto value = parse_unsigned<std::uint32_t>(argv[16]);
        if (!value || *value > 1u) {
            return EXIT_FAILURE;
        }
        probe_full_frame = *value != 0u;
    }
  auto scheduler = psycles::luisa_backend::LuisaPathScheduler::megakernel;
    if (argc > 17) {
        const auto parsed =
        psycles::luisa_backend::parse_luisa_path_scheduler(argv[17]);
        if (!parsed) {
      std::cerr << "error: invalid path scheduler '" << argv[17]
                << "' (expected megakernel, megakernel-per-sample, "
                   "wavefront, wavefront-graph, wavefront-staged, or "
                   "persistent)\n";
            return EXIT_FAILURE;
        }
        scheduler = *parsed;
    }
    auto wavefront_execution_block_size = std::uint32_t{32u};
    if (argc > 18) {
        auto value = parse_unsigned<std::uint32_t>(argv[18]);
        if (!value ||
        !psycles::luisa_backend::valid_luisa_wavefront_execution_block_size(
            *value)) {
      std::cerr << "error: wavefront execution block size must be a "
                   "multiple of 32 in [32, 1024]\n";
            return EXIT_FAILURE;
        }
        wavefront_execution_block_size = *value;
    }
    auto persistent_worker_count = std::uint32_t{1u << 15u};
    if (argc > 19) {
        auto value = parse_unsigned<std::uint32_t>(argv[19]);
        if (!value || *value == 0u) {
      std::cerr << "error: persistent worker count must be positive\n";
            return EXIT_FAILURE;
        }
        persistent_worker_count = *value;
    }
    auto persistent_block_size = std::uint32_t{32u};
    if (argc > 20) {
        auto value = parse_unsigned<std::uint32_t>(argv[20]);
        if (!value ||
        !psycles::luisa_backend::valid_luisa_execution_block_size(*value)) {
      std::cerr << "error: persistent block size must be a multiple of 32 "
                   "in [32, 1024]\n";
            return EXIT_FAILURE;
        }
        persistent_block_size = *value;
    }
    auto persistent_fetch_size = std::uint32_t{1u};
    if (argc > 21) {
        auto value = parse_unsigned<std::uint32_t>(argv[21]);
        if (!value || *value == 0u) {
      std::cerr << "error: persistent fetch size must be positive\n";
            return EXIT_FAILURE;
        }
        persistent_fetch_size = *value;
    }
    auto staged_surface_sorting = true;
    if (argc > 22) {
        auto value = parse_unsigned<std::uint32_t>(argv[22]);
        if (!value || *value > 1u) {
      std::cerr << "error: staged surface sorting must be 0 or 1\n";
            return EXIT_FAILURE;
        }
        staged_surface_sorting = *value != 0u;
    }
    auto staged_direct_light_queue = false;
  if (argc > 23) {
    auto value = parse_unsigned<std::uint32_t>(argv[23]);
    if (!value || *value > 1u) {
      std::cerr << "error: staged direct-light queue must be 0 or 1\n";
      return EXIT_FAILURE;
    }
    staged_direct_light_queue = *value != 0u;
  }
  auto wavefront_counter_readback_batch_size = std::uint32_t{4u};
  if (argc > 24) {
    auto value = parse_unsigned<std::uint32_t>(argv[24]);
    if (!value || *value == 0u) {
      std::cerr << "error: wavefront counter readback batch size must be "
                   "positive\n";
      return EXIT_FAILURE;
    }
    wavefront_counter_readback_batch_size = *value;
  }
  auto wavefront_counter_readback_pipeline_depth = std::uint32_t{2u};
  if (argc > 25) {
    auto value = parse_unsigned<std::uint32_t>(argv[25]);
    if (!value || *value == 0u) {
      std::cerr << "error: wavefront counter readback pipeline depth must be "
                   "positive\n";
      return EXIT_FAILURE;
    }
    wavefront_counter_readback_pipeline_depth = *value;
  }
  auto wavefront_tail_megakernel_threshold = std::uint32_t{4096u};
  if (argc > 26) {
    auto value = parse_unsigned<std::uint32_t>(argv[26]);
    if (!value) {
      std::cerr << "error: invalid wavefront tail megakernel threshold\n";
      return EXIT_FAILURE;
    }
    wavefront_tail_megakernel_threshold = *value;
  }
  if (!psycles::luisa_backend::valid_luisa_persistent_scheduler_shape(
          persistent_worker_count, persistent_block_size,
                persistent_fetch_size)) {
    std::cerr << "error: persistent block-size times fetch-size exceeds "
               "the uint scheduler ABI\n";
        return EXIT_FAILURE;
    }
  auto path_trace_sink = path_trace_output
            ? std::make_shared<MemoryPathTraceSink>()
            : std::shared_ptr<MemoryPathTraceSink>{};
  std::optional<psycles::luisa_backend::LuisaPathTraceRequest>
        path_trace_request;
    if (path_trace_sink) {
    path_trace_request = psycles::luisa_backend::LuisaPathTraceRequest{
                .pixel_x = path_trace_x,
                .pixel_y = path_trace_y,
                .sample = path_trace_sample,
                .sink = path_trace_sink};
    }

    enforce_vulkan_native_xir_spirv(backend_name);
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend_name);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .scheduler = scheduler,
       .wavefront_execution_block_size = wavefront_execution_block_size,
       .wavefront_counter_readback_batch_size =
           wavefront_counter_readback_batch_size,
       .wavefront_counter_readback_pipeline_depth =
           wavefront_counter_readback_pipeline_depth,
       .wavefront_tail_megakernel_threshold =
           wavefront_tail_megakernel_threshold,
       .staged_surface_sorting = staged_surface_sorting,
       .staged_direct_light_queue = staged_direct_light_queue,
         .persistent_worker_count = persistent_worker_count,
         .persistent_block_size = persistent_block_size,
         .persistent_fetch_size = persistent_fetch_size,
       .max_samples_per_dispatch = max_samples_per_dispatch,
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
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    compile_begin)
            .count();

    const psycles::contract::RenderSettings settings{
        .full_extent = {.width = width, .height = height},
      .window =
          sample_chunk_output && !probe_full_frame
              ? psycles::contract::PixelWindow{.x = path_trace_x,
                            .y = height - 1u - path_trace_y,
                            .width = 1u,
                            .height = 1u}
                      : psycles::contract::PixelWindow{},
        .seed = imported.seed,
      .transparent_background = imported.transparent_background,
        .pixel_filter = imported.pixel_filter,
        .filter_width = imported.filter_width,
      .pass_alpha_threshold = imported.pass_alpha_threshold,
        .integrator = imported.integrator,
      .passes = {{.kind = psycles::contract::PassKind::combined,
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
                 {.kind = psycles::contract::PassKind::glossy_color,
             .name = "GlossCol",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::transmission_color,
             .name = "TransCol",
             .light_group = {},
             .channels = 3u},
            {.kind = psycles::contract::PassKind::emission,
             .name = "Emit",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::environment,
             .name = "Env",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::diffuse_direct,
             .name = "DiffDir",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::diffuse_indirect,
             .name = "DiffInd",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::glossy_direct,
             .name = "GlossDir",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::glossy_indirect,
             .name = "GlossInd",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::transmission_direct,
             .name = "TransDir",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::transmission_indirect,
             .name = "TransInd",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::volume_direct,
             .name = "VolumeDir",
             .light_group = {},
             .channels = 3u},
                 {.kind = psycles::contract::PassKind::volume_indirect,
             .name = "VolumeInd",
             .light_group = {},
             .channels = 3u}}};
  const auto session_begin = std::chrono::steady_clock::now();
  auto session = renderer.create_session(*compilation.scene, settings);
    if (!session) {
        return EXIT_FAILURE;
    }
    const auto session_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    session_begin)
            .count();
    if (sample_chunk_output) {
        const auto raster_y = height - 1u - path_trace_y;
    psycles::io::PixelOutputSink pixel_sink{path_trace_x, raster_y};
        psycles::io::ProgressivePixelAccumulator accumulator;
        std::vector<psycles::io::ProgressiveSampleChunk> records;
    records.reserve((sample_count + probe_chunk_size - 1u) / probe_chunk_size);
    const auto render_begin = std::chrono::steady_clock::now();
    for (std::uint32_t offset = 0u; offset < sample_count;) {
            const auto absolute_sample = sample_first + offset;
      const auto chunk_count =
          std::min(probe_chunk_size, sample_count - offset);
      if (!session->render_samples({.first = absolute_sample,
                     .count = chunk_count,
                     .offset = 0u,
                     .total = samples},
                    pixel_sink) ||
                pixel_sink.cancelled()) {
        std::cerr << "error: sample-chunk pixel probe failed at "
                    << absolute_sample << '\n';
                return EXIT_FAILURE;
            }
            const auto &captures = pixel_sink.passes();
            if (captures.size() != settings.passes.size()) {
        std::cerr << "error: sample-chunk pixel probe returned "
                    << captures.size() << " passes, expected "
                    << settings.passes.size() << '\n';
                return EXIT_FAILURE;
            }
      auto record = accumulator.append(absolute_sample, chunk_count, captures);
            if (!record) {
        std::cerr << "error: sample-chunk pixel probe changed pass "
                       "layout at "
                    << absolute_sample << '\n';
                return EXIT_FAILURE;
            }
            records.emplace_back(std::move(*record));
            offset += chunk_count;
        }
        if (!sample_chunk_output->parent_path().empty()) {
      std::filesystem::create_directories(sample_chunk_output->parent_path());
        }
    if (!write_sample_chunk_probe(*sample_chunk_output, path_trace_x,
                                  path_trace_y, samples, records)) {
            return EXIT_FAILURE;
        }
        if (path_trace_output) {
            if (!path_trace_output->parent_path().empty()) {
        std::filesystem::create_directories(path_trace_output->parent_path());
            }
            if (!path_trace_sink->trace() ||
          !write_raw_path_trace(*path_trace_sink->trace(),
                    *path_trace_output)) {
                return EXIT_FAILURE;
            }
        }
        const auto render_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      render_begin)
                .count();
    std::cout << "Luisa/" << backend_name << '/'
              << psycles::luisa_backend::luisa_path_scheduler_name(scheduler)
              << " compiled " << imported.scene->geometries.size()
              << " geometries, " << imported.scene->instances.size()
              << " instances, " << imported.scene->materials.size()
              << " materials in " << compile_seconds << " s\n"
              << "Luisa shader JIT completed in " << session_seconds << " s\n"
            << "Captured " << records.size()
              << " sample-chunk records for pixel (" << path_trace_x << ", "
              << path_trace_y << ") in " << render_seconds
              << " s: " << *sample_chunk_output << '\n';
        return EXIT_SUCCESS;
    }
    psycles::io::MemoryOutputSink sink;
    const auto render_begin = std::chrono::steady_clock::now();
  if (!session->render_samples({.first = sample_first,
             .count = sample_count,
             .offset = 0u,
             .total = samples},
            sink)) {
    std::cerr << "error: rendering failed or returned incomplete "
               "sample coverage\n";
        return EXIT_FAILURE;
    }
    if (path_trace_output) {
        if (!path_trace_sink->trace()) {
      std::cerr << "error: requested path trace was not produced\n";
            return EXIT_FAILURE;
        }
        if (!path_trace_output->parent_path().empty()) {
      std::filesystem::create_directories(path_trace_output->parent_path());
        }
    if (!write_raw_path_trace(*path_trace_sink->trace(), *path_trace_output)) {
            return EXIT_FAILURE;
        }
    }
    const auto render_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    render_begin)
            .count();
  const auto *combined = sink.find(psycles::contract::PassKind::combined);
  const auto *normal = sink.find(psycles::contract::PassKind::normal);
  const auto *albedo = sink.find(psycles::contract::PassKind::albedo);
  if (combined == nullptr || normal == nullptr || albedo == nullptr) {
        return EXIT_FAILURE;
    }
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
  if (!psycles::io::write_ppm(*combined, output,
            {.exposure_stops = -2.0f,
             .apply_aces_tonemap = true,
             .apply_srgb_transfer = true})) {
        return EXIT_FAILURE;
    }
    const auto stem = output.parent_path() / output.stem();
    const auto combined_path =
        std::filesystem::path{stem.string() + "-combined.pfm"};
  const auto normal_path = std::filesystem::path{stem.string() + "-normal.pfm"};
  const auto albedo_path = std::filesystem::path{stem.string() + "-albedo.pfm"};
    if (!psycles::io::write_pfm(*combined, combined_path) ||
        !psycles::io::write_pfm(*normal, normal_path) ||
        !psycles::io::write_pfm(*albedo, albedo_path)) {
        return EXIT_FAILURE;
    }
    constexpr std::array component_passes{
      std::pair{psycles::contract::PassKind::glossy_color,
            std::string_view{"glossy-color"}},
      std::pair{psycles::contract::PassKind::transmission_color,
            std::string_view{"transmission-color"}},
      std::pair{psycles::contract::PassKind::emission,
            std::string_view{"emission"}},
      std::pair{psycles::contract::PassKind::environment,
            std::string_view{"environment"}},
      std::pair{psycles::contract::PassKind::diffuse_direct,
            std::string_view{"diffuse-direct"}},
      std::pair{psycles::contract::PassKind::diffuse_indirect,
            std::string_view{"diffuse-indirect"}},
      std::pair{psycles::contract::PassKind::glossy_direct,
            std::string_view{"glossy-direct"}},
      std::pair{psycles::contract::PassKind::glossy_indirect,
            std::string_view{"glossy-indirect"}},
      std::pair{psycles::contract::PassKind::transmission_direct,
            std::string_view{"transmission-direct"}},
      std::pair{psycles::contract::PassKind::transmission_indirect,
            std::string_view{"transmission-indirect"}},
      std::pair{psycles::contract::PassKind::volume_direct,
            std::string_view{"volume-direct"}},
      std::pair{psycles::contract::PassKind::volume_indirect,
            std::string_view{"volume-indirect"}}};
    for (const auto &[kind, suffix] : component_passes) {
        const auto *pass = sink.find(kind);
    const auto path = std::filesystem::path{stem.string() + "-" +
                                            std::string{suffix} + ".pfm"};
    if (pass == nullptr || !psycles::io::write_pfm(*pass, path)) {
            return EXIT_FAILURE;
        }
    }

#if defined(PSYCLES_WITH_OPENIMAGEIO)
  const auto exr_path = std::filesystem::path{stem.string() + ".exr"};
    std::string exr_error;
  if (!psycles::io::write_multilayer_exr(sink.images(), exr_path, "ViewLayer",
            &exr_error)) {
    std::cerr << "error: could not write multilayer OpenEXR: " << exr_error
              << '\n';
        return EXIT_FAILURE;
    }
#endif

  std::cout << "Luisa/" << backend_name << '/'
            << psycles::luisa_backend::luisa_path_scheduler_name(scheduler)
            << " compiled " << imported.scene->geometries.size()
            << " geometries, " << imported.scene->instances.size()
            << " instances, " << imported.scene->materials.size()
            << " materials in " << compile_seconds << " s\n"
            << "Luisa shader JIT completed in " << session_seconds << " s\n"
            << "Rendered " << width << 'x' << height << " at " << sample_count
            << " spp from absolute sample range [" << sample_first << ", "
            << sample_first + sample_count << ") of " << samples << " in "
            << render_seconds << " s: " << output << '\n'
        << "Linear Combined: " << combined_path << '\n'
        << "Linear Normal:   " << normal_path << '\n'
        << "Linear Albedo:   " << albedo_path << '\n'
        << (path_trace_output
                    ? "Path trace:      " + path_trace_output->string() + "\n"
                : std::string{})
#if defined(PSYCLES_WITH_OPENIMAGEIO)
        << "Multilayer EXR:  " << exr_path << '\n'
#endif
        ;
    return EXIT_SUCCESS;
}
