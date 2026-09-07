#include <psycles/adapter/blender_scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <luisa/runtime/context.h>

namespace {
namespace contract = psycles::contract;
namespace backend = psycles::luisa_backend;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-volume-film-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  Bundle() {
    require(std::filesystem::create_directory(path), "cannot create bundle");
    std::filesystem::copy_file(PSYCLES_VOLUME_FILM_SCENE, path / "scene.json");
    // Exact external geometry bytes, text-encoded for reviewability only.
    std::ifstream encoded{PSYCLES_VOLUME_FILM_GEOMETRY};
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    unsigned byte{}, count{};
    while (encoded >> std::hex >> byte) {
      require(byte <= 255u, "invalid geometry byte");
      geometry.put(static_cast<char>(byte));
      ++count;
    }
    require(encoded.eof() && count == 2320u && geometry.good(), "truncated geometry fixture");
  }
  ~Bundle() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void render(luisa::compute::Context &context, std::string_view device_name,
            const psycles::adapter::BlenderSceneImport &imported,
            backend::LuisaPathScheduler scheduler) {
  backend::LuisaPathTracerBackend renderer{
      context.create_device(device_name),
      {.scheduler = scheduler, .wavefront_frame_capacity = 64u,
       .max_samples_per_dispatch = 16u}};
  auto compiled = renderer.compile_scene(*imported.scene);
  for (const auto &diagnostic : compiled.diagnostics) { std::cerr << diagnostic.message << '\n'; }
  require(compiled.ok(), "original volume scene did not compile");
  contract::RenderSettings settings{
      .full_extent = {8u, 8u}, .seed = imported.seed,
      .transparent_background = imported.transparent_background,
      .pixel_filter = imported.pixel_filter, .filter_width = imported.filter_width,
      .pass_alpha_threshold = imported.pass_alpha_threshold,
      .integrator = imported.integrator};
  settings.passes = {{.kind = contract::PassKind::combined, .channels = 4u}};
  auto session = renderer.create_session(*compiled.scene, settings);
  require(bool(session), "volume session creation failed");
  psycles::io::MemoryOutputSink output;
  require(session->render_samples({.first = 0u, .count = 16u, .offset = 0u, .total = 16u}, output),
          "volume render failed");
  const auto *image = output.find(contract::PassKind::combined);
  require(image && image->extent.width == 8u && image->extent.height == 8u,
          "volume film is missing");
  for (const auto value : image->pixels) { require(std::isfinite(value), "nonfinite volume film"); }
  std::ifstream oracle{PSYCLES_VOLUME_FILM_ORACLE};
  // Interior/corner rays avoid stochastic edge-coverage differences. This is
  // the complete renderer, including enter/exit, volume integration, and film,
  // not a hand-evaluated emission formula or a transparent closure substitute.
  for (auto i = 0u; i < 8u; ++i) {
    unsigned x{}, y{};
    float expected[3]{};
    oracle >> x >> y >> expected[0] >> expected[1] >> expected[2];
    require(bool(oracle) && x < 8u && y < 8u, "invalid external film oracle");
    for (auto lane = 0u; lane < 3u; ++lane) {
      const auto actual = image->pixels[(y * 8u + x) * image->channels + lane];
      if (std::abs(actual - expected[lane]) > 2.0e-5f) {
        std::cerr << "pixel=" << x << ',' << y << " lane=" << lane
                  << " actual=" << actual << " Cycles=" << expected[lane] << '\n';
        throw std::runtime_error{"volume film differs from original Cycles HIP"};
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
#if defined(_WIN32)
    _putenv_s("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1");
    _putenv_s("PSYCLES_DISABLE_SHADER_CACHE", "1");
#else
    setenv("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1", 1);
    setenv("PSYCLES_DISABLE_SHADER_CACHE", "1", 1);
#endif
    Bundle fixture;
    const auto imported = psycles::adapter::load_blender_scene_bundle(fixture.path);
    require(imported.ok(), "external volume bundle failed to import");
    luisa::compute::Context context{argv[0]};
    const std::string_view device = argc > 1 ? argv[1] : "hip";
    render(context, device, imported, backend::LuisaPathScheduler::megakernel);
    render(context, device, imported, backend::LuisaPathScheduler::wavefront_staged);
    std::cout << "External Cycles volume film passed megakernel and staged coroutine paths\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
