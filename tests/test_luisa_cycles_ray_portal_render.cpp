#include <psycles/adapter/blender_scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <luisa/runtime/context.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace contract = psycles::contract;
namespace backend = psycles::luisa_backend;

void require(bool condition, const char *message) {
  if (!condition) { throw std::runtime_error{message}; }
}

struct Case {
  const char *name;
  unsigned geometry_bytes;
};
constexpr std::array cases{
    Case{"transparent-control", 496u}, Case{"ray-portal", 496u},
    Case{"portal-depth-world", 496u}, Case{"portal-depth-surface", 976u},
    Case{"portal-depth-chain", 976u}, Case{"portal-signed-weight", 496u},
    Case{"portal-transparent-mix", 976u}, Case{"portal-limit-zero", 496u},
    Case{"portal-depth-nee", 976u}, Case{"portal-depth-shadow", 1456u},
    Case{"portal-default-position", 496u}};

struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-ray-portal-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(const Case &input) {
    require(std::filesystem::create_directory(path), "cannot create test bundle");
    const auto prefix = std::filesystem::path{PSYCLES_RAY_PORTAL_RENDER_FIXTURES} / input.name;
    std::filesystem::copy_file(prefix.string() + "-scene.json", path / "scene.json");
    // Decode the exact original exported bytes; no host geometry or renderer
    // model is substituted for the Blender input.
    std::ifstream encoded{prefix.string() + "-geometry.txt"};
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    unsigned byte{}, count{};
    while (encoded >> std::hex >> byte) {
      require(byte <= 255u, "invalid geometry byte");
      geometry.put(static_cast<char>(byte));
      ++count;
    }
    require(encoded.eof() && count == input.geometry_bytes && geometry.good(),
            "truncated original geometry");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

bool render(luisa::compute::Device &device, const Case &input,
            const contract::CompiledScene &compiled,
            const psycles::adapter::BlenderSceneImport &imported,
            backend::LuisaPathScheduler scheduler) {
  backend::LuisaPathTracerBackend renderer{
      luisa::compute::Device{device.impl_shared()},
      {.scheduler = scheduler, .wavefront_frame_capacity = 64u,
       .max_samples_per_dispatch = 1u}};
  contract::RenderSettings settings{
      .full_extent = {16u, 16u}, .seed = imported.seed,
      .transparent_background = imported.transparent_background,
      .pixel_filter = imported.pixel_filter, .filter_width = imported.filter_width,
      .pass_alpha_threshold = imported.pass_alpha_threshold,
      .integrator = imported.integrator};
  settings.passes = {{.kind = contract::PassKind::combined, .channels = 4u}};
  auto session = renderer.create_session(compiled, settings);
  require(bool(session), "ray portal session creation failed");
  psycles::io::MemoryOutputSink output;
  require(session->render_samples({.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, output),
          "ray portal render failed");
  const auto *image = output.find(contract::PassKind::combined);
  require(image && image->extent.width == 16u && image->extent.height == 16u &&
              image->channels == 4u, "missing complete ray portal film");
  std::ifstream oracle{(std::filesystem::path{PSYCLES_RAY_PORTAL_RENDER_FIXTURES} /
                        (std::string{input.name} + "-film.txt"))};
  unsigned mismatches{};
  for (unsigned row = 0; row < 256u; ++row) {
    unsigned x{}, y{};
    std::array<float, 3u> expected{};
    oracle >> x >> y >> expected[0] >> expected[1] >> expected[2];
    require(bool(oracle) && x == row % 16u && y == row / 16u,
            "invalid original HIP film row");
    for (unsigned lane = 0; lane < 3u; ++lane) {
      const auto actual = image->pixels[row * image->channels + lane];
      const auto tolerance = 2.0e-6f + 2.0e-6f * std::abs(expected[lane]);
      const auto mismatch = !std::isfinite(actual) || std::abs(actual - expected[lane]) > tolerance;
      if (mismatch && mismatches < 12u) {
        std::cerr << input.name << " pixel=(" << x << ',' << y << ") lane=" << lane
                  << " actual=" << actual << " original=" << expected[lane] << '\n';
      }
      mismatches += mismatch;
    }
  }
  std::string extra;
  require(!(oracle >> extra) && oracle.eof(), "extra original HIP film rows");
  std::cout << input.name << " scheduler=" << backend::luisa_path_scheduler_name(scheduler)
            << " original RGB lanes=768 mismatches=" << mismatches << '\n';
  return mismatches == 0u;
}
} // namespace

int main(int argc, char **argv) {
  try {
#if defined(_WIN32)
    _putenv_s("PSYCLES_DISABLE_SHADER_CACHE", "1");
#else
    setenv("PSYCLES_DISABLE_SHADER_CACHE", "1", 1);
#endif
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    backend::LuisaPathTracerBackend compiler{luisa::compute::Device{device.impl_shared()}};
    bool passed = true;
    for (const auto &input : cases) {
      Bundle fixture{input};
      const auto imported = psycles::adapter::load_blender_scene_bundle(fixture.path);
      require(imported.ok(), "original ray portal bundle failed to import");
      const auto compiled = compiler.compile_scene(*imported.scene);
      for (const auto &diagnostic : compiled.diagnostics) {
        std::cerr << diagnostic.message << '\n';
      }
      require(compiled.ok(), "original ray portal scene failed to compile");
      for (const auto scheduler : {backend::LuisaPathScheduler::megakernel,
                                   backend::LuisaPathScheduler::wavefront_staged}) {
        passed = render(device, input, *compiled.scene, imported, scheduler) && passed;
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
