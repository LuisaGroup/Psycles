#include "cycles_holdout_fixture.h"

#include <psycles/adapter/blender_scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>
#include <luisa/runtime/context.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace f = psycles::test_support::holdout;
namespace contract = psycles::contract;
namespace backend = psycles::luisa_backend;
using contract::PassKind;
constexpr std::array passes{PassKind::combined, PassKind::normal, PassKind::albedo,
    PassKind::glossy_color, PassKind::transmission_color, PassKind::diffuse_direct,
    PassKind::diffuse_indirect, PassKind::glossy_direct, PassKind::glossy_indirect,
    PassKind::transmission_direct, PassKind::transmission_indirect, PassKind::emission,
    PassKind::environment, PassKind::volume_direct, PassKind::volume_indirect};

bool render(luisa::compute::Device &device, std::string_view name,
            const contract::CompiledScene &compiled,
            const psycles::adapter::BlenderSceneImport &imported,
            backend::LuisaPathScheduler scheduler) {
  backend::LuisaPathTracerBackend renderer{luisa::compute::Device{device.impl_shared()},
      {.scheduler = scheduler, .wavefront_frame_capacity = 64u, .max_samples_per_dispatch = 1u}};
  contract::RenderSettings settings{
      .full_extent = {16u, 16u}, .seed = imported.seed,
      .transparent_background = imported.transparent_background,
      .pixel_filter = imported.pixel_filter, .filter_width = imported.filter_width,
      .pass_alpha_threshold = imported.pass_alpha_threshold, .integrator = imported.integrator};
  for (const auto kind : passes) {
    settings.passes.push_back({.kind = kind, .channels = kind == PassKind::combined ? 4u : 3u});
  }
  auto session = renderer.create_session(compiled, settings);
  f::require(bool(session), "holdout render session failed");
  psycles::io::MemoryOutputSink output;
  f::require(session->render_samples({.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, output),
             "holdout render failed");
  std::ifstream oracle{f::fixture(name, "-film.txt")};
  unsigned mismatches{};
  for (auto row = 0u; row < 256u; ++row) {
    unsigned x{}, y{};
    oracle >> x >> y;
    f::require(bool(oracle) && x == row % 16u && y == row / 16u, "invalid original film row");
    for (const auto kind : passes) {
      const auto *image = output.find(kind);
      const auto channels = kind == PassKind::combined ? 4u : 3u;
      f::require(image && image->extent.width == 16u && image->extent.height == 16u &&
                     image->channels == channels, "missing original-contract output pass");
      for (auto lane = 0u; lane < channels; ++lane) {
        float expected{};
        oracle >> expected;
        f::require(bool(oracle), "truncated original film");
        const auto actual = image->pixels[row * channels + lane];
        if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-6f + 2e-6f * std::abs(expected)) {
          if (mismatches < 12u) {
            std::cerr << name << " pixel=(" << x << ',' << y << ") pass=" << unsigned(kind)
                      << " lane=" << lane << " actual=" << actual << " original=" << expected << '\n';
          }
          ++mismatches;
        }
      }
    }
  }
  std::string extra;
  f::require(!(oracle >> extra) && oracle.eof(), "extra original film rows");
  std::cout << name << " scheduler=" << backend::luisa_path_scheduler_name(scheduler)
            << " original film lanes=11776 mismatches=" << mismatches << '\n';
  return mismatches == 0u;
}

int main(int argc, char **argv) {
#if defined(_WIN32)
  _putenv_s("PSYCLES_DISABLE_SHADER_CACHE", "1");
#else
  setenv("PSYCLES_DISABLE_SHADER_CACHE", "1", 1);
#endif
  luisa::compute::Context context{argv[0]};
  auto device = context.create_device(argc > 1 ? argv[1] : "hip");
  backend::LuisaPathTracerBackend compiler{luisa::compute::Device{device.impl_shared()}};
  bool passed = true;
  for (const auto name : f::cases) {
    try {
      f::Bundle bundle{name};
      const auto imported = psycles::adapter::load_blender_scene_bundle(bundle.path);
      for (const auto &d : imported.diagnostics) { std::cerr << d.message << '\n'; }
      f::require(imported.ok(), "original holdout import failed");
      const auto compiled = compiler.compile_scene(*imported.scene);
      for (const auto &d : compiled.diagnostics) { std::cerr << d.message << '\n'; }
      f::require(compiled.ok(), "original holdout compilation failed");
      for (const auto scheduler : {backend::LuisaPathScheduler::megakernel,
                                   backend::LuisaPathScheduler::wavefront_staged}) {
        passed = render(device, name, *compiled.scene, imported, scheduler) && passed;
      }
    } catch (const std::exception &error) {
      std::cerr << name << ": " << error.what() << '\n';
      passed = false;
    }
  }
  return passed ? 0 : 1;
}
