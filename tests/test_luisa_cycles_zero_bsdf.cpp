#include <psycles/adapter/blender_scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include "path_tracer_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <luisa/runtime/context.h>

namespace {
namespace contract = psycles::contract;
namespace backend = psycles::luisa_backend;
namespace trace_schema = backend::path_trace_schema;

void require(bool condition, const char *message) {
  if (!condition) { throw std::runtime_error{message}; }
}

struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-zero-bsdf-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  Bundle() {
    require(std::filesystem::create_directory(path), "cannot create bundle");
    std::filesystem::copy_file(PSYCLES_ZERO_BSDF_SCENE, path / "scene.json");
    std::ifstream encoded{PSYCLES_ZERO_BSDF_GEOMETRY};
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    unsigned byte{}, count{};
    while (encoded >> std::hex >> byte) {
      require(byte <= 255u, "invalid geometry byte");
      geometry.put(static_cast<char>(byte));
      ++count;
    }
    require(encoded.eof() && count == 1456u && geometry.good(),
            "truncated original geometry");
  }
  ~Bundle() { std::error_code error; std::filesystem::remove_all(path, error); }
};

class TraceSink final : public backend::LuisaPathTraceSink {
public:
  std::optional<backend::LuisaPathTrace> value;
  void write(const backend::LuisaPathTrace &trace) override { value = trace; }
};

bool render(luisa::compute::Device &device,
            const contract::CompiledScene &compiled,
            const psycles::adapter::BlenderSceneImport &imported,
            backend::LuisaPathScheduler scheduler, unsigned pixel_x) {
  const auto sink = std::make_shared<TraceSink>();
  backend::LuisaPathTracerBackend renderer{
      luisa::compute::Device{device.impl_shared()},
      {.scheduler = scheduler, .wavefront_frame_capacity = 64u,
       .max_samples_per_dispatch = 16u,
       .path_trace = backend::LuisaPathTraceRequest{
           .pixel_x = pixel_x, .pixel_y = 2u, .sample = 0u, .sink = sink}}};
  contract::RenderSettings settings{
      .full_extent = {12u, 4u},
      .window = contract::PixelWindow{.x = pixel_x, .y = 1u, .width = 1u, .height = 1u},
      .seed = imported.seed,
      .transparent_background = imported.transparent_background,
      .pixel_filter = imported.pixel_filter, .filter_width = imported.filter_width,
      .pass_alpha_threshold = imported.pass_alpha_threshold,
      .integrator = imported.integrator};
  settings.passes = {{.kind = contract::PassKind::combined, .channels = 4u}};
  auto session = renderer.create_session(compiled, settings);
  require(bool(session), "zero-BSDF session creation failed");
  psycles::io::MemoryOutputSink output;
  require(session->render_samples({.first = 0u, .count = 1u, .offset = 0u, .total = 16u}, output),
          "zero-BSDF render failed");
  require(sink->value.has_value(), "missing production path trace");
  const auto &trace = *sink->value;
  // Original HIP AOV observations, not locally reconstructed BSDF values.
  // Discrete bounce state and the write-presence lane are exact; radiometric
  // and directional values allow ordinary fast-math differences.
  std::ifstream oracle{PSYCLES_ZERO_BSDF_TRACE};
  require(bool(oracle), "missing Cycles HIP trace");
  unsigned x{}, slot{}, checked{};
  bool passed = true;
  while (oracle >> x >> slot) {
    float expected[4]{};
    for (auto &v : expected) { oracle >> v; }
    require(bool(oracle) && slot < trace_schema::slot_count, "invalid trace row");
    if (x != pixel_x) { continue; }
    ++checked;
    for (unsigned lane = 0u; lane < 4u; ++lane) {
      const auto actual = trace.slots[slot][lane];
      const bool exact = lane == 3u || slot == 40u || slot == 44u || slot == 47u;
      const auto tolerance = exact ? 0.0f :
          5.0e-6f + 5.0e-5f * std::max(std::abs(actual), std::abs(expected[lane]));
      if (!std::isfinite(actual) || std::abs(actual - expected[lane]) > tolerance) {
        std::cerr << backend::luisa_path_scheduler_name(scheduler)
                  << " pixel=" << x << " slot=" << slot << " lane=" << lane
                  << " actual=" << actual << " Cycles=" << expected[lane] << '\n';
        passed = false;
      }
    }
  }
  require(oracle.eof() && checked == 15u, "incomplete trace oracle");
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  try {
#if defined(_WIN32)
    _putenv_s("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "");
    _putenv_s("PSYCLES_DISABLE_SHADER_CACHE", "1");
#else
    unsetenv("PSYCLES_NATIVE_CYCLES_SVM_SURFACE");
    setenv("PSYCLES_DISABLE_SHADER_CACHE", "1", 1);
#endif
    Bundle fixture;
    auto imported = psycles::adapter::load_blender_scene_bundle(fixture.path);
    require(imported.ok(), "external zero-BSDF fixture failed to import");
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    backend::LuisaPathTracerBackend compiler{luisa::compute::Device{device.impl_shared()}};
    auto compiled = compiler.compile_scene(*imported.scene);
    for (const auto &d : compiled.diagnostics) { std::cerr << d.message << '\n'; }
    require(compiled.ok(), "external zero-BSDF fixture did not compile");
    const auto &native = dynamic_cast<const backend::detail::LuisaCompiledScene &>(
        *compiled.scene).data();
    require(native->native_cycles_svm_surface && native->cycles_svm,
            "Cycles SVM must be the default without an environment opt-in");
    // A stale shell setting must not resurrect the retired surface route.
#if defined(_WIN32)
    _putenv_s("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "0");
#else
    setenv("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "0", 1);
#endif
    const auto disabled = compiler.compile_scene(*imported.scene);
    require(disabled.ok(), "retired opt-out broke scene compilation");
    require(dynamic_cast<const backend::detail::LuisaCompiledScene &>(
                *disabled.scene).data()->native_cycles_svm_surface,
            "retired opt-out selected the legacy evaluator");
    // Renderer-authored scenes have no Blender indices. Reserve a sparse
    // explicit index as well: container order is not the native object ID.
    auto authored = *imported.scene;
    authored.cycles_object_count.reset();
    authored.cycles_background_object_index.reset();
    for (auto &[id, instance] : authored.instances) {
      static_cast<void>(id);
      instance.cycles_object_index.reset();
    }
    authored.instances.begin()->second.cycles_object_index = 8u;
    for (auto &[id, light] : authored.lights) {
      static_cast<void>(id);
      light.cycles_object_index.reset();
    }
    const auto assigned = compiler.compile_scene(authored);
    require(assigned.ok(), "authored native identity fixture did not compile");
    const auto &assigned_data = dynamic_cast<const backend::detail::LuisaCompiledScene &>(
        *assigned.scene).data();
    std::vector<backend::detail::InstanceGpu> instances(authored.instances.size());
    auto inspect = device.create_stream();
    inspect << assigned_data->instance_buffer.copy_to(instances.data())
            << luisa::compute::synchronize();
    auto resource = std::size_t{0u};
    for (const auto &[id, instance] : authored.instances) {
      static_cast<void>(instance);
      require(instances[resource++].cycles_object_index ==
                  assigned_data->cycles_svm->object_identities.instance_indices.at(id),
              "instance resource diverged from native KernelObject identity");
    }
    bool passed = true;
    for (auto scheduler : {backend::LuisaPathScheduler::megakernel,
                           backend::LuisaPathScheduler::wavefront_staged}) {
      for (unsigned x : {2u, 6u, 10u}) {
        passed = render(device, *compiled.scene, imported, scheduler, x) && passed;
      }
    }
    if (!passed) { return 1; }
    std::cout << "Cycles zero/finite/single-channel BSDF termination passed both path schedulers\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
