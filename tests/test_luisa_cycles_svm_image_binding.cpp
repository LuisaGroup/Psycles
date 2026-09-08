#include "cycles_svm_image_sampling.h"
#include "path_tracer_cycles_svm_scene.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/path_tracer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
constexpr ImageId source_image{37u};
constexpr std::array interpolations{"Closest", "Linear", "Cubic", "Smart"};
constexpr std::array extensions{"REPEAT", "EXTEND", "CLIP", "MIRROR"};
constexpr std::array oracle_interpolations{
    abi::ImageInterpolation::closest, abi::ImageInterpolation::linear,
    abi::ImageInterpolation::cubic, abi::ImageInterpolation::smart};

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

SceneSnapshot snapshot() {
  SceneSnapshot result;
  // Input pixels, not reference shading results: a two-texel binary PPM.
  constexpr std::string_view header{"P6\n2 1\n255\n"};
  ImageDesc image{.name = "shared.ppm", .width = 2u, .height = 1u};
  image.encoded_data.assign(header.begin(), header.end());
  image.encoded_data.insert(image.encoded_data.end(), {255u, 0u, 0u, 0u, 255u, 0u});
  result.images.emplace(source_image, std::move(image));
  TriangleMeshDesc mesh;
  mesh.positions = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  for (const auto *interpolation : interpolations) {
    for (const auto *extension : extensions) {
      ShaderGraph graph;
      const auto texture = graph.add_node(node_type::image_texture, "Image");
      const auto emission = graph.add_node(node_type::emission, "Emission");
      require(graph.set_property(texture, "Image", SocketValue::unsigned_integer(source_image.value)) &&
                  graph.set_property(texture, "Interpolation", SocketValue::string(interpolation)) &&
                  graph.set_property(texture, "Extension", SocketValue::string(extension)) &&
                  graph.set_property(texture, "Projection", SocketValue::string("FLAT")) &&
                  graph.set_property(texture, "ProjectionBlend", SocketValue::floating(0.0f)) &&
                  graph.set_property(texture, "ColorSpace", SocketValue::string("Non-Color")) &&
                  graph.set_property(texture, "UnassociateAlpha", SocketValue::boolean(false)) &&
                  graph.set_input(texture, "Vector", SocketValue::vector({0.375f, 0.5f, 0.0f})) &&
                  graph.connect({texture, "Color"}, emission, "Color"),
              "cannot configure image handle");
      graph.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});
      const auto index = static_cast<std::uint32_t>(mesh.material_slots.size());
      const MaterialId material{index + 1u};
      result.materials.emplace(material, MaterialDesc{
          .name = std::string{interpolation} + extension, .shader = std::move(graph)});
      mesh.material_slots.push_back(material);
      mesh.triangles.push_back({0u, 1u, 2u});
      mesh.triangle_material_slots.push_back(index);
    }
  }
  result.geometries.emplace(GeometryId{1u}, std::move(mesh));
  result.instances.emplace(InstanceId{1u}, InstanceDesc{.geometry = GeometryId{1u}});
  result.cameras.emplace(CameraId{1u}, CameraDesc{});
  result.active_camera = CameraId{1u};
  result.world_sampling = WorldSampling::none;
  return result;
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    LuisaPathTracerBackend renderer{Device{device.impl_shared()}, {}};
    const auto compiled = renderer.compile_scene(snapshot());
    for (const auto &diagnostic : compiled.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    require(compiled.ok(), "image binding scene was rejected");
    const auto *native = dynamic_cast<const LuisaCompiledScene *>(compiled.scene.get());
    require(native != nullptr, "missing native scene");
    const auto &scene = *native->data();
    const auto &runtime = *scene.cycles_svm;
    const auto &handles = runtime.compilation.images;
    require(handles.size() == 16u && runtime.image_bindings.size() == 16u,
            "distinct ImageManager samplers did not retain their handles");
    require(scene.images.size() == 2u, "sampler aliases duplicated pixel storage");
    std::set<std::uint32_t> descriptors;
    for (auto i = std::size_t{}; i < handles.size(); ++i) {
      require(handles[i].resource_id == source_image.value && !handles[i].nishita,
              "source image identity changed");
      require(runtime.image_bindings[i].texture_slot > source_image.value &&
                  descriptors.insert(runtime.image_bindings[i].texture_slot).second,
              "native sampler descriptor aliases another sampler or the legacy slot");
      require(runtime.image_bindings[i].sampler == cycles_svm_image_sampler(
                  handles[i].interpolation, handles[i].extension),
              "immutable sampler metadata changed");
    }

    // The actual side uses the complete scene compiler/upload. The control
    // uses independent explicit native sampler arguments over the same image;
    // no CPU texture evaluator is involved. Cycles' GPU texture-object policy
    // is pinned in device/hip/device_impl.cpp and kernel/device/gpu/image.h.
    constexpr std::array coordinates{
        luisa::float2{0.375f, 0.5f}, luisa::float2{-0.125f, 0.5f},
        luisa::float2{1.125f, 0.5f}, luisa::float2{0.125f, -0.5f},
        luisa::float2{0.875f, 1.5f}};
    const auto count = static_cast<std::uint32_t>(handles.size() * coordinates.size());
    auto coordinate_buffer = device.create_buffer<luisa::float2>(coordinates.size());
    auto output_buffer = device.create_buffer<luisa::float4>(count * 2u);
    const Kernel1D<BindlessArray, Buffer<CyclesSvmImageBindingGpu>,
                   Buffer<luisa::float2>, Buffer<luisa::float4>> kernel =
        [&](BindlessVar textures, BufferVar<CyclesSvmImageBindingGpu> bindings,
            BufferFloat2 probes, BufferFloat4 output) noexcept {
      const UInt index = dispatch_id().x;
      const auto handle = index / static_cast<std::uint32_t>(coordinates.size());
      const svm::Dual2 uv{
          .val = probes.read(index % static_cast<std::uint32_t>(coordinates.size())),
          .dx = make_float2(0.0f), .dy = make_float2(0.0f)};
      output.write(2u * index, svm::detail::sample_scene_image_2d(
          textures, bindings, handle.cast<int>(), uv));
      for (auto i = std::size_t{}; i < handles.size(); ++i) {
        $if(handle == static_cast<std::uint32_t>(i)) {
          output.write(2u * index + 1u, svm::detail::sample_image_2d(
              textures, static_cast<std::uint32_t>(source_image.value), uv,
              handles[i].interpolation, handles[i].extension));
        };
      }
    };
    auto shader = device.compile(kernel, ShaderOption{
        .enable_cache = false, .enable_fast_math = true});
    std::vector<luisa::float4> outputs(count * 2u);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << coordinate_buffer.copy_from(coordinates.data())
           << shader(scene.texture_heap, *runtime.image_binding_buffer,
                     coordinate_buffer, output_buffer).dispatch(count)
           << output_buffer.copy_to(outputs.data()) << synchronize();
    std::array<luisa::float4, 80u> oracle{};
    std::ifstream oracle_file{PSYCLES_IMAGE_BINDING_ORACLE};
    for (auto i = 0u; i < oracle.size(); ++i) {
      unsigned row{};
      auto &v = oracle[i];
      require(static_cast<bool>(oracle_file >> row >> v.x >> v.y >> v.z >> v.w) && row == i,
              "invalid original Cycles image oracle");
    }
    for (auto i = std::size_t{}; i < count; ++i) {
      const auto &handle = handles[i / coordinates.size()];
      const auto interpolation = static_cast<std::size_t>(std::find(
          oracle_interpolations.begin(), oracle_interpolations.end(),
          handle.interpolation) - oracle_interpolations.begin());
      require(interpolation < oracle_interpolations.size(), "unknown original interpolation");
      const auto oracle_index = (interpolation * extensions.size() +
          static_cast<std::size_t>(handle.extension)) * coordinates.size() + i % coordinates.size();
      for (auto lane = 0u; lane < 4u; ++lane) {
        const auto actual = outputs[i * 2u][lane];
        const auto expected = outputs[i * 2u + 1u][lane];
        require(std::isfinite(actual) && std::isfinite(expected) &&
                    std::abs(actual - expected) < 2.0e-6f,
                "scene-uploaded sampler differs from explicit native sampler");
        // Cross-backend texture filtering precision is not bit-identical.
        // The original HIP oracle uses the same device/storage coordinates;
        // all backends independently retain the explicit-sampler control.
        if (backend == "hip" && std::abs(actual - oracle[oracle_index][lane]) >= 2.0e-6f) {
          std::cerr << "case=" << i << " oracle=" << oracle_index << " lane=" << lane
                    << " actual=" << actual << " control=" << expected
                    << " Cycles=" << oracle[oracle_index][lane] << '\n';
        }
        if (backend == "hip") {
          require(std::isfinite(oracle[oracle_index][lane]) &&
                      std::abs(actual - oracle[oracle_index][lane]) < 2.0e-6f,
                  "scene-uploaded sampler differs from original Cycles 5.2.1 GPU");
        }
      }
    }
    std::cout << "Shared image / 16 sampler descriptors: "
              << (backend == "hip" ? 640u : 320u) << " checks passed on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
