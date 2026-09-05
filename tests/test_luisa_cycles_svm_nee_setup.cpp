#include "cycles_svm_nee_setup_fixture.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_light.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <type_traits>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::nee_setup_cases;
using psycles::test_support::NeeSetupKind;
namespace abi = psycles::compiler::cycles_svm;
constexpr auto case_count = static_cast<unsigned>(nee_setup_cases.size());

template <typename T, std::size_t Extent>
auto upload(Device &device, Stream &stream, std::span<T, Extent> values) {
  auto buffer = device.create_buffer<std::remove_const_t<T>>(values.size());
  stream << buffer.copy_from(values.data()) << synchronize();
  return buffer;
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->cycles_svm->geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  scene->cycles_svm->objects = std::make_unique<CyclesSvmObjectRuntime>();
  scene->light_count = case_count;
  scene->emissive_triangle_count = case_count;
  scene->cycles_background_shader_id = 0u;
  scene->cycles_background_object_index = case_count;
  scene->background_map_width = 512u;
  scene->background_map_height = 256u;
  scene->background_map_weight = 1.0f;

  std::array<abi::KernelObject, case_count> objects{};
  std::array<unsigned, case_count> flags{}, shaders{};
  std::array<abi::packed_uint3, case_count> indices{};
  std::array<abi::packed_float3, case_count * 3u> vertices{};
  std::array<abi::packed_normal, case_count * 3u> normals{};
  std::array<LightGpu, case_count> lights{};
  std::array<luisa::float4, case_count * 3u> rays{};
  std::array<luisa::uint4, case_count> identities{};
  for (auto i = 0u; i < case_count; ++i) {
    const auto &input = nee_setup_cases[i];
    auto &object = objects[i];
    object.tfm = input.tfm;
    object.itfm = input.itfm;
    object.position_offset = static_cast<int>(i * 3u);
    object.normal_offset =
        (input.object_flags & abi::SD_OBJECT_HAS_CORNER_NORMALS)
            ? 0
            : static_cast<int>(i * 3u);
    object.primitive_type = input.kind == NeeSetupKind::triangle
                                ? abi::PRIMITIVE_TRIANGLE
                                : abi::PRIMITIVE_LAMP;
    flags[i] = input.object_flags;
    shaders[i] = input.shader;
    indices[i] = {0u, 1u, 2u};
    for (auto v = 0u; v < 3u; ++v) {
      vertices[i * 3u + v] = input.vertices[v];
      normals[i * 3u + v] = abi::pack_geometry_normal(input.normals[v]);
    }
    auto &light = lights[i];
    light.type = static_cast<unsigned>(
        input.kind == NeeSetupKind::spot   ? psycles::contract::LightType::spot
        : input.kind == NeeSetupKind::area ? psycles::contract::LightType::area
        : input.kind == NeeSetupKind::sun
            ? psycles::contract::LightType::distant
            : psycles::contract::LightType::point);
    light.position =
        luisa::make_float3(input.tfm.x.w, input.tfm.y.w, input.tfm.z.w);
    light.axis_x = luisa::make_float3(1.0f, 0.0f, 0.0f);
    light.axis_y = luisa::make_float3(0.0f, 1.0f, 0.0f);
    light.axis_z = luisa::make_float3(0.0f, 0.0f, 1.0f);
    light.axis_scale =
        luisa::make_float3(input.tfm.x.x, input.tfm.y.y, input.tfm.z.z);
    light.radius = input.radius;
    light.size_u = input.size_u;
    light.size_v = input.size_v;
    light.spot_angle = input.angle;
    light.angle = input.angle;
    light.flags = input.sphere ? light_flag_sphere : 0u;
    light.cycles_shader_id = 0u;
    rays[i * 3u] = luisa::make_float4(input.ray_P.x, input.ray_P.y,
                                      input.ray_P.z, input.time);
    rays[i * 3u + 1u] = luisa::make_float4(input.ray_D.x, input.ray_D.y,
                                           input.ray_D.z, input.distance);
    rays[i * 3u + 2u] = luisa::make_float4(input.dP, input.dD, 0.0f, 0.0f);
    identities[i] = luisa::make_uint4(
        input.sample, input.rng_hash, input.rng_offset,
        input.kind == NeeSetupKind::background ? case_count : i);
  }
  auto &runtime = *scene->cycles_svm;
  runtime.objects->object_buffer = upload(device, stream, std::span{objects});
  runtime.objects->object_flag_buffer =
      upload(device, stream, std::span{flags});
  runtime.geometry->triangle_vertex_buffer =
      upload(device, stream, std::span{vertices});
  runtime.geometry->triangle_index_buffer =
      upload(device, stream, std::span{indices});
  runtime.geometry->triangle_shader_buffer =
      upload(device, stream, std::span{shaders});
  runtime.geometry->attribute_normal_buffer =
      upload(device, stream, std::span{normals});
  const std::array<abi::KernelShader, 1u> shader_records{};
  runtime.kernel_shader_buffer =
      upload(device, stream, std::span{shader_records});
  scene->light_buffer = upload(device, stream, std::span{lights});
  auto ray_buffer = upload(device, stream, std::span{rays});
  auto identity_buffer = upload(device, stream, std::span{identities});

  Kernel1D<Buffer<luisa::float4>, Buffer<luisa::uint4>, Buffer<luisa::float4>,
           Buffer<luisa::uint4>>
      kernel = [scene](BufferFloat4 rays, BufferUInt4 identities,
                       BufferFloat4 out, BufferUInt4 meta) noexcept {
        const auto i = dispatch_id().x;
        const auto p = rays.read(i * 3u);
        const auto d = rays.read(i * 3u + 1u);
        const auto differential = rays.read(i * 3u + 2u);
        const auto id = identities.read(i);
        Var<DirectLightTaskCall> task;
        task.ray_origin = p.xyz();
        task.ray_time = p.w;
        task.ray_direction = d.xyz();
        task.ray_maximum = d.w;
        task.ray_dP = differential.x;
        task.ray_dD = differential.y;
        task.sample_index = id.x;
        task.rng_hash = id.y;
        task.rng_offset = id.z;
        task.light_object = id.w;
        task.light_primitive = i;
        Var<RenderKernelParameters> parameters;
        parameters.camera_transform = make_float4x4(1.0f);
        parameters.camera_inverse_transform = make_float4x4(1.0f);
        const PathCyclesSvmKernelGlobals kg{
            scene, parameters, psycles::contract::CameraProjection{}, true,
            true};
        const auto light =
            setup_cycles_svm_light_shader_data(scene, kg, task, parameters);
        const auto &sd = light.shader_data;
        out.write(i * 8u, make_float4(sd.P, sd.time));
        out.write(i * 8u + 1u, make_float4(sd.N, sd.ray_length));
        out.write(i * 8u + 2u, make_float4(sd.Ng, sd.dP));
        out.write(i * 8u + 3u, make_float4(sd.wi, sd.dI));
        out.write(i * 8u + 4u, make_float4(sd.dPdu, sd.u));
        out.write(i * 8u + 5u, make_float4(sd.dPdv, sd.v));
        out.write(i * 8u + 6u,
                  make_float4(sd.du.dx, sd.du.dy, sd.dv.dx, sd.dv.dy));
        out.write(i * 8u + 7u, make_float4(select(make_float3(0.0f), sd.ray_P,
                                                  light.background),
                                           light.background.cast<float>()));
        meta.write(i * 2u,
                   make_uint4(sd.shader, sd.flag, sd.object_flag, sd.type));
        meta.write(i * 2u + 1u,
                   make_uint4(sd.object, sd.prim, sd.lcg_state, 0u));
      };
  auto shader = device.compile(kernel);
  auto output = device.create_buffer<luisa::float4>(case_count * 8u);
  auto metadata = device.create_buffer<luisa::uint4>(case_count * 2u);
  std::array<luisa::float4, case_count * 8u> values{};
  std::array<luisa::uint4, case_count * 2u> integer_values{};
  stream << shader(ray_buffer, identity_buffer, output, metadata)
                .dispatch(case_count)
         << output.copy_to(values.data())
         << metadata.copy_to(integer_values.data()) << synchronize();
  std::ifstream oracle{PSYCLES_NEE_SETUP_ORACLE};
  if (!oracle) {
    std::cerr << "Missing Cycles NEE setup oracle\n";
    return false;
  }
  bool passed = true;
  for (auto i = 0u; i < case_count; ++i) {
    unsigned scenario{};
    oracle >> scenario;
    passed &= scenario == i;
    for (auto j = 0u; j < 32u; ++j) {
      float expected{};
      oracle >> expected;
      const auto actual = values[i * 8u + j / 4u][j % 4u];
      // UVs have a unit coordinate scale, including barycentric subtraction
      // near zero. Native Vulkan acos/atan2 need not match HIP's libm ULPs:
      // acos(0.8f) alone explains the point-sphere UV discrepancy. Use the
      // same 5e-5 coordinate budget for both UV lanes in every scenario and
      // backend; keep the stricter relative checks on geometric state.
      const auto scale_floor = (j == 19u || j == 23u) ? 1.0f : 1.0e-5f;
      if (!std::isfinite(actual) ||
          std::abs(actual - expected) >
              5.0e-5f * std::max(std::abs(expected), scale_floor)) {
        std::cerr << "NEE setup case=" << i << " float=" << j
                  << " actual=" << actual << " expected=" << expected << '\n';
        passed = false;
      }
    }
    for (auto j = 0u; j < 8u; ++j) {
      unsigned expected{};
      oracle >> expected;
      const auto actual = integer_values[i * 2u + j / 4u][j % 4u];
      if (actual != expected) {
        std::cerr << "NEE setup case=" << i << " uint=" << j
                  << " actual=" << actual << " expected=" << expected << '\n';
        passed = false;
      }
    }
    if (!oracle) {
      std::cerr << "Malformed NEE setup oracle\n";
      return false;
    }
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
