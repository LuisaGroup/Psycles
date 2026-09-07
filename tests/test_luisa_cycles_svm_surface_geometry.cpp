#include "cycles_svm_native_shadow_fixture.h"
#include "path_kernel_surface_primitive.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <span>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::native_shadow_inputs;
namespace abi = psycles::compiler::cycles_svm;
constexpr auto count = unsigned(native_shadow_inputs.size());

template <typename T, std::size_t N>
auto upload(Device &device, Stream &stream, const std::array<T, N> &data) {
  auto result = device.create_buffer<T>(N);
  stream << result.copy_from(data.data()) << synchronize();
  return result;
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->native_cycles_svm_surface = true;
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &runtime = *scene->cycles_svm;
  runtime.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  runtime.objects = std::make_unique<CyclesSvmObjectRuntime>();
  std::array<abi::KernelObject, count> objects{};
  std::array<unsigned, count> flags{}, shaders{};
  std::array<abi::KernelShader, count> shader_records{};
  std::array<InstanceGpu, count> instances{};
  std::array<GeometryGpu, count> geometries{};
  std::array<MaterialBindingGpu, count> materials{};
  std::array<abi::packed_uint3, count> indices{};
  std::array<abi::packed_float3, count * 3u> vertices{};
  std::array<abi::packed_normal, count * 3u> normals{};
  std::array<luisa::float4, count * 3u> rays{};
  for (auto i = 0u; i < count; ++i) {
    const auto &c = native_shadow_inputs[i];
    const auto &g = c.geometry;
    objects[i].tfm = g.tfm;
    objects[i].itfm = g.itfm;
    objects[i].position_offset = int(i * 3u);
    objects[i].normal_offset =
        (g.object_flags & abi::SD_OBJECT_HAS_CORNER_NORMALS) ? 0 : int(i * 3u);
    objects[i].primitive_type = abi::PRIMITIVE_TRIANGLE;
    flags[i] = g.object_flags;
    shaders[i] = g.shader;
    shader_records[i].flags = c.shader_flags;
    instances[count - 1u - i].cycles_object_index = i;
    instances[count - 1u - i].cycles_primitive_offset = i;
    instances[count - 1u - i].geometry_index = i;
    geometries[i].cycles_primitive_offset = i;
    geometries[i].material_count = 1u;
    geometries[i].material_offset = i;
    materials[i].cycles_shader_index = i;
    indices[i] = {0u, 1u, 2u};
    for (auto j = 0u; j < 3u; ++j) {
      vertices[i * 3u + j] = g.vertices[j];
      normals[i * 3u + j] = abi::pack_geometry_normal(g.normals[j]);
    }
    rays[i * 3u] = luisa::make_float4(g.ray_P.x, g.ray_P.y, g.ray_P.z, g.time);
    rays[i * 3u + 1u] = luisa::make_float4(g.ray_D.x, g.ray_D.y, g.ray_D.z, g.distance);
    rays[i * 3u + 2u] = luisa::make_float4(g.dP, g.dD, c.u, c.v);
  }
  runtime.objects->object_buffer = upload(device, stream, objects);
  runtime.objects->object_flag_buffer = upload(device, stream, flags);
  runtime.geometry->triangle_vertex_buffer = upload(device, stream, vertices);
  runtime.geometry->triangle_index_buffer = upload(device, stream, indices);
  runtime.geometry->triangle_shader_buffer = upload(device, stream, shaders);
  runtime.geometry->attribute_normal_buffer = upload(device, stream, normals);
  runtime.kernel_shader_buffer = upload(device, stream, shader_records);
  scene->instance_buffer = upload(device, stream, instances);
  scene->geometry_buffer = upload(device, stream, geometries);
  scene->geometry_material_buffer = upload(device, stream, materials);
  scene->override_material_buffer = upload(device, stream, materials);
  auto triangle = upload(device, stream, std::array{Triangle{0u, 1u, 2u}});
  auto slot = upload(device, stream, std::array{0u});
  auto smooth = upload(device, stream, std::array{1u});
  scene->heap = device.create_bindless_array(geometry_bindless_stride);
  scene->heap.emplace_on_update(0u, triangle);
  scene->heap.emplace_on_update(4u, slot);
  scene->heap.emplace_on_update(8u, smooth);
  stream << scene->heap.update() << synchronize();
  // Deliberately no Accel, legacy normal, UV, generated or tangent buffers:
  // main surface setup must use precomputed KernelObject transforms and the
  // same packed Cycles attributes as shadow setup, not the legacy resolver.
  const auto geometry = make_surface_primitive_geometry_component({true, false});
  const SafeNormalizeCallable unused{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
  Kernel1D kernel = [scene, geometry, unused](BufferFloat4 input,
                                              BufferFloat4 output,
                                              BufferUInt4 metadata) {
    const auto i = dispatch_x();
    const auto p = input.read(i * 3u), d = input.read(i * 3u + 1u);
    const auto uv = input.read(i * 3u + 2u);
    auto ray = make_ray(p.xyz(), d.xyz(), 0.0f, 100.0f);
    Var<CommittedHit> hit;
    hit.inst = count - 1u - i;
    hit.prim = 0u;
    hit.bary = uv.zw();
    hit.hit_type = 1u;
    hit.committed_ray_t = d.w;
    Var<RenderKernelParameters> params;
    params.camera_transform = params.camera_inverse_transform = make_float4x4(1.0f);
    const auto surface = geometry->emit(scene, hit, ray, uv.x, uv.y, p.w, params, unused);
    const auto &point = surface.point;
    output.write(i * 7u, make_float4(point.position, point.time));
    output.write(i * 7u + 1u, make_float4(point.shading_normal, point.ray_length));
    output.write(i * 7u + 2u, make_float4(point.geometric_normal, surface.differential_radius));
    output.write(i * 7u + 3u, make_float4(point.incoming, uv.y));
    output.write(i * 7u + 4u, make_float4(point.dpdu, point.barycentric.x));
    output.write(i * 7u + 5u, make_float4(point.dpdv, point.barycentric.y));
    output.write(i * 7u + 6u, make_float4(point.barycentric_dx.x, point.barycentric_dy.x,
                                          point.barycentric_dx.y, point.barycentric_dy.y));
    metadata.write(i, make_uint4(surface.cycles_surface_shader,
                                 surface.cycles_object_index,
                                 surface.cycles_primitive_index,
                                 point.back_facing.cast<unsigned>()));
  };
  auto shader = device.compile(kernel);
  auto input = upload(device, stream, rays);
  auto output = device.create_buffer<luisa::float4>(count * 7u);
  auto metadata = device.create_buffer<luisa::uint4>(count);
  std::array<luisa::float4, count * 7u> actual{};
  std::array<luisa::uint4, count> meta{};
  stream << shader(input, output, metadata).dispatch(count)
         << output.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
  std::ifstream oracle{PSYCLES_NATIVE_SHADOW_ORACLE};
  if (!oracle) {
    std::cerr << "Cannot read Cycles oracle: " << PSYCLES_NATIVE_SHADOW_ORACLE << '\n';
    return false;
  }
  bool passed = true;
  for (auto i = 0u; i < count; ++i) {
    char kind{}; unsigned row{};
    if (!(oracle >> kind >> row) || kind != 'S' || row != i) { return false; }
    for (auto j = 0u; j < 32u; ++j) {
      float expected{};
      if (!(oracle >> expected)) { return false; }
      if (j >= 28u) { continue; } // Shadow transparency is not main geometry.
      const auto value = actual[i * 7u + j / 4u][j % 4u];
      if (!std::isfinite(value) || std::abs(value - expected) >
                                     5e-5f * std::max(std::abs(expected), 1e-5f)) {
        std::cerr << "Main surface " << i << " float " << j << ": " << value
                  << " expected " << expected << '\n';
        passed = false;
      }
    }
    std::array<unsigned, 8u> expected{};
    for (auto &v : expected) { if (!(oracle >> v)) { return false; } }
    const std::array expected_meta{expected[0], expected[4], expected[5],
        unsigned((expected[1] & abi::SD_BACKFACING) != 0u)};
    for (auto j = 0u; j < 4u; ++j) {
      if (meta[i][j] != expected_meta[j]) {
        std::cerr << "Main surface " << i << " uint " << j << ": " << meta[i][j]
                  << " expected " << expected_meta[j] << '\n';
        passed = false;
      }
    }
  }
  if (passed) { std::cout << "Native main surface: 25 original Cycles HIP geometry cases passed\n"; }
  return passed;
}
} // namespace
int main(int argc, char **argv) { return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1; }
