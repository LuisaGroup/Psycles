#include "cycles_svm_curve_setup_fixture.h"
#include "path_kernel_surface_primitive.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shadow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;
namespace abi = psycles::compiler::cycles_svm;
constexpr unsigned count = curve_setup_inputs.size();

template <typename T> auto upload(Device &device, Stream &stream, const std::vector<T> &data) {
  auto buffer = device.create_buffer<T>(data.size());
  stream << buffer.copy_from(data.data()) << synchronize();
  return buffer;
}
bool run(const char *program, const char *backend) {
  std::array<std::array<float, 28u>, count> expected{};
  std::array<std::array<unsigned, 8u>, count> expected_meta{};
  std::ifstream oracle{PSYCLES_CURVE_SETUP_ORACLE};
  for (auto i = 0u; i < count; ++i) {
    char kind{}; unsigned row{};
    if (!(oracle >> kind >> row) || kind != 'C' || row != i) { return false; }
    for (auto &v : expected[i]) { if (!(oracle >> v)) { return false; } }
    for (auto &v : expected_meta[i]) { if (!(oracle >> v)) { return false; } }
  }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->curve_geometries.emplace_back();
  auto &runtime = *scene->cycles_svm;
  runtime.objects = std::make_unique<CyclesSvmObjectRuntime>();
  runtime.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  std::vector<abi::KernelObject> objects(count);
  std::vector<unsigned> flags(count);
  std::vector<abi::KernelCurve> curves(curve_setup_primitive_base + count);
  std::vector<abi::packed_float4> keys(count * 6u);
  std::vector<abi::KernelShader> shaders(count);
  std::vector<InstanceGpu> instances(count);
  std::vector<GeometryGpu> geometries(count);
  std::vector<MaterialBindingGpu> materials(count);
  std::vector<luisa::float4> rays(count * 3u);
  std::vector<Buffer<CurveSegmentGpu>> segments;
  scene->heap = device.create_bindless_array(count * geometry_bindless_stride);
  auto material_slot = upload(device, stream, std::vector<unsigned>{0u});
  for (auto i = 0u; i < count; ++i) {
    const auto &c = curve_setup_inputs[i];
    auto &object = objects[count - 1u - i];
    object.tfm = c.tfm; object.itfm = c.itfm;
    object.primitive_type = abi::PRIMITIVE_CURVE_RIBBON;
    object.position_offset = i * 6u;
    flags[count - 1u - i] = c.object_flags;
    curves[curve_setup_primitive_base + i] = {int(i | (1u << 30u)), 2, 4, abi::PRIMITIVE_CURVE_RIBBON};
    shaders[i].flags = c.shader_flags;
    std::copy(c.keys.begin(), c.keys.end(), keys.begin() + i * 6u);
    instances[i].cycles_object_index = count - 1u - i;
    instances[i].geometry_index = i;
    geometries[i].primitive_kind = geometry_kind_curve;
    geometries[i].bindless_base = i * geometry_bindless_stride;
    geometries[i].curve_subdivision_level = c.subdivision_level;
    geometries[i].material_offset = i;
    geometries[i].material_count = 1u;
    materials[i].cycles_shader_index = i;
    CurveSegmentGpu segment{};
    segment.key_begin = 2u + c.segment;
    segment.cycles_curve_index = curve_setup_primitive_base + i;
    segments.emplace_back(upload(device, stream, std::vector{segment}));
    scene->heap.emplace_on_update(geometries[i].bindless_base, segments.back());
    scene->heap.emplace_on_update(geometries[i].bindless_base + 4u, material_slot);
    rays[i * 3u] = luisa::make_float4(c.ray_P.x, c.ray_P.y, c.ray_P.z, c.time);
    // The committed distance is input from the original Cycles intersection.
    // Native setup independently reconstructs u/v from the same ray and keys.
    rays[i * 3u + 1u] = luisa::make_float4(c.ray_D.x, c.ray_D.y, c.ray_D.z, expected[i][7u]);
    rays[i * 3u + 2u] = luisa::make_float4(c.dP, c.dD, 0.0f, 0.0f);
  }
  stream << scene->heap.update() << synchronize();
  runtime.objects->object_buffer = upload(device, stream, objects);
  runtime.objects->object_flag_buffer = upload(device, stream, flags);
  runtime.geometry->curve_buffer = upload(device, stream, curves);
  runtime.geometry->curve_key_buffer = upload(device, stream, keys);
  runtime.kernel_shader_buffer = upload(device, stream, shaders);
  scene->instance_buffer = upload(device, stream, instances);
  scene->geometry_buffer = upload(device, stream, geometries);
  scene->geometry_material_buffer = upload(device, stream, materials);
  scene->override_material_buffer = upload(device, stream, materials);
  // No acceleration structure, triangle data, legacy curve keys/attributes,
  // SurfaceProgram, parameter arrays, or legacy normal/UV buffers exist.
  auto input = upload(device, stream, rays);
  auto output = device.create_buffer<luisa::float4>(count * 7u);
  auto metadata = device.create_buffer<luisa::uint4>(count * 2u);
  bool passed = true;
  for (const auto main_surface : {false, true}) {
    const auto geometry = make_surface_primitive_geometry_component({false, true});
    const SafeNormalizeCallable unused{luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
    Kernel1D kernel = [scene, geometry, unused, main_surface](BufferFloat4 input, BufferFloat4 output, BufferUInt4 meta) {
      const auto i = dispatch_x();
      const auto P = input.read(i * 3u), D = input.read(i * 3u + 1u), diff = input.read(i * 3u + 2u);
      const auto ray = make_ray(P.xyz(), D.xyz(), 0.0f, 100.0f);
      Var<RenderKernelParameters> params;
      params.camera_transform = params.camera_inverse_transform = make_float4x4(1.0f);
      if (main_surface) {
        Var<CommittedHit> hit;
        hit.inst = i; hit.prim = 0u; hit.hit_type = 2u; hit.committed_ray_t = D.w;
        const auto surface = geometry->emit(scene, hit, ray, diff.x, diff.y, P.w, params, unused);
        const auto &p = surface.point;
        output.write(i * 7u, make_float4(p.position, p.time));
        output.write(i * 7u + 1u, make_float4(p.shading_normal, p.ray_length));
        output.write(i * 7u + 2u, make_float4(p.geometric_normal, surface.differential_radius));
        output.write(i * 7u + 3u, make_float4(p.incoming, diff.y));
        output.write(i * 7u + 4u, make_float4(p.dpdu, p.barycentric.x));
        output.write(i * 7u + 5u, make_float4(p.dpdv, p.barycentric.y));
        output.write(i * 7u + 6u, make_float4(p.barycentric_dx.x, p.barycentric_dy.x, p.barycentric_dx.y, p.barycentric_dy.y));
        const auto flag = select(0u, unsigned(abi::SD_BACKFACING), p.back_facing) |
                          select(0u, unsigned(abi::SD_USE_BUMP_MAP_CORRECTION), p.use_bump_map_correction);
        meta.write(i * 2u, make_uint4(surface.cycles_surface_shader, flag,
            scene->cycles_svm->objects->object_flag_buffer->read(surface.cycles_object_index), surface.cycles_primitive_type));
        meta.write(i * 2u + 1u, make_uint4(surface.cycles_object_index, surface.cycles_primitive_index, p.is_curve.cast<unsigned>(), 0u));
      } else {
        const PathCyclesSvmKernelGlobals kg{scene, params, scene->camera.projection, true, true};
        const auto setup = setup_cycles_svm_ray_shader_data(scene, kg, ray, i, 0u, make_float2(0.0f), D.w, diff.x, diff.y, P.w, 0u, params);
        const auto &sd = setup.shader_data;
        output.write(i * 7u, make_float4(sd.P, sd.time));
        output.write(i * 7u + 1u, make_float4(sd.N, sd.ray_length));
        output.write(i * 7u + 2u, make_float4(sd.Ng, sd.dP));
        output.write(i * 7u + 3u, make_float4(sd.wi, sd.dI));
        output.write(i * 7u + 4u, make_float4(sd.dPdu, sd.u));
        output.write(i * 7u + 5u, make_float4(sd.dPdv, sd.v));
        output.write(i * 7u + 6u, make_float4(sd.du.dx, sd.du.dy, sd.dv.dx, sd.dv.dy));
        meta.write(i * 2u, make_uint4(sd.shader, sd.flag, sd.object_flag, sd.type));
        meta.write(i * 2u + 1u, make_uint4(sd.object, sd.prim, 1u, 0u));
      }
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::array<luisa::float4, count * 7u> actual{};
    std::array<luisa::uint4, count * 2u> meta{};
    stream << shader(input, output, metadata).dispatch(count)
           << output.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
    for (auto i = 0u; i < count; ++i) {
      for (auto j = 0u; j < 28u; ++j) {
        const auto v = actual[i * 7u + j / 4u][j % 4u];
        const auto e = expected[i][j];
        if (!std::isfinite(v) || std::abs(v - e) > 1e-6f + 8e-5f * std::abs(e)) {
          std::cerr << "Curve " << i << " main=" << main_surface << " float " << j << ": " << v << " expected " << e << '\n';
          passed = false;
        }
      }
      for (auto j = 0u; j < 8u; ++j) {
        if (meta[i * 2u + j / 4u][j % 4u] != expected_meta[i][j]) {
          std::cerr << "Curve " << i << " main=" << main_surface << " metadata " << j << " mismatch\n";
          passed = false;
        }
      }
    }
  }
  if (passed) { std::cout << "Native curve ShaderData: 24 original Cycles HIP cases, main and shadow setup passed\n"; }
  return passed;
}
} // namespace
int main(int argc, char **argv) { return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1; }
