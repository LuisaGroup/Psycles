#include "cycles_svm_volume_stack_fixture.h"
#include "path_tracer_cycles_svm_volume.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace f = psycles::test_support::volume_stack_fixture;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace abi = psycles::compiler::cycles_svm;
constexpr unsigned float_count = 32u + 9u + 16u * 8u + 8u * 8u;

template<typename T> auto upload(Device &d, Stream &s, const T &values) {
  auto b = d.create_buffer<typename T::value_type>(values.size());
  s << b.copy_from(values.data()) << synchronize();
  return b;
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &r = *scene->cycles_svm;
  r.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  r.objects = std::make_unique<CyclesSvmObjectRuntime>();
  const auto dummy = [&]<typename T>(Buffer<T> &buffer) {
    buffer = upload(device, stream, std::array<T, 1u>{});
  };
  auto &g = *r.geometry;
  dummy(g.attribute_map_buffer); dummy(g.attribute_float_buffer);
  dummy(g.attribute_float2_buffer); dummy(g.attribute_float3_buffer);
  dummy(g.attribute_float4_buffer); dummy(g.attribute_uchar4_buffer);
  dummy(g.attribute_normal_buffer); dummy(g.triangle_vertex_buffer);
  dummy(g.triangle_index_buffer); dummy(g.triangle_shader_buffer);
  dummy(g.curve_key_buffer); dummy(g.curve_buffer); dummy(g.point_buffer);
  std::array<abi::KernelObject, 4u> objects{};
  std::array<unsigned, 4u> object_flags{};
  for (unsigned o = 0; o < 4u; ++o) {
    auto &v = objects[o];
    v.volume_density = f::densities[o];
    v.visibility = o == 2u ? abi::PATH_RAY_VISIBILITY_CAMERA : abi::PATH_RAY_VISIBILITY_ALL;
    v.tfm = {{2, 0, 0, 0.5f}, {0, -4, 0, -0.25f}, {0, 0, 0.5f, 1}};
    v.itfm = {{0.5f, 0, 0, -0.25f}, {0, -0.25f, 0, -0.0625f}, {0, 0, 2, -2}};
    object_flags[o] = o % 2u ? abi::SD_OBJECT_NEGATIVE_SCALE : 0u;
  }
  r.objects->object_buffer = upload(device, stream, objects);
  r.objects->object_flag_buffer = upload(device, stream, object_flags);
  std::array<abi::KernelShader, f::shader_count> shaders{};
  for (unsigned s = 0; s < shaders.size(); ++s) { shaders[s].flags = f::shader_flags(s); }
  r.kernel_shader_buffer = upload(device, stream, shaders);
  r.compilation.table.words = f::words();
  r.compilation.table.peak_stack_usage = 4u;
  r.word_buffer = upload(device, stream, r.compilation.table.words);
  r.kernel_features = svm::kernel_feature_volume;
  for (auto node : {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_CLOSURE_SET_WEIGHT,
                    abi::NODE_CLOSURE_VOLUME, abi::NODE_VOLUME_COEFFICIENTS, abi::NODE_TEX_COORD,
                    abi::NODE_GEOMETRY, abi::NODE_LIGHT_PATH, abi::NODE_EMISSION_WEIGHT,
                    abi::NODE_CLOSURE_EMISSION}) { r.compilation.table.node_types_used[node] = true; }
  std::vector<luisa::uint2> entries;
  std::vector<luisa::uint2> states;
  for (unsigned row = 0u; row < f::stacks.size(); ++row) {
    for (unsigned e = 0; e < 3u; ++e) {
      entries.push_back(luisa::make_uint2(f::object(row, e), unsigned(f::stacks[row][e])));
    }
  }
  for (unsigned mode = 0; mode < f::state_count; ++mode) {
    states.push_back(luisa::make_uint2(f::visibility(mode), f::flag(mode)));
  }
  auto entry_buffer = upload(device, stream, entries);
  auto state_buffer = upload(device, stream, states);
  auto floats = device.create_buffer<float>(f::count * float_count);
  auto metadata = device.create_buffer<unsigned>(f::count * 8u);
  std::ifstream oracle{PSYCLES_VOLUME_STACK_ORACLE};
  bool passed = true;
  for (unsigned capacity : f::capacities) {
    r.compilation.max_closures = capacity;
    Kernel1D kernel = [scene, capacity](BufferUInt2 entries, BufferUInt2 states,
                                        BufferFloat out, BufferUInt meta) {
      const auto i = dispatch_x(), row = i / f::state_count, mode = i % f::state_count;
      const auto s = states.read(mode);
      const svm::PathState state{s.x, s.y, 3u};
      VolumeStack stack{4u};
      for (unsigned e = 0; e < 3u; ++e) {
        const auto pair = entries.read(row * 3u + e);
        const VolumeStackEntry entry{pair.x, pair.y, ~0u, 0u, ~0u,
                                      volume_sample_distance, pair.y != ~0u};
        stack.initialize_background(entry, entry.valid);
      }
      svm::ClosurePool closures{capacity};
      auto sd = setup_cycles_svm_volume_shader_data(
          make_float3(1.0f, -2.0f, 3.0f), make_float3(0.0f, 0.0f, 1.0f),
          0.25f, 0.375f, svm::object_none, &closures);
      const auto put = [&](unsigned field, Float value) { out.write(i * float_count + field, value); };
      const auto put3 = [&](unsigned field, Float3 v) {
        put(field, v.x); put(field + 1u, v.y); put(field + 2u, v.z);
      };
      put3(0, sd.P); put3(3, sd.ray_P); put3(6, sd.N); put3(9, sd.Ng); put3(12, sd.wi);
      put3(15, sd.dPdu); put3(18, sd.dPdv);
      put(21, sd.time); put(22, sd.ray_length); put(23, sd.dP); put(24, sd.dI);
      put(25, sd.u); put(26, sd.v); put(27, sd.du.dx); put(28, sd.du.dy);
      put(29, sd.dv.dx); put(30, sd.dv.dy); put(31, 0.0f);
      sd.flag = abi::SD_CACHE_MISS | abi::SD_EXTINCTION | abi::SD_EMISSION;
      sd.closure_transparent_extinction = make_float3(-9.0f);
      sd.closure_emission_background = make_float3(-7.0f);
      Var<RenderKernelParameters> params;
      params.camera_transform = params.camera_inverse_transform = make_float4x4(1.0f);
      $if(mode == 2u) { evaluate_cycles_svm_volume_stack(scene, params, stack, sd, state, true); }
      $else { evaluate_cycles_svm_volume_stack(scene, params, stack, sd, state, false); };
      const auto coefficients = cycles_svm_volume_coefficients(sd);
      put3(32, coefficients.sigma_t); put3(35, coefficients.sigma_s); put3(38, coefficients.emission);
      VolumePhaseSet phases{8u};
      phases.copy_from(closures);
      meta.write(i * 8u, sd.flag); meta.write(i * 8u + 1u, sd.object_flag);
      meta.write(i * 8u + 2u, sd.type); meta.write(i * 8u + 3u, sd.object);
      meta.write(i * 8u + 4u, sd.shader); meta.write(i * 8u + 5u, closures.count());
      meta.write(i * 8u + 6u, closures.left()); meta.write(i * 8u + 7u, phases.count());
      for (unsigned j = 0; j < 16u; ++j) {
        const auto base = 41u + j * 8u;
        for (unsigned k = 0; k < 8u; ++k) { put(base + k, 0.0f); }
        $if(j < closures.count()) {
          const auto c = closures.volume_common(j);
          put(base, c.type.cast<float>()); put3(base + 1u, c.weight);
          put(base + 4u, c.sample_weight); put3(base + 5u, closures.volume_phase_parameters(j));
        };
      }
      for (unsigned j = 0; j < 8u; ++j) {
        const auto base = 41u + 16u * 8u + j * 8u;
        const auto c = phases.entry(j);
        put(base, c.type.cast<float>()); put3(base + 1u, c.weight);
        put(base + 4u, c.sample_weight); put3(base + 5u, c.parameters);
      }
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::vector<float> actual(f::count * float_count);
    std::vector<unsigned> meta(f::count * 8u);
    stream << shader(entry_buffer, state_buffer, floats, metadata).dispatch(f::count)
           << floats.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
    for (unsigned row = 0; row < f::count; ++row) {
      char tag{}; unsigned cap{}, index{};
      if (!(oracle >> tag >> cap >> index) || tag != 'R' || cap != capacity || index != row) { return false; }
      for (unsigned field = 0; field < 8u; ++field) {
        unsigned expected{};
        if (!(oracle >> expected)) { return false; }
        if (meta[row * 8u + field] != expected) {
          std::cerr << "Volume stack " << capacity << ':' << row << " integer " << field
                    << ": " << meta[row * 8u + field] << " expected " << expected << '\n';
          passed = false;
        }
      }
      for (unsigned field = 0; field < float_count; ++field) {
        float expected{};
        if (!(oracle >> expected)) { return false; }
        const auto value = actual[row * float_count + field];
        if (!std::isfinite(value) || std::abs(value - expected) > 1.0e-6f + 1.0e-4f * std::abs(expected)) {
          std::cerr << "Volume stack " << capacity << ':' << row << " float " << field
                    << ": " << value << " expected " << expected << '\n';
          passed = false;
        }
      }
    }
  }
  oracle >> std::ws;
  return passed && oracle.eof();
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
