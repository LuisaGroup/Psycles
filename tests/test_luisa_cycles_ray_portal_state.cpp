#include "cycles_ray_portal_fixture.h"
#include "cycles_svm_surface_integrator.h"
#include "cycles_svm_surface_shader.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <psycles/luisa/cycles_closure.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace luisa::compute;
namespace f = psycles::test_support::ray_portal;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace native = svm::detail;
namespace path = psycles::luisa_backend::cycles_path_state;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace abi = psycles::compiler::cycles_svm;
constexpr auto count = unsigned(f::inputs.size());
constexpr unsigned input_float_rows = 6, input_integer_rows = 3;

class Globals final : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  Bool _transform_applied;
  UInt _triangle;
public:
  Globals(Bool applied, UInt triangle) : _transform_applied{applied}, _triangle{triangle} {}
  [[nodiscard]] svm::TriangleVertices triangle_vertices(
      Expr<unsigned>, Expr<unsigned>) const noexcept override {
    const auto z = select(0.0f, 2.0f, _transform_applied);
    svm::TriangleVertices result{make_float3(-2.0f, -2.0f, z), make_float3(3.0f, -2.0f, z),
                                make_float3(-2.0f, 3.0f, z)};
    $if(_triangle >= 2u) {
      result.v0 = make_float3(-2.0f, -2.0f, 3.0f);
      result.v1 = select(make_float3(2.0f, -2.0f, 3.0f),
                         make_float3(2.0f, 2.0f, 3.0f), _triangle == 3u);
      result.v2 = select(make_float3(2.0f, 2.0f, 3.0f),
                         make_float3(-2.0f, 2.0f, 3.0f), _triangle == 3u);
    };
    return result;
  }
};

[[nodiscard]] auto kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<luisa::uint4>,
                  Buffer<luisa::float4>, Buffer<luisa::uint4>>{
      [](BufferFloat4 inputs, BufferUInt4 controls,
         BufferFloat4 output, BufferUInt4 metadata) noexcept {
    const auto id = dispatch_id().x;
    const auto flags = controls.read(id * input_integer_rows);
    const auto transition = controls.read(id * input_integer_rows + 1u);
    const auto types = controls.read(id * input_integer_rows + 2u);
    const auto point = inputs.read(id * input_float_rows + 3u);
    const auto portal_P = inputs.read(id * input_float_rows + 4u).xyz();
    const auto portal_D = inputs.read(id * input_float_rows + 5u).xyz();
    svm::ClosurePool pool{3u};
    UInt i = 0u;
    $while(i < flags.x) {
      const auto data = inputs.read(id * input_float_rows + i);
      const auto allocated = pool.allocate(types[i], data.xyz());
      pool.set_sample_weight(allocated.index, data.w);
      pool.set_normal(allocated.index, make_float3(0.0f, 0.0f, 1.0f));
      $if(types[i] == closure::type_ray_portal) {
        pool.set_ray_portal_param(allocated.index, {.P = portal_P, .D = portal_D});
      };
      i += 1u;
    };
    const auto identity = make_float4x4(1.0f);
    svm::ShaderData sd{
        point.xyz(), make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f),
        select(unsigned(abi::PRIMITIVE_CURVE), unsigned(abi::PRIMITIVE_TRIANGLE), flags.y != 0u),
        0u, unsigned(abi::SD_BSDF) | select(0u, unsigned(abi::SD_RAY_PORTAL), flags.w != 0u),
        select(0u, unsigned(abi::SD_OBJECT_TRANSFORM_APPLIED), flags.z != 0u),
        0u, 0.0f, 0.0f, 0u, 0.0f, 1.0f, 0.375f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f), identity, identity, 0u, &pool};
    native::RayPortalState state{
        .P = make_float3(7.0f, 8.0f, 9.0f), .D = make_float3(0.0f, 0.0f, 1.0f),
        .tmin = 0.125f, .tmax = 10.0f, .dP = 0.25f,
        .throughput = make_float3(0.2f, 0.4f, 0.8f), .isect_object = 0u,
        .path = {.flag = path::flag_reflect | path::flag_transparent_background,
                 .visibility = path::visibility_diffuse, .bounce = 0u,
                 .diffuse_bounce = 1u, .glossy_bounce = 2u,
                 .transmission_bounce = 3u, .transparent_bounce = transition.y,
                 .rng_offset = 49u, .portal_bounce = transition.z}};
    const path::Limits limits{.maximum = 4u, .maximum_diffuse = 2u,
                             .maximum_glossy = 4u, .maximum_transmission = 5u,
                             .maximum_transparent = transition.x};
    native::SurfaceShaderClosurePick pick{0u, make_float3(0.125f, 0.25f, point.w)};
    $if(types.w == 0u) {
      pick = native::surface_shader_bsdf_bssrdf_pick(sd, pick.random);
    };
    UInt label = transition.w;
    $if(label != 0u) {
      state.path = path::next_surface(
          state.path, label, (sd.flag & unsigned(abi::SD_BSDF_HAS_TRANSMISSION)) != 0u,
          (sd.flag & unsigned(abi::SD_RAY_PORTAL)) != 0u, limits);
    }
    $else {
      Globals kg{flags.z != 0u, flags.y};
      // Exact immutable original KernelObject::itfm, in Luisa column layout.
      const auto inverse = make_float4x4(
          make_float4(0.5f, 0.0f, 0.0f, 0.0f),
          make_float4(0.0f, 2.0f / 3.0f, 0.0f, 0.0f),
          make_float4(0.0f, 0.0f, 1.0f, 0.0f),
          make_float4(-0.125f, 1.0f / 3.0f, -2.0f, 1.0f));
      label = native::integrate_surface_ray_portal(kg, sd, pick.index, inverse, state, limits);
    };
    const auto base = id * f::float_rows;
    output.write(base, make_float4(state.P, state.tmin));
    output.write(base + 1u, make_float4(state.D, state.tmax));
    output.write(base + 2u, make_float4(state.throughput, state.dP));
    // These original fields are outside the portal operation's write set.
    output.write(base + 3u, make_float4(0.0625f, 0.33f, 0.125f, pick.random.z));
    output.write(base + 4u, make_float4(1.0f, 0.0f, 0.0f, 0.875f));
    output.write(base + 5u, make_float4(0.25f, 0.5f, 0.75f, pool.common(pick.index).sample_weight));
    output.write(base + 6u, make_float4(0.75f, 0.5f, 0.25f, 0.0f));
    const auto meta = id * f::integer_rows;
    metadata.write(meta, make_uint4(label, pick.index, state.path.flag, state.path.visibility));
    metadata.write(meta + 1u, make_uint4(state.path.bounce, state.path.diffuse_bounce,
        state.path.glossy_bounce, state.path.transmission_bounce));
    metadata.write(meta + 2u, make_uint4(state.path.transparent_bounce, state.path.portal_bounce,
        state.path.rng_offset, state.isect_object));
    metadata.write(meta + 3u, make_uint4(17u, unsigned(abi::PRIMITIVE_TRIANGLE),
        select(0u, 1u, label != 0u), sd.flag));
  }};
}

bool run(char **argv) {
  std::array<luisa::float4, count * input_float_rows> input{};
  std::array<luisa::uint4, count * input_integer_rows> control{};
  constexpr std::array<unsigned, 5> types{closure::type_ray_portal,
      closure::type_diffuse, closure::type_transparent, closure::type_bssrdf_burley,
      unsigned(abi::CLOSURE_HOLDOUT_ID)};
  for (unsigned i = 0; i < count; ++i) {
    const auto &in = f::inputs[i];
    for (unsigned c = 0; c < 3; ++c) {
      const auto &sc = in.closures[c];
      input[i * input_float_rows + c] = {sc.weight[0], sc.weight[1], sc.weight[2], sc.sample_weight};
      control[i * input_integer_rows + 2][c] = types.at(unsigned(sc.kind));
    }
    input[i * input_float_rows + 3] = {in.point[0], in.point[1], in.point[2], in.random_z};
    input[i * input_float_rows + 4] = {in.portal_position[0], in.portal_position[1], in.portal_position[2], 0};
    input[i * input_float_rows + 5] = {in.portal_direction[0], in.portal_direction[1], in.portal_direction[2], 0};
    control[i * input_integer_rows] = {in.count, in.triangle, in.transform_applied, in.portal_shader};
    control[i * input_integer_rows + 1] = {in.transparent_limit, in.transparent_bounce, in.portal_bounce, in.transition_label};
    control[i * input_integer_rows + 2].w = in.force_first_closure;
  }
  Context context{argv[0]};
  auto device = context.create_device(argv[1]);
  auto stream = device.create_stream();
  auto inputs = device.create_buffer<luisa::float4>(input.size());
  auto controls = device.create_buffer<luisa::uint4>(control.size());
  auto output = device.create_buffer<luisa::float4>(count * f::float_rows);
  auto metadata = device.create_buffer<luisa::uint4>(count * f::integer_rows);
  const auto shader = device.compile(kernel(), ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, count * f::float_rows> actual{};
  std::array<luisa::uint4, count * f::integer_rows> meta{};
  stream << inputs.copy_from(input.data()) << controls.copy_from(control.data())
         << shader(inputs, controls, output, metadata).dispatch(count)
         << output.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
  std::ifstream oracle{PSYCLES_RAY_PORTAL_STATE_ORACLE};
  if (!oracle) throw std::runtime_error{"missing original Cycles portal state"};
  unsigned mismatches = 0;
  for (unsigned i = 0; i < count; ++i) {
    unsigned index;
    if (!(oracle >> index) || index != i) throw std::runtime_error{"invalid oracle index"};
    for (unsigned j = 0; j < 4 * f::float_rows; ++j) {
      float expected;
      if (!(oracle >> expected)) throw std::runtime_error{"truncated float oracle"};
      const auto value = actual[i * f::float_rows + j / 4][j % 4];
      if (!std::isfinite(value) || std::abs(value - expected) > 2e-6f + 2e-6f * std::abs(expected)) {
        std::cerr << "case " << i << " float " << j << ": " << value << " != " << expected << '\n';
        ++mismatches;
      }
    }
    for (unsigned j = 0; j < 4 * f::integer_rows; ++j) {
      unsigned expected;
      if (!(oracle >> expected)) throw std::runtime_error{"truncated integer oracle"};
      const auto value = meta[i * f::integer_rows + j / 4][j % 4];
      if (value != expected) {
        std::cerr << "case " << i << " uint " << j << ": " << value << " != " << expected << '\n';
        ++mismatches;
      }
    }
  }
  std::string extra;
  if (oracle >> extra) throw std::runtime_error{"unexpected trailing oracle data"};
  std::cout << "original Cycles portal/path state: " << count << " cases, " << mismatches << " mismatches\n";
  return mismatches == 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try { return run(argv) ? 0 : 1; }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
