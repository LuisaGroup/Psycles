#include "cycles_light_endpoint_fixture.h"
#include "cycles_light_visibility.h"
#include "path_kernel_builder.h"

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
namespace f = psycles::test_support::light_endpoint;
namespace path = psycles::luisa_backend::cycles_path_state;

template <typename Function> Function unbound() {
  return Function{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

PathKernelConfig make_config(std::shared_ptr<LuisaSceneData> scene) {
  return {.scene = std::move(scene),
          .light_transport = {unbound<SafeNormalizeCallable>(),
                              unbound<ForwardLightWeightCallable>(),
                              unbound<NeeLightWeightCallable>(),
                              unbound<ClampLightContributionCallable>(),
                              unbound<LightSampleRouletteCallable>(),
                              unbound<LightComponentRatioCallable>(),
                              unbound<SplitScatteredLightCallable>()},
          .light_distribution_sample =
              unbound<LightDistributionSampleCallable>(),
          .light_tree = {unbound<LightTreeSampleCallable>(),
                         unbound<LightTreeSampleCallable>(),
                         unbound<LightTreePdfCallable>(),
                         unbound<LightTreePdfCallable>(),
                         unbound<LightTreeForwardPdfCallable>(),
                         unbound<LightTreeTriangleEmitterCallable>()},
          .surfaces = {nullptr, unbound<SurfacePreparationCallable>(),
                       unbound<SurfaceEvaluateLightCallable>(),
                       unbound<SurfaceConstantEmissionCallable>(),
                       unbound<SurfaceEmissionCallable>(),
                       unbound<SurfaceSampleCallable>(),
                       unbound<SurfaceClosureTraceCallable>(),
                       unbound<SurfaceSampleTraceCallable>(),
                       unbound<SurfaceBssrdfNormalCallable>()},
          .shade_shadow_surface = unbound<EvaluateShadowSurfaceCallable>(),
          .trace_shadow = unbound<TraceShadowCallable>()};
}

bool run(const char *program, const char *backend) {
  std::array<luisa::float4, f::count> expected{};
  std::array<unsigned, 64u> expected_visibility{};
  std::ifstream oracle{PSYCLES_LIGHT_ENDPOINT_ORACLE};
  for (unsigned i = 0u; i < f::count; ++i) {
    unsigned index{};
    auto &v = expected[i];
    if (!(oracle >> index >> v.x >> v.y >> v.z >> v.w) || index != i) {
      std::cerr << "Incomplete Cycles GPU oracle at row " << i << '\n';
      return false;
    }
  }
  for (unsigned i = 0u; i < expected_visibility.size(); ++i) {
    char kind{};
    unsigned index{};
    if (!(oracle >> kind >> index >> expected_visibility[i]) || kind != 'V' ||
        index != i) {
      std::cerr << "Incomplete Cycles GPU visibility oracle at row " << i
                << '\n';
      return false;
    }
  }
  std::string extra;
  if (oracle >> extra) {
    return false;
  }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->light_count = 1u;
  scene->light_buffer = device.create_buffer<LightGpu>(1u);
  auto config = make_config(scene);
  const auto closest = make_closest_event_stage(true, {}, false);
  auto records = device.create_buffer<luisa::float4>(f::count);
  auto unused_float = device.create_buffer<float>(1u);
  auto unused_uint = device.create_buffer<unsigned>(1u);
  Kernel1D kernel = [&](BufferFloat4 output, BufferFloat dummy,
                        BufferUInt counts, Float3 origin, Float3 direction,
                        Float minimum, UInt visibility, UInt index) {
    UInt first = 0u;
    Var<RenderKernelParameters> parameters;
    parameters.sobol_sequence_size = 1u;
    PathKernelInvocation invocation{.config = config,
                                    .film_accumulation =
                                        PathFilmAccumulation::serial,
                                    .combined = output,
                                    .normal = output,
                                    .albedo = output,
                                    .light_passes = output,
                                    .sample_count = counts,
                                    .volume_guiding_raw = output,
                                    .volume_guiding_denoised = counts,
                                    .path_trace = output,
                                    .sample_first = first,
                                    .sobol_table = output,
                                    .filter_table = dummy,
                                    .parameters = parameters};
    PathSampleContext sample{.invocation = invocation};
    sample.ray = make_ray(origin, direction, minimum, f::maximum);
    sample.cycles_path_visibility = visibility;
    sample.previous_mis_origin_normal = make_float3(0, 0, 1);
    UInt step = 0u;
    Var<CommittedHit> hit;
    hit->hit_type = static_cast<unsigned>(HitType::Miss);
    hit->committed_ray_t = f::maximum;
    PathBounceContext bounce{.sample = sample,
                             .path_step = step,
                             .random_state = nullptr,
                             .hit = hit,
                             .subsurface_exit = false};
    // Exercise the actual endpoint stage, including its runtime visibility
    // predicate, not a copy of the intended branch in the regression.
    const auto event = closest->emit(bounce, ~0u);
    output.write(index,
                 make_float4(select(0.0f, 1.0f, event.analytic_light),
                             event.distance, event.light_evaluation_factor,
                             event.light_pdf));
  };
  auto shader = device.compile(
      kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
  auto visibility_output = device.create_buffer<unsigned>(64u);
  Kernel1D visibility_kernel = [](BufferUInt output) {
    namespace s = cycles_shader_identity;
    const auto i = dispatch_x();
    const auto shader_flags = select(0u, s::exclude_diffuse, (i & 1u) != 0u) |
                              select(0u, s::exclude_glossy, (i & 2u) != 0u) |
                              select(0u, s::exclude_transmit, (i & 4u) != 0u) |
                              select(0u, s::exclude_camera, (i & 8u) != 0u) |
                              select(0u, s::exclude_scatter, (i & 16u) != 0u);
    UInt mask = 0u;
    $for(visibility, 32u) {
      const auto flag = select(0u, path::flag_reflect, i >= 32u);
      $if(cycles_light_shader_visible(shader_flags, visibility, flag)) {
        mask |= 1u << visibility;
      };
    };
    output.write(i, mask);
  };
  auto visibility_shader =
      device.compile(visibility_kernel, ShaderOption{.enable_cache = false,
                                                     .enable_fast_math = true});
  for (unsigned row = 0u; row < f::inputs.size(); ++row) {
    const auto &input = f::inputs[row];
    for (unsigned mode = 0u; mode < f::modes; ++mode) {
      LightGpu light{};
      light.type = unsigned(input.type == 0u   ? LightType::point
                            : input.type == 1u ? LightType::spot
                                               : LightType::area);
      light.axis_x = luisa::make_float3(1, 0, 0);
      light.axis_y = luisa::make_float3(0, 1, 0);
      light.axis_z = luisa::make_float3(0, 0, 1);
      light.radius = input.radius;
      light.size_u = light.size_v = f::length;
      light.spot_angle = f::spot_angle;
      light.spot_smooth = f::spot_smooth;
      light.spread = f::spread;
      light.flags = light_flag_normalize |
                    (input.sphere ? light_flag_sphere : 0u) |
                    (input.ellipse ? light_flag_ellipse : 0u) |
                    (mode != 4u ? light_flag_use_mis : 0u);
      light.visibility_mask = psycles::contract::all_ray_visibility;
      if (mode == 2u) {
        light.visibility_mask &= ~diffuse_visibility;
      }
      if (mode == 3u) {
        light.visibility_mask &= ~camera_visibility;
      }
      const auto visibility = mode == 0u || mode == 3u
                                  ? path::visibility_camera
                                  : path::visibility_diffuse;
      stream << scene->light_buffer.copy_from(&light)
             << shader(records, unused_float, unused_uint,
                       luisa::make_float3(input.origin[0], input.origin[1],
                                          input.origin[2]),
                       luisa::make_float3(input.direction[0],
                                          input.direction[1],
                                          input.direction[2]),
                       input.minimum, visibility, row * f::modes + mode)
                    .dispatch(1u)
             << synchronize();
    }
  }
  std::array<luisa::float4, f::count> actual{};
  std::array<unsigned, 64u> actual_visibility{};
  stream << records.copy_to(actual.data())
         << visibility_shader(visibility_output).dispatch(64u)
         << visibility_output.copy_to(actual_visibility.data())
         << synchronize();
  unsigned failures = 0u;
  for (unsigned i = 0u; i < f::count; ++i) {
    for (unsigned lane = 0u; lane < 4u; ++lane) {
      const auto a = actual[i][lane], e = expected[i][lane];
      const auto tolerance =
          lane == 0u ? 0.0f : 1.0e-4f * std::max(1.0f, std::abs(e));
      if (!std::isfinite(a) || std::abs(a - e) > tolerance) {
        std::cerr << "case=" << i << " lane=" << lane << " expected=" << e
                  << " actual=" << a << '\n';
        ++failures;
      }
    }
  }
  for (unsigned i = 0u; i < expected_visibility.size(); ++i) {
    if (actual_visibility[i] != expected_visibility[i]) {
      std::cerr << "visibility row=" << i
                << " expected=" << expected_visibility[i]
                << " actual=" << actual_visibility[i] << '\n';
      ++failures;
    }
  }
  std::cout
      << f::count * 4u
      << " Cycles GPU endpoint checks, 2048 visibility predicates; failures="
      << failures << '\n';
  return failures == 0u;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
