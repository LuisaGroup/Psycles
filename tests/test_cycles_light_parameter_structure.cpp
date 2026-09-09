// Inspect production device consumers, not a host light evaluator. Parameters
// owned by Cycles scene synchronization must not record per-ray trigonometry.
#include <psycles/luisa/area_light_sampling.h>
#include <psycles/luisa/volume_light_interval.h>
#include <luisa/luisa-compute.h>

#include <iostream>
#include <set>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
namespace sampling = analytic_light_sampling;

unsigned count(Function function, bool cosine, std::set<std::uint64_t> &seen) {
  if (!seen.insert(function.hash()).second) { return 0; }
  unsigned total = 0;
  traverse_expressions<true>(function.body(), [&](const Expression *expr) {
    if (expr->tag() != Expression::Tag::CALL) { return; }
    const auto call = static_cast<const CallExpr *>(expr);
    if (call->is_custom()) { total += count(call->custom(), cosine, seen); }
    else if (call->op() == CallOp::TAN || (cosine && call->op() == CallOp::COS)) { ++total; }
  }, [](auto) {}, [](auto) {});
  return total;
}

bool check(unsigned mode) {
  Kernel1D kernel = [&](BufferFloat4 values, BufferFloat4 output,
                        Var<Buffer<AreaLightParameters>> areas,
                        Var<Buffer<SpotLightParameters>> spots) {
    const auto a = values.read(0u), b = values.read(1u), c = values.read(2u);
    const auto x = make_float3(1, 0, 0), y = make_float3(0, 1, 0), z = make_float3(0, 0, 1);
    const auto center = make_float3(0);
    const auto area = areas.read(dispatch_x());
    const auto spot = spots.read(dispatch_x());
    if (mode == 0) {
      output.write(0u, make_float4(sampling::spot_light_attenuation(a.xyz(), spot)));
    } else if (mode == 1) {
      output.write(0u, make_float4(sampling::spot_light_uv(a.xyz(), spot.half_cot_half_spot_angle), 0.0f, 0.0f));
    } else if (mode == 2) {
      output.write(0u, make_float4(sampling::area_spread_attenuation(a.xyz(), z, area)));
    } else if (mode == 3) {
      const auto value = sampling::sample_spot_light(a.xyz(), z, false, center, b.z,
          true, x, y, z, c.xyz(), spot, a.xy(), true);
      output.write(0u, make_float4(value.direction, value.conditional_pdf));
    } else if (mode == 4) {
      const AreaLightSampleInput input{a.xyz(), center, x, y, z, b.y, b.z,
          area, false, a.xy(), true};
      const auto value = AreaLightSampling{}.from_position(input);
      output.write(0u, make_float4(value.direction, value.conditional_pdf));
    } else if (mode == 5) {
      const auto value = VolumeLightInterval{}.spot({a.xyz(), c.xyz(), {0, 10},
          center, x, y, z, c.xyz(), spot});
      output.write(0u, make_float4(value.interval.minimum, value.interval.maximum,
          cast<float>(value.valid), 0.0f));
    } else {
      const auto value = VolumeLightInterval{}.area({a.xyz(), c.xyz(), {0, 10},
          center, x, y, z, b.y, b.z, area, false});
      output.write(0u, make_float4(value.interval.minimum, value.interval.maximum,
          cast<float>(value.valid), 0.0f));
    }
  };
  std::set<std::uint64_t> seen;
  // Sampling legitimately uses sine/cosine of RNG- or geometry-dependent
  // angles. TAN is scene-owned in these consumers; the scalar helpers and
  // volume cone interval contain no legitimate per-ray COS either.
  const auto calls = count(Function{kernel.function().get()}, mode < 3 || mode == 5, seen);
  std::cout << "consumer=" << mode << " scene-owned trig calls=" << calls << '\n';
  return calls == 0;
}
} // namespace

int main() {
  unsigned passed = 0;
  for (unsigned mode = 0; mode < 7; ++mode) { passed += check(mode); }
  std::cout << "Cycles scene-owned light parameter structure: " << passed << "/7\n";
  return passed == 7 ? 0 : 1;
}
