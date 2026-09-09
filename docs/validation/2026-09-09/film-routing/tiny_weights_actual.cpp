#include "tiny_weight_inputs.h"
#include "path_tracer_lighting.h"
#include <luisa/luisa-compute.h>
#include <array>
#include <cstdio>

int main(int argc, char **argv) {
  using namespace luisa::compute;
  namespace d = psycles::luisa_backend::detail;
  Context context{argv[0]};
  auto device = context.create_device(argc > 1 ? argv[1] : "hip");
  auto stream = device.create_stream();
  auto callables = d::make_light_transport_callables(
      psycles::contract::DirectLightSampling::next_event_estimation);
  Kernel1D kernel = [&](BufferUInt out, UInt numerator, UInt denominator, UInt zero_bits) {
    const auto n = numerator.as<float>(), v = denominator.as<float>(), z = zero_bits.as<float>();
    auto a = callables.light_component_ratio(make_float3(z, n, n), make_float3(v));
    auto b = callables.light_component_ratio(make_float3(n, z, n), make_float3(v));
    for (unsigned i = 0; i < 3; ++i) {
      out.write(i, a[i].as<unsigned>()); out.write(i + 3, b[i].as<unsigned>());
    }
  };
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});

  auto out = device.create_buffer<unsigned>(6);
  for (const auto &v : tiny_weight_inputs) {
    std::array<unsigned, 6> result{};
    stream << shader(out, v[0], v[1], 0u).dispatch(1u)
           << out.copy_to(result.data()) << synchronize();
    std::printf("%08x %08x", v[0], v[1]);
    for (auto bits : result) { std::printf(" %08x", bits); }
    std::printf("\n");
  }
}
