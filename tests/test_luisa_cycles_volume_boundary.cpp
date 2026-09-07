#include "cycles_volume_boundary_fixture.h"

#include <psycles/luisa/cycles_volume_boundary.h>
#include <luisa/luisa-compute.h>

#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace luisa::compute;
using namespace psycles::test_support;
namespace boundary = psycles::luisa_backend::cycles_volume_boundary;
namespace closure = psycles::luisa_backend::cycles_closure;

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    auto stream = device.create_stream();
    constexpr auto count = volume_boundary_inputs.size();
    std::array<luisa::float4, count> parameters{};
    for (auto i = 0u; i < count; ++i) {
      const auto input = volume_boundary_inputs[i];
      parameters[i] = {std::bit_cast<float>(input.bounds), std::bit_cast<float>(input.flag),
                        input.probability, input.distance};
    }
    auto inputs = device.create_buffer<luisa::float4>(count);
    auto output = device.create_buffer<luisa::float4>(count);
    auto metadata = device.create_buffer<luisa::uint4>(count);
    Kernel1D<Buffer<luisa::float4>, Buffer<luisa::float4>, Buffer<luisa::uint4>> kernel =
        [](BufferFloat4 input, BufferFloat4 out, BufferVar<luisa::uint4> meta) {
          const auto i = dispatch_x();
          const auto value = input.read(i);
          const auto result = boundary::advance(make_float3(0.25f, 0.5f, 0.75f),
              value.x.bitcast<unsigned>(), 33u, 0.125f, value.w, value.z,
              value.y.bitcast<unsigned>());
          out.write(i, make_float4(result.throughput, result.ray_tmin));
          meta.write(i, make_uint4(result.bounds_bounce, result.rng_offset,
              select(0u, closure::label_transmit | closure::label_transparent, result.valid), 0u));
        };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    std::array<luisa::float4, count> actual{};
    std::array<luisa::uint4, count> states{};
    stream << inputs.copy_from(parameters.data()) << shader(inputs, output, metadata).dispatch(count)
           << output.copy_to(actual.data()) << metadata.copy_to(states.data()) << synchronize();
    std::ifstream oracle{PSYCLES_VOLUME_BOUNDARY_ORACLE};
    for (auto i = 0u; i < count; ++i) {
      unsigned index{};
      luisa::float4 value{};
      luisa::uint4 state{};
      oracle >> index >> value.x >> value.y >> value.z >> value.w >> state.x >> state.y >> state.z;
      if (!oracle || index != i || luisa::any(states[i] != state)) {
        throw std::runtime_error{"volume boundary integer state differs from Cycles HIP"};
      }
      for (auto lane = 0u; lane < 4u; ++lane) {
        // Keep tmin's integer next-float operation exact. Fallback's native
        // reciprocal approximation can differ by 1 ULP even for /0.25; this
        // is allowed, not a reason to add a slower division implementation.
        const auto a = std::bit_cast<unsigned>(actual[i][lane]);
        const auto e = std::bit_cast<unsigned>(value[lane]);
        const auto ulps = a > e ? a - e : e - a;
        if (ulps > (lane == 3u ? 0u : 1u)) {
          std::cerr << "case=" << i << " lane=" << lane << std::hex
                    << " actual=0x" << std::bit_cast<unsigned>(actual[i][lane])
                    << " expected=0x" << std::bit_cast<unsigned>(value[lane]) << std::dec << '\n';
          throw std::runtime_error{"volume boundary float state differs from Cycles HIP"};
        }
      }
    }
    std::cout << "36 original Cycles volume-only boundary transitions passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
