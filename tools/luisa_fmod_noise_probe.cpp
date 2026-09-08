// Compare the production Luisa/Psycles operations with cycles_fmod_noise_oracle.hip.
// The host only transfers dynamic input/output; no expected shader formula.
#include <psycles/luisa/cycles_noise.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include <array>
#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  using namespace luisa;
  using namespace luisa::compute;
  namespace noise = psycles::luisa_backend::cycles_noise;
  if (argc != 3 && argc != 4) { return 2; }
  const bool fast = std::string_view{argv[2]} == "fast";
  const bool early_ocml = argc == 4 && std::string_view{argv[3]} == "ocml";
  Context context{argv[1]};
  auto device = context.create_device("hip");
  auto stream = device.create_stream();
  constexpr unsigned count = 8;
  const std::array<float4, count> coordinates{{
      {5.75f, -5.75f, 0.125f, -0.125f},
      {1.0e17f, -1.0e17f, 1.0e12f, -1.0e12f},
      {99999.5f, 100000.0f, 100000.5f, -100000.5f},
      {999999.5f, 1000000.0f, 1000000.5f, -1000000.5f},
      {1.0e17f, -1.0e17f, 1.0e30f, -1.0e30f},
      {0.0f, -0.0f, 2.0f, -2.0f},
      {0.3f, 0.7f, -0.3f, -0.7f},
      {12345.625f, -12345.625f, 1024.25f, -1024.25f}}};
  const std::array<float, count> divisors{{2.0f, 6.283185307179586f, 100000.0f,
      100000.0f, 100000.0f, 2.0f, 0.1f, 7.0f}};
  auto device_coordinates = device.create_buffer<float4>(count);
  auto device_divisors = device.create_buffer<float>(count);
  auto device_outputs = device.create_buffer<float4>(count * 4);
  Kernel1D kernel = [early_ocml](BufferFloat4 coordinates, BufferFloat divisors, BufferFloat4 outputs) {
    const auto index = dispatch_x();
    const auto p = coordinates.read(index);
    const auto divisor = divisors.read(index);
    // Diagnostic only: reuse the backend's existing OCML body before JIT IPO.
    // Production Noise is unchanged, and no remainder formula is substituted.
    ExternalCallable<float(float, float)> ocml_fmod{"__ocml_fmod_f32"};
    const auto remainder = [&](Float x, Float y) {
      return early_ocml ? ocml_fmod(x, y) : fmod(x, y);
    };
    const auto vector_remainder = [&](Float4 x, Float y) {
      return early_ocml ? make_float4(remainder(x.x, y), remainder(x.y, y),
                                     remainder(x.z, y), remainder(x.w, y))
                        : fmod(x, make_float4(y));
    };
    const auto vector3 = early_ocml ? vector_remainder(p, divisor).xyz()
                                   : fmod(p.xyz(), make_float3(divisor));
    outputs.write(index * 4u, make_float4(remainder(p.x, divisor), vector3));
    outputs.write(index * 4u + 1u, vector_remainder(p, divisor));
    outputs.write(index * 4u + 2u, make_float4(noise::signed_noise(p.x),
        noise::signed_noise(p.xy()), noise::signed_noise(p.xyz()), noise::signed_noise(p)));
    outputs.write(index * 4u + 3u, vector_remainder(p, 100000.0f));
  };
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = fast});
  std::array<float4, count * 4> output;
  stream << device_coordinates.copy_from(luisa::span{coordinates})
         << device_divisors.copy_from(luisa::span{divisors})
         << shader(device_coordinates, device_divisors, device_outputs).dispatch(count)
         << device_outputs.copy_to(luisa::span{output}) << synchronize();
  for (unsigned index = 0; index < count; ++index) {
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto value = output[index * 4 + slot];
      std::array<unsigned, 4> bits;
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(bits.data(), &value, sizeof(value));
      std::printf("%u %u %08x %08x %08x %08x %.9g %.9g %.9g %.9g\n", index, slot,
          bits[0], bits[1], bits[2], bits[3], value.x, value.y, value.z, value.w);
    }
  }
}
