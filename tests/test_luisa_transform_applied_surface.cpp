#include "cycles_triangle_surface_component.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output_bits = device.create_buffer<luisa::uint4>(2u);
  auto output_values = device.create_buffer<luisa::float4>(2u);

  const auto component = make_cycles_triangle_surface_component();
  Kernel1D evaluate = [component](BufferUInt4 bits,
                                  BufferFloat4 values) noexcept {
    SafeNormalizeCallable safe_normalize = [](Float3 value,
                                              Float3 fallback) noexcept {
      const auto valid = dot(value, value) > 1.0e-20f;
      const auto selected = select(fallback, value, valid);
      return normalize(select(make_float3(0.0f, 0.0f, 1.0f), selected,
                              dot(selected, selected) > 1.0e-20f));
    };

    // Barbershop object 114, primitive 1146566 at sample zero of film pixel
    // (626, 217). Cycles CPU reports SD_OBJECT_TRANSFORM_APPLIED and these
    // exact final world-space support bits. Reconstructing from the original
    // vertices through the instance matrix instead changes x by one ULP and
    // places the direct-light origin behind its coincident sibling.
    const auto object_to_world = make_float4x4(
        make_float4(0.45969536900520325f, -3.2663324746085324e-15f, 0.0f, 0.0f),
        make_float4(5.006332754088497e-15f, 0.7045786380767822f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.45969536900520325f, 0.0f),
        make_float4(0.6094699501991272f, 4.068269729614258f,
                    0.08646600693464279f, 1.0f));
    const auto p0 = make_float3(0.07778621464967728f, -0.5496593713760376f,
                                1.9021942615509033f);
    const auto p1 =
        make_float3(0.07778624445199966f, 0.0f, 1.9021943807601929f);
    const auto p2 =
        make_float3(0.07778630405664444f, 0.0f, 1.9509856700897217f);
    const auto final_p0 = make_float3(0.6452279090881348f, 3.6809914112091064f,
                                      0.9608958959579468f);
    const auto final_p1 = make_float3(0.6452279090881348f, 4.068269729614258f,
                                      0.9608959555625916f);
    const auto final_p2 = make_float3(0.6452279686927795f, 4.068269729614258f,
                                      0.9833250641822815f);
    const auto object_ng = normalize(cross(p1 - p0, p2 - p0));
    const CyclesTriangleSurfaceInput input{
        .object_to_world = object_to_world,
        .normal_to_world = transpose(inverse(object_to_world)),
        .transform_applied = true,
        .barycentric = make_float2(0.28864699602127075f, 0.5069726705551147f),
        .ray_direction = make_float3(
            -0.36165210604667664f, -0.9280669689178467f, -0.08887875825166702f),
        .smooth = false,
        .p0 = p0,
        .p1 = p1,
        .p2 = p2,
        .final_p0 = final_p0,
        .final_p1 = final_p1,
        .final_p2 = final_p2,
        .n0 = object_ng,
        .n1 = object_ng,
        .n2 = object_ng};
    const auto applied = component->resolve(input, safe_normalize);
    bits.write(0u, make_uint4(as<uint>(applied.position.x),
                              as<uint>(applied.position.y),
                              as<uint>(applied.position.z),
                              as<uint>(applied.p0.x)));
    values.write(0u, make_float4(applied.geometric_normal,
                                 select(0.0f, 1.0f, applied.back_facing)));

    auto ordinary_input = input;
    ordinary_input.transform_applied = false;
    const auto ordinary = component->resolve(ordinary_input, safe_normalize);
    bits.write(1u, make_uint4(as<uint>(ordinary.position.x),
                              as<uint>(ordinary.position.y),
                              as<uint>(ordinary.position.z),
                              as<uint>(ordinary.p0.x)));
    values.write(1u, make_float4(ordinary.geometric_normal,
                                 select(0.0f, 1.0f, ordinary.back_facing)));
  };
  auto shader = device.compile(evaluate);

  std::array<luisa::uint4, 2u> actual_bits{};
  std::array<luisa::float4, 2u> actual_values{};
  stream << shader(output_bits, output_values).dispatch(1u)
         << output_bits.copy_to(luisa::span{actual_bits})
         << output_values.copy_to(luisa::span{actual_values}) << synchronize();

  constexpr auto applied_x = std::uint32_t{0x3f252da9u};
  constexpr auto applied_p0_x = std::uint32_t{0x3f252da8u};
  if (actual_bits[0u].x != applied_x || actual_bits[0u].w != applied_p0_x ||
      !near(actual_values[0u].x, 1.0f, 1.0e-7f) ||
      !near(actual_values[0u].y, 4.0900164e-13f, 1.0e-11f) ||
      !near(actual_values[0u].z, -2.6574685e-6f, 2.0e-9f) ||
      actual_values[0u].w != 0.0f) {
    std::cerr << "transform-applied Cycles support failed on " << backend
              << ": position bits {" << std::hex << actual_bits[0u].x << ", "
              << actual_bits[0u].y << ", " << actual_bits[0u].z << "}, p0.x "
              << actual_bits[0u].w << std::dec << ", Ng {"
              << actual_values[0u].x << ", " << actual_values[0u].y << ", "
              << actual_values[0u].z << "}, back " << actual_values[0u].w
              << "\n";
    return EXIT_FAILURE;
  }
  if (actual_bits[1u].x == actual_bits[0u].x ||
      actual_bits[1u].w == applied_p0_x) {
    std::cerr << "ordinary object-space path collapsed into final support on "
              << backend << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
