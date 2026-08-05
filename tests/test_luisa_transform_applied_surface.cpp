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
  auto output_bits = device.create_buffer<luisa::uint4>(3u);
  auto output_values = device.create_buffer<luisa::float4>(3u);

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

    // Barbershop cupboard object 115, primitive 1147418 at sample 6 of film
    // pixel (634, 209). Cycles CPU and HIP both contract the ordinary object
    // transform into a fused multiply-add and reconstruct x as 0x3f24212b.
    // A generic matrix lowering rounds the product before the translation,
    // producing 0x3f24212c and moving the shadow origin through its exactly
    // coincident sibling.
    const auto cupboard_transform = make_float4x4(
        make_float4(0.16329917311668396f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.16329915821552277f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.08164958655834198f, 0.0f),
        make_float4(0.4866490066051483f, 4.260610580444336f,
                    0.8810397982597351f, 1.0f));
    const auto cupboard_p0 = make_float3(0.9460067749023438f, 0.0f, 0.0f);
    const auto cupboard_p1 =
        make_float3(0.9460067749023438f, 0.0f, 0.9330167770385742f);
    const auto cupboard_p2 = make_float3(
        0.9460067749023438f, -1.111984372138977f, 0.9330167770385742f);
    const auto cupboard_ng =
        normalize(cross(cupboard_p1 - cupboard_p0,
                        cupboard_p2 - cupboard_p0));
    const auto cupboard = component->resolve(
        {.object_to_world = cupboard_transform,
         .normal_to_world = transpose(inverse(cupboard_transform)),
         .transform_applied = false,
         .barycentric =
             make_float2(0.03073400817811489f, 0.3425517976284027f),
         .ray_direction = make_float3(-0.3722669184207916f,
                                      -0.9224809408187866f,
                                      -0.1022074967622757f),
         .smooth = false,
         .p0 = cupboard_p0,
         .p1 = cupboard_p1,
         .p2 = cupboard_p2,
         .final_p0 = cupboard_p0,
         .final_p1 = cupboard_p1,
         .final_p2 = cupboard_p2,
         .n0 = cupboard_ng,
         .n1 = cupboard_ng,
         .n2 = cupboard_ng},
        safe_normalize);
    bits.write(2u, make_uint4(as<uint>(cupboard.position.x),
                              as<uint>(cupboard.world_p0.x),
                              as<uint>(cupboard.world_p1.x),
                              as<uint>(cupboard.world_p2.x)));
    values.write(2u, make_float4(cupboard.position, cupboard.object_position.x));
  };
  auto shader = device.compile(evaluate);

  std::array<luisa::uint4, 3u> actual_bits{};
  std::array<luisa::float4, 3u> actual_values{};
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
  constexpr auto cupboard_world_x = std::uint32_t{0x3f24212bu};
  if (actual_bits[2u].x != cupboard_world_x ||
      actual_bits[2u].y != cupboard_world_x ||
      actual_bits[2u].z != cupboard_world_x ||
      actual_bits[2u].w != cupboard_world_x) {
    std::cerr << "ordinary Cycles FMA surface reconstruction failed on "
              << backend << ": position/world support x bits {" << std::hex
              << actual_bits[2u].x << ", " << actual_bits[2u].y << ", "
              << actual_bits[2u].z << ", " << actual_bits[2u].w << std::dec
              << "}, values {" << actual_values[2u].x << ", "
              << actual_values[2u].y << ", " << actual_values[2u].z << ", "
              << actual_values[2u].w << "}\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
