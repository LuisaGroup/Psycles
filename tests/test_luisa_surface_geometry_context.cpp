#include "luisa_surface_test_support.h"
#include "surface_geometry_context.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected) noexcept {
  return approximately_equal(actual.x, expected.x) &&
         approximately_equal(actual.y, expected.y) &&
         approximately_equal(actual.z, expected.z);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};

  Kernel1D evaluate = [](BufferFloat3 output) noexcept {
    auto triangle = make_surface_point();
    triangle.generated = make_float3(0.75f, 0.75f, 0.4f);
    triangle.normal_to_world_x = make_float3(2.0f, 0.0f, 0.0f);
    triangle.normal_to_world_y = make_float3(0.0f, 1.0f, 0.0f);
    triangle.normal_to_world_z = make_float3(0.0f, 0.0f, 0.5f);
    triangle.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    triangle.is_curve = false;
    output.write(0u, surface_geometry_tangent(triangle));

    auto curve = triangle;
    curve.is_curve = true;
    curve.dpdu = make_float3(3.0f, 4.0f, 0.0f);
    output.write(1u, surface_geometry_tangent(curve));

    auto objectless = triangle;
    objectless.geometry_index = ~0u;
    objectless.dpdu = make_float3(0.0f, 5.0f, 0.0f);
    output.write(2u, surface_geometry_tangent(objectless));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output = device.create_buffer<luisa::float3>(3u);
  auto kernel = device.compile(evaluate);
  std::array<luisa::float3, 3u> actual{};
  stream << kernel(output).dispatch(1u) << output.copy_to(luisa::span{actual})
         << synchronize();

  // Blender 5.2.1 Cycles primitive_tangent oracle:
  // mesh = cross(N, normalize(cross(normalize(M^-T * radial), N)));
  // curve/objectless ShaderData = normalize(dPdu).
  constexpr std::array expected{
      luisa::float3{-0.894427191f, 0.447213596f, 0.0f},
      luisa::float3{0.6f, 0.8f, 0.0f}, luisa::float3{0.0f, 1.0f, 0.0f}};
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (!near(actual[index], expected[index])) {
      std::cerr << "Cycles Geometry Tangent oracle " << index << " failed on "
                << backend << ": got {" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << "}\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
