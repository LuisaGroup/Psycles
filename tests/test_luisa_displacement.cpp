#include "surface_displacement.h"

#include "luisa_surface_test_support.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;

[[nodiscard]] bool
approximately_equal_float3(luisa::float3 actual, luisa::float3 expected,
                           float tolerance = 3.0e-6f) noexcept {
  return approximately_equal(actual.x, expected.x, tolerance) &&
         approximately_equal(actual.y, expected.y, tolerance) &&
         approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] SurfaceDisplacementInput make_input() noexcept {
  // object_to_world M = diag(2, 1, 0.5), hence the SurfacePoint normal
  // transform A = (M^-1)^T = diag(0.5, 1, 2).
  return {.height = 1.0f,
          .midlevel = 0.4f,
          .scale = 0.5f,
          .normal = normalize(make_float3(1.0f)),
          .normal_to_world_x = make_float3(0.5f, 0.0f, 0.0f),
          .normal_to_world_y = make_float3(0.0f, 1.0f, 0.0f),
          .normal_to_world_z = make_float3(0.0f, 0.0f, 2.0f)};
}

[[nodiscard]] SurfaceDisplacementInput make_sheared_input() noexcept {
  // This is A = (M^-1)^T for the deliberately non-symmetric transform
  //
  //   M = {1.50,  0.25, -0.10,
  //        0.20,  0.80,  0.30,
  //        0.05, -0.15,  1.20}.
  //
  // A shear is required here: a diagonal-only oracle cannot distinguish
  // inverse, transpose, row-major, and column-major mistakes.
  return {.height = 0.85f,
          .midlevel = 0.2f,
          .scale = 0.4f,
          .normal = make_float3(0.3f, -0.4f, 0.8660254038f),
          .normal_to_world_x =
              make_float3(0.68918224f, -0.19543974f, 0.10629179f),
          .normal_to_world_y =
              make_float3(-0.15429453f, 1.23778502f, -0.32230413f),
          .normal_to_world_z =
              make_float3(-0.04800274f, 0.16286645f, 0.78861649f)};
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};

  Kernel1D kernel = [](BufferFloat3 output) noexcept {
    const auto input = make_input();
    const auto sheared_input = make_sheared_input();
    output.write(0u, displacement_world_inline(input));
    output.write(1u, displacement_object_inline(input));
    output.write(2u, displacement_world_inline(sheared_input));
    output.write(3u, displacement_object_inline(sheared_input));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output = device.create_buffer<luisa::float3>(4u);
  auto shader = device.compile(kernel);
  std::array<luisa::float3, 4u> actual{};
  stream << shader(output).dispatch(1u) << output.copy_to(luisa::span{actual})
         << synchronize();

  // These are the closed-form outputs of Cycles 5.2
  // svm_node_displacement for the transform above. OBJECT must not collapse
  // to WORLD under non-uniform scale:
  //   normalize(M^T N) = (2, 1, 0.5) / sqrt(5.25)
  //   M * normalize(M^T N) * 0.3
  const auto expected_world =
      luisa::make_float3(0.1732050808f, 0.1732050808f, 0.1732050808f);
  const auto expected_object =
      luisa::make_float3(0.5237229366f, 0.1309307341f, 0.0327326835f);
  // Closed-form Cycles result for the shear above:
  //   amount       = (height - midlevel) * scale
  //   object_normal = normalize(M^T * normal)
  //   offset        = M * object_normal * amount
  const auto expected_sheared_world =
      luisa::make_float3(0.078f, -0.104f, 0.225166605f);
  const auto expected_sheared_object =
      luisa::make_float3(0.10830410f, 0.01226093f, 0.28332174f);
  if (!approximately_equal_float3(actual[0u], expected_world) ||
      !approximately_equal_float3(actual[1u], expected_object) ||
      !approximately_equal_float3(actual[2u], expected_sheared_world) ||
      !approximately_equal_float3(actual[3u], expected_sheared_object) ||
      approximately_equal_float3(actual[0u], actual[1u])) {
    std::cerr << "Cycles scalar Displacement mismatch on " << backend
              << ": world={" << actual[0u].x << ", " << actual[0u].y << ", "
              << actual[0u].z << "}, object={" << actual[1u].x << ", "
              << actual[1u].y << ", " << actual[1u].z << "}, sheared world={"
              << actual[2u].x << ", " << actual[2u].y << ", " << actual[2u].z
              << "}, sheared object={" << actual[3u].x << ", " << actual[3u].y
              << ", " << actual[3u].z << "}\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
