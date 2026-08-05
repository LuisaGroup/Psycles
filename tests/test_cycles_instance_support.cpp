#include "../src/luisa/path_tracer_instance_support.h"
#include "../src/luisa/path_tracer_internal.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using psycles::contract::GeometryId;
using psycles::contract::SceneSnapshot;
using psycles::contract::TriangleMeshDesc;
using psycles::luisa_backend::detail::classify_cycles_final_triangle_supports;
using psycles::luisa_backend::detail::GeometryUpload;
using psycles::luisa_backend::detail::Triangle;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] GeometryUpload make_support() {
  GeometryUpload upload;
  upload.positions = {luisa::make_float3(0.0f, 0.0f, 0.0f),
                      luisa::make_float3(1.0f, 0.0f, 0.0f),
                      luisa::make_float3(0.0f, 1.0f, 0.0f)};
  upload.triangles = {Triangle{0u, 1u, 2u}};
  return upload;
}

void test_final_support_equivalence() {
  SceneSnapshot scene;
  for (auto id = 1u; id <= 4u; ++id) {
    scene.geometries.emplace(
        GeometryId{id},
        TriangleMeshDesc{.name = "support-" + std::to_string(id)});
  }
  std::vector<GeometryUpload> uploads;
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads[2u].positions[0u].z = std::nextafter(0.0f, 1.0f);
  uploads[3u].triangles[0u] = Triangle{0u, 2u, 1u};
  const std::map<GeometryId, std::uint32_t> indices{{GeometryId{1u}, 0u},
                                                    {GeometryId{2u}, 1u},
                                                    {GeometryId{3u}, 2u},
                                                    {GeometryId{4u}, 3u}};

  const auto classes =
      classify_cycles_final_triangle_supports(scene, indices, uploads);
  require(classes.ok(), "final support classification failed");
  require(classes.by_geometry.at(GeometryId{1u}) ==
              classes.by_geometry.at(GeometryId{2u}),
          "bitwise-equal final supports were split");
  require(classes.by_geometry.at(GeometryId{1u}) !=
              classes.by_geometry.at(GeometryId{3u}),
          "one-bit post-displacement position change was grouped");
  require(classes.by_geometry.at(GeometryId{1u}) !=
              classes.by_geometry.at(GeometryId{4u}),
          "triangle-index order change was grouped");

  // Simulate two source-identical meshes whose material/object contexts
  // produce different true-displacement results. Alias identity must follow
  // the final accelerator support, never the source mesh.
  uploads[1u].positions[1u].z = 0.125f;
  const auto displaced_classes =
      classify_cycles_final_triangle_supports(scene, indices, uploads);
  require(displaced_classes.ok(), "displaced support classification failed");
  require(displaced_classes.by_geometry.at(GeometryId{1u}) !=
              displaced_classes.by_geometry.at(GeometryId{2u}),
          "source-identical meshes remained aliased after displacement");
}

} // namespace

int main() {
  test_final_support_equivalence();
  std::cout << "Cycles final instance-support tests passed\n";
  return EXIT_SUCCESS;
}
