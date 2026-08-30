#include <psycles/adapter/blender_scene.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using psycles::contract::CurveShape;
using psycles::contract::GeometryId;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

void expect_near(float actual, float expected, const std::string &message) {
  expect(std::abs(actual - expected) <= 1.0e-6f,
         message + ": got " + std::to_string(actual) + ", expected " +
             std::to_string(expected));
}

class TemporaryDirectory {

private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-curve-import-" + std::to_string(nonce));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return _path;
  }
};

struct Section {
  std::uint64_t offset{};
  std::uint64_t bytes{};
};

template <typename T>
[[nodiscard]] Section write_values(std::ofstream &stream,
                                   const std::vector<T> &values) {
  const auto offset = static_cast<std::uint64_t>(stream.tellp());
  const auto bytes = static_cast<std::uint64_t>(values.size() * sizeof(T));
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(bytes));
  return {.offset = offset, .bytes = bytes};
}

void write_section(std::ofstream &stream, const Section &section) {
  stream << "{\"offset\":" << section.offset << ",\"bytes\":" << section.bytes
         << '}';
}

void test_native_curve_bundle_round_trip() {
  TemporaryDirectory temporary;
  Section keys;
  Section first_key;
  Section material_slots;
  Section uv;
  Section color;
  Section intercept;
  Section length;
  Section random;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    const std::array<std::uint32_t, 2u> version{2u, 0u};
    geometry.write(reinterpret_cast<const char *>(version.data()),
                   static_cast<std::streamsize>(sizeof(version)));
    keys =
        write_values<float>(geometry, {0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 1.0f,
                                       0.07f, 0.25f, 0.0f, 2.0f, 0.0f});
    first_key = write_values<std::uint32_t>(geometry, {0u});
    material_slots = write_values<std::uint32_t>(geometry, {0u});
    uv = write_values<float>(geometry, {0.25f, 0.75f});
    color = write_values<float>(geometry, {0.1f, 0.2f, 0.3f, 1.0f});
    intercept = write_values<float>(geometry, {0.0f, 0.45f, 1.0f});
    length = write_values<float>(geometry, {2.0311289f});
    random = write_values<float>(geometry, {0.86031276f});
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v2",
  "images":[],
  "node_groups":[],
  "materials":[],
  "render":{"width":64,"height":32,"percentage":100,"cycles":{}},
  "camera":{
    "name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0
  },
  "geometries":[],
  "curve_geometries":[{
    "name":"Particle Hair","shape":"RIBBON","subdivisions":2,
    "key_count":3,"curve_count":1,"segment_count":2,
    "keys":)JSON";
    write_section(scene, keys);
    scene << R"JSON(,
    "curve_first_key":)JSON";
    write_section(scene, first_key);
    scene << R"JSON(,
    "curve_material_slots":)JSON";
    write_section(scene, material_slots);
    scene << R"JSON(,
    "default_uv_layer":"RootUV",
    "uv_layers":[{"name":"RootUV","values":)JSON";
    write_section(scene, uv);
    scene << R"JSON(}],
    "color_attributes":[{"name":"RootColor","values":)JSON";
    write_section(scene, color);
    scene << R"JSON(}],
    "intercept":)JSON";
    write_section(scene, intercept);
    scene << R"JSON(,
    "length":)JSON";
    write_section(scene, length);
    scene << R"JSON(,
    "random":)JSON";
    write_section(scene, random);
    scene << R"JSON(,
    "material_slots":[null],
    "cycles_sync":{"curve_offset":7,"segment_offset":19}
  }],
  "instances":[{
    "name":"Particle Hair","geometry_type":"CURVE","geometry":0,
    "is_instance":true,
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,2,3,4,1],
    "random_id":0,"particle_index":0,
    "cycles_sync":{"object_index":11,"light_group":-1}
  }],
  "lights":[],"world":null,"world_environment":null
})JSON";
  }

  const auto imported = load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "native curve bundle did not import");
  expect(imported.scene->geometries.empty(), "curve became a triangle mesh");
  expect(imported.scene->curve_geometries.size() == 1u,
         "native curve geometry is missing");
  const auto &curves = imported.scene->curve_geometries.at(GeometryId{1u});
  expect(curves.shape == CurveShape::ribbon, "curve shape mismatch");
  expect(curves.subdivisions == 2u, "curve subdivisions mismatch");
  expect(curves.keys.size() == 3u, "curve key count mismatch");
  expect(curves.curve_first_key == std::vector<std::uint32_t>{0u},
         "curve first-key table mismatch");
  expect(curves.curve_material_slots == std::vector<std::uint32_t>{0u},
         "curve material table mismatch");
  expect(curves.material_slots.size() == 1u,
         "default curve material was not resolved");
  expect(curves.default_uv_layer == "RootUV",
         "default curve UV layer mismatch");
  expect(curves.uv_layers.at("RootUV").size() == 1u,
         "curve UV domain mismatch");
  expect_near(curves.uv_layers.at("RootUV")[0u].x, 0.25f,
              "curve root U");
  expect_near(curves.uv_layers.at("RootUV")[0u].y, 0.75f,
              "curve root V");
  expect(curves.color_attributes.at("RootColor").size() == 1u,
         "curve color domain mismatch");
  expect_near(curves.color_attributes.at("RootColor")[0u].x, 0.1f,
              "curve root color R");
  expect_near(curves.color_attributes.at("RootColor")[0u].w, 1.0f,
              "curve root color alpha");
  expect_near(curves.keys[1u].w, 0.07f, "middle key radius");
  expect_near(curves.intercept[1u], 0.45f, "middle Intercept");
  expect_near(curves.length[0u], 2.0311289f, "curve Length");
  expect_near(curves.random[0u], 0.86031276f, "curve Random");
  expect(curves.cycles_curve_offset == 7u &&
             curves.cycles_segment_offset == 19u,
         "Cycles curve primitive offsets mismatch");
  expect(imported.scene->instances.size() == 1u, "curve instance is missing");
  const auto &instance = imported.scene->instances.begin()->second;
  expect(instance.geometry == GeometryId{1u},
         "curve-local geometry index was not resolved");
  expect(instance.cycles_object_index == 11u, "hair object identity mismatch");
  expect(instance.is_blender_instance,
         "dependency-graph instance representation was dropped");
}

} // namespace

int main() {
  try {
    test_native_curve_bundle_round_trip();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
