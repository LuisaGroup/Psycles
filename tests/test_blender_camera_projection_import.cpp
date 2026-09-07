#include <psycles/adapter/blender_scene.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
      ("psycles-camera-projection-import-" + std::to_string(nonce));
  std::filesystem::create_directories(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
  } cleanup{directory};
  { std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8); }
  bool passed = true;
  for (unsigned variant = 0u; variant < 3u; ++variant) {
    {
      std::ofstream scene{directory / "scene.json"};
      scene << R"({"schema":"psycles.blender-scene.v1","images":[],"node_groups":[],
        "materials":[],"render":{"width":900,"height":1600,"cycles":{}},
        "camera":{"name":"Physical","type":"PERSP","sensor_fit":"AUTO",
        "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
        "lens":25.5,"angle_x":1.2,"angle_y":0.8,
        "pixel_aspect_x":1.5,"pixel_aspect_y":1.0)";
      if (variant != 0u) {
        scene << ",\"sensor_width\":" << (variant == 1u ? 36 : 0)
              << ",\"sensor_height\":24";
      }
      scene << R"(},"geometries":[],"curve_geometries":[],"instances":[],
        "lights":[],"world":null,"world_environment":null})";
    }
    const auto imported = psycles::adapter::load_blender_scene_bundle(directory);
    if (variant == 2u) { passed &= !imported.ok(); continue; }
    if (!imported.ok() || imported.scene->cameras.size() != 1u) { return 1; }
    const auto &camera = imported.scene->cameras.begin()->second;
    passed &= camera.sensor_fit == psycles::contract::CameraSensorFit::automatic;
    passed &= camera.pixel_aspect == psycles::Vec2f{1.5f, 1.0f};
    passed &= camera.sensor.has_value() == (variant == 1u);
    if (camera.sensor) {
      passed &= camera.sensor->lens_mm == 25.5f && camera.sensor->width_mm == 36.0f &&
                camera.sensor->height_mm == 24.0f;
    }
  }
  if (!passed) { std::cerr << "Physical/legacy camera import contract failed\n"; }
  return passed ? 0 : 1;
}
