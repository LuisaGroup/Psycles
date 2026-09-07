#include <psycles/adapter/blender_scene.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace psycles::contract;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

struct TemporaryDirectory {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("psycles-failed-image-import-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  TemporaryDirectory() {
    require(std::filesystem::create_directory(path),
            "temporary directory exists");
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
} // namespace

int main() {
  try {
    TemporaryDirectory temporary;
    {
      std::ofstream geometry{temporary.path / "geometry.bin", std::ios::binary};
      geometry.write("PSYGEO1\0", 8);
      // Resource-boundary fixture only: no expected SVM words or reference
      // shading implementation. There is deliberately no source pixel file.
      std::ofstream json{temporary.path / "scene.json"};
      json << R"JSON({"schema":"psycles.blender-scene.v1",
        "images":[
          {"name":"Missing sRGB","source":"FILE","colorspace":"sRGB",
           "alpha_mode":"STRAIGHT","width":0,"height":0,"load_failed":true},
          {"name":"Missing data","source":"FILE","colorspace":"Non-Color",
           "alpha_mode":"CHANNEL_PACKED","width":32,"height":32,"load_failed":true}],
        "materials":[],"node_groups":[],"geometries":[],"curve_geometries":[],
        "instances":[],"lights":[],"camera":{},"world":{},"render":{}
      })JSON";
    }
    const auto loaded =
        psycles::adapter::load_blender_scene_bundle(temporary.path);
    for (const auto &diagnostic : loaded.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    require(loaded.ok(),
            "failed-image metadata did not import without a pixel file");
    const auto &images = loaded.scene->images;
    require(images.size() == 2u, "failed image identities were collapsed");
    const auto &srgb = images.at(ImageId{1u});
    const auto &data = images.at(ImageId{2u});
    require(srgb.name == "Missing sRGB" &&
                srgb.color_space == ImageColorSpace::srgb &&
                data.name == "Missing data" &&
                data.color_space == ImageColorSpace::data &&
                data.alpha_type == ImageAlphaType::channel_packed,
            "failed image source metadata was lost");
    require(srgb.load_failed && data.load_failed && srgb.encoded_data.empty() &&
                data.encoded_data.empty(),
            "failed image acquired a fake payload");
    SceneDatabase database;
    SceneDelta delta;
    for (const auto &[id, image] : images) {
      delta.emplace<UpsertImage>(id, image);
    }
    require(database.apply(delta).committed,
            "scene contract rejected explicit failed-image state");
    auto inconsistent = srgb;
    inconsistent.encoded_data = {1u};
    SceneDelta invalid{.base_revision = database.snapshot().revision};
    invalid.emplace<UpsertImage>(ImageId{1u}, inconsistent);
    require(!database.apply(invalid).committed,
            "failed-image state accepted contradictory pixels");
    inconsistent.encoded_data.clear();
    inconsistent.load_failed = false;
    invalid.commands.clear();
    invalid.emplace<UpsertImage>(ImageId{1u}, inconsistent);
    require(!database.apply(invalid).committed,
            "ordinary empty image bypassed validation");
    std::cout << "Failed-image import and contract passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
