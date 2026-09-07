#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

struct Bundle {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("psycles-native-volume-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(const std::filesystem::path &scene) {
    require(std::filesystem::create_directory(path),
            "cannot create test bundle");
    std::filesystem::copy_file(scene, path / "scene.json");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write empty geometry header");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

bool requests(const ShaderImage &image, AttributeStandard standard,
              std::string_view name = {}) {
  return std::ranges::any_of(
      image.attribute_requests, [&](const auto &request) {
        return request.standard == standard && request.name == name;
      });
}

void run(std::string_view stem, unsigned expected_count) {
  const auto prefix = std::filesystem::path{PSYCLES_VOLUME_FIXTURE_DIR} / stem;
  Bundle bundle{prefix.string() + "_scene.json"};
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(bundle.path);
  for (const auto &diagnostic : imported.diagnostics) {
    std::cerr << diagnostic.message << '\n';
  }
  require(imported.ok(), "external volume fixture did not import");
  const ShaderCompiler compiler{make_core_node_registry()};
  std::ifstream oracle{prefix.string() + "_words.txt"};
  unsigned count{};
  oracle >> count;
  require(oracle.good() && count == expected_count,
          "invalid oracle cardinality");
  for (auto index = 0u; index < count; ++index) {
    std::string name;
    unsigned size{};
    oracle >> std::quoted(name) >> std::dec >> size;
    require(oracle.good() && size >= 7u && size < 1024u,
            "invalid oracle image");
    std::vector<std::uint32_t> expected(size);
    oracle >> std::hex;
    for (auto &word : expected) {
      oracle >> word;
    }
    require(!oracle.fail(), "truncated external word image");
    const auto material =
        std::ranges::find_if(imported.scene->materials, [&](const auto &entry) {
          return entry.second.cycles_shader_index == index + 5u;
        });
    require(material != imported.scene->materials.end(),
            "external shader identity missing");
    require(material->second.name == name, "external shader name changed");
    const auto shader = compiler.compile(material->second.shader);
    require(shader.ok(), "volume source graph failed validation");
    AttributeIDMap attributes;
    const auto image =
        compile_shader(*shader.program, attributes, ShaderCompileContext{});
    if (!image.valid) {
      throw std::runtime_error{name + ": " + image.diagnostic};
    }
    // The observer-enabled original Cycles compiler supplies the complete
    // expected tail. Only its three global jump targets are relocated; payload
    // padding, input kinds, named attribute IDs and stack addresses are intact.
    if (image.words != expected) {
      std::cerr << name << ": actual " << image.words.size() << ", Cycles "
                << expected.size() << " words\n";
      for (auto i = 0u; i < std::max(image.words.size(), expected.size());
           ++i) {
        std::cerr << i << ": " << std::hex
                  << (i < image.words.size() ? image.words[i] : ~0u) << " / "
                  << (i < expected.size() ? expected[i] : ~0u) << std::dec
                  << '\n';
      }
      throw std::runtime_error{
          "volume image differs from original Cycles 5.2.1"};
    }
    require(image.metadata.has_volume && !image.metadata.has_surface &&
                (image.metadata.kernel_features & kernel_feature_node_volume) !=
                    0u,
            "volume domain or feature metadata was lost");
    const auto mixed = name.ends_with("Linked") || name.ends_with("Repeated");
    require(image.metadata.num_closures == (mixed ? 64u : 32u),
            "volume graph closure budget differs from Cycles");
    const auto principled = name.starts_with("Principled Volume");
    require(image.node_types_used[principled ? NODE_PRINCIPLED_VOLUME
                                             : NODE_VOLUME_COEFFICIENTS],
            "native volume opcode was not recorded");
    if (principled) {
      require(image.metadata.has_volume_attribute_dependency,
              "Principled Volume attribute dependency was lost");
      require(requests(image, ATTR_STD_GENERATED_TRANSFORM),
              "generated transform missing");
      if (name.ends_with("Named")) {
        for (auto attribute :
             {"probe_density", "probe_color", "probe_temperature"}) {
          require(requests(image, ATTR_STD_NONE, attribute),
                  "symbolic attribute name lost");
        }
        require(!requests(image, ATTR_STD_VOLUME_DENSITY) &&
                    !requests(image, ATTR_STD_VOLUME_TEMPERATURE),
                "custom grids silently replaced by default grids");
      } else {
        require(requests(image, ATTR_STD_VOLUME_DENSITY) &&
                    requests(image, ATTR_STD_VOLUME_TEMPERATURE) &&
                    !requests(image, ATTR_STD_VOLUME_COLOR),
                "default density/temperature/empty color requests changed");
      }
    }
    std::cout << name << ": " << size << " exact Cycles words\n";
  }
  std::string trailing;
  require(!(oracle >> trailing), "trailing external fixture data");
}
} // namespace

int main() {
  try {
    run("volume_coefficients_svm", 7u);
    run("principled_volume_svm", 1u);
    run("principled_volume_named_svm", 1u);
    run("principled_volume_linked_svm", 1u);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
