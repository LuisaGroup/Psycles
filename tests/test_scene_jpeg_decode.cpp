#include "path_tracer_image_decode.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <vector>

namespace {
std::vector<std::uint8_t> read(const std::filesystem::path &path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) { throw std::runtime_error("Cannot read fixture " + path.string()); }
  return {std::istreambuf_iterator<char>{input}, {}};
}
} // namespace

int main() {
  const auto directory = std::filesystem::path{PSYCLES_JPEG_ORACLE_DIRECTORY};
  const auto encoded = read(directory / "blender_jpeg_decode.jpg");
  const auto expected = read(directory / "blender_jpeg_decode.rgba");
  if (expected.size() != 32u * 24u * 4u) { return 1; }
  // A resource may have no suffix or an unhelpful file name. Select the
  // Blender-compatible JPEG decoder from the encoded magic, not the hint.
  for (const std::string_view hint : {"texture.jpg", "texture.bin", ""}) {
    const auto image = psycles::luisa_backend::detail::decode_scene_image(encoded, hint);
    if (!image || image->width != 32u || image->height != 24u ||
        image->storage != psycles::io::DecodedImageStorage::unorm8 ||
        image->unorm8_pixels != expected) {
      std::cerr << "JPEG bytes differ from Blender 5.2.1 (hint '" << hint << "')\n";
      return 1;
    }
  }
  std::cout << "Blender JPEG decoder: 3072 exact bytes, three resource hints passed\n";
}
