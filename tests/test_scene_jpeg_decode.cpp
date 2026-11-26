#include "path_tracer_image_decode.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
std::vector<std::uint8_t> read(const std::filesystem::path &path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) { throw std::runtime_error("Cannot read fixture " + path.string()); }
  return {std::istreambuf_iterator<char>{input}, {}};
}

void expect(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void test_texture_limit_downscale() {
  using namespace psycles::contract;
  using namespace psycles::io;
  using namespace psycles::luisa_backend::detail;

  expect(compute_texture_limit_extent(4096u, 8192u, 2048u) ==
             std::pair<std::uint32_t, std::uint32_t>{1024u, 2048u},
         "4096x8192 limit 2048 extent mismatch");
  expect(compute_texture_limit_extent(257u, 257u, 128u) ==
             std::pair<std::uint32_t, std::uint32_t>{64u, 64u},
         "257x257 limit 128 extent mismatch");
  expect(compute_texture_limit_extent(256u, 256u, 0u) ==
             std::pair<std::uint32_t, std::uint32_t>{256u, 256u},
         "unlimited texture extent mismatch");

  // Unorm8 straight alpha premul-before-resize + raw-byte box rounding.
  DecodedImageRgba img_u8{
      .width = 256u,
      .height = 256u,
      .storage = DecodedImageStorage::unorm8,
      .unorm8_pixels = std::vector<std::uint8_t>(256u * 256u * 4u, 0u)};
  // Cell (0, 0): (255,255,255,255) + (255,255,255,0) -> premul puts 255 once -> round(255/4) = 64.
  img_u8.unorm8_pixels[0u] = 255u;
  img_u8.unorm8_pixels[1u] = 255u;
  img_u8.unorm8_pixels[2u] = 255u;
  img_u8.unorm8_pixels[3u] = 255u;
  img_u8.unorm8_pixels[4u] = 255u;
  img_u8.unorm8_pixels[5u] = 255u;
  img_u8.unorm8_pixels[6u] = 255u;
  img_u8.unorm8_pixels[7u] = 0u;
  // Cell (1, 0) (x=2..3, y=0..1): two 255 red pixels -> (510 + 2)/4 = 128.
  for (const std::size_t px : {2u, 3u}) {
    img_u8.unorm8_pixels[px * 4u + 0u] = 255u;
    img_u8.unorm8_pixels[px * 4u + 3u] = 255u;
  }
  apply_scene_image_alpha(img_u8, ImageAlphaType::straight, ImageColorSpace::srgb);
  downscale_scene_image_to_limit(img_u8, 128u);
  expect(img_u8.width == 128u && img_u8.height == 128u,
         "256x256 downscale dimensions mismatch");
  expect(img_u8.unorm8_pixels[0u] == 64u && img_u8.unorm8_pixels[3u] == 64u,
         "unorm8 premul-before-resize or rounding mismatch");
  expect(img_u8.unorm8_pixels[4u] == 128u,
         "unorm8 box rounding (510/4 -> 128) mismatch");

  // Odd 255x255 float32 ramp -> 127x127 (x=254 excluded, top-down y=0 excluded, bottom y=254 paired).
  DecodedImageRgba img_f32{
      .width = 255u,
      .height = 255u,
      .storage = DecodedImageStorage::float32,
      .float_pixels = std::vector<float>(255u * 255u * 4u, 0.0f)};
  for (std::uint32_t y = 0u; y < 255u; ++y) {
    for (std::uint32_t x = 0u; x < 255u; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * 255u + x) * 4u;
      img_f32.float_pixels[off + 0u] = static_cast<float>(x) / 255.0f;
      img_f32.float_pixels[off + 1u] =
          (x == 254u || (y == 254u && x < 2u)) ? 1.0f : 0.0f;
      img_f32.float_pixels[off + 2u] = (y == 0u) ? 1.0f : 0.0f;
      img_f32.float_pixels[off + 3u] = 1.0f;
    }
  }
  downscale_scene_image_to_limit(img_f32, 128u);
  expect(img_f32.width == 127u && img_f32.height == 127u,
         "255x255 downscale dimensions mismatch");
  expect(std::abs(img_f32.float_pixels[0u] - (0.5f / 255.0f)) < 1.0e-6f &&
             img_f32.float_pixels[2u] == 0.0f,
         "255x255 float ramp col 0 or dropped top row B mismatch");
  const std::size_t last_off = 126u * 4u;
  expect(std::abs(img_f32.float_pixels[last_off + 0u] - (252.5f / 255.0f)) < 1.0e-6f &&
             img_f32.float_pixels[last_off + 1u] == 0.0f,
         "255x255 float trailing column exclusion mismatch");
  const std::size_t bottom_row_off = (126u * 127u) * 4u;
  expect(std::abs(img_f32.float_pixels[bottom_row_off + 1u] - 0.5f) < 1.0e-6f,
         "255x255 bottom row pairing G mismatch");

  // No-op when under limit or limit == 0.
  downscale_scene_image_to_limit(img_f32, 0u);
  downscale_scene_image_to_limit(img_f32, 128u);
  expect(img_f32.width == 127u && img_f32.height == 127u,
         "downscale should be no-op under limit or unlimited");
}
} // namespace

int main() {
  test_texture_limit_downscale();
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
