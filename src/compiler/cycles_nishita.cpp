#include <psycles/compiler/cycles_nishita.h>

#include "sky_nishita.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace psycles::compiler::cycles_svm {

NishitaSunData precompute_nishita_sun(const NishitaImageBinding &image,
                                      float angular_diameter) {
  std::array<float, 3u> bottom{};
  std::array<float, 3u> top{};
  if (image.multiple_scattering) {
    SKY_multiple_scattering_precompute_sun(
        image.sun_elevation(), angular_diameter, image.altitude(),
        image.air_density(), image.aerosol_density(), image.ozone_density(),
        bottom.data(), top.data());
  } else {
    SKY_single_scattering_precompute_sun(
        image.sun_elevation(), angular_diameter, image.altitude(),
        image.air_density(), image.aerosol_density(), bottom.data(),
        top.data());
  }
  return {.pixel_bottom = {bottom[0u], bottom[1u], bottom[2u]},
          .pixel_top = {top[0u], top[1u], top[2u]},
          .earth_intersection_angle =
              -SKY_earth_intersection_angle(image.altitude())};
}

std::vector<float>
precompute_nishita_texture(const NishitaImageBinding &image) {
  constexpr auto pixel_count =
      nishita_texture_width * nishita_texture_height;
  constexpr auto source_channels = std::size_t{3u};
  std::vector<float> source(pixel_count * source_channels);

  // SkyLoader produces texture-space rows: row zero is the lower hemisphere
  // and the last row approaches the zenith. Psycles' ordinary decoded images
  // are stored top-down, and the shared native image sampler accounts for that
  // convention by mapping v to 1-v. Reverse the generated rows once on the
  // host so Nishita uses the same storage contract without a shader-side
  // special case or a second texture-sampling implementation.
  if (image.multiple_scattering) {
    SKY_multiple_scattering_precompute_texture(
        source.data(), static_cast<int>(source_channels),
        static_cast<int>(nishita_texture_width),
        static_cast<int>(nishita_texture_height), image.sun_elevation(),
        image.altitude(), image.air_density(), image.aerosol_density(),
        image.ozone_density());
  } else {
    SKY_single_scattering_precompute_texture(
        source.data(), static_cast<int>(source_channels),
        static_cast<int>(nishita_texture_width),
        static_cast<int>(nishita_texture_height), image.sun_elevation(),
        image.altitude(), image.air_density(), image.aerosol_density(),
        image.ozone_density());
  }

  std::vector<float> pixels(pixel_count * nishita_texture_channels);
  for (auto source_y = std::size_t{};
       source_y < nishita_texture_height; ++source_y) {
    const auto destination_y =
        nishita_texture_height - source_y - 1u;
    for (auto x = std::size_t{}; x < nishita_texture_width; ++x) {
      const auto source_offset =
          (source_y * nishita_texture_width + x) * source_channels;
      const auto destination =
          (destination_y * nishita_texture_width + x) *
          nishita_texture_channels;
      const auto finite = std::isfinite(source[source_offset]) &&
                          std::isfinite(source[source_offset + 1u]) &&
                          std::isfinite(source[source_offset + 2u]);
      pixels[destination] = finite ? source[source_offset] : 0.0f;
      pixels[destination + 1u] =
          finite ? source[source_offset + 1u] : 0.0f;
      pixels[destination + 2u] =
          finite ? source[source_offset + 2u] : 0.0f;
      pixels[destination + 3u] = finite ? 1.0f : 0.0f;
    }
  }
  return pixels;
}

} // namespace psycles::compiler::cycles_svm
