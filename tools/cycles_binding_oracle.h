// Diagnostic-only registry observer for the version-pinned original Cycles.
// Include after native scene headers; this contains no shader evaluator.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>

namespace psycles_binding_oracle {

class Writer {
  std::ofstream stream;
  const char *path;

public:
  explicit Writer(const char *output) : stream{output, std::ios::binary | std::ios::trunc}, path{output} {}
  void bytes(const char *data, std::size_t size) { stream.write(data, std::streamsize(size)); }
  void u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32u; shift += 8u) { stream.put(char((value >> shift) & 255u)); }
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64u; shift += 8u) { stream.put(char((value >> shift) & 255u)); }
  }
  void f32(float value) {
    static_assert(sizeof(value) == sizeof(std::uint32_t));
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }
  void text(std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
      stream.setstate(std::ios::failbit);
      return;
    }
    u32(std::uint32_t(value.size()));
    bytes(value.data(), value.size());
  }
  void finish() {
    stream.flush();
    if (!stream) { std::fprintf(stderr, "Psycles binding observer write failed: %s\n", path); }
  }
};

// Called only after the native image task pool and descriptor upload finish.
// Do not load, finalize, assign or convert anything while observing.
template<typename Images, typename Udims>
void images(const Images &images, const Udims &udims) {
  const auto path = std::getenv("PSYCLES_CYCLES_IMAGE_BINDING_DUMP");
  if (path == nullptr || *path == '\0') { return; }
  Writer out{path};
  out.bytes("PSYIMG52", 8u);
  out.u32(1u);
  out.u32(std::uint32_t(images.size()));
  for (auto [index, image] : images.enumerate()) {
    out.u32(std::uint32_t(index));
    out.u32(image != nullptr);
    if (image == nullptr) { continue; }
    const auto &params = image->params;
    const auto &metadata = image->metadata;
    out.u32(std::uint32_t(image->image_texture_id));
    out.u32(unsigned(image->builtin) | (unsigned(image->need_load) << 1u) |
            (unsigned(image->need_metadata) << 2u));
    out.u32(std::uint32_t(params.interpolation));
    out.u32(std::uint32_t(params.extension));
    out.u32(std::uint32_t(params.alpha_type));
    out.u32(unsigned(params.animated));
    out.f32(params.frame);
    out.u32(std::uint32_t(image->miplevel_offset));
    out.u64(std::uint64_t(metadata.width));
    out.u64(std::uint64_t(metadata.height));
    out.u32(std::uint32_t(metadata.channels));
    out.u32(std::uint32_t(metadata.type));
    out.u32(unsigned(metadata.is_compressible_as_srgb) |
            (unsigned(metadata.is_unassociated_alpha) << 1u) |
            (unsigned(metadata.ignore_alpha) << 2u) |
            (unsigned(metadata.is_channel_packed) << 3u) |
            (unsigned(metadata.is_tx_file) << 4u) |
            (unsigned(metadata.has_tiles_and_mipmaps) << 5u));
    out.u32(metadata.tile_size);
    out.text(image->loader->name());
    out.text(params.colorspace.c_str());
    out.text(metadata.colorspace.c_str());
    out.u32(std::uint32_t(image->loader->get_tile_number()));
  }
  out.u32(std::uint32_t(udims.size()));
  for (auto [index, udim] : udims.enumerate()) {
    out.u32(std::uint32_t(index));
    out.u32(udim != nullptr);
    if (udim == nullptr) { continue; }
    out.u32(std::uint32_t(udim->id));
    out.u32(std::uint32_t(udim->tiles.size()));
    for (const auto &[tile, handle] : udim->tiles) {
      out.u32(std::uint32_t(tile));
      out.u32(std::uint32_t(handle.kernel_id()));
    }
  }
  out.finish();
}

// Called after native insertion while its existing attribute_lock_ is held.
template<typename Attributes>
void attributes(const Attributes &attributes) {
  const auto path = std::getenv("PSYCLES_CYCLES_ATTRIBUTE_BINDING_DUMP");
  if (path == nullptr || *path == '\0') { return; }
  Writer out{path};
  out.bytes("PSYATT52", 8u);
  out.u32(1u);
  out.u32(std::uint32_t(attributes.size()));
  for (const auto &[name, id] : attributes) {
    out.u64(id);
    out.text(name.c_str());
  }
  out.finish();
}

} // namespace psycles_binding_oracle
