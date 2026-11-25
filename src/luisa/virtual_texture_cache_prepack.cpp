#include "virtual_texture_cache_prepack.h"

#include "virtual_texture_cache_formats.h"
#include "virtual_texture_cache_gpu_bc.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <thread>

#include <luisa/core/stl/hash.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::array<char, 8u> kVtPagesMagic{'P', 'S', 'V', 'T',
                                                    'P', 'G', '0', '1'};

struct VtPagesFileHeader {
  char magic[8]{};
  std::uint32_t version{};
  std::uint32_t header_bytes{};
  std::uint64_t content_hash_lo{};
  std::uint64_t content_hash_hi{};
  std::uint64_t encoded_byte_size{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint8_t storage{};
  std::uint8_t alpha_type{};
  std::uint8_t color_space{};
  std::uint8_t compression{};
  std::uint32_t extension{};
  std::uint32_t physical_tile_size{};
  std::uint32_t tile_border{};
  std::uint32_t tiles_x{};
  std::uint32_t tiles_y{};
  std::uint32_t tile_bytes{};
  std::uint32_t flags{};
};

static_assert(sizeof(VtPagesFileHeader) <=
              VtPrepackedDiskStore::kHeaderAlignedBytes);

[[nodiscard]] std::uint32_t compute_tile_bytes(
    const VtPrepackedImageKey &img_key) noexcept {
  return vt_tile_format_bytes(
      img_key.physical_tile_size,
      vt_tile_format_from_disk_key(img_key.source.storage, img_key.compression));
}

void close_open_handle(VtPrepackedDiskStore::OpenFileHandle &h) noexcept {
#ifdef _WIN32
  if (h.os_handle != nullptr && h.os_handle != INVALID_HANDLE_VALUE) {
    ::CloseHandle(static_cast<HANDLE>(h.os_handle));
  }
  h.os_handle = nullptr;
#else
  if (h.posix_fd >= 0) {
    ::close(h.posix_fd);
  }
  h.posix_fd = -1;
#endif
}

#ifdef _WIN32
struct ThreadLocalIoEvent {
  HANDLE event{nullptr};
  ThreadLocalIoEvent() noexcept {
    event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  }
  ~ThreadLocalIoEvent() noexcept {
    if (event != nullptr) {
      ::CloseHandle(event);
      event = nullptr;
    }
  }
  ThreadLocalIoEvent(const ThreadLocalIoEvent &) = delete;
  ThreadLocalIoEvent &operator=(const ThreadLocalIoEvent &) = delete;
};

[[nodiscard]] HANDLE get_thread_io_event() noexcept {
  thread_local ThreadLocalIoEvent tl_ev;
  return tl_ev.event;
}
#endif

} // namespace

std::size_t VtPrepackedImageKeyHash::operator()(
    const VtPrepackedImageKey &k) const noexcept {
  const std::array<std::uint64_t, 7u> words{
      k.source.content_hash_lo,
      k.source.content_hash_hi,
      k.source.encoded_byte_size,
      (static_cast<std::uint64_t>(k.source.width) << 32u) | k.source.height,
      static_cast<std::uint64_t>(k.source.storage) |
          (static_cast<std::uint64_t>(k.source.alpha_type) << 8u) |
          (static_cast<std::uint64_t>(k.source.color_space) << 16u),
      (static_cast<std::uint64_t>(k.extension) << 32u) | k.physical_tile_size,
      (static_cast<std::uint64_t>(k.compression) << 32u) |
          ((static_cast<std::uint64_t>(k.mip_level) << 16u) ^ k.tile_border),
  };
  return static_cast<std::size_t>(luisa::hash64(
      words.data(), words.size() * sizeof(std::uint64_t),
      luisa::hash64_default_seed));
}

std::size_t VtPrepackedPageKeyHash::operator()(
    const VtPrepackedPageKey &k) const noexcept {
  const std::uint64_t img_h = VtPrepackedImageKeyHash{}(k.image);
  const std::array<std::uint64_t, 2u> words{
      img_h,
      (static_cast<std::uint64_t>(k.tile_x) << 32u) | k.tile_y,
  };
  return static_cast<std::size_t>(luisa::hash64(
      words.data(), words.size() * sizeof(std::uint64_t),
      0x9e3779b97f4a7c15ull));
}

VtSourceIdentityKey vt_make_source_identity_key(
    std::span<const std::uint8_t> encoded_data,
    std::uint32_t width,
    std::uint32_t height,
    io::DecodedImageStorage storage,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space) noexcept {
  VtSourceIdentityKey key{};
  key.encoded_byte_size = static_cast<std::uint64_t>(encoded_data.size());
  key.width = width;
  key.height = height;
  key.storage = static_cast<std::uint8_t>(storage);
  key.alpha_type = static_cast<std::uint8_t>(alpha_type);
  key.color_space = static_cast<std::uint8_t>(color_space);
  if (!encoded_data.empty()) {
    key.content_hash_lo = luisa::hash64(
        encoded_data.data(), encoded_data.size(), 0x243f6a8885a308d3ull);
    key.content_hash_hi = luisa::hash64(
        encoded_data.data(), encoded_data.size(), 0x13198a2e03707344ull);
  } else {
    const std::array<std::uint64_t, 2u> fallback{
        (static_cast<std::uint64_t>(width) << 32u) | height,
        static_cast<std::uint64_t>(key.storage) |
            (static_cast<std::uint64_t>(key.alpha_type) << 8u) |
            (static_cast<std::uint64_t>(key.color_space) << 16u),
    };
    key.content_hash_lo = luisa::hash64(
        fallback.data(), sizeof(fallback), 0x243f6a8885a308d3ull);
    key.content_hash_hi = luisa::hash64(
        fallback.data(), sizeof(fallback), 0x13198a2e03707344ull);
  }
  return key;
}

std::int32_t vt_wrap_index(
    std::int32_t coord,
    std::int32_t extent,
    std::uint32_t ext_mode) noexcept {
  switch (ext_mode) {
    case 0u: {
      std::int32_t m = coord % extent;
      if (m < 0) {
        m += extent;
      }
      return m;
    }
    case 1u: {
      if (coord < 0 || coord >= extent) {
        return -1;
      }
      return coord;
    }
    case 2u: {
      return std::clamp<std::int32_t>(coord, 0, extent - 1);
    }
    case 3u: {
      const std::int32_t period = extent * 2;
      std::int32_t m = coord % period;
      if (m < 0) {
        m += period;
      }
      if (m >= extent) {
        m = period - 1 - m;
      }
      return std::clamp<std::int32_t>(m, 0, extent - 1);
    }
    default:
      return std::clamp<std::int32_t>(coord, 0, extent - 1);
  }
}

void vt_pack_byte4_tile(
    const std::uint8_t *pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t tile_x,
    std::uint32_t tile_y,
    std::uint32_t extension,
    std::uint32_t physical_tile_size,
    std::uint32_t logical_tile_size,
    std::uint32_t border,
    std::span<luisa::uint> dst_tile) noexcept {
  if (pixels == nullptr || width == 0u || height == 0u) {
    std::fill(dst_tile.begin(), dst_tile.end(), 0u);
    return;
  }
  const auto w = static_cast<std::int32_t>(width);
  const auto h = static_cast<std::int32_t>(height);
  const auto base_x =
      static_cast<std::int32_t>(tile_x * logical_tile_size) -
      static_cast<std::int32_t>(border);
  const auto base_y =
      static_cast<std::int32_t>(tile_y * logical_tile_size) -
      static_cast<std::int32_t>(border);
  const auto P = static_cast<std::int32_t>(physical_tile_size);

  const bool interior_x = (base_x >= 0) && (base_x + P <= w);
  const bool interior_y = (base_y >= 0) && (base_y + P <= h);
  const std::size_t row_bytes = static_cast<std::size_t>(P) * 4u;

  if (interior_x && interior_y) {
    for (std::int32_t dy = 0; dy < P; ++dy) {
      const auto sy = static_cast<std::size_t>(base_y + dy);
      const auto sx = static_cast<std::size_t>(base_x);
      const std::uint8_t *src_row =
          pixels + (sy * static_cast<std::size_t>(width) + sx) * 4u;
      std::memcpy(dst_tile.data() + static_cast<std::size_t>(dy) * P, src_row,
                  row_bytes);
    }
    return;
  }

  std::array<std::int32_t, 128u> sx_table{};
  for (std::int32_t dx = 0; dx < P; ++dx) {
    sx_table[static_cast<std::size_t>(dx)] =
        vt_wrap_index(base_x + dx, w, extension);
  }
  for (std::int32_t dy = 0; dy < P; ++dy) {
    const auto sy = vt_wrap_index(base_y + dy, h, extension);
    luisa::uint *dst_row =
        dst_tile.data() + static_cast<std::size_t>(dy) * P;
    if (sy < 0) {
      std::memset(dst_row, 0, row_bytes);
      continue;
    }
    const std::uint8_t *src_row =
        pixels + static_cast<std::size_t>(sy) * width * 4u;
    if (interior_x) {
      std::memcpy(dst_row, src_row + static_cast<std::size_t>(base_x) * 4u,
                  row_bytes);
    } else {
      for (std::int32_t dx = 0; dx < P; ++dx) {
        const auto sx = sx_table[static_cast<std::size_t>(dx)];
        if (sx < 0) {
          dst_row[static_cast<std::size_t>(dx)] = 0u;
        } else {
          std::uint32_t packed = 0u;
          std::memcpy(&packed, src_row + static_cast<std::size_t>(sx) * 4u, 4u);
          dst_row[static_cast<std::size_t>(dx)] = packed;
        }
      }
    }
  }
}

void vt_pack_float4_tile(
    const float *pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t tile_x,
    std::uint32_t tile_y,
    std::uint32_t extension,
    std::uint32_t physical_tile_size,
    std::uint32_t logical_tile_size,
    std::uint32_t border,
    std::span<luisa::float4> dst_tile) noexcept {
  if (pixels == nullptr || width == 0u || height == 0u) {
    std::fill(dst_tile.begin(), dst_tile.end(), luisa::make_float4(0.0f));
    return;
  }
  const auto w = static_cast<std::int32_t>(width);
  const auto h = static_cast<std::int32_t>(height);
  const auto base_x =
      static_cast<std::int32_t>(tile_x * logical_tile_size) -
      static_cast<std::int32_t>(border);
  const auto base_y =
      static_cast<std::int32_t>(tile_y * logical_tile_size) -
      static_cast<std::int32_t>(border);
  const auto P = static_cast<std::int32_t>(physical_tile_size);

  const bool interior_x = (base_x >= 0) && (base_x + P <= w);
  const bool interior_y = (base_y >= 0) && (base_y + P <= h);
  const std::size_t row_bytes =
      static_cast<std::size_t>(P) * sizeof(luisa::float4);

  if (interior_x && interior_y) {
    for (std::int32_t dy = 0; dy < P; ++dy) {
      const auto sy = static_cast<std::size_t>(base_y + dy);
      const auto sx = static_cast<std::size_t>(base_x);
      const float *src_row =
          pixels + (sy * static_cast<std::size_t>(width) + sx) * 4u;
      std::memcpy(dst_tile.data() + static_cast<std::size_t>(dy) * P, src_row,
                  row_bytes);
    }
    return;
  }

  std::array<std::int32_t, 128u> sx_table{};
  for (std::int32_t dx = 0; dx < P; ++dx) {
    sx_table[static_cast<std::size_t>(dx)] =
        vt_wrap_index(base_x + dx, w, extension);
  }
  for (std::int32_t dy = 0; dy < P; ++dy) {
    const auto sy = vt_wrap_index(base_y + dy, h, extension);
    luisa::float4 *dst_row =
        dst_tile.data() + static_cast<std::size_t>(dy) * P;
    if (sy < 0) {
      std::memset(dst_row, 0, row_bytes);
      continue;
    }
    const float *src_row =
        pixels + static_cast<std::size_t>(sy) * width * 4u;
    if (interior_x) {
      std::memcpy(dst_row, src_row + static_cast<std::size_t>(base_x) * 4u,
                  row_bytes);
    } else {
      for (std::int32_t dx = 0; dx < P; ++dx) {
        const auto sx = sx_table[static_cast<std::size_t>(dx)];
        if (sx < 0) {
          dst_row[static_cast<std::size_t>(dx)] = luisa::make_float4(0.0f);
        } else {
          const float *p = src_row + static_cast<std::size_t>(sx) * 4u;
          dst_row[static_cast<std::size_t>(dx)] =
              luisa::make_float4(p[0], p[1], p[2], p[3]);
        }
      }
    }
  }
}

void VtPrepackedPageCache::ImageIndex::init_grid(
    const VtPrepackedImageKey &img_key) {
  const std::uint32_t P = img_key.physical_tile_size;
  const std::uint32_t B = img_key.tile_border;
  const std::uint32_t mip_w =
      img_key.mip_level >= 31u
          ? 1u
          : std::max(1u, img_key.source.width >> img_key.mip_level);
  const std::uint32_t mip_h =
      img_key.mip_level >= 31u
          ? 1u
          : std::max(1u, img_key.source.height >> img_key.mip_level);
  if (P > 2u * B && img_key.source.width > 0u && img_key.source.height > 0u) {
    const std::uint32_t logical = P - 2u * B;
    tiles_x = std::max(1u, (mip_w + logical - 1u) / logical);
    tiles_y = std::max(1u, (mip_h + logical - 1u) / logical);
  } else {
    tiles_x = 1u;
    tiles_y = 1u;
  }
  tile_to_slot.assign(static_cast<std::size_t>(tiles_x) * tiles_y,
                      kInvalidSlot32);
}

void VtPrepackedPageCache::ImageIndex::ensure_tile_capacity(
    std::uint32_t tx,
    std::uint32_t ty) {
  if (tx < tiles_x && ty < tiles_y) {
    return;
  }
  const std::uint32_t new_tx = std::max(tiles_x, tx + 1u);
  const std::uint32_t new_ty = std::max(tiles_y, ty + 1u);
  std::vector<std::uint32_t> grown(
      static_cast<std::size_t>(new_tx) * new_ty, kInvalidSlot32);
  for (std::uint32_t y = 0u; y < tiles_y; ++y) {
    std::memcpy(grown.data() + static_cast<std::size_t>(y) * new_tx,
                tile_to_slot.data() + static_cast<std::size_t>(y) * tiles_x,
                static_cast<std::size_t>(tiles_x) * sizeof(std::uint32_t));
  }
  tile_to_slot.swap(grown);
  tiles_x = new_tx;
  tiles_y = new_ty;
}

VtPrepackedPageCache::ImageIndex *
VtPrepackedPageCache::get_or_create_image_locked(
    const VtPrepackedImageKey &img_key) {
  const auto it = _images.find(img_key);
  if (it != _images.end()) {
    return it->second.get();
  }
  auto idx = std::make_unique<ImageIndex>();
  idx->init_grid(img_key);
  ImageIndex *raw = idx.get();
  _images.emplace(img_key, std::move(idx));
  return raw;
}

std::uint8_t *VtPrepackedPageCache::acquire_block_locked(
    std::size_t size_bytes) {
  if (size_bytes == 0u) {
    return nullptr;
  }
  SlabPool *pool = nullptr;
  for (auto &s : _slabs) {
    if (s.tile_bytes == size_bytes) {
      pool = &s;
      break;
    }
  }
  if (pool == nullptr) {
    _slabs.push_back(SlabPool{.tile_bytes = size_bytes});
    pool = &_slabs.back();
  }
  if (pool->free_blocks.empty()) {
    const std::size_t max_budget_tiles = std::max<std::size_t>(
        1u, static_cast<std::size_t>(_budget_bytes / size_bytes));
    const std::size_t target_4mib_tiles =
        std::max<std::size_t>(1u, (4ull * 1024ull * 1024ull) / size_bytes);
    const std::size_t chunk_tiles =
        std::clamp<std::size_t>(max_budget_tiles, 1u, target_4mib_tiles);
    auto chunk =
        std::unique_ptr<std::uint8_t[]>(new std::uint8_t[chunk_tiles * size_bytes]);
    std::uint8_t *base = chunk.get();
    pool->free_blocks.reserve(pool->free_blocks.size() + chunk_tiles);
    for (std::size_t i = chunk_tiles; i > 0u; --i) {
      pool->free_blocks.push_back(base + (i - 1u) * size_bytes);
    }
    pool->chunks.push_back(std::move(chunk));
  }
  std::uint8_t *ptr = pool->free_blocks.back();
  pool->free_blocks.pop_back();
  return ptr;
}

void VtPrepackedPageCache::release_block_locked(
    std::uint8_t *ptr,
    std::size_t size_bytes) noexcept {
  if (ptr == nullptr || size_bytes == 0u) {
    return;
  }
  for (auto &s : _slabs) {
    if (s.tile_bytes == size_bytes) {
      s.free_blocks.push_back(ptr);
      return;
    }
  }
}

bool VtPrepackedPageCache::evict_one_unpinned_locked() noexcept {
  const std::size_t n = _entries.size();
  if (n == 0u) {
    return false;
  }
  if (_clock_hand >= n) {
    _clock_hand = 0u;
  }
  const std::size_t max_steps = n * 2u;
  for (std::size_t visited = 0u; visited < max_steps; ++visited) {
    const std::size_t idx = _clock_hand;
    _clock_hand = (_clock_hand + 1u == n) ? 0u : (_clock_hand + 1u);
    auto &entry = _entries[idx];
    if (entry.active == 0u || entry.pin_count > 0u) {
      continue;
    }
    if (entry.referenced != 0u) {
      entry.referenced = 0u;
      continue;
    }
    if (entry.owner_img != nullptr) {
      const std::size_t flat_idx =
          static_cast<std::size_t>(entry.owner_ty) * entry.owner_img->tiles_x +
          entry.owner_tx;
      if (flat_idx < entry.owner_img->tile_to_slot.size()) {
        entry.owner_img->tile_to_slot[flat_idx] = kInvalidSlot32;
      }
    }
    _resident_bytes = (_resident_bytes >= entry.size_bytes)
                          ? (_resident_bytes - entry.size_bytes)
                          : 0ull;
    release_block_locked(entry.data, entry.size_bytes);
    entry.owner_img = nullptr;
    entry.owner_tx = 0u;
    entry.owner_ty = 0u;
    entry.data = nullptr;
    entry.size_bytes = 0u;
    entry.active = 0u;
    entry.populated = 0u;
    entry.referenced = 0u;
    _free_entry_indices.push_back(idx);
    if (_entry_count > 0u) {
      --_entry_count;
    }
    ++_evictions;
    return true;
  }
  return false;
}

void VtPrepackedPageCache::set_budget_bytes(
    std::uint64_t budget_bytes) noexcept {
  std::lock_guard lock{_mutex};
  _budget_bytes = budget_bytes;
  while (_resident_bytes > _budget_bytes) {
    if (!evict_one_unpinned_locked()) {
      break;
    }
  }
}

void VtPrepackedPageCache::clear() noexcept {
  std::lock_guard lock{_mutex};
  _images.clear();
  _entries.clear();
  _free_entry_indices.clear();
  _slabs.clear();
  _resident_bytes = 0ull;
  _entry_count = 0u;
  _clock_hand = 0u;
}

void VtPrepackedPageCache::resolve_batch_common(
    std::span<const VtPrepackedPageKey> keys,
    const std::uint32_t *opt_indices,
    std::size_t count,
    std::size_t tile_bytes,
    std::vector<ResolvedTileSlot> &out) noexcept {
  out.resize(count);
  std::fill(out.begin(), out.end(), ResolvedTileSlot{});
  std::lock_guard lock{_mutex};
  const VtPrepackedImageKey *last_img_key = nullptr;
  ImageIndex *last_img = nullptr;

  for (std::size_t i = 0u; i < count; ++i) {
    const std::size_t key_idx = opt_indices != nullptr ? opt_indices[i] : i;
    const auto &key = keys[key_idx];
    ImageIndex *img = nullptr;
    if (last_img_key != nullptr && *last_img_key == key.image) {
      img = last_img;
    } else {
      img = get_or_create_image_locked(key.image);
      last_img_key = &key.image;
      last_img = img;
    }
    img->ensure_tile_capacity(key.tile_x, key.tile_y);
    const std::size_t flat_idx =
        static_cast<std::size_t>(key.tile_y) * img->tiles_x + key.tile_x;
    const std::uint32_t existing_slot = img->tile_to_slot[flat_idx];
    if (existing_slot != kInvalidSlot32) {
      const std::size_t entry_idx = static_cast<std::size_t>(existing_slot);
      auto &entry = _entries[entry_idx];
      if (entry.populated != 0u) {
        entry.referenced = 1u;
        ++entry.pin_count;
        ++_hits;
        out[i] = ResolvedTileSlot{
            .hit_ptr = entry.data,
            .fill_ptr = nullptr,
            .bytes = entry.size_bytes,
            .entry_index = entry_idx,
            .duplicate_of = -1,
            .is_hit = true};
        continue;
      }
      // Duplicate miss within the same resolve batch: find owner slot.
      std::int32_t owner_slot = -1;
      for (std::size_t prev = 0u; prev < i; ++prev) {
        if (out[prev].entry_index == entry_idx && !out[prev].is_hit &&
            out[prev].duplicate_of < 0) {
          owner_slot = static_cast<std::int32_t>(prev);
          break;
        }
      }
      ++_misses;
      out[i] = ResolvedTileSlot{
          .hit_ptr = nullptr,
          .fill_ptr = nullptr,
          .bytes = tile_bytes,
          .entry_index = kInvalidSlot,
          .duplicate_of = owner_slot,
          .is_hit = false};
      continue;
    }

    ++_misses;
    if (tile_bytes == 0u || tile_bytes > _budget_bytes) {
      continue;
    }
    while (_resident_bytes + tile_bytes > _budget_bytes) {
      if (!evict_one_unpinned_locked()) {
        break;
      }
    }
    if (_resident_bytes + tile_bytes > _budget_bytes) {
      continue;
    }

    std::size_t entry_idx = kInvalidSlot;
    if (!_free_entry_indices.empty()) {
      entry_idx = _free_entry_indices.back();
      _free_entry_indices.pop_back();
    } else {
      entry_idx = _entries.size();
      _entries.emplace_back();
    }
    std::uint8_t *blk = acquire_block_locked(tile_bytes);
    auto &entry = _entries[entry_idx];
    entry.owner_img = img;
    entry.owner_tx = key.tile_x;
    entry.owner_ty = key.tile_y;
    entry.data = blk;
    entry.size_bytes = tile_bytes;
    entry.pin_count = 1u;
    entry.active = 1u;
    entry.populated = 0u;
    entry.referenced = 1u;
    img->tile_to_slot[flat_idx] = static_cast<std::uint32_t>(entry_idx);
    ++_entry_count;
    _resident_bytes += tile_bytes;
    out[i] = ResolvedTileSlot{
        .hit_ptr = nullptr,
        .fill_ptr = blk,
        .bytes = tile_bytes,
        .entry_index = entry_idx,
        .duplicate_of = -1,
        .is_hit = false};
  }
}

void VtPrepackedPageCache::resolve_batch(
    std::span<const VtPrepackedPageKey> keys,
    std::size_t tile_bytes,
    std::vector<ResolvedTileSlot> &out) noexcept {
  resolve_batch_common(keys, nullptr, keys.size(), tile_bytes, out);
}

void VtPrepackedPageCache::resolve_batch_indexed(
    std::span<const VtPrepackedPageKey> all_keys,
    std::span<const std::uint32_t> indices,
    std::size_t tile_bytes,
    std::vector<ResolvedTileSlot> &out) noexcept {
  resolve_batch_common(all_keys, indices.data(), indices.size(), tile_bytes,
                       out);
}

void VtPrepackedPageCache::unpin_batch(
    std::span<const ResolvedTileSlot> slots) noexcept {
  std::lock_guard lock{_mutex};
  for (const auto &s : slots) {
    if (s.entry_index == kInvalidSlot || s.entry_index >= _entries.size()) {
      continue;
    }
    auto &entry = _entries[s.entry_index];
    if (entry.active == 0u) {
      continue;
    }
    if (!s.is_hit) {
      entry.populated = 1u;
    }
    if (entry.pin_count > 0u) {
      --entry.pin_count;
    }
  }
}

bool VtPrepackedPageCache::insert_warm_tile(
    const VtPrepackedPageKey &key,
    std::span<const std::uint8_t> bytes,
    bool evict_if_needed) noexcept {
  if (bytes.empty()) {
    return false;
  }
  std::lock_guard lock{_mutex};
  ImageIndex *img = get_or_create_image_locked(key.image);
  img->ensure_tile_capacity(key.tile_x, key.tile_y);
  const std::size_t flat_idx =
      static_cast<std::size_t>(key.tile_y) * img->tiles_x + key.tile_x;
  if (img->tile_to_slot[flat_idx] != kInvalidSlot32) {
    _entries[img->tile_to_slot[flat_idx]].referenced = 1u;
    return true;
  }
  if (bytes.size() > _budget_bytes) {
    return false;
  }
  while (_resident_bytes + bytes.size() > _budget_bytes) {
    if (!evict_if_needed || !evict_one_unpinned_locked()) {
      return false;
    }
  }
  std::size_t entry_idx = kInvalidSlot;
  if (!_free_entry_indices.empty()) {
    entry_idx = _free_entry_indices.back();
    _free_entry_indices.pop_back();
  } else {
    entry_idx = _entries.size();
    _entries.emplace_back();
  }
  std::uint8_t *blk = acquire_block_locked(bytes.size());
  std::memcpy(blk, bytes.data(), bytes.size());
  auto &entry = _entries[entry_idx];
  entry.owner_img = img;
  entry.owner_tx = key.tile_x;
  entry.owner_ty = key.tile_y;
  entry.data = blk;
  entry.size_bytes = bytes.size();
  entry.pin_count = 0u;
  entry.active = 1u;
  entry.populated = 1u;
  entry.referenced = 1u;
  img->tile_to_slot[flat_idx] = static_cast<std::uint32_t>(entry_idx);
  ++_entry_count;
  _resident_bytes += bytes.size();
  return true;
}

VtPrepackedCacheStats VtPrepackedPageCache::stats() const noexcept {
  std::lock_guard lock{_mutex};
  return VtPrepackedCacheStats{
      .hits = _hits,
      .misses = _misses,
      .evictions = _evictions,
      .resident_bytes = _resident_bytes,
      .budget_bytes = _budget_bytes,
      .entry_count = _entry_count};
}

VtPrepackedDiskStore::VtPrepackedDiskStore(std::filesystem::path dir) noexcept
    : _dir{std::move(dir)} {}

VtPrepackedDiskStore::~VtPrepackedDiskStore() noexcept {
  std::lock_guard lock{_mutex};
  for (auto &[key, handle] : _open_files) {
    if (handle) {
      close_open_handle(*handle);
    }
  }
  _open_files.clear();
}

void VtPrepackedDiskStore::set_directory(std::filesystem::path dir) noexcept {
  std::lock_guard lock{_mutex};
  for (auto &[key, handle] : _open_files) {
    if (handle) {
      close_open_handle(*handle);
    }
  }
  _open_files.clear();
  _known_missing.clear();
  _dir = std::move(dir);
}

std::filesystem::path VtPrepackedDiskStore::file_path_for(
    const VtPrepackedImageKey &img_key) const {
  char name[256];
  char comp_buf[24]{};
  if (img_key.compression != 0u) {
    std::snprintf(comp_buf, sizeof(comp_buf), "_bc%u",
                  static_cast<unsigned>(img_key.compression));
  }
  char mip_buf[24]{};
  if (img_key.mip_level != 0u) {
    std::snprintf(mip_buf, sizeof(mip_buf), "_m%u",
                  static_cast<unsigned>(img_key.mip_level));
  }
  std::snprintf(
      name, sizeof(name),
      "vt_%016llx_%016llx_w%u_h%u_s%u_a%u_c%u_e%u_p%u_b%u%s%s.vtpages",
      static_cast<unsigned long long>(img_key.source.content_hash_lo),
      static_cast<unsigned long long>(img_key.source.content_hash_hi),
      img_key.source.width, img_key.source.height,
      static_cast<unsigned>(img_key.source.storage),
      static_cast<unsigned>(img_key.source.alpha_type),
      static_cast<unsigned>(img_key.source.color_space),
      img_key.extension, img_key.physical_tile_size, img_key.tile_border,
      comp_buf, mip_buf);
  return _dir / name;
}

std::shared_ptr<VtPrepackedDiskStore::OpenFileHandle>
VtPrepackedDiskStore::try_open_existing_locked(
    const VtPrepackedImageKey &img_key,
    std::uint32_t tiles_x,
    std::uint32_t tiles_y) noexcept {
  if (_dir.empty()) {
    return nullptr;
  }
  if (const auto it = _open_files.find(img_key); it != _open_files.end()) {
    return it->second;
  }
  if (_known_missing.find(img_key) != _known_missing.end()) {
    return nullptr;
  }
  const auto path = file_path_for(img_key);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    _known_missing.emplace(img_key, 1u);
    return nullptr;
  }
  const std::uint32_t expected_tile_bytes = compute_tile_bytes(img_key);
  const std::uint64_t expected_size =
      static_cast<std::uint64_t>(kHeaderAlignedBytes) +
      static_cast<std::uint64_t>(tiles_x) * tiles_y * expected_tile_bytes;
  const auto fsize = std::filesystem::file_size(path, ec);
  if (ec || fsize != expected_size) {
    _known_missing.emplace(img_key, 1u);
    return nullptr;
  }

  VtPagesFileHeader hdr{};
  {
    std::ifstream ifs{path, std::ios::binary};
    if (!ifs.is_open()) {
      _known_missing.emplace(img_key, 1u);
      return nullptr;
    }
    ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!ifs.good()) {
      _known_missing.emplace(img_key, 1u);
      return nullptr;
    }
  }
  const bool valid_hdr =
      (std::memcmp(hdr.magic, kVtPagesMagic.data(), kVtPagesMagic.size()) ==
       0) &&
      hdr.version == kFormatVersion &&
      hdr.header_bytes == kHeaderAlignedBytes &&
      hdr.content_hash_lo == img_key.source.content_hash_lo &&
      hdr.content_hash_hi == img_key.source.content_hash_hi &&
      hdr.encoded_byte_size == img_key.source.encoded_byte_size &&
      hdr.width == img_key.source.width &&
      hdr.height == img_key.source.height &&
      hdr.storage == img_key.source.storage &&
      hdr.alpha_type == img_key.source.alpha_type &&
      hdr.color_space == img_key.source.color_space &&
      hdr.compression == img_key.compression &&
      hdr.extension == img_key.extension &&
      hdr.physical_tile_size == img_key.physical_tile_size &&
      hdr.tile_border == img_key.tile_border &&
      hdr.tiles_x == tiles_x &&
      hdr.tiles_y == tiles_y &&
      hdr.tile_bytes == expected_tile_bytes &&
      hdr.flags == img_key.mip_level;
  if (!valid_hdr) {
    _known_missing.emplace(img_key, 1u);
    return nullptr;
  }

  auto handle = std::make_shared<OpenFileHandle>();
  handle->tiles_x = tiles_x;
  handle->tiles_y = tiles_y;
  handle->tile_bytes = expected_tile_bytes;
#ifdef _WIN32
  HANDLE h = ::CreateFileW(
      path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
      nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    _known_missing.emplace(img_key, 1u);
    return nullptr;
  }
  handle->os_handle = h;
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    _known_missing.emplace(img_key, 1u);
    return nullptr;
  }
  handle->posix_fd = fd;
#endif
  _open_files.emplace(img_key, handle);
  return handle;
}

bool VtPrepackedDiskStore::has_valid_file(
    const VtPrepackedImageKey &img_key,
    std::uint32_t tiles_x,
    std::uint32_t tiles_y) noexcept {
  std::lock_guard lock{_mutex};
  return try_open_existing_locked(img_key, tiles_x, tiles_y) != nullptr;
}

bool VtPrepackedDiskStore::ensure_image_file(
    const VtPrepackedImageKey &img_key,
    const std::uint8_t *unorm8_pixels,
    const float *float_pixels,
    std::uint32_t tiles_x,
    std::uint32_t tiles_y,
    VtGpuBcCompressor *gpu_bc) noexcept {
  if (_dir.empty() || tiles_x == 0u || tiles_y == 0u) {
    return false;
  }
  {
    std::lock_guard lock{_mutex};
    if (try_open_existing_locked(img_key, tiles_x, tiles_y) != nullptr) {
      return true;
    }
  }
  if (img_key.physical_tile_size <= 2u * img_key.tile_border) {
    return false;
  }
  const std::uint32_t P = img_key.physical_tile_size;
  const std::uint32_t border = img_key.tile_border;
  const std::uint32_t logical = P - 2u * border;
  const VtTileFormat fmt =
      vt_tile_format_from_disk_key(img_key.source.storage, img_key.compression);
  const std::uint32_t tile_bytes = vt_tile_format_bytes(P, fmt);
  const std::size_t total_tiles =
      static_cast<std::size_t>(tiles_x) * tiles_y;
  const bool high_prec = vt_tile_format_is_high_precision(fmt);
  if ((high_prec && float_pixels == nullptr) ||
      (!high_prec && unorm8_pixels == nullptr)) {
    return false;
  }

  const std::uint32_t mip_w =
      img_key.mip_level >= 31u
          ? 1u
          : std::max(1u, img_key.source.width >> img_key.mip_level);
  const std::uint32_t mip_h =
      img_key.mip_level >= 31u
          ? 1u
          : std::max(1u, img_key.source.height >> img_key.mip_level);
  std::vector<std::uint8_t> packed_all(total_tiles * tile_bytes);
  bool packed_by_gpu = false;
  if (gpu_bc != nullptr && gpu_bc->available() &&
      VtGpuBcCompressor::supports_format(fmt) &&
      total_tiles >= gpu_bc->min_batch_tiles()) {
    std::vector<VtGpuBcTileJob> jobs(total_tiles);
    for (std::size_t linear_idx = 0u; linear_idx < total_tiles; ++linear_idx) {
      const std::uint32_t ty = static_cast<std::uint32_t>(linear_idx / tiles_x);
      const std::uint32_t tx = static_cast<std::uint32_t>(linear_idx % tiles_x);
      jobs[linear_idx] = VtGpuBcTileJob{
          .unorm8_pixels = unorm8_pixels,
          .float_pixels = float_pixels,
          .width = mip_w,
          .height = mip_h,
          .tile_x = tx,
          .tile_y = ty,
          .extension = img_key.extension,
          .physical_tile_size = P,
          .logical_tile_size = logical,
          .border = border,
          .format = fmt,
          .dst = packed_all.data() + linear_idx * tile_bytes,
          .dst_bytes = tile_bytes};
    }
    packed_by_gpu = gpu_bc->compress_batch(jobs);
  }
  if (!packed_by_gpu) {
    auto pack_one = [&](std::size_t linear_idx) noexcept {
      const std::uint32_t ty = static_cast<std::uint32_t>(linear_idx / tiles_x);
      const std::uint32_t tx = static_cast<std::uint32_t>(linear_idx % tiles_x);
      std::uint8_t *dst_ptr = packed_all.data() + linear_idx * tile_bytes;
      vt_pack_tile_formatted(
          unorm8_pixels, float_pixels, mip_w, mip_h, tx, ty, img_key.extension,
          P, logical, border, fmt,
          std::span<std::uint8_t>{dst_ptr, tile_bytes});
    };
    if (total_tiles >= 2u) {
      psycles::parallel_for(total_tiles, pack_one);
    } else {
      for (std::size_t i = 0u; i < total_tiles; ++i) {
        pack_one(i);
      }
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(_dir, ec);
  if (ec) {
    return false;
  }

  const auto final_path = file_path_for(img_key);
  static std::atomic<std::uint64_t> tmp_seq{0ull};
  const auto tmp_path = std::filesystem::path{
      final_path.string() + ".tmp." +
      std::to_string(tmp_seq.fetch_add(1ull, std::memory_order_relaxed))};

  std::array<char, kHeaderAlignedBytes> header_block{};
  VtPagesFileHeader hdr{};
  std::memcpy(hdr.magic, kVtPagesMagic.data(), kVtPagesMagic.size());
  hdr.version = kFormatVersion;
  hdr.header_bytes = static_cast<std::uint32_t>(kHeaderAlignedBytes);
  hdr.content_hash_lo = img_key.source.content_hash_lo;
  hdr.content_hash_hi = img_key.source.content_hash_hi;
  hdr.encoded_byte_size = img_key.source.encoded_byte_size;
  hdr.width = img_key.source.width;
  hdr.height = img_key.source.height;
  hdr.storage = img_key.source.storage;
  hdr.alpha_type = img_key.source.alpha_type;
  hdr.color_space = img_key.source.color_space;
  hdr.compression = img_key.compression;
  hdr.extension = img_key.extension;
  hdr.physical_tile_size = P;
  hdr.tile_border = border;
  hdr.tiles_x = tiles_x;
  hdr.tiles_y = tiles_y;
  hdr.tile_bytes = tile_bytes;
  hdr.flags = img_key.mip_level;
  std::memcpy(header_block.data(), &hdr, sizeof(hdr));

  {
    std::ofstream ofs{tmp_path, std::ios::binary | std::ios::trunc};
    if (!ofs.is_open()) {
      return false;
    }
    ofs.write(header_block.data(),
              static_cast<std::streamsize>(header_block.size()));
    ofs.write(reinterpret_cast<const char *>(packed_all.data()),
              static_cast<std::streamsize>(packed_all.size()));
    ofs.flush();
    if (!ofs.good()) {
      ofs.close();
      std::filesystem::remove(tmp_path, ec);
      return false;
    }
  }

  std::filesystem::rename(tmp_path, final_path, ec);
  if (ec) {
    // Another thread or process may have created final_path concurrently.
    std::filesystem::remove(tmp_path, ec);
  }

  std::lock_guard lock{_mutex};
  _known_missing.erase(img_key);
  return try_open_existing_locked(img_key, tiles_x, tiles_y) != nullptr;
}

bool VtPrepackedDiskStore::read_single_page(
    const VtPrepackedPageKey &page_key,
    std::uint32_t tiles_x,
    std::uint32_t tiles_y,
    std::span<std::uint8_t> dst) noexcept {
  if (page_key.tile_x >= tiles_x || page_key.tile_y >= tiles_y) {
    return false;
  }
  std::shared_ptr<OpenFileHandle> handle;
  {
    std::lock_guard lock{_mutex};
    handle = try_open_existing_locked(page_key.image, tiles_x, tiles_y);
  }
  if (!handle || dst.size() < handle->tile_bytes) {
    return false;
  }
  const std::uint64_t linear_page =
      static_cast<std::uint64_t>(page_key.tile_y) * tiles_x + page_key.tile_x;
  const std::uint64_t offset =
      static_cast<std::uint64_t>(kHeaderAlignedBytes) +
      linear_page * handle->tile_bytes;

#ifdef _WIN32
  HANDLE ev = get_thread_io_event();
  if (ev == nullptr || handle->os_handle == nullptr) {
    return false;
  }
  ::ResetEvent(ev);
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(offset & 0xffffffffull);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32u);
  ov.hEvent = ev;
  DWORD bytes_read = 0u;
  BOOL ok = ::ReadFile(static_cast<HANDLE>(handle->os_handle), dst.data(),
                       static_cast<DWORD>(handle->tile_bytes), &bytes_read,
                       &ov);
  if (!ok) {
    if (::GetLastError() != ERROR_IO_PENDING) {
      return false;
    }
    if (!::GetOverlappedResult(static_cast<HANDLE>(handle->os_handle), &ov,
                               &bytes_read, TRUE)) {
      return false;
    }
  }
  return bytes_read == handle->tile_bytes;
#else
  if (handle->posix_fd < 0) {
    return false;
  }
  std::size_t total = 0u;
  while (total < handle->tile_bytes) {
    const ssize_t n = ::pread(
        handle->posix_fd, dst.data() + total, handle->tile_bytes - total,
        static_cast<off_t>(offset + total));
    if (n <= 0) {
      return false;
    }
    total += static_cast<std::size_t>(n);
  }
  return true;
#endif
}

} // namespace psycles::luisa_backend::detail
