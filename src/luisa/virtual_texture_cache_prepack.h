#pragma once

#include <psycles/contract/scene.h>
#include <psycles/io/image.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include <luisa/core/basic_types.h>

namespace psycles::luisa_backend::detail {

struct VtSourceIdentityKey {
  std::uint64_t content_hash_lo{};
  std::uint64_t content_hash_hi{};
  std::uint64_t encoded_byte_size{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint8_t storage{};
  std::uint8_t alpha_type{};
  std::uint8_t color_space{};
  std::uint8_t pad{};

  [[nodiscard]] bool operator==(const VtSourceIdentityKey &) const noexcept = default;
};

struct VtPrepackedImageKey {
  VtSourceIdentityKey source{};
  std::uint32_t extension{};
  std::uint32_t physical_tile_size{};
  std::uint32_t tile_border{};
  std::uint8_t compression{0u};
  std::uint32_t mip_level{0u};

  [[nodiscard]] bool operator==(const VtPrepackedImageKey &) const noexcept = default;
};

struct VtPrepackedPageKey {
  VtPrepackedImageKey image{};
  std::uint32_t tile_x{};
  std::uint32_t tile_y{};

  [[nodiscard]] bool operator==(const VtPrepackedPageKey &) const noexcept = default;
};

struct VtPrepackedImageKeyHash {
  [[nodiscard]] std::size_t operator()(
      const VtPrepackedImageKey &k) const noexcept;
};

struct VtPrepackedPageKeyHash {
  [[nodiscard]] std::size_t operator()(
      const VtPrepackedPageKey &k) const noexcept;
};

[[nodiscard]] VtSourceIdentityKey vt_make_source_identity_key(
    std::span<const std::uint8_t> encoded_data,
    std::uint32_t width,
    std::uint32_t height,
    io::DecodedImageStorage storage,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space) noexcept;

[[nodiscard]] std::int32_t vt_wrap_index(
    std::int32_t coord,
    std::int32_t extent,
    std::uint32_t ext_mode) noexcept;

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
    std::span<luisa::uint> dst_tile) noexcept;

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
    std::span<luisa::float4> dst_tile) noexcept;

struct VtPrepackedCacheStats {
  std::uint64_t hits{};
  std::uint64_t misses{};
  std::uint64_t evictions{};
  std::uint64_t resident_bytes{};
  std::uint64_t budget_bytes{};
  std::size_t entry_count{};
};

class VtPrepackedPageCache final {
public:
  static constexpr std::size_t kInvalidSlot = ~std::size_t{0u};
  static constexpr std::uint32_t kInvalidSlot32 = ~std::uint32_t{0u};

  struct ResolvedTileSlot {
    const std::uint8_t *hit_ptr{nullptr};
    std::uint8_t *fill_ptr{nullptr};
    std::size_t bytes{0u};
    std::size_t entry_index{kInvalidSlot};
    std::int32_t duplicate_of{-1};
    bool is_hit{false};
  };

private:
  struct ImageIndex {
    std::uint32_t tiles_x{1u};
    std::uint32_t tiles_y{1u};
    std::vector<std::uint32_t> tile_to_slot;

    void init_grid(const VtPrepackedImageKey &img_key);
    void ensure_tile_capacity(std::uint32_t tx, std::uint32_t ty);
  };

  struct SlabPool {
    std::size_t tile_bytes{0u};
    std::vector<std::unique_ptr<std::uint8_t[]>> chunks;
    std::vector<std::uint8_t *> free_blocks;
  };

  struct CacheEntry {
    ImageIndex *owner_img{nullptr};
    std::uint32_t owner_tx{0u};
    std::uint32_t owner_ty{0u};
    std::uint8_t *data{nullptr};
    std::size_t size_bytes{0u};
    std::uint32_t pin_count{0u};
    std::uint8_t active{0u};
    std::uint8_t populated{0u};
    std::uint8_t referenced{0u};
  };

  mutable std::mutex _mutex;
  std::uint64_t _budget_bytes{0ull};
  std::uint64_t _resident_bytes{0ull};
  std::uint64_t _hits{0ull};
  std::uint64_t _misses{0ull};
  std::uint64_t _evictions{0ull};
  std::size_t _entry_count{0u};
  std::size_t _clock_hand{0u};
  std::vector<CacheEntry> _entries;
  std::vector<std::size_t> _free_entry_indices;
  std::vector<SlabPool> _slabs;
  std::unordered_map<VtPrepackedImageKey, std::unique_ptr<ImageIndex>,
                     VtPrepackedImageKeyHash>
      _images;

  [[nodiscard]] ImageIndex *get_or_create_image_locked(
      const VtPrepackedImageKey &img_key);
  [[nodiscard]] std::uint8_t *acquire_block_locked(std::size_t size_bytes);
  void release_block_locked(std::uint8_t *ptr, std::size_t size_bytes) noexcept;
  [[nodiscard]] bool evict_one_unpinned_locked() noexcept;
  void resolve_batch_common(
      std::span<const VtPrepackedPageKey> keys,
      const std::uint32_t *opt_indices,
      std::size_t count,
      std::size_t tile_bytes,
      std::vector<ResolvedTileSlot> &out) noexcept;

public:
  explicit VtPrepackedPageCache(std::uint64_t budget_bytes = 0ull) noexcept
      : _budget_bytes{budget_bytes} {}

  void set_budget_bytes(std::uint64_t budget_bytes) noexcept;
  void clear() noexcept;

  // Resolves a chunk of keys under a single lock acquisition. Hits receive a
  // pinned read pointer valid until unpin_batch(). Unique misses receive a
  // writable fill_ptr (also pinned) when capacity permits, or nullptr when
  // budget cannot accommodate the slot without unpinning active entries.
  void resolve_batch(
      std::span<const VtPrepackedPageKey> keys,
      std::size_t tile_bytes,
      std::vector<ResolvedTileSlot> &out) noexcept;

  void resolve_batch_indexed(
      std::span<const VtPrepackedPageKey> all_keys,
      std::span<const std::uint32_t> indices,
      std::size_t tile_bytes,
      std::vector<ResolvedTileSlot> &out) noexcept;

  void unpin_batch(std::span<const ResolvedTileSlot> slots) noexcept;

  // Inserts a pre-populated tile directly (respecting budget and skipping if
  // already cached or budget full when evict_if_needed is false).
  [[nodiscard]] bool insert_warm_tile(
      const VtPrepackedPageKey &key,
      std::span<const std::uint8_t> bytes,
      bool evict_if_needed) noexcept;

  [[nodiscard]] VtPrepackedCacheStats stats() const noexcept;
};

class VtPrepackedDiskStore final {
public:
  static constexpr std::uint32_t kFormatVersion = 1u;
  static constexpr std::size_t kHeaderAlignedBytes = 4096u;

  struct OpenFileHandle {
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    std::uint32_t tile_bytes{};
    void *os_handle{nullptr};
    int posix_fd{-1};
  };

private:
  std::filesystem::path _dir;
  mutable std::mutex _mutex;
  std::unordered_map<VtPrepackedImageKey, std::shared_ptr<OpenFileHandle>,
                     VtPrepackedImageKeyHash>
      _open_files;
  std::unordered_map<VtPrepackedImageKey, std::uint8_t, VtPrepackedImageKeyHash>
      _known_missing;

  [[nodiscard]] std::shared_ptr<OpenFileHandle>
  try_open_existing_locked(
      const VtPrepackedImageKey &img_key,
      std::uint32_t tiles_x,
      std::uint32_t tiles_y) noexcept;

public:
  explicit VtPrepackedDiskStore(std::filesystem::path dir = {}) noexcept;
  ~VtPrepackedDiskStore() noexcept;
  VtPrepackedDiskStore(const VtPrepackedDiskStore &) = delete;
  VtPrepackedDiskStore &operator=(const VtPrepackedDiskStore &) = delete;

  void set_directory(std::filesystem::path dir) noexcept;
  [[nodiscard]] bool enabled() const noexcept { return !_dir.empty(); }
  [[nodiscard]] const std::filesystem::path &directory() const noexcept {
    return _dir;
  }

  [[nodiscard]] std::filesystem::path
  file_path_for(const VtPrepackedImageKey &img_key) const;

  [[nodiscard]] bool has_valid_file(
      const VtPrepackedImageKey &img_key,
      std::uint32_t tiles_x,
      std::uint32_t tiles_y) noexcept;

  [[nodiscard]] bool ensure_image_file(
      const VtPrepackedImageKey &img_key,
      const std::uint8_t *unorm8_pixels,
      const float *float_pixels,
      std::uint32_t tiles_x,
      std::uint32_t tiles_y,
      class VtGpuBcCompressor *gpu_bc = nullptr) noexcept;

  // Reads exactly one prepacked physical page at its fixed 4 KiB-aligned
  // offset. Thread-safe across concurrent worker threads once opened.
  [[nodiscard]] bool read_single_page(
      const VtPrepackedPageKey &page_key,
      std::uint32_t tiles_x,
      std::uint32_t tiles_y,
      std::span<std::uint8_t> dst) noexcept;
};

} // namespace psycles::luisa_backend::detail
