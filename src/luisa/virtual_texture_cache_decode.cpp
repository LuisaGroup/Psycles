#include "virtual_texture_cache.h"

#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"
#include "virtual_texture_cache_mipmap.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
using DecodedSourcePayload = VirtualTextureCache::DecodedSourcePayload;

[[nodiscard]] std::uint8_t float_to_unorm8(float v) noexcept {
  const float c = std::clamp(v, 0.0f, 1.0f) * 255.0f;
  return static_cast<std::uint8_t>(std::clamp(std::lround(c), 0l, 255l));
}

[[nodiscard]] DecodedSourcePayload decode_source_payload(
    const VirtualTextureCache::SourceImageEntry &src,
    bool build_mips) noexcept {
  DecodedSourcePayload out;
  auto decoded = decode_scene_image(src.encoded_data, src.name);
  if (!decoded) {
    return out;
  }
  apply_scene_image_alpha(*decoded, src.alpha_type, src.color_space);
  downscale_scene_image_to_limit(*decoded, src.texture_limit);
  out.ok = true;
  out.width = decoded->width;
  out.height = decoded->height;
  if (src.storage == io::DecodedImageStorage::unorm8) {
    if (decoded->storage == io::DecodedImageStorage::unorm8) {
      out.unorm8_pixels = std::move(decoded->unorm8_pixels);
    } else {
      out.unorm8_pixels.resize(decoded->float_pixels.size());
      for (std::size_t i = 0u; i < decoded->float_pixels.size(); ++i) {
        out.unorm8_pixels[i] = float_to_unorm8(decoded->float_pixels[i]);
      }
    }
    out.decoded_bytes = out.unorm8_pixels.size();
    if (build_mips) {
      out.unorm8_mips = vt_build_unorm8_tail_mips(
          out.unorm8_pixels, out.width, out.height,
          src.color_space == contract::ImageColorSpace::srgb);
      for (const auto &m : out.unorm8_mips) {
        out.decoded_bytes += m.rgba.size();
      }
    }
  } else {
    if (decoded->storage == io::DecodedImageStorage::float32) {
      out.float_pixels = std::move(decoded->float_pixels);
    } else {
      out.float_pixels.resize(decoded->unorm8_pixels.size());
      constexpr float inv255 = 1.0f / 255.0f;
      for (std::size_t i = 0u; i < decoded->unorm8_pixels.size(); ++i) {
        out.float_pixels[i] =
            static_cast<float>(decoded->unorm8_pixels[i]) * inv255;
      }
    }
    out.decoded_bytes = out.float_pixels.size() * sizeof(float);
    if (build_mips) {
      out.float4_mips = vt_build_float4_tail_mips(
          out.float_pixels, out.width, out.height);
      for (const auto &m : out.float4_mips) {
        out.decoded_bytes += m.rgba.size() * sizeof(float);
      }
    }
  }
  return out;
}

} // namespace

void VirtualTextureCache::apply_decoded_source_payload(
    SourceImageEntry &src,
    DecodedSourcePayload &&payload) noexcept {
  src.last_use_seq = ++_decode_use_seq;
  if (src.decoded || src.load_failed) {
    return;
  }
  if (!payload.ok) {
    src.load_failed = true;
    src.decoded = true;
  } else {
    ++_stats.images_decoded;
    src.width = payload.width;
    src.height = payload.height;
    src.unorm8_pixels = std::move(payload.unorm8_pixels);
    src.float_pixels = std::move(payload.float_pixels);
    src.unorm8_mips = std::move(payload.unorm8_mips);
    src.float4_mips = std::move(payload.float4_mips);
    src.decoded_bytes = payload.decoded_bytes;
    src.decoded = true;
    _host_decoded_bytes += src.decoded_bytes;
    _stats.peak_host_decoded_bytes =
        std::max(_stats.peak_host_decoded_bytes, _host_decoded_bytes);
  }
}

void VirtualTextureCache::decode_sources_batch(
    std::span<const std::uint32_t> source_indices) noexcept {
  if (source_indices.empty()) {
    return;
  }
  const bool build_mips =
      (_options.mip_mode == LuisaTextureMipMode::isotropic);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<DecodedSourcePayload> payloads(source_indices.size());
  if (source_indices.size() > 1u) {
    psycles::parallel_for(source_indices.size(), [&](std::size_t idx) {
      payloads[idx] =
          decode_source_payload(_sources[source_indices[idx]], build_mips);
    });
  } else {
    payloads[0u] =
        decode_source_payload(_sources[source_indices[0u]], build_mips);
  }
  const auto t1 = std::chrono::steady_clock::now();
  _stats.decode_ms +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (std::size_t idx = 0u; idx < source_indices.size(); ++idx) {
    apply_decoded_source_payload(
        _sources[source_indices[idx]], std::move(payloads[idx]));
  }
}

void VirtualTextureCache::bg_decode_worker_loop() noexcept {
  while (true) {
    std::uint32_t idx = std::numeric_limits<std::uint32_t>::max();
    {
      std::unique_lock lock(_bg_mutex);
      _bg_cv.wait(lock, [&] {
        return _bg_stop || !_bg_pending_queue.empty();
      });
      if (_bg_stop) {
        return;
      }
      idx = _bg_pending_queue.front();
      _bg_pending_queue.pop_front();
      _bg_source_queued[idx] = 2u;
    }
    const bool build_mips =
        (_options.mip_mode == LuisaTextureMipMode::isotropic);
    DecodedSourcePayload payload =
        decode_source_payload(_sources[idx], build_mips);
    {
      std::lock_guard lock(_bg_mutex);
      if (_bg_stop) {
        _bg_source_queued[idx] = 0u;
        _bg_cv.notify_all();
        return;
      }
      _bg_ready_payloads[idx] = std::move(payload);
      _bg_source_queued[idx] = 3u;
    }
    _bg_cv.notify_all();
  }
}

void VirtualTextureCache::ensure_bg_decode_workers_started() noexcept {
  if (!_bg_workers.empty() || _bg_stop) {
    return;
  }
  const std::size_t worker_count = std::min<std::size_t>(
      2u, std::max<std::size_t>(1u, std::thread::hardware_concurrency() / 4u));
  _bg_workers.reserve(worker_count);
  for (std::size_t i = 0u; i < worker_count; ++i) {
    _bg_workers.emplace_back([this]() noexcept { bg_decode_worker_loop(); });
  }
}

void VirtualTextureCache::stop_bg_decode_workers() noexcept {
  {
    std::lock_guard lock(_bg_mutex);
    _bg_stop = true;
    _bg_pending_queue.clear();
  }
  _bg_cv.notify_all();
  for (auto &w : _bg_workers) {
    if (w.joinable()) {
      w.join();
    }
  }
  _bg_workers.clear();
}

void VirtualTextureCache::enqueue_bg_decode(std::uint32_t source_index) noexcept {
  if (source_index >= _sources.size()) {
    return;
  }
  const auto &src = _sources[source_index];
  if (src.decoded || src.load_failed || src.encoded_data.empty()) {
    return;
  }
  ensure_bg_decode_workers_started();
  std::lock_guard lock(_bg_mutex);
  if (_bg_stop || _bg_source_queued[source_index] != 0u ||
      _bg_pending_queue.size() >= 8u) {
    return;
  }
  _bg_source_queued[source_index] = 1u;
  _bg_pending_queue.push_back(source_index);
  _bg_cv.notify_one();
}

void VirtualTextureCache::decode_source_on_demand(
    std::uint32_t source_index) noexcept {
  if (source_index >= _sources.size()) {
    return;
  }
  auto &src = _sources[source_index];
  src.last_use_seq = ++_decode_use_seq;
  if (src.decoded || src.load_failed) {
    return;
  }
  bool need_sync_decode = false;
  DecodedSourcePayload bg_payload;
  bool have_bg_payload = false;
  if (!_bg_source_queued.empty()) {
    const auto wait_t0 = std::chrono::steady_clock::now();
    std::unique_lock lock(_bg_mutex);
    const auto state = _bg_source_queued[source_index];
    if (state == 3u && _bg_ready_payloads[source_index].has_value()) {
      bg_payload = std::move(*_bg_ready_payloads[source_index]);
      _bg_ready_payloads[source_index].reset();
      _bg_source_queued[source_index] = 0u;
      have_bg_payload = true;
    } else if (state == 1u) {
      const auto it = std::find(_bg_pending_queue.begin(),
                                _bg_pending_queue.end(), source_index);
      if (it != _bg_pending_queue.end()) {
        _bg_pending_queue.erase(it);
      }
      _bg_source_queued[source_index] = 0u;
      need_sync_decode = true;
    } else if (state == 2u) {
      _bg_cv.wait(lock, [&] {
        return _bg_stop || _bg_source_queued[source_index] != 2u;
      });
      if (_bg_source_queued[source_index] == 3u &&
          _bg_ready_payloads[source_index].has_value()) {
        bg_payload = std::move(*_bg_ready_payloads[source_index]);
        _bg_ready_payloads[source_index].reset();
        _bg_source_queued[source_index] = 0u;
        have_bg_payload = true;
      } else {
        _bg_source_queued[source_index] = 0u;
        need_sync_decode = true;
      }
    } else {
      need_sync_decode = true;
    }
    if (have_bg_payload) {
      const auto wait_t1 = std::chrono::steady_clock::now();
      _stats.decode_ms +=
          std::chrono::duration<double, std::milli>(wait_t1 - wait_t0).count();
    }
  } else {
    need_sync_decode = true;
  }
  if (have_bg_payload) {
    apply_decoded_source_payload(src, std::move(bg_payload));
    return;
  }
  if (need_sync_decode) {
    const std::array<std::uint32_t, 1u> single{source_index};
    decode_sources_batch(single);
  }
}

void VirtualTextureCache::trim_host_decoded_cache(
    std::uint32_t protect_source_index) noexcept {
  const std::array<std::uint32_t, 1u> single{protect_source_index};
  trim_host_decoded_cache(std::span<const std::uint32_t>{single});
}

void VirtualTextureCache::trim_host_decoded_cache(
    std::span<const std::uint32_t> protected_sources) noexcept {
  const auto budget = _options.host_decoded_cache_budget_bytes;
  if (budget == 0ull || _host_decoded_bytes <= budget) {
    return;
  }
  while (_host_decoded_bytes > budget) {
    std::uint32_t victim = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t i = 0u; i < _sources.size(); ++i) {
      if (std::find(protected_sources.begin(), protected_sources.end(), i) !=
          protected_sources.end()) {
        continue;
      }
      const auto &s = _sources[i];
      if (!s.decoded || s.decoded_bytes == 0ull || s.encoded_data.empty()) {
        continue;
      }
      if (s.last_use_seq < best_seq) {
        best_seq = s.last_use_seq;
        victim = i;
      }
    }
    if (victim == std::numeric_limits<std::uint32_t>::max()) {
      break;
    }
    auto &v = _sources[victim];
    _host_decoded_bytes =
        (_host_decoded_bytes >= v.decoded_bytes)
            ? (_host_decoded_bytes - v.decoded_bytes)
            : 0ull;
    v.decoded_bytes = 0ull;
    v.decoded = false;
    {
      std::vector<std::uint8_t> empty;
      v.unorm8_pixels.swap(empty);
    }
    {
      std::vector<float> empty;
      v.float_pixels.swap(empty);
    }
    {
      std::vector<VtUnorm8MipLevel> empty;
      v.unorm8_mips.swap(empty);
    }
    {
      std::vector<VtFloat4MipLevel> empty;
      v.float4_mips.swap(empty);
    }
    ++_stats.host_decode_evictions;
  }
}

void VirtualTextureCache::ensure_eager_direct_textures_uploaded(
    LuisaSceneData &scene) {
  if (_eager_fallback_uploaded || _sources.empty() || !scene.cycles_svm) {
    _eager_fallback_uploaded = true;
    return;
  }
  auto stream = scene.device.create_stream();
  bool uploaded_any = false;
  std::uint32_t last_decoded_src = 0u;
  const auto &handles = scene.cycles_svm->compilation.images;
  const auto &bindings = scene.cycles_svm->image_bindings;

  if (_options.host_decoded_cache_budget_bytes == 0ull) {
    std::vector<std::uint32_t> pending_sources;
    for (std::uint32_t idx = 0u; idx < _sources.size(); ++idx) {
      const auto &src = _sources[idx];
      if (!src.load_failed && !src.force_direct && !src.decoded) {
        pending_sources.push_back(idx);
      }
    }
    decode_sources_batch(pending_sources);
  }

  for (std::uint32_t idx = 0u; idx < _sources.size(); ++idx) {
    auto &src = _sources[idx];
    if (src.load_failed || src.force_direct) {
      continue;
    }
    decode_source_on_demand(idx);
    last_decoded_src = idx;
    if (src.load_failed || src.width == 0u || src.height == 0u) {
      continue;
    }
    const bool is_unorm8 =
        (src.storage == io::DecodedImageStorage::unorm8) &&
        !src.unorm8_pixels.empty();
    const bool is_float32 =
        (src.storage == io::DecodedImageStorage::float32) &&
        !src.float_pixels.empty();
    if (!is_unorm8 && !is_float32) {
      continue;
    }
    const auto storage = is_unorm8 ? PixelStorage::BYTE4 : PixelStorage::FLOAT4;
    auto &resource = scene.images.emplace_back(
        scene.device.create_image<float>(storage, src.width, src.height));
    if (is_unorm8) {
      stream << resource.copy_from(src.unorm8_pixels.data());
    } else {
      stream << resource.copy_from(src.float_pixels.data());
    }
    scene.texture_heap.emplace_on_update(
        static_cast<std::uint32_t>(src.id.value), resource,
        Sampler::linear_point_repeat());
    for (std::size_t h_idx = 0u; h_idx < handles.size(); ++h_idx) {
      const auto &handle = handles[h_idx];
      if (!handle.nishita && handle.resource_id == src.id.value &&
          h_idx < bindings.size()) {
        scene.texture_heap.emplace_on_update(
            bindings[h_idx].texture_slot, resource,
            cycles_svm_texture_sampler(handle.interpolation, handle.extension));
      }
    }
    uploaded_any = true;
  }

  if (uploaded_any) {
    stream << scene.texture_heap.update();
  }
  stream << synchronize();
  if (uploaded_any) {
    trim_host_decoded_cache(last_decoded_src);
  }
  _eager_fallback_uploaded = true;
}

void VirtualTextureCache::populate_temporary_direct_textures(
    LuisaSceneData &scene,
    Stream &stream,
    const std::set<contract::ImageId> &image_ids,
    std::vector<Image<float>> &temp_images,
    std::vector<std::uint32_t> &rebound_slots) {
  if (image_ids.empty() || !scene.cycles_svm) {
    return;
  }
  const auto &handles = scene.cycles_svm->compilation.images;
  const auto &bindings = scene.cycles_svm->image_bindings;
  bool updated = false;
  std::uint32_t last_src = 0u;

  for (const auto image_id : image_ids) {
    const auto it = _id_to_source.find(image_id.value);
    if (it == _id_to_source.end()) {
      continue;
    }
    const auto src_idx = it->second;
    auto &src = _sources[src_idx];
    if (src.load_failed || src.force_direct) {
      continue;
    }
    decode_source_on_demand(src_idx);
    last_src = src_idx;
    if (src.load_failed || src.width == 0u || src.height == 0u) {
      continue;
    }
    const bool is_unorm8 =
        (src.storage == io::DecodedImageStorage::unorm8) &&
        !src.unorm8_pixels.empty();
    const bool is_float32 =
        (src.storage == io::DecodedImageStorage::float32) &&
        !src.float_pixels.empty();
    if (!is_unorm8 && !is_float32) {
      continue;
    }
    const auto storage = is_unorm8 ? PixelStorage::BYTE4 : PixelStorage::FLOAT4;
    const std::uint32_t mip_levels =
        is_unorm8
            ? static_cast<std::uint32_t>(1u + src.unorm8_mips.size())
            : static_cast<std::uint32_t>(1u + src.float4_mips.size());
    auto &resource = temp_images.emplace_back(
        scene.device.create_image<float>(
            storage, src.width, src.height, mip_levels));
    if (is_unorm8) {
      stream << resource.view(0u).copy_from(src.unorm8_pixels.data());
      for (std::size_t m = 0u; m < src.unorm8_mips.size(); ++m) {
        stream << resource.view(static_cast<std::uint32_t>(m + 1u))
                      .copy_from(src.unorm8_mips[m].rgba.data());
      }
    } else {
      stream << resource.view(0u).copy_from(src.float_pixels.data());
      for (std::size_t m = 0u; m < src.float4_mips.size(); ++m) {
        stream << resource.view(static_cast<std::uint32_t>(m + 1u))
                      .copy_from(src.float4_mips[m].rgba.data());
      }
    }
    const auto primary_slot = static_cast<std::uint32_t>(image_id.value);
    scene.texture_heap.emplace_on_update(primary_slot, resource,
                                         Sampler::linear_point_repeat());
    rebound_slots.push_back(primary_slot);
    for (std::size_t h_idx = 0u; h_idx < handles.size(); ++h_idx) {
      const auto &handle = handles[h_idx];
      if (!handle.nishita && handle.resource_id == image_id.value &&
          h_idx < bindings.size()) {
        const auto slot = bindings[h_idx].texture_slot;
        scene.texture_heap.emplace_on_update(
            slot, resource,
            cycles_svm_texture_sampler(handle.interpolation, handle.extension));
        rebound_slots.push_back(slot);
      }
    }
    updated = true;
  }

  if (updated) {
    stream << scene.texture_heap.update() << synchronize();
    trim_host_decoded_cache(last_src);
  }
}

} // namespace psycles::luisa_backend::detail
