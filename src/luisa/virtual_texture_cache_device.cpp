#include "virtual_texture_cache.h"

#include "cycles_texture_sampling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;

void emit_vt_stamp_slot_epoch(
    const Var<VtPoolConfigGpu> &cfg,
    const BufferUInt &slot_epoch,
    const BufferUInt &current_epoch,
    const UInt &pool_idx,
    const UInt &entry) noexcept {
  const auto slot = entry & vt_page_slot_mask;
  const auto epoch = current_epoch.read(pool_idx);
  $if(epoch != 0u) {
    const auto epoch_idx = cfg.epoch_offset + slot;
    $if(slot_epoch.read(epoch_idx) != epoch) {
      slot_epoch.write(epoch_idx, epoch);
    };
  };
}

void emit_vt_pool_hit_coords(
    const Var<VtPoolConfigGpu> &cfg,
    const BufferUInt &slot_epoch,
    const BufferUInt &current_epoch,
    const UInt &pool_idx,
    const UInt &entry,
    std::uint32_t physical_tile_size,
    Float &ox_f,
    Float &oy_f,
    UInt &bindless_base,
    Float &inv_w,
    Float &inv_h) noexcept {
  const auto slot = entry & vt_page_slot_mask;
  emit_vt_stamp_slot_epoch(cfg, slot_epoch, current_epoch, pool_idx, entry);
  const auto atlas_idx = slot >> cfg.slots_per_atlas_log2;
  const auto slot_in_atlas = slot & cfg.slots_per_atlas_mask;
  inv_w = cfg.full_inv_w;
  inv_h = ite(atlas_idx + 1u == cfg.num_atlases, cfg.last_inv_h,
              cfg.full_inv_h);
  const auto slot_tx = slot_in_atlas & cfg.atlas_tiles_x_mask;
  const auto slot_ty = slot_in_atlas >> cfg.atlas_tiles_x_log2;
  ox_f = cast<float>(slot_tx * physical_tile_size);
  oy_f = cast<float>(slot_ty * physical_tile_size);
  bindless_base = cfg.bindless_base + (atlas_idx << 1u);
}

[[nodiscard]] Float4 apply_vt_pool_swizzle(
    const Float4 &c,
    const UInt &swizzle) noexcept {
  Float4 out = c;
  $if(swizzle == static_cast<luisa::uint>(VtChannelSwizzle::rrr1)) {
    out = make_float4(c.x, c.x, c.x, 1.0f);
  } $elif(swizzle == static_cast<luisa::uint>(VtChannelSwizzle::rrrg)) {
    out = make_float4(c.x, c.x, c.x, c.y);
  };
  return out;
}

[[nodiscard]] Float emit_vt_wrap_axis(
    const Float &coord,
    const Float &extent_f,
    const UInt &dyn_ext,
    const UInt *dyn_interp,
    Bool &clipped) noexcept {
  auto apply_repeat = [&] {
    const auto f = coord - floor(coord);
    const auto safe_f = ite((f >= 0.0f) & (f < 1.0f), f, 0.0f);
    return safe_f * extent_f;
  };
  auto apply_clip = [&] {
    const auto raw = coord * extent_f;
    if (dyn_interp == nullptr) {
      clipped = clipped | !((raw > -0.5f) & (raw < extent_f + 0.5f));
    } else {
      $if(*dyn_interp == 0u) {
        clipped = clipped | !((coord >= 0.0f) & (coord < 1.0f));
      } $elif(*dyn_interp == 1u) {
        clipped = clipped | !((raw > -0.5f) & (raw < extent_f + 0.5f));
      } $else {
        clipped = clipped | !((raw > -1.5f) & (raw < extent_f + 1.5f));
      };
    }
    return raw;
  };
  auto apply_extend = [&] {
    const auto safe_c = ite(coord == coord, coord, 0.0f);
    const auto clamped_raw = clamp(safe_c, -2.0f, 3.0f) * extent_f;
    return clamp(clamped_raw, -1.5f, extent_f + 1.5f);
  };
  auto apply_mirror = [&] {
    const auto f = coord - 2.0f * floor(coord * 0.5f);
    const auto safe_f = ite((f >= 0.0f) & (f < 2.0f), f, 0.0f);
    const auto folded = ite(safe_f >= 1.0f, 2.0f - safe_f, safe_f);
    return clamp(folded, 0.0f, 1.0f) * extent_f;
  };

  Float wrapped = def(0.0f);
  $switch(dyn_ext) {
    $case(0u) { wrapped = apply_repeat(); };
    $case(1u) { wrapped = apply_clip(); };
    $case(2u) { wrapped = apply_extend(); };
    $default { wrapped = apply_mirror(); };
  };
  return wrapped;
}

[[nodiscard]] Float4 emit_vt_sample_linear(
    const BindlessVar &atlas_heap,
    const UInt &bindless_base,
    const Float &ox_f,
    const Float &oy_f,
    const Float &inv_w,
    const Float &inv_h,
    const Float &tex_x,
    const Float &tex_y,
    const UInt &tile_origin_x,
    const UInt &tile_origin_y,
    const UInt &valid_w,
    const UInt &valid_h,
    std::uint32_t tile_border) noexcept {
  const auto lx = tex_x - cast<float>(tile_origin_x);
  const auto ly = tex_y - cast<float>(tile_origin_y);
  const auto min_x = ox_f + 0.5f;
  const auto max_x = ox_f + cast<float>(valid_w) +
                     static_cast<float>(2u * tile_border) - 0.5f;
  const auto min_y = oy_f + 0.5f;
  const auto max_y = oy_f + cast<float>(valid_h) +
                     static_cast<float>(2u * tile_border) - 0.5f;
  const auto ax = clamp(ox_f + static_cast<float>(tile_border) + lx, min_x, max_x);
  const auto ay = clamp(oy_f + static_cast<float>(tile_border) + ly, min_y, max_y);
  return atlas_heap.tex2d(bindless_base).sample(make_float2(ax * inv_w, ay * inv_h));
}

void emit_vt_record_miss_bitset(const BufferUInt &miss_bitset,
                                 const UInt &page_id) noexcept {
  const auto word = page_id >> 5u;
  const auto bit = 1u << (page_id & 31u);
  $if((miss_bitset.read(word) & bit) == 0u) {
    miss_bitset.atomic(word).fetch_or(bit);
  };
}

} // namespace

VirtualTextureCache::SampleCallable
VirtualTextureCache::record_linear_repeat_sampler(
    std::uint32_t physical_tile_size,
    std::uint32_t tile_border,
    std::uint32_t logical_tile_size,
    luisa::string_view name) noexcept {
  VirtualTextureCache::SampleCallable fn{
      [physical_tile_size, tile_border, logical_tile_size](
          BufferVar<VtPoolConfigGpu> pool_cfgs,
          BufferVar<VtImageDescGpu> descs,
          BufferUInt page_table,
          BufferUInt slot_epoch,
          BufferUInt current_epoch,
          BufferUInt miss_bitset,
          BindlessVar atlas_heap,
          UInt pool_idx,
          UInt binding_id,
          Float2 uv) noexcept {
        Var<VtSampleResult> out;
        out.color = make_float4(0.0f);
        out.miss_page = vt_invalid_page;
        const auto cfg = pool_cfgs.read(pool_idx);
        $if(cfg.num_atlases == 0u) {
          out.color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        } $else {
          const auto desc = descs.read(binding_id);
          const auto w_u = desc.width;
          const auto h_u = desc.height;
          const auto fx = uv.x - floor(uv.x);
          const auto fy = uv.y - floor(uv.y);
          const auto safe_fx = ite((fx >= 0.0f) & (fx < 1.0f), fx, 0.0f);
          const auto safe_fy = ite((fy >= 0.0f) & (fy < 1.0f), fy, 0.0f);
          const auto tex_x = safe_fx * cast<float>(w_u);
          const auto tex_y = safe_fy * cast<float>(h_u);
          const auto ix = clamp(cast<int>(floor(tex_x)), 0, cast<int>(w_u) - 1);
          const auto iy = clamp(cast<int>(floor(tex_y)), 0, cast<int>(h_u) - 1);
          const auto T = UInt(logical_tile_size);
          const auto tx = cast<uint>(ix) / T;
          const auto ty = cast<uint>(iy) / T;
          const auto page_id = desc.page_table_offset + ty * desc.tiles_x + tx;
          const auto entry = page_table.read(page_id);
          $if((entry & vt_page_resident_bit) != 0u) {
            Float ox_f;
            Float oy_f;
            UInt bindless_base;
            Float inv_w;
            Float inv_h;
            emit_vt_pool_hit_coords(
                cfg, slot_epoch, current_epoch, pool_idx, entry,
                physical_tile_size, ox_f, oy_f, bindless_base, inv_w, inv_h);
            const auto ax = ox_f + static_cast<float>(tile_border) +
                            (tex_x - cast<float>(tx * T));
            const auto ay = oy_f + static_cast<float>(tile_border) +
                            (tex_y - cast<float>(ty * T));
            out.color = apply_vt_pool_swizzle(
                atlas_heap.tex2d(bindless_base)
                    .sample(make_float2(ax * inv_w, ay * inv_h)),
                cfg.channel_swizzle);
          } $else {
            out.miss_page = page_id;
            emit_vt_record_miss_bitset(miss_bitset, page_id);
          };
        };
        return out;
      }};
  fn.set_name(name);
  return fn;
}

VirtualTextureCache::SampleCallable
VirtualTextureCache::record_linear_any_ext_sampler(
    std::uint32_t physical_tile_size,
    std::uint32_t tile_border,
    std::uint32_t logical_tile_size,
    luisa::string_view name) noexcept {
  VirtualTextureCache::SampleCallable fn{
      [physical_tile_size, tile_border, logical_tile_size](
          BufferVar<VtPoolConfigGpu> pool_cfgs,
          BufferVar<VtImageDescGpu> descs,
          BufferUInt page_table,
          BufferUInt slot_epoch,
          BufferUInt current_epoch,
          BufferUInt miss_bitset,
          BindlessVar atlas_heap,
          UInt pool_idx,
          UInt binding_id,
          Float2 uv) noexcept {
        Var<VtSampleResult> out;
        out.color = make_float4(0.0f);
        out.miss_page = vt_invalid_page;
        const auto cfg = pool_cfgs.read(pool_idx);
        $if(cfg.num_atlases == 0u) {
          out.color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        } $else {
          const auto desc = descs.read(binding_id);
          const auto w_u = desc.width;
          const auto h_u = desc.height;
          const auto w_f = cast<float>(w_u);
          const auto h_f = cast<float>(h_u);
          const auto tx_count = desc.tiles_x;
          const auto ty_count = desc.tiles_y;
          const auto pt_offset = desc.page_table_offset;
          const auto dyn_ext =
              (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;

          Bool clipped_out = def(false);
          const auto tex_x =
              emit_vt_wrap_axis(uv.x, w_f, dyn_ext, nullptr, clipped_out);
          const auto tex_y =
              emit_vt_wrap_axis(uv.y, h_f, dyn_ext, nullptr, clipped_out);

          $if(!clipped_out) {
            const auto ix = clamp(cast<int>(floor(tex_x)), 0, cast<int>(w_u) - 1);
            const auto iy = clamp(cast<int>(floor(tex_y)), 0, cast<int>(h_u) - 1);
            const auto T = UInt(logical_tile_size);
            const auto tx = min(cast<uint>(ix) / T, tx_count - 1u);
            const auto ty = min(cast<uint>(iy) / T, ty_count - 1u);
            const auto tile_origin_x = tx * T;
            const auto tile_origin_y = ty * T;
            const auto valid_w = min(T, w_u - tile_origin_x);
            const auto valid_h = min(T, h_u - tile_origin_y);
            const auto page_id = pt_offset + ty * tx_count + tx;

            const auto entry = page_table.read(page_id);
            $if((entry & vt_page_resident_bit) != 0u) {
              Float ox_f;
              Float oy_f;
              UInt bindless_base;
              Float inv_w;
              Float inv_h;
              emit_vt_pool_hit_coords(
                  cfg, slot_epoch, current_epoch, pool_idx, entry,
                  physical_tile_size, ox_f, oy_f, bindless_base, inv_w, inv_h);
              out.color = apply_vt_pool_swizzle(
                  emit_vt_sample_linear(
                      atlas_heap, bindless_base, ox_f, oy_f, inv_w, inv_h,
                      tex_x, tex_y, tile_origin_x, tile_origin_y, valid_w,
                      valid_h, tile_border),
                  cfg.channel_swizzle);
            } $else {
              out.miss_page = page_id;
              emit_vt_record_miss_bitset(miss_bitset, page_id);
            };
          };
        };
        return out;
      }};
  fn.set_name(name);
  return fn;
}

VirtualTextureCache::SampleCallable
VirtualTextureCache::record_general_sampler(
    std::uint32_t physical_tile_size,
    std::uint32_t tile_border,
    std::uint32_t logical_tile_size,
    luisa::string_view name) noexcept {
  VirtualTextureCache::SampleCallable fn{
      [physical_tile_size, tile_border, logical_tile_size](
          BufferVar<VtPoolConfigGpu> pool_cfgs,
          BufferVar<VtImageDescGpu> descs,
          BufferUInt page_table,
          BufferUInt slot_epoch,
          BufferUInt current_epoch,
          BufferUInt miss_bitset,
          BindlessVar atlas_heap,
          UInt pool_idx,
          UInt binding_id,
          Float2 uv) noexcept {
        Var<VtSampleResult> out;
        out.color = make_float4(0.0f);
        out.miss_page = vt_invalid_page;
        const auto cfg = pool_cfgs.read(pool_idx);
        $if(cfg.num_atlases == 0u) {
          out.color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        } $else {
          const auto desc = descs.read(binding_id);
          const auto w_u = desc.width;
          const auto h_u = desc.height;
          const auto w_f = cast<float>(w_u);
          const auto h_f = cast<float>(h_u);
          const auto tx_count = desc.tiles_x;
          const auto ty_count = desc.tiles_y;
          const auto pt_offset = desc.page_table_offset;
          const auto dyn_interp = desc.flags & vt_desc_interp_mask;
          const auto dyn_ext =
              (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;

          Bool clipped_out = def(false);
          const auto tex_x =
              emit_vt_wrap_axis(uv.x, w_f, dyn_ext, &dyn_interp, clipped_out);
          const auto tex_y =
              emit_vt_wrap_axis(uv.y, h_f, dyn_ext, &dyn_interp, clipped_out);

          $if(!clipped_out) {
            const auto ix = clamp(cast<int>(floor(tex_x)), 0, cast<int>(w_u) - 1);
            const auto iy = clamp(cast<int>(floor(tex_y)), 0, cast<int>(h_u) - 1);
            const auto T = UInt(logical_tile_size);
            const auto tx = min(cast<uint>(ix) / T, tx_count - 1u);
            const auto ty = min(cast<uint>(iy) / T, ty_count - 1u);
            const auto tile_origin_x = tx * T;
            const auto tile_origin_y = ty * T;
            const auto valid_w = min(T, w_u - tile_origin_x);
            const auto valid_h = min(T, h_u - tile_origin_y);
            const auto page_id = pt_offset + ty * tx_count + tx;

            const auto entry = page_table.read(page_id);
            $if((entry & vt_page_resident_bit) != 0u) {
              Float ox_f;
              Float oy_f;
              UInt bindless_base;
              Float inv_w;
              Float inv_h;
              emit_vt_pool_hit_coords(
                  cfg, slot_epoch, current_epoch, pool_idx, entry,
                  physical_tile_size, ox_f, oy_f, bindless_base, inv_w, inv_h);

              const auto sample_closest = [&] {
                const auto lx_u = cast<uint>(ix) - tile_origin_x;
                const auto ly_u = cast<uint>(iy) - tile_origin_y;
                const auto u_atlas =
                    (ox_f + static_cast<float>(tile_border) +
                     cast<float>(lx_u) + 0.5f) *
                    inv_w;
                const auto v_atlas =
                    (oy_f + static_cast<float>(tile_border) +
                     cast<float>(ly_u) + 0.5f) *
                    inv_h;
                return atlas_heap.tex2d(bindless_base + 1u)
                    .sample(make_float2(u_atlas, v_atlas));
              };

              const auto sample_cubic = [&] {
                auto cubic_axis = [&](Float ax, Float min_c, Float max_c,
                                      Float inv_ext) noexcept {
                  const auto x = ax - 0.5f;
                  const auto px = floor(x);
                  const auto a = x - px;
                  const auto w0 =
                      (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
                  const auto w1 =
                      (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
                  const auto w2 =
                      (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
                  const auto w3 = (1.0f / 6.0f) * a * a * a;
                  const auto g0 = w0 + w1;
                  const auto g1 = w2 + w3;
                  const auto h0 = w1 / g0 - 1.0f;
                  const auto h1 = w3 / g1 + 1.0f;
                  const auto c0 = clamp(px + h0 + 0.5f, min_c, max_c) * inv_ext;
                  const auto c1 = clamp(px + h1 + 0.5f, min_c, max_c) * inv_ext;
                  return make_float4(c0, c1, g0, g1);
                };
                const auto lx = tex_x - cast<float>(tile_origin_x);
                const auto ly = tex_y - cast<float>(tile_origin_y);
                const auto min_x = ox_f + 0.5f;
                const auto max_x = ox_f + cast<float>(valid_w) +
                                   static_cast<float>(2u * tile_border) - 0.5f;
                const auto min_y = oy_f + 0.5f;
                const auto max_y = oy_f + cast<float>(valid_h) +
                                   static_cast<float>(2u * tile_border) - 0.5f;
                const auto cx = cubic_axis(
                    ox_f + static_cast<float>(tile_border) + lx, min_x, max_x,
                    inv_w);
                const auto cy = cubic_axis(
                    oy_f + static_cast<float>(tile_border) + ly, min_y, max_y,
                    inv_h);
                auto tex = atlas_heap.tex2d(bindless_base);
                const auto s00 = tex.sample(make_float2(cx.x, cy.x));
                const auto s10 = tex.sample(make_float2(cx.y, cy.x));
                const auto s01 = tex.sample(make_float2(cx.x, cy.y));
                const auto s11 = tex.sample(make_float2(cx.y, cy.y));
                return cy.z * (cx.z * s00 + cx.w * s10) +
                       cy.w * (cx.z * s01 + cx.w * s11);
              };

              $if(dyn_interp == 0u) {
                out.color =
                    apply_vt_pool_swizzle(sample_closest(), cfg.channel_swizzle);
              } $elif(dyn_interp == 1u) {
                out.color = apply_vt_pool_swizzle(
                    emit_vt_sample_linear(
                        atlas_heap, bindless_base, ox_f, oy_f, inv_w, inv_h,
                        tex_x, tex_y, tile_origin_x, tile_origin_y, valid_w,
                        valid_h, tile_border),
                    cfg.channel_swizzle);
              } $else {
                out.color =
                    apply_vt_pool_swizzle(sample_cubic(), cfg.channel_swizzle);
              };
            } $else {
              out.miss_page = page_id;
              emit_vt_record_miss_bitset(miss_bitset, page_id);
            };
          };
        };
        return out;
      }};
  fn.set_name(name);
  return fn;
}

Bool VirtualTextureCache::is_resident(
    Expr<luisa::uint> page_id) const noexcept {
  if (_pages.empty()) {
    return def(true);
  }
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  const auto valid = page_id < total_pages;
  const auto safe_idx = ite(valid, page_id, 0u);
  return ite(valid,
             (_page_table->read(safe_idx) & vt_page_resident_bit) != 0u,
             true);
}

Float4 VirtualTextureCache::sample_2d_const(
    std::int32_t binding_id,
    Expr<luisa::float2> uv,
    UInt *miss_page_out,
    bool record_bitset) const noexcept {
  static_cast<void>(record_bitset);
  if (binding_id < 0 ||
      static_cast<std::size_t>(binding_id) >= _descs.size()) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const auto &desc = _descs[static_cast<std::size_t>(binding_id)];
  if ((desc.flags & vt_desc_flag_load_failed) != 0u ||
      (desc.flags & vt_desc_flag_virtual) == 0u) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const std::uint32_t interp = desc.flags & vt_desc_interp_mask;
  const std::uint32_t ext = (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;
  const std::optional<SampleCallable> *callable_ptr = nullptr;
  if (interp == 1u && ext == 0u && _sample_linear_repeat) {
    callable_ptr = &_sample_linear_repeat;
  } else if (interp == 1u && _sample_linear_any_ext) {
    callable_ptr = &_sample_linear_any_ext;
  } else if (_sample_general) {
    callable_ptr = &_sample_general;
  }
  if (callable_ptr == nullptr || !callable_ptr->has_value()) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const auto pool_idx = UInt(desc.pool_index);
  const auto res = (**callable_ptr)(
      _pool_configs, _image_descs_buffer, _page_table, _slot_epoch,
      _current_epoch, _miss_bitset, _atlas_heap, pool_idx,
      UInt(static_cast<std::uint32_t>(binding_id)), uv);
  if (miss_page_out != nullptr) {
    *miss_page_out =
        ite(*miss_page_out == vt_invalid_page, res.miss_page, *miss_page_out);
  }
  return res.color;
}

Float4 VirtualTextureCache::sample_2d_dynamic(
    Expr<std::int32_t> image_texture_id,
    Expr<luisa::float2> uv,
    const BindlessArray &direct_heap,
    UInt *miss_page_out,
    bool record_bitset) const noexcept {
  static_cast<void>(record_bitset);
  Float4 result = def(make_float4(1.0f, 0.0f, 1.0f, 1.0f));
  if (_descs.empty()) {
    return result;
  }
  const auto count = static_cast<std::int32_t>(_descs.size());
  $if((image_texture_id >= 0) & (image_texture_id < count)) {
    const auto desc_idx = cast<luisa::uint>(image_texture_id);
    const auto desc = _image_descs_buffer->read(desc_idx);
    $if((desc.flags & vt_desc_flag_load_failed) == 0u) {
      const auto interp = desc.flags & vt_desc_interp_mask;
      $if((desc.flags & vt_desc_flag_virtual) != 0u) {
        if (_sample_general) {
          const auto pool_idx = desc.pool_index;
          const auto res = (*_sample_general)(
              _pool_configs, _image_descs_buffer, _page_table, _slot_epoch,
              _current_epoch, _miss_bitset, _atlas_heap, pool_idx, desc_idx,
              uv);
          if (miss_page_out != nullptr) {
            *miss_page_out = ite(*miss_page_out == vt_invalid_page,
                                 res.miss_page, *miss_page_out);
          }
          result = res.color;
        }
      } $else {
        auto tex = direct_heap->tex2d(desc.direct_texture_slot);
        $if(interp == 2u) {
          result = sample_cycles_texture_2d_bicubic(
              tex, uv, [&](Float u, Float v) noexcept {
                return tex.sample(make_float2(u, v));
              });
        } $else {
          result = tex.sample(uv);
        };
      };
    };
  };
  return result;
}

void VirtualTextureCache::record_aux_miss(
    Expr<luisa::uint> page_id) const noexcept {
  if (_pages.empty()) {
    return;
  }
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  $if(page_id < total_pages) {
    _aux_demand_hist->atomic(page_id).fetch_add(1u);
  };
}

VirtualTextureCache::PreflightDemandCallable
VirtualTextureCache::record_preflight_demand_callable(
    std::uint32_t logical_tile_size,
    luisa::string_view name) noexcept {
  VirtualTextureCache::PreflightDemandCallable fn{
      [logical_tile_size](
          BufferVar<VtImageDescGpu> descs,
          BufferUInt aux_demand_hist,
          UInt binding_id,
          Float2 uv,
          UInt weight) noexcept {
        const auto desc = descs.read(binding_id);
        $if(((desc.flags & vt_desc_flag_load_failed) == 0u) &
            ((desc.flags & vt_desc_flag_virtual) != 0u)) {
          const auto w_u = desc.width;
          const auto h_u = desc.height;
          const auto w_f = cast<float>(w_u);
          const auto h_f = cast<float>(h_u);
          const auto tx_count = desc.tiles_x;
          const auto ty_count = desc.tiles_y;
          const auto pt_offset = desc.page_table_offset;
          const auto dyn_interp = desc.flags & vt_desc_interp_mask;
          const auto dyn_ext =
              (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;

          Bool clipped_out = def(false);
          const auto tex_x =
              emit_vt_wrap_axis(uv.x, w_f, dyn_ext, &dyn_interp, clipped_out);
          const auto tex_y =
              emit_vt_wrap_axis(uv.y, h_f, dyn_ext, &dyn_interp, clipped_out);

          $if(!clipped_out) {
            const auto ix = clamp(cast<int>(floor(tex_x)), 0, cast<int>(w_u) - 1);
            const auto iy = clamp(cast<int>(floor(tex_y)), 0, cast<int>(h_u) - 1);
            const auto T = UInt(logical_tile_size);
            const auto tx = min(cast<uint>(ix) / T, tx_count - 1u);
            const auto ty = min(cast<uint>(iy) / T, ty_count - 1u);
            const auto page_id = pt_offset + ty * tx_count + tx;
            aux_demand_hist.atomic(page_id).fetch_add(weight);
          };
        };
      }};
  fn.set_name(name);
  return fn;
}

void VirtualTextureCache::record_preflight_demand_2d(
    Expr<std::int32_t> image_texture_id,
    Expr<luisa::float2> uv,
    Expr<luisa::uint> weight) const noexcept {
  if (_descs.empty() || !_preflight_demand_callable || _pages.empty()) {
    return;
  }
  const auto count = static_cast<std::int32_t>(_descs.size());
  $if((image_texture_id >= 0) & (image_texture_id < count)) {
    (*_preflight_demand_callable)(
        _image_descs_buffer, _aux_demand_hist,
        cast<luisa::uint>(image_texture_id), uv, weight);
  };
}

VirtualTextureCache::SampleMipCallable
VirtualTextureCache::record_mip_sampler(
    std::uint32_t physical_tile_size,
    std::uint32_t tile_border,
    std::uint32_t logical_tile_size,
    bool linear_repeat_only,
    luisa::string_view name) noexcept {
  VirtualTextureCache::SampleMipCallable fn{
      [physical_tile_size, tile_border, logical_tile_size, linear_repeat_only](
          BufferVar<VtPoolConfigGpu> pool_cfgs,
          BufferVar<VtImageDescGpu> descs,
          BufferVar<VtMipLevelGpu> mip_levels,
          BufferUInt page_table,
          BufferUInt slot_epoch,
          BufferUInt current_epoch,
          BufferUInt miss_bitset,
          BindlessVar atlas_heap,
          UInt pool_idx,
          UInt binding_id,
          Float2 uv,
          Float2 duv_dx,
          Float2 duv_dy) noexcept {
        Var<VtSampleResult> out;
        out.color = make_float4(0.0f);
        out.miss_page = vt_invalid_page;
        const auto cfg = pool_cfgs.read(pool_idx);
        $if(cfg.num_atlases == 0u) {
          out.color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        } $else {
          const auto desc = descs.read(binding_id);
          const auto dyn_interp = desc.flags & vt_desc_interp_mask;
          const auto dyn_ext =
              (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;
          const auto mip_count = max(desc.mip_count, 1u);
          const auto lod = vt_compute_isotropic_lod(
              duv_dx, duv_dy, desc.width, desc.height, mip_count);
          const Bool is_closest =
              linear_repeat_only ? Bool(false) : (dyn_interp == 0u);
          const auto sel = vt_select_mip_levels(lod, mip_count, is_closest);

          auto resolve_for_mip = [&](const Var<VtMipLevelGpu> &ml) noexcept {
            const UInt ext_mode = linear_repeat_only ? UInt(0u) : dyn_ext;
            const UInt *interp_ptr = linear_repeat_only ? nullptr : &dyn_interp;
            return vt_resolve_mip_page_coords(
                uv, ml.width, ml.height, ml.tiles_x, ml.tiles_y,
                ml.page_table_offset, logical_tile_size, ext_mode, interp_ptr);
          };

          auto sample_resident_coords =
              [&](const UInt &entry, const VtMipPageCoords &coords) noexcept {
                Float ox_f;
                Float oy_f;
                UInt bindless_base;
                Float inv_w;
                Float inv_h;
                emit_vt_pool_hit_coords(
                    cfg, slot_epoch, current_epoch, pool_idx, entry,
                    physical_tile_size, ox_f, oy_f, bindless_base, inv_w,
                    inv_h);
                if (linear_repeat_only) {
                  return apply_vt_pool_swizzle(
                      emit_vt_sample_linear(
                          atlas_heap, bindless_base, ox_f, oy_f, inv_w, inv_h,
                          coords.tex_x, coords.tex_y, coords.tile_origin_x,
                          coords.tile_origin_y, coords.valid_w, coords.valid_h,
                          tile_border),
                      cfg.channel_swizzle);
                }
                auto sample_closest = [&] {
                  const auto lx_u = coords.texel_x - coords.tile_origin_x;
                  const auto ly_u = coords.texel_y - coords.tile_origin_y;
                  const auto u_atlas =
                      (ox_f + static_cast<float>(tile_border) +
                       cast<float>(lx_u) + 0.5f) *
                      inv_w;
                  const auto v_atlas =
                      (oy_f + static_cast<float>(tile_border) +
                       cast<float>(ly_u) + 0.5f) *
                      inv_h;
                  return atlas_heap.tex2d(bindless_base + 1u)
                      .sample(make_float2(u_atlas, v_atlas));
                };
                auto sample_cubic = [&] {
                  auto cubic_axis = [&](Float ax, Float min_c, Float max_c,
                                        Float inv_ext) noexcept {
                    const auto x = ax - 0.5f;
                    const auto px = floor(x);
                    const auto a = x - px;
                    const auto w0 =
                        (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
                    const auto w1 =
                        (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
                    const auto w2 =
                        (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
                    const auto w3 = (1.0f / 6.0f) * a * a * a;
                    const auto g0 = w0 + w1;
                    const auto g1 = w2 + w3;
                    const auto h0 = w1 / g0 - 1.0f;
                    const auto h1 = w3 / g1 + 1.0f;
                    const auto c0 = clamp(px + h0 + 0.5f, min_c, max_c) * inv_ext;
                    const auto c1 = clamp(px + h1 + 0.5f, min_c, max_c) * inv_ext;
                    return make_float4(c0, c1, g0, g1);
                  };
                  const auto lx =
                      coords.tex_x - cast<float>(coords.tile_origin_x);
                  const auto ly =
                      coords.tex_y - cast<float>(coords.tile_origin_y);
                  const auto min_x = ox_f + 0.5f;
                  const auto max_x = ox_f + cast<float>(coords.valid_w) +
                                     static_cast<float>(2u * tile_border) - 0.5f;
                  const auto min_y = oy_f + 0.5f;
                  const auto max_y = oy_f + cast<float>(coords.valid_h) +
                                     static_cast<float>(2u * tile_border) - 0.5f;
                  const auto cx = cubic_axis(
                      ox_f + static_cast<float>(tile_border) + lx, min_x, max_x,
                      inv_w);
                  const auto cy = cubic_axis(
                      oy_f + static_cast<float>(tile_border) + ly, min_y, max_y,
                      inv_h);
                  auto tex = atlas_heap.tex2d(bindless_base);
                  const auto s00 = tex.sample(make_float2(cx.x, cy.x));
                  const auto s10 = tex.sample(make_float2(cx.y, cy.x));
                  const auto s01 = tex.sample(make_float2(cx.x, cy.y));
                  const auto s11 = tex.sample(make_float2(cx.y, cy.y));
                  return cy.z * (cx.z * s00 + cx.w * s10) +
                         cy.w * (cx.z * s01 + cx.w * s11);
                };
                Float4 raw_col = make_float4(0.0f);
                $if(dyn_interp == 0u) {
                  raw_col = sample_closest();
                } $elif(dyn_interp == 1u) {
                  raw_col = emit_vt_sample_linear(
                      atlas_heap, bindless_base, ox_f, oy_f, inv_w, inv_h,
                      coords.tex_x, coords.tex_y, coords.tile_origin_x,
                      coords.tile_origin_y, coords.valid_w, coords.valid_h,
                      tile_border);
                } $else {
                  raw_col = sample_cubic();
                };
                return apply_vt_pool_swizzle(raw_col, cfg.channel_swizzle);
              };

          const auto ml0 = mip_levels.read(desc.mip_table_offset + sel.mip0);
          const auto c0 = resolve_for_mip(ml0);
          $if(sel.mip1 == sel.mip0) {
            $if(!c0.clipped) {
              const auto entry0 = page_table.read(c0.page_id);
              $if((entry0 & vt_page_resident_bit) != 0u) {
                out.color = sample_resident_coords(entry0, c0);
              } $else {
                out.miss_page = c0.page_id;
                emit_vt_record_miss_bitset(miss_bitset, c0.page_id);
              };
            };
          } $else {
            const auto ml1 = mip_levels.read(desc.mip_table_offset + sel.mip1);
            const auto c1 = resolve_for_mip(ml1);
            UInt entry0 = 0u;
            UInt entry1 = 0u;
            $if(!c0.clipped) { entry0 = page_table.read(c0.page_id); };
            $if(!c1.clipped) { entry1 = page_table.read(c1.page_id); };
            const Bool res0 = (!c0.clipped) & ((entry0 & vt_page_resident_bit) != 0u);
            const Bool res1 = (!c1.clipped) & ((entry1 & vt_page_resident_bit) != 0u);
            const Bool hit0 = c0.clipped | res0;
            const Bool hit1 = c1.clipped | res1;
            $if(hit0 & hit1) {
              Float4 col0 = make_float4(0.0f);
              Float4 col1 = make_float4(0.0f);
              $if(!c0.clipped) { col0 = sample_resident_coords(entry0, c0); };
              $if(!c1.clipped) { col1 = sample_resident_coords(entry1, c1); };
              out.color = lerp(col0, col1, sel.frac);
            } $else {
              // Never substitute a resident coarse mip in final renders; stamp
              // slot_epoch on any resident partner so it survives while waiting.
              $if(res0) {
                emit_vt_stamp_slot_epoch(
                    cfg, slot_epoch, current_epoch, pool_idx, entry0);
              };
              $if(res1) {
                emit_vt_stamp_slot_epoch(
                    cfg, slot_epoch, current_epoch, pool_idx, entry1);
              };
              $if(!hit0) {
                out.miss_page = c0.page_id;
                emit_vt_record_miss_bitset(miss_bitset, c0.page_id);
              };
              $if(!hit1) {
                $if(out.miss_page == vt_invalid_page) {
                  out.miss_page = c1.page_id;
                };
                emit_vt_record_miss_bitset(miss_bitset, c1.page_id);
              };
            };
          };
        };
        return out;
      }};
  fn.set_name(name);
  return fn;
}

VirtualTextureCache::PreflightMipDemandCallable
VirtualTextureCache::record_preflight_mip_demand_callable(
    std::uint32_t logical_tile_size,
    luisa::string_view name) noexcept {
  VirtualTextureCache::PreflightMipDemandCallable fn{
      [logical_tile_size](
          BufferVar<VtImageDescGpu> descs,
          BufferVar<VtMipLevelGpu> mip_levels,
          BufferUInt aux_demand_hist,
          UInt binding_id,
          Float2 uv,
          Float2 duv_dx,
          Float2 duv_dy,
          UInt weight) noexcept {
        const auto desc = descs.read(binding_id);
        $if(((desc.flags & vt_desc_flag_load_failed) == 0u) &
            ((desc.flags & vt_desc_flag_virtual) != 0u)) {
          const auto dyn_interp = desc.flags & vt_desc_interp_mask;
          const auto dyn_ext =
              (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;
          const auto mip_count = max(desc.mip_count, 1u);
          const auto lod = vt_compute_isotropic_lod(
              duv_dx, duv_dy, desc.width, desc.height, mip_count);
          const auto sel =
              vt_select_mip_levels(lod, mip_count, dyn_interp == 0u);
          const auto ml0 = mip_levels.read(desc.mip_table_offset + sel.mip0);
          const auto c0 = vt_resolve_mip_page_coords(
              uv, ml0.width, ml0.height, ml0.tiles_x, ml0.tiles_y,
              ml0.page_table_offset, logical_tile_size, dyn_ext, &dyn_interp);
          $if(!c0.clipped) {
            aux_demand_hist.atomic(c0.page_id).fetch_add(weight);
          };
          $if(sel.mip1 != sel.mip0) {
            const auto ml1 = mip_levels.read(desc.mip_table_offset + sel.mip1);
            const auto c1 = vt_resolve_mip_page_coords(
                uv, ml1.width, ml1.height, ml1.tiles_x, ml1.tiles_y,
                ml1.page_table_offset, logical_tile_size, dyn_ext, &dyn_interp);
            $if((!c1.clipped) & (c1.clipped | (c1.page_id != c0.page_id) | c0.clipped)) {
              aux_demand_hist.atomic(c1.page_id).fetch_add(weight);
            };
          };
        };
      }};
  fn.set_name(name);
  return fn;
}

Float4 VirtualTextureCache::sample_2d_const_mip(
    std::int32_t binding_id,
    Expr<luisa::float2> uv,
    Expr<luisa::float2> duv_dx,
    Expr<luisa::float2> duv_dy,
    UInt *miss_page_out,
    bool record_bitset) const noexcept {
  if (_options.mip_mode != LuisaTextureMipMode::isotropic) {
    return sample_2d_const(binding_id, uv, miss_page_out, record_bitset);
  }
  static_cast<void>(record_bitset);
  if (binding_id < 0 ||
      static_cast<std::size_t>(binding_id) >= _descs.size()) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const auto &desc = _descs[static_cast<std::size_t>(binding_id)];
  if ((desc.flags & vt_desc_flag_load_failed) != 0u ||
      (desc.flags & vt_desc_flag_virtual) == 0u) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const std::uint32_t interp = desc.flags & vt_desc_interp_mask;
  const std::uint32_t ext = (desc.flags & vt_desc_ext_mask) >> vt_desc_ext_shift;
  const std::optional<SampleMipCallable> *callable_ptr = nullptr;
  if (interp == 1u && ext == 0u && _sample_mip_linear_repeat) {
    callable_ptr = &_sample_mip_linear_repeat;
  } else if (_sample_mip_general) {
    callable_ptr = &_sample_mip_general;
  }
  if (callable_ptr == nullptr || !callable_ptr->has_value()) {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }
  const auto res = (**callable_ptr)(
      _pool_configs, _image_descs_buffer, _mip_levels_buffer, _page_table,
      _slot_epoch, _current_epoch, _miss_bitset, _atlas_heap,
      UInt(desc.pool_index), UInt(static_cast<std::uint32_t>(binding_id)), uv,
      duv_dx, duv_dy);
  if (miss_page_out != nullptr) {
    *miss_page_out =
        ite(*miss_page_out == vt_invalid_page, res.miss_page, *miss_page_out);
  }
  return res.color;
}

Float4 VirtualTextureCache::sample_2d_dynamic_mip(
    Expr<std::int32_t> image_texture_id,
    Expr<luisa::float2> uv,
    Expr<luisa::float2> duv_dx,
    Expr<luisa::float2> duv_dy,
    const BindlessArray &direct_heap,
    UInt *miss_page_out,
    bool record_bitset) const noexcept {
  if (_options.mip_mode != LuisaTextureMipMode::isotropic) {
    return sample_2d_dynamic(
        image_texture_id, uv, direct_heap, miss_page_out, record_bitset);
  }
  static_cast<void>(record_bitset);
  Float4 result = def(make_float4(1.0f, 0.0f, 1.0f, 1.0f));
  if (_descs.empty()) {
    return result;
  }
  const auto count = static_cast<std::int32_t>(_descs.size());
  $if((image_texture_id >= 0) & (image_texture_id < count)) {
    const auto desc_idx = cast<luisa::uint>(image_texture_id);
    const auto desc = _image_descs_buffer->read(desc_idx);
    $if((desc.flags & vt_desc_flag_load_failed) == 0u) {
      const auto interp = desc.flags & vt_desc_interp_mask;
      $if((desc.flags & vt_desc_flag_virtual) != 0u) {
        if (_sample_mip_general) {
          const auto res = (*_sample_mip_general)(
              _pool_configs, _image_descs_buffer, _mip_levels_buffer,
              _page_table, _slot_epoch, _current_epoch, _miss_bitset,
              _atlas_heap, desc.pool_index, desc_idx, uv, duv_dx, duv_dy);
          if (miss_page_out != nullptr) {
            *miss_page_out = ite(*miss_page_out == vt_invalid_page,
                                 res.miss_page, *miss_page_out);
          }
          result = res.color;
        }
      } $else {
        result = sample_direct_texture_2d_isotropic(
            direct_heap->tex2d(desc.direct_texture_slot), uv, duv_dx, duv_dy,
            desc.width, desc.height, desc.mip_count, interp);
      };
    };
  };
  return result;
}

void VirtualTextureCache::record_preflight_demand_2d(
    Expr<std::int32_t> image_texture_id,
    Expr<luisa::float2> uv,
    Expr<luisa::float2> duv_dx,
    Expr<luisa::float2> duv_dy,
    Expr<luisa::uint> weight) const noexcept {
  if (_options.mip_mode != LuisaTextureMipMode::isotropic ||
      !_preflight_mip_demand_callable) {
    record_preflight_demand_2d(image_texture_id, uv, weight);
    return;
  }
  if (_descs.empty() || _pages.empty()) {
    return;
  }
  const auto count = static_cast<std::int32_t>(_descs.size());
  $if((image_texture_id >= 0) & (image_texture_id < count)) {
    (*_preflight_mip_demand_callable)(
        _image_descs_buffer, _mip_levels_buffer, _aux_demand_hist,
        cast<luisa::uint>(image_texture_id), uv, duv_dx, duv_dy, weight);
  };
}

} // namespace psycles::luisa_backend::detail
