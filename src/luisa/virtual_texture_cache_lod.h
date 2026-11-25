#pragma once

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>

#include <cstdint>

namespace psycles::luisa_backend::detail {

[[nodiscard]] inline luisa::compute::UInt vt_mip_extent_dsl(
    luisa::compute::Expr<luisa::uint> base_extent,
    luisa::compute::Expr<luisa::uint> mip) noexcept {
  using namespace luisa::compute;
  return dsl::max(1u, base_extent >> dsl::min(mip, 31u));
}

[[nodiscard]] inline luisa::compute::Float vt_compute_isotropic_lod(
    luisa::compute::Expr<luisa::float2> duv_dx,
    luisa::compute::Expr<luisa::float2> duv_dy,
    luisa::compute::Expr<luisa::uint> width,
    luisa::compute::Expr<luisa::uint> height,
    luisa::compute::Expr<luisa::uint> mip_count) noexcept {
  using namespace luisa::compute;
  const auto dims = make_float2(
      dsl::max(cast<float>(width), 1.0f),
      dsl::max(cast<float>(height), 1.0f));
  const auto dtx_dx = duv_dx * dims;
  const auto dtx_dy = duv_dy * dims;
  const auto len2_x = dsl::dot(dtx_dx, dtx_dx);
  const auto len2_y = dsl::dot(dtx_dy, dtx_dy);
  const auto max_len2 = dsl::max(len2_x, len2_y);
  const auto finite_grads =
      !(dsl::isnan(len2_x) | dsl::isinf(len2_x) |
        dsl::isnan(len2_y) | dsl::isinf(len2_y));
  const auto valid_footprint = finite_grads & (max_len2 > 0.0f);
  const auto raw_lod =
      dsl::select(0.0f, 0.5f * dsl::log2(dsl::max(max_len2, 1.0e-30f)), valid_footprint);
  const auto max_lod =
      cast<float>(dsl::select(0u, mip_count - 1u, mip_count > 0u));
  return dsl::clamp(raw_lod, 0.0f, max_lod);
}

struct VtMipSelection {
  luisa::compute::UInt mip0;
  luisa::compute::UInt mip1;
  luisa::compute::Float frac;
};

[[nodiscard]] inline VtMipSelection vt_select_mip_levels(
    luisa::compute::Expr<float> lod,
    luisa::compute::Expr<luisa::uint> mip_count,
    luisa::compute::Expr<bool> is_closest) noexcept {
  using namespace luisa::compute;
  const auto max_mip = dsl::select(0u, mip_count - 1u, mip_count > 0u);
  const auto clamped_lod = dsl::clamp(lod, 0.0f, cast<float>(max_mip));
  VtMipSelection sel{
      .mip0 = UInt(0u),
      .mip1 = UInt(0u),
      .frac = Float(0.0f)};
  $if(max_mip > 0u) {
    $if(is_closest) {
      const auto nearest = dsl::min(cast<luisa::uint>(dsl::round(clamped_lod)), max_mip);
      sel.mip0 = nearest;
      sel.mip1 = nearest;
      sel.frac = 0.0f;
    } $else {
      const auto m0 = dsl::min(cast<luisa::uint>(dsl::floor(clamped_lod)), max_mip);
      const auto m1 = dsl::min(m0 + 1u, max_mip);
      const auto f = dsl::clamp(clamped_lod - cast<float>(m0), 0.0f, 1.0f);
      $if((f < 1.0e-4f) | (m0 == m1)) {
        sel.mip0 = m0;
        sel.mip1 = m0;
        sel.frac = 0.0f;
      } $elif(f > (1.0f - 1.0e-4f)) {
        sel.mip0 = m1;
        sel.mip1 = m1;
        sel.frac = 0.0f;
      } $else {
        sel.mip0 = m0;
        sel.mip1 = m1;
        sel.frac = f;
      };
    };
  };
  return sel;
}

[[nodiscard]] inline luisa::compute::Float vt_wrap_axis(
    const luisa::compute::Float &coord,
    const luisa::compute::Float &extent_f,
    const luisa::compute::UInt &dyn_ext,
    const luisa::compute::UInt *dyn_interp,
    luisa::compute::Bool &clipped) noexcept {
  using namespace luisa::compute;
  auto apply_repeat = [&] {
    const auto f = coord - dsl::floor(coord);
    const auto safe_f = dsl::ite((f >= 0.0f) & (f < 1.0f), f, 0.0f);
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
    const auto safe_c = dsl::ite(coord == coord, coord, 0.0f);
    const auto clamped_raw = dsl::clamp(safe_c, -2.0f, 3.0f) * extent_f;
    return dsl::clamp(clamped_raw, -1.5f, extent_f + 1.5f);
  };
  auto apply_mirror = [&] {
    const auto f = coord - 2.0f * dsl::floor(coord * 0.5f);
    const auto safe_f = dsl::ite((f >= 0.0f) & (f < 2.0f), f, 0.0f);
    const auto folded = dsl::ite(safe_f >= 1.0f, 2.0f - safe_f, safe_f);
    return dsl::clamp(folded, 0.0f, 1.0f) * extent_f;
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

struct VtMipPageCoords {
  luisa::compute::UInt page_id;
  luisa::compute::Float tex_x;
  luisa::compute::Float tex_y;
  luisa::compute::UInt texel_x;
  luisa::compute::UInt texel_y;
  luisa::compute::UInt tile_origin_x;
  luisa::compute::UInt tile_origin_y;
  luisa::compute::UInt valid_w;
  luisa::compute::UInt valid_h;
  luisa::compute::Bool clipped;
};

[[nodiscard]] inline VtMipPageCoords vt_resolve_mip_page_coords(
    const luisa::compute::Float2 &uv,
    const luisa::compute::UInt &w_u,
    const luisa::compute::UInt &h_u,
    const luisa::compute::UInt &tx_count,
    const luisa::compute::UInt &ty_count,
    const luisa::compute::UInt &pt_offset,
    std::uint32_t logical_tile_size,
    const luisa::compute::UInt &dyn_ext,
    const luisa::compute::UInt *dyn_interp) noexcept {
  using namespace luisa::compute;
  const auto w_f = cast<float>(w_u);
  const auto h_f = cast<float>(h_u);
  Bool clipped = def(false);
  const auto tex_x = vt_wrap_axis(uv.x, w_f, dyn_ext, dyn_interp, clipped);
  const auto tex_y = vt_wrap_axis(uv.y, h_f, dyn_ext, dyn_interp, clipped);

  VtMipPageCoords res{
      .page_id = UInt(0u),
      .tex_x = tex_x,
      .tex_y = tex_y,
      .texel_x = UInt(0u),
      .texel_y = UInt(0u),
      .tile_origin_x = UInt(0u),
      .tile_origin_y = UInt(0u),
      .valid_w = UInt(0u),
      .valid_h = UInt(0u),
      .clipped = clipped};
  $if(!res.clipped) {
    const auto ix = dsl::clamp(cast<int>(dsl::floor(tex_x)), 0, cast<int>(w_u) - 1);
    const auto iy = dsl::clamp(cast<int>(dsl::floor(tex_y)), 0, cast<int>(h_u) - 1);
    res.texel_x = cast<uint>(ix);
    res.texel_y = cast<uint>(iy);
    const auto T = UInt(logical_tile_size);
    const auto tx = dsl::min(res.texel_x / T, dsl::max(tx_count, 1u) - 1u);
    const auto ty = dsl::min(res.texel_y / T, dsl::max(ty_count, 1u) - 1u);
    res.tile_origin_x = tx * T;
    res.tile_origin_y = ty * T;
    res.valid_w = dsl::min(T, w_u - res.tile_origin_x);
    res.valid_h = dsl::min(T, h_u - res.tile_origin_y);
    res.page_id = pt_offset + ty * tx_count + tx;
  };
  return res;
}

} // namespace psycles::luisa_backend::detail
