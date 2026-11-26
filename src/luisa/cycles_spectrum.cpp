#include <psycles/luisa/cycles_spectrum.h>

#include <cmath>

namespace psycles::luisa_backend::cycles_spectrum {

using namespace luisa::compute;

namespace {

[[nodiscard]] Float safe_atanh(Expr<float> x) noexcept {
  const auto xc = clamp(x, -0.999999f, 0.999999f);
  return 0.5f * log((1.0f + xc) / (1.0f - xc));
}

[[nodiscard]] Float4 safe_cosh4(Expr<luisa::float4> x) noexcept {
  const auto e = exp(clamp(abs(x), make_float4(0.0f), make_float4(20.0f)));
  return 0.5f * (e + 1.0f / e);
}

[[nodiscard]] Float inv_smoothstep_device(Expr<float> v) noexcept {
  const auto arg = clamp(1.0f - 2.0f * clamp(v, 0.0f, 1.0f), -1.0f, 1.0f);
  return 0.5f - sin(asin(arg) * (1.0f / 3.0f));
}

[[nodiscard]] Float4 sample_cie_illuminant(
    Expr<Buffer<luisa::float4>> cie_buf, Expr<float> lambda_nm) noexcept {
  const auto pos = clamp(lambda_nm - spectral_lambda_min, 0.0f,
                         spectral_lambda_span);
  const auto idx0 = min(cast<std::uint32_t>(pos),
                        spectral_cie_sample_count - 2u);
  const auto frac = clamp(pos - cast<float>(idx0), 0.0f, 1.0f);
  const auto c0 = cie_buf.read(idx0);
  const auto c1 = cie_buf.read(idx0 + 1u);
  return lerp(c0, c1, frac);
}

[[nodiscard]] Float4 sample_illuminant_4_impl(
    Expr<Buffer<luisa::float4>> cie_buf, Expr<luisa::float4> lambdas) noexcept {
  return make_float4(sample_cie_illuminant(cie_buf, lambdas.x).w,
                     sample_cie_illuminant(cie_buf, lambdas.y).w,
                     sample_cie_illuminant(cie_buf, lambdas.z).w,
                     sample_cie_illuminant(cie_buf, lambdas.w).w);
}

[[nodiscard]] Float3 lookup_rgb2spec_coeffs(
    Expr<Buffer<luisa::float4>> rgb2spec_buf, Expr<luisa::float3> rgb) noexcept {
  const auto r = max(rgb.x, 0.0f);
  const auto g = max(rgb.y, 0.0f);
  const auto b = max(rgb.z, 0.0f);

  const auto r_max = (r >= g) & (r >= b);
  const auto g_max = (!r_max) & (g >= b);
  const auto face = select(select(UInt{2u}, UInt{1u}, g_max), UInt{0u}, r_max);
  const auto m = luisa::compute::max(r, luisa::compute::max(g, b));
  const auto inv_m = select(0.0f, 1.0f / luisa::compute::max(m, 1.0e-20f), m > 0.0f);
  const auto c1 = select(select(r, b, g_max), g, r_max) * inv_m;
  const auto c2 = select(select(g, r, g_max), b, r_max) * inv_m;

  constexpr float max_coord =
      static_cast<float>(rgb2spec_table_resolution - 1u);
  const auto u = inv_smoothstep_device(c1) * max_coord;
  const auto v = inv_smoothstep_device(c2) * max_coord;
  const auto w = inv_smoothstep_device(m) * max_coord;

  const auto uc = clamp(u, 0.0f, max_coord);
  const auto vc = clamp(v, 0.0f, max_coord);
  const auto wc = clamp(w, 0.0f, max_coord);

  const auto ix = min(cast<std::uint32_t>(uc), rgb2spec_table_resolution - 2u);
  const auto iy = min(cast<std::uint32_t>(vc), rgb2spec_table_resolution - 2u);
  const auto iz = min(cast<std::uint32_t>(wc), rgb2spec_table_resolution - 2u);

  const auto fx = clamp(uc - cast<float>(ix), 0.0f, 1.0f);
  const auto fy = clamp(vc - cast<float>(iy), 0.0f, 1.0f);
  const auto fz = clamp(wc - cast<float>(iz), 0.0f, 1.0f);

  constexpr std::uint32_t res = rgb2spec_table_resolution;
  constexpr std::uint32_t stride_x = res * res;
  constexpr std::uint32_t stride_face = res * res * res;
  const auto base = face * stride_face + ix * stride_x + iy * res + iz;

  const auto c000 = rgb2spec_buf.read(base).xyz();
  const auto c001 = rgb2spec_buf.read(base + 1u).xyz();
  const auto c010 = rgb2spec_buf.read(base + res).xyz();
  const auto c011 = rgb2spec_buf.read(base + res + 1u).xyz();
  const auto c100 = rgb2spec_buf.read(base + stride_x).xyz();
  const auto c101 = rgb2spec_buf.read(base + stride_x + 1u).xyz();
  const auto c110 = rgb2spec_buf.read(base + stride_x + res).xyz();
  const auto c111 = rgb2spec_buf.read(base + stride_x + res + 1u).xyz();

  const auto c00 = lerp(c000, c001, fz);
  const auto c01 = lerp(c010, c011, fz);
  const auto c10 = lerp(c100, c101, fz);
  const auto c11 = lerp(c110, c111, fz);
  const auto c0 = lerp(c00, c01, fy);
  const auto c1_v = lerp(c10, c11, fy);
  return lerp(c0, c1_v, fx);
}

[[nodiscard]] Float4 eval_sigmoid_poly(
    Expr<luisa::float3> coeffs, Expr<luisa::float4> lambdas) noexcept {
  const auto t =
      (clamp(lambdas, make_float4(spectral_lambda_min),
             make_float4(spectral_lambda_max)) -
       spectral_lambda_min) *
      (1.0f / spectral_lambda_span);
  const auto p = (coeffs.x * t + coeffs.y) * t + coeffs.z;
  return 0.5f + 0.5f * p * rsqrt(1.0f + p * p);
}

[[nodiscard]] Float4 wrap_fract_stratified(Expr<float> u) noexcept {
  const auto base = clamp(u, 0.0f, 0.99999994f);
  const auto u1 = base + 0.25f;
  const auto u2 = base + 0.50f;
  const auto u3 = base + 0.75f;
  return make_float4(base,
                     select(u1, u1 - 1.0f, u1 >= 1.0f),
                     select(u2, u2 - 1.0f, u2 >= 1.0f),
                     select(u3, u3 - 1.0f, u3 >= 1.0f));
}

[[nodiscard]] const Callable<luisa::float4(luisa::float4)> &
visible_wavelength_pdfs_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Float4 lambdas) noexcept {
      constexpr float a = static_cast<float>(wavelength_sample_a);
      constexpr float center = static_cast<float>(wavelength_sample_center);
      constexpr float pdf_scale = static_cast<float>(wavelength_sample_pdf_scale);
      const auto c = safe_cosh4(a * (lambdas - make_float4(center)));
      const auto pdf = make_float4(pdf_scale) / (c * c);
      const auto valid = (lambdas >= spectral_lambda_min) &&
                         (lambdas <= spectral_lambda_max);
      return select(make_float4(0.0f), pdf, valid);
    }};
    result.set_name("cycles_visible_wavelength_pdfs");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(float)> &
sample_visible_wavelengths_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Float u_lambda) noexcept {
      const auto u = wrap_fract_stratified(u_lambda);
      constexpr float inv_a = static_cast<float>(1.0 / wavelength_sample_a);
      constexpr float center = static_cast<float>(wavelength_sample_center);
      constexpr float t_lo = static_cast<float>(wavelength_sample_t_lo);
      constexpr float span = static_cast<float>(wavelength_sample_span);

      const auto l0 =
          clamp(center + inv_a * safe_atanh(t_lo + span * u.x),
                spectral_lambda_min, spectral_lambda_max);
      const auto l1 =
          clamp(center + inv_a * safe_atanh(t_lo + span * u.y),
                spectral_lambda_min, spectral_lambda_max);
      const auto l2 =
          clamp(center + inv_a * safe_atanh(t_lo + span * u.z),
                spectral_lambda_min, spectral_lambda_max);
      const auto l3 =
          clamp(center + inv_a * safe_atanh(t_lo + span * u.w),
                spectral_lambda_min, spectral_lambda_max);
      return make_float4(l0, l1, l2, l3);
    }};
    result.set_name("cycles_sample_visible_wavelengths");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(Buffer<luisa::float4>, luisa::float4)> &
sample_illuminant_4_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<luisa::float4> cie_buf, Float4 lambdas) noexcept {
      return sample_illuminant_4_impl(cie_buf, lambdas);
    }};
    result.set_name("cycles_sample_illuminant_4");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(
    Buffer<luisa::float4>, luisa::float3, luisa::float4)> &
rgb_to_spectrum_reflectance_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<luisa::float4> rgb2spec_buf, Float3 rgb,
                       Float4 lambdas) noexcept {
      const auto clamped = luisa::compute::max(rgb, make_float3(0.0f));
      const auto m = luisa::compute::max(
          clamped.x, luisa::compute::max(clamped.y, clamped.z));
      const auto mn = luisa::compute::min(
          clamped.x, luisa::compute::min(clamped.y, clamped.z));
      const auto is_gray = (m - mn) <= 1.0e-6f * luisa::compute::max(m, 1.0f);
      Float4 out = make_float4(m);
      $if(!is_gray) {
        const auto near_white = (mn > 0.5f) & (m <= 1.0f) & (m > 0.9f);
        const auto hdr = m > 1.0f;
        const auto base = select(0.0f, mn, near_white);
        const auto scale = select(select(1.0f, 2.0f * m, hdr),
                                  max(1.0f - mn, 1.0e-6f), near_white);
        const auto scaled = (clamped - make_float3(base)) / scale;
        const auto coeffs = lookup_rgb2spec_coeffs(rgb2spec_buf, scaled);
        out = make_float4(base) + scale * eval_sigmoid_poly(coeffs, lambdas);
      };
      return out;
    }};
    result.set_name("cycles_rgb_to_spectrum_reflectance");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float3(
    Buffer<luisa::float4>, luisa::float4, luisa::float4, luisa::float4, bool,
    luisa::float3, luisa::float3, luisa::float3)> &
spectrum_to_rgb_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<luisa::float4> cie_buf, Float4 radiance,
                       Float4 lambdas, Float4 pdfs, Bool secondary_terminated,
                       Float3 spec_xyz_to_r, Float3 spec_xyz_to_g,
                       Float3 spec_xyz_to_b) noexcept {
      const auto w0 =
          select(0.0f, radiance.x / pdfs.x, pdfs.x > 0.0f);
      const auto c0 = sample_cie_illuminant(cie_buf, lambdas.x).xyz();
      Float3 xyz = w0 * c0;
      $if(!secondary_terminated) {
        const auto w1 =
            select(0.0f, radiance.y / pdfs.y, pdfs.y > 0.0f);
        const auto w2 =
            select(0.0f, radiance.z / pdfs.z, pdfs.z > 0.0f);
        const auto w3 =
            select(0.0f, radiance.w / pdfs.w, pdfs.w > 0.0f);
        const auto c1 = sample_cie_illuminant(cie_buf, lambdas.y).xyz();
        const auto c2 = sample_cie_illuminant(cie_buf, lambdas.z).xyz();
        const auto c3 = sample_cie_illuminant(cie_buf, lambdas.w).xyz();
        xyz = (xyz + w1 * c1 + w2 * c2 + w3 * c3) * 0.25f;
      };
      const Float3 rgb = make_float3(dot(spec_xyz_to_r, xyz),
                                     dot(spec_xyz_to_g, xyz),
                                     dot(spec_xyz_to_b, xyz));
      const auto safe = !luisa::compute::dsl::isnan(rgb) &&
                        !luisa::compute::dsl::isinf(rgb);
      return select(make_float3(0.0f), rgb, safe);
    }};
    result.set_name("cycles_spectrum_to_rgb");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<float(
    Buffer<luisa::float4>, luisa::float4, luisa::float4, luisa::float4, bool)> &
spectrum_luminance_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<luisa::float4> cie_buf, Float4 reflectance,
                       Float4 lambdas, Float4 pdfs,
                       Bool secondary_terminated) noexcept {
      const auto illum = sample_illuminant_4_impl(cie_buf, lambdas);
      const auto y0 = sample_cie_illuminant(cie_buf, lambdas.x).y;
      const auto w0 =
          select(0.0f, (reflectance.x * illum.x) / pdfs.x, pdfs.x > 0.0f);
      Float lum = w0 * y0;
      $if(!secondary_terminated) {
        const auto w1 =
            select(0.0f, (reflectance.y * illum.y) / pdfs.y, pdfs.y > 0.0f);
        const auto w2 =
            select(0.0f, (reflectance.z * illum.z) / pdfs.z, pdfs.z > 0.0f);
        const auto w3 =
            select(0.0f, (reflectance.w * illum.w) / pdfs.w, pdfs.w > 0.0f);
        const auto y1 = sample_cie_illuminant(cie_buf, lambdas.y).y;
        const auto y2 = sample_cie_illuminant(cie_buf, lambdas.z).y;
        const auto y3 = sample_cie_illuminant(cie_buf, lambdas.w).y;
        lum = (lum + w1 * y1 + w2 * y2 + w3 * y3) * 0.25f;
      };
      return select(0.0f, lum,
                    !luisa::compute::dsl::isnan(lum) &
                        !luisa::compute::dsl::isinf(lum));
    }};
    result.set_name("cycles_spectrum_luminance");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(Buffer<float>, float, luisa::float4)> &
blackbody_spd_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<float> norm_buf, Float temperature_kelvin,
                       Float4 lambdas) noexcept {
      const auto temp = clamp(temperature_kelvin, blackbody_temp_min,
                              blackbody_temp_max);
      constexpr float c2 = 1.438776877e7f;
      const auto peak_lambda = clamp(2.897771955e6f / temp, spectral_lambda_min,
                                     spectral_lambda_max);
      // Overflow-free Planck ratio:
      // (lambda_peak/lambda)^5 * exp(x_peak - x) * (1 - exp(-x_peak)) / (1 - exp(-x)).
      const auto x_peak = c2 / (peak_lambda * temp);
      const auto denom_peak =
          select(x_peak * (1.0f - 0.5f * x_peak), 1.0f - exp(-x_peak),
                 x_peak > 1.0e-4f);
      const auto x = make_float4(c2) / (lambdas * temp);
      const auto denom =
          select(x * (make_float4(1.0f) - 0.5f * x), make_float4(1.0f) - exp(-x),
                 x > 1.0e-4f);
      const auto r = lambdas / peak_lambda;
      const auto r2 = r * r;
      const auto r5 = r2 * r2 * r;
      const auto exp_ratio = exp(clamp(make_float4(x_peak) - x, make_float4(-88.0f),
                                       make_float4(80.0f)));
      const auto rel =
          (exp_ratio * denom_peak) / max(r5 * denom, make_float4(1.0e-30f));
      const float log_min = std::log(blackbody_temp_min);
      const float inv_log_span =
          1.0f / (std::log(blackbody_temp_max) - std::log(blackbody_temp_min));
      const float max_idx = static_cast<float>(blackbody_lut_count - 1u);
      const auto pos =
          clamp((log(temp) - log_min) * inv_log_span * max_idx, 0.0f, max_idx);
      const auto i0 = min(cast<std::uint32_t>(pos), blackbody_lut_count - 2u);
      const auto frac = clamp(pos - cast<float>(i0), 0.0f, 1.0f);
      const auto n0 = norm_buf.read(i0);
      const auto n1 = norm_buf.read(i0 + 1u);
      const auto norm = exp(lerp(log(n0), log(n1), frac));
      return rel * norm;
    }};
    result.set_name("cycles_blackbody_spd");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(float, float, luisa::float4)> &
cauchy_ior_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Float ior_d, Float abbe_v, Float4 lambdas) noexcept {
      Float4 out = make_float4(ior_d);
      $if(abbe_v > 0.0f) {
        // Handle both entering (ior_d > 1) and exiting (ior_d < 1) media cleanly.
        const auto exiting = ior_d < 1.0f;
        const auto n_d = max(select(ior_d, 1.0f / max(ior_d, 1.0e-6f), exiting),
                             1.0001f);
        constexpr float inv_ld2 = 1.0f / (587.5618f * 587.5618f);
        constexpr float inv_diff =
            (1.0f / (486.1327f * 486.1327f)) - (1.0f / (656.2725f * 656.2725f));
        const auto b_cauchy = (n_d - 1.0f) / (abbe_v * inv_diff);
        const auto a_cauchy = n_d - b_cauchy * inv_ld2;
        const auto n_wl =
            max(make_float4(a_cauchy) + make_float4(b_cauchy) / (lambdas * lambdas),
                make_float4(1.0001f));
        out = select(n_wl, make_float4(1.0f) / n_wl, exiting);
      };
      return out;
    }};
    result.set_name("cycles_cauchy_ior");
    return result;
  }();
  return callable;
}

} // namespace

Float4 sample_illuminant_4(const SpectralTablesGpu &tables,
                           Expr<luisa::float4> lambdas) noexcept {
  return sample_illuminant_4_callable()(tables.cie_illuminant_buffer, lambdas);
}

SampledWavelengths sample_visible_wavelengths(Expr<float> u_lambda) noexcept {
  const Float4 lambdas = sample_visible_wavelengths_callable()(u_lambda);
  return {.lambdas = lambdas, .pdfs = visible_wavelength_pdfs(lambdas)};
}

Float4 visible_wavelength_pdfs(Expr<luisa::float4> lambdas) noexcept {
  return visible_wavelength_pdfs_callable()(lambdas);
}

Float4 rgb_to_spectrum_reflectance(const SpectralTablesGpu &tables,
                                   Expr<luisa::float3> rgb,
                                   Expr<luisa::float4> lambdas) noexcept {
  return rgb_to_spectrum_reflectance_callable()(tables.rgb2spec_buffer, rgb,
                                                lambdas);
}

Float4 rgb_to_spectrum_illuminant(const SpectralTablesGpu &tables,
                                  Expr<luisa::float3> rgb,
                                  Expr<luisa::float4> lambdas) noexcept {
  const auto illum = sample_illuminant_4(tables, lambdas);
  return rgb_to_spectrum_reflectance(tables, rgb, lambdas) * illum;
}

Float3 spectrum_to_rgb(const SpectralTablesGpu &tables,
                       Expr<luisa::float4> radiance,
                       const SampledWavelengths &wavelengths,
                       Expr<bool> secondary_terminated) noexcept {
  return spectrum_to_rgb_callable()(
      tables.cie_illuminant_buffer, radiance, wavelengths.lambdas,
      wavelengths.pdfs, secondary_terminated,
      make_float3(tables.spec_xyz_to_r), make_float3(tables.spec_xyz_to_g),
      make_float3(tables.spec_xyz_to_b));
}

Float3 reflectance_to_rgb(const SpectralTablesGpu &tables,
                          Expr<luisa::float4> reflectance,
                          const SampledWavelengths &wavelengths,
                          Expr<bool> secondary_terminated) noexcept {
  const auto illum = sample_illuminant_4(tables, wavelengths.lambdas);
  return spectrum_to_rgb(tables, reflectance * illum, wavelengths,
                         secondary_terminated);
}

Float spectrum_luminance(const SpectralTablesGpu &tables,
                         Expr<luisa::float4> reflectance,
                         const SampledWavelengths &wavelengths,
                         Expr<bool> secondary_terminated) noexcept {
  return spectrum_luminance_callable()(
      tables.cie_illuminant_buffer, reflectance, wavelengths.lambdas,
      wavelengths.pdfs, secondary_terminated);
}

Float4 blackbody_spd(const SpectralTablesGpu &tables,
                     Expr<float> temperature_kelvin,
                     Expr<luisa::float4> lambdas) noexcept {
  return blackbody_spd_callable()(tables.blackbody_norm_buffer,
                                  temperature_kelvin, lambdas);
}

Float4 wavelength_spd(Expr<float> center_nm,
                      Expr<luisa::float4> lambdas) noexcept {
  // Narrow-band Gaussian scaled to match Cycles' NODE_WAVELENGTH integral.
  constexpr float sigma = 5.0f;
  constexpr float inv_sigma = 1.0f / sigma;
  constexpr float norm = 106.856895f / (2.52f * 2.50662827f * sigma);
  const auto d = (lambdas - make_float4(center_nm)) * inv_sigma;
  const auto spd = norm * exp(-0.5f * d * d);
  const auto valid = (center_nm >= 380.0f) & (center_nm <= 780.0f);
  return select(make_float4(0.0f), spd, valid);
}

Float4 cauchy_ior(Expr<float> ior_d, Expr<float> abbe_v,
                  Expr<luisa::float4> lambdas) noexcept {
  return cauchy_ior_callable()(ior_d, abbe_v, lambdas);
}

} // namespace psycles::luisa_backend::cycles_spectrum
