#include <psycles/luisa/cycles_spectrum.h>
#include "thin_film_fresnel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::contract;
using namespace psycles::luisa_backend::cycles_spectrum;
using namespace psycles::luisa_backend::detail;

[[nodiscard]] bool approx(float a, float b, float tol) noexcept {
  return std::isfinite(a) && std::isfinite(b) &&
         std::abs(a - b) <= tol * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}

[[nodiscard]] bool approx3(luisa::float3 a, luisa::float3 b, float tol) noexcept {
  return approx(a.x, b.x, tol) && approx(a.y, b.y, tol) && approx(a.z, b.z, tol);
}

[[nodiscard]] double eval_planck_rel_ref(double lambda_nm, double temp_k) noexcept {
  constexpr double c2 = 1.438776877e7;
  const double peak_lambda = std::clamp(2.897771955e6 / temp_k, 360.0, 830.0);
  const double x_peak = c2 / (peak_lambda * temp_k);
  const double dp =
      x_peak > 1.0e-6 ? 1.0 - std::exp(-x_peak) : x_peak * (1.0 - 0.5 * x_peak);
  const double x = c2 / (lambda_nm * temp_k);
  const double d = x > 1.0e-6 ? 1.0 - std::exp(-x) : x * (1.0 - 0.5 * x);
  const double r = lambda_nm / peak_lambda;
  const double r5 = r * r * r * r * r;
  return (std::exp(std::clamp(x_peak - x, -700.0, 80.0)) * dp) /
         std::max(r5 * d, 1.0e-30);
}

[[nodiscard]] bool run_host_tests(const SpectralTablesHost &tables) {
  // 1. Wavelength importance sampling CDF bounds, monotonicity & PDF integral.
  const float l0 = sample_visible_wavelength_host(0.0f);
  const float l1 = sample_visible_wavelength_host(0.999999f);
  if (!approx(l0, spectral_lambda_min, 1.0e-3f) ||
      l1 < 829.0f || l1 > spectral_lambda_max + 1.0e-3f) {
    std::cerr << "FAIL wavelength bounds: l0=" << l0 << " l1=" << l1 << "\n";
    return false;
  }
  float prev = 0.0f;
  for (int i = 0; i <= 100; ++i) {
    const float u = static_cast<float>(i) / 100.0f * 0.999999f;
    const float wl = sample_visible_wavelength_host(u);
    if (wl < prev - 1.0e-4f || wl < spectral_lambda_min - 1.0e-3f ||
        wl > spectral_lambda_max + 1.0e-3f) {
      std::cerr << "FAIL wavelength monotonicity at u=" << u << " wl=" << wl << "\n";
      return false;
    }
    prev = wl;
  }

  constexpr int bins = 10000;
  const double dlambda =
      static_cast<double>(spectral_lambda_max - spectral_lambda_min) / bins;
  double pdf_integral = 0.0;
  for (int i = 0; i < bins; ++i) {
    const float wl =
        spectral_lambda_min + static_cast<float>((i + 0.5) * dlambda);
    pdf_integral += static_cast<double>(visible_wavelength_pdf_host(wl)) * dlambda;
  }
  if (std::abs(pdf_integral - 1.0) > 1.0e-4) {
    std::cerr << "FAIL visible wavelength PDF integral=" << pdf_integral << "\n";
    return false;
  }
  if (!(visible_wavelength_pdf_host(538.0f) > visible_wavelength_pdf_host(400.0f) &&
        visible_wavelength_pdf_host(538.0f) > visible_wavelength_pdf_host(700.0f))) {
    std::cerr << "FAIL visible wavelength PDF peak ordering\n";
    return false;
  }

  // 2. rgb2spec 17x17x17 cube + extreme edge round-trip error < 2.5e-4.
  float max_err = 0.0f;
  luisa::float3 worst_rgb{};
  constexpr int N = 17;
  for (int r = 0; r < N; ++r) {
    for (int g = 0; g < N; ++g) {
      for (int b = 0; b < N; ++b) {
        const luisa::float3 in{static_cast<float>(r) / (N - 1),
                               static_cast<float>(g) / (N - 1),
                               static_cast<float>(b) / (N - 1)};
        const auto xyz = integrate_reflectance_xyz_host(tables, in);
        const luisa::float3 out{
            tables.spec_xyz_to_r.x * xyz.x + tables.spec_xyz_to_r.y * xyz.y +
                tables.spec_xyz_to_r.z * xyz.z,
            tables.spec_xyz_to_g.x * xyz.x + tables.spec_xyz_to_g.y * xyz.y +
                tables.spec_xyz_to_g.z * xyz.z,
            tables.spec_xyz_to_b.x * xyz.x + tables.spec_xyz_to_b.y * xyz.y +
                tables.spec_xyz_to_b.z * xyz.z};
        const float err = std::max({std::abs(out.x - in.x),
                                    std::abs(out.y - in.y),
                                    std::abs(out.z - in.z)});
        if (err > max_err) {
          max_err = err;
          worst_rgb = in;
        }
      }
    }
  }
  // Check near-white decomposition (0.99, 0.999, 0.95).
  for (const luisa::float3 nw :
       {luisa::float3{0.99f, 0.99f, 0.99f},
        luisa::float3{0.999f, 0.999f, 0.999f},
        luisa::float3{0.99f, 0.95f, 0.97f}}) {
    const auto xyz = integrate_reflectance_xyz_host(tables, nw);
    const luisa::float3 out{
        tables.spec_xyz_to_r.x * xyz.x + tables.spec_xyz_to_r.y * xyz.y +
            tables.spec_xyz_to_r.z * xyz.z,
        tables.spec_xyz_to_g.x * xyz.x + tables.spec_xyz_to_g.y * xyz.y +
            tables.spec_xyz_to_g.z * xyz.z,
        tables.spec_xyz_to_b.x * xyz.x + tables.spec_xyz_to_b.y * xyz.y +
            tables.spec_xyz_to_b.z * xyz.z};
    const float err = std::max({std::abs(out.x - nw.x),
                                std::abs(out.y - nw.y),
                                std::abs(out.z - nw.z)});
    if (err > 1.0e-5f) {
      std::cerr << "FAIL near-white err=" << err << " on {" << nw.x << ","
                << nw.y << "," << nw.z << "}\n";
      return false;
    }
  }
  if (max_err > 2.5e-4f) {
    std::cerr << "FAIL rgb2spec 17^3 max_err=" << max_err << " at {"
              << worst_rgb.x << "," << worst_rgb.y << "," << worst_rgb.z
              << "}\n";
    return false;
  }

  // 3. Verify white D65 illuminant (1,1,1) maps to RGB (1,1,1).
  luisa::float3 d65_xyz{};
  for (std::size_t i = 0; i < spectral_cie_sample_count; ++i) {
    const float w =
        (i == 0 || i + 1 == spectral_cie_sample_count) ? 0.5f : 1.0f;
    const auto &e = tables.cie_illuminant[i];
    d65_xyz.x += w * e.x * e.w;
    d65_xyz.y += w * e.y * e.w;
    d65_xyz.z += w * e.z * e.w;
  }
  const luisa::float3 d65_rgb{
      tables.spec_xyz_to_r.x * d65_xyz.x + tables.spec_xyz_to_r.y * d65_xyz.y +
          tables.spec_xyz_to_r.z * d65_xyz.z,
      tables.spec_xyz_to_g.x * d65_xyz.x + tables.spec_xyz_to_g.y * d65_xyz.y +
          tables.spec_xyz_to_g.z * d65_xyz.z,
      tables.spec_xyz_to_b.x * d65_xyz.x + tables.spec_xyz_to_b.y * d65_xyz.y +
          tables.spec_xyz_to_b.z * d65_xyz.z};
  if (!approx3(d65_rgb, {1.0f, 1.0f, 1.0f}, 2.0e-5f)) {
    std::cerr << "FAIL D65 white rgb={" << d65_rgb.x << "," << d65_rgb.y
              << "," << d65_rgb.z << "}\n";
    return false;
  }

  // 4. Verify blackbody_norm integrates Y_rel to 1.0.
  const double log_min = std::log(static_cast<double>(blackbody_temp_min));
  const double inv_log_span =
      1.0 / (std::log(static_cast<double>(blackbody_temp_max)) - log_min);
  for (const float T : {650.0f, 1500.0f, 3200.0f, 6500.0f, 14000.0f}) {
    const double pos =
        std::clamp((std::log(static_cast<double>(T)) - log_min) * inv_log_span *
                       (blackbody_lut_count - 1u),
                   0.0, static_cast<double>(blackbody_lut_count - 1u));
    const auto i0 = std::min(static_cast<std::size_t>(pos),
                             static_cast<std::size_t>(blackbody_lut_count - 2u));
    const double frac = pos - static_cast<double>(i0);
    const double norm =
        std::exp(std::log(static_cast<double>(tables.blackbody_norm[i0])) *
                     (1.0 - frac) +
                 std::log(static_cast<double>(tables.blackbody_norm[i0 + 1u])) *
                     frac);
    double Y = 0.0;
    for (std::size_t i = 0; i < spectral_cie_sample_count; ++i) {
      const double wl = static_cast<double>(spectral_lambda_min + i);
      const double w =
          (i == 0 || i + 1 == spectral_cie_sample_count) ? 0.5 : 1.0;
      Y += w * tables.cie_illuminant[i].y * eval_planck_rel_ref(wl, T) * norm;
    }
    if (std::abs(Y - 1.0) > 5.0e-4) {
      std::cerr << "FAIL blackbody Y integral at T=" << T << " Y=" << Y << "\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const std::string_view backend = argc > 1 ? argv[1] : "fallback";
  const ShaderColorSpace cs{};
  const auto host_tables = build_spectral_tables_host(cs);
  if (!run_host_tests(host_tables)) {
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto gpu_tables = upload_spectral_tables(device, stream, cs);

  // Test colors for Monte Carlo round-trip on GPU.
  constexpr std::array<luisa::float3, 8> test_colors{{
      {1.0f, 1.0f, 1.0f},
      {0.5f, 0.5f, 0.5f},
      {0.82f, 0.11f, 0.14f},
      {0.12f, 0.72f, 0.16f},
      {0.10f, 0.22f, 0.85f},
      {0.91f, 0.84f, 0.12f},
      {0.99f, 0.99f, 0.99f},
      {0.18f, 0.65f, 0.78f},
  }};
  constexpr std::uint32_t color_count =
      static_cast<std::uint32_t>(test_colors.size());
  constexpr std::uint32_t sample_count = 4096u;

  auto color_buf = device.create_buffer<luisa::float3>(color_count);
  auto refl_out = device.create_buffer<luisa::float3>(color_count * sample_count);
  auto illum_out = device.create_buffer<luisa::float3>(color_count * sample_count);
  auto bb_out = device.create_buffer<luisa::float4>(4u * sample_count);
  auto misc_out = device.create_buffer<luisa::float4>(6u);

  const std::array<float, 4> bb_temps{800.0f, 3000.0f, 6500.0f, 15000.0f};
  auto bb_temp_buf = device.create_buffer<float>(4u);

  Kernel1D mc_kernel = [&](BufferVar<luisa::float3> colors,
                           BufferVar<float> temps,
                           BufferVar<luisa::float3> out_refl,
                           BufferVar<luisa::float3> out_illum,
                           BufferVar<luisa::float4> out_bb) {
    const UInt gid = dispatch_id().x;
    const UInt sample_idx = gid % sample_count;
    const Float u_lambda = (cast<float>(sample_idx) + 0.5f) /
                           static_cast<float>(sample_count);
    const auto sw = sample_visible_wavelengths(u_lambda);
    const UInt color_idx = gid / sample_count;
    $if(color_idx < color_count) {
      const Float3 rgb = colors->read(color_idx);
      const Spectrum s_refl = Spectrum::from_rgb_reflectance(
          gpu_tables.get(), rgb, &sw, psycles::contract::SpectralMode::hero4);
      const Spectrum s_illum = Spectrum::from_rgb_illuminant(
          gpu_tables.get(), rgb, &sw, psycles::contract::SpectralMode::hero4);
      out_refl->write(gid, reflectance_to_rgb(s_refl, gpu_tables.get(), &sw,
                                              false));
      out_illum->write(gid, to_rgb(s_illum, gpu_tables.get(), &sw, false));
    };
    $if(color_idx < 4u) {
      const Float T = temps->read(color_idx);
      const Float4 spd = blackbody_spd(*gpu_tables, T, sw.lambdas);
      const Float3 rgb = spectrum_to_rgb(*gpu_tables, spd, sw, false);
      const Float lum = dot(make_float3(gpu_tables->rgb_to_y), rgb);
      out_bb->write(color_idx * sample_count + sample_idx,
                    make_float4(rgb.x, rgb.y, rgb.z, lum));
    };
  };

  Kernel1D optics_kernel = [&](BufferVar<luisa::float4> out) {
    const UInt idx = dispatch_id().x;
    $if(idx == 0u) {
      const Float4 lambdas = make_float4(450.0f, 520.0f, 587.5618f, 680.0f);
      const Float4 enter = cauchy_ior(1.5f, 30.0f, lambdas);
      const Float4 exit = cauchy_ior(1.0f / 1.5f, 30.0f, lambdas);
      out->write(0u, enter);
      out->write(1u, enter * exit);

      // Quarter-wave MgF2 (n2=1.38) on glass (n3=1.52) at lambda=550nm:
      // d = 550 / (4 * 1.38) = 99.63768nm -> anti-reflection min ~0.012598.
      // At half-wave d = 199.27536nm -> absentee layer ~0.042580.
      const Float4 wls_550 = make_float4(550.0f, 550.0f, 550.0f, 550.0f);
      const auto qw = thin_film_dielectric_fresnel_spectral(
          99.63768f, 1.38f, make_float4(1.52f),
          make_float4(0.04258025f), 1.0f, wls_550);
      const auto hw = thin_film_dielectric_fresnel_spectral(
          199.27536f, 1.38f, make_float4(1.52f),
          make_float4(0.04258025f), 1.0f, wls_550);
      const auto f82 = thin_film_f82_fresnel_spectral(
          320.0f, 1.45f, make_float4(0.95f, 0.85f, 0.6f, 0.4f),
          make_float4(1.0f, 2.0f, 3.0f, 4.0f), 0.85f, lambdas);
      const auto cond = thin_film_conductor_fresnel_spectral(
          280.0f, 1.38f, make_float4(0.2f, 0.5f, 1.2f, 1.5f),
          make_float4(3.5f, 2.5f, 2.0f, 1.8f), 0.7f, lambdas);
      out->write(2u, qw.reflectance);
      out->write(3u, hw.reflectance);
      out->write(4u, f82);
      out->write(5u, cond);
    };
  };

  auto mc_shader = device.compile(mc_kernel);
  auto optics_shader = device.compile(optics_kernel);

  std::vector<luisa::float3> host_refl(color_count * sample_count);
  std::vector<luisa::float3> host_illum(color_count * sample_count);
  std::vector<luisa::float4> host_bb(4u * sample_count);
  std::array<luisa::float4, 6u> host_misc{};

  stream << color_buf.copy_from(luisa::span{test_colors})
         << bb_temp_buf.copy_from(luisa::span{bb_temps})
         << mc_shader(color_buf, bb_temp_buf, refl_out, illum_out, bb_out)
                .dispatch(color_count * sample_count)
         << optics_shader(misc_out).dispatch(1u)
         << refl_out.copy_to(luisa::span{host_refl})
         << illum_out.copy_to(luisa::span{host_illum})
         << bb_out.copy_to(luisa::span{host_bb})
         << misc_out.copy_to(luisa::span{host_misc})
         << synchronize();

  // Verify MC round-trips (< 5e-4 across 4096 stratified samples).
  for (std::uint32_t c = 0; c < color_count; ++c) {
    double rx = 0.0, ry = 0.0, rz = 0.0;
    double ix = 0.0, iy = 0.0, iz = 0.0;
    for (std::uint32_t s = 0; s < sample_count; ++s) {
      const auto &vr = host_refl[c * sample_count + s];
      const auto &vi = host_illum[c * sample_count + s];
      rx += vr.x;
      ry += vr.y;
      rz += vr.z;
      ix += vi.x;
      iy += vi.y;
      iz += vi.z;
    }
    const luisa::float3 mean_r{static_cast<float>(rx / sample_count),
                               static_cast<float>(ry / sample_count),
                               static_cast<float>(rz / sample_count)};
    const luisa::float3 mean_i{static_cast<float>(ix / sample_count),
                               static_cast<float>(iy / sample_count),
                               static_cast<float>(iz / sample_count)};
    if (!approx3(mean_r, test_colors[c], 5.0e-4f) ||
        !approx3(mean_i, test_colors[c], 5.0e-4f)) {
      std::cerr << "FAIL GPU MC round-trip on " << backend << " color " << c
                << " refl={" << mean_r.x << "," << mean_r.y << "," << mean_r.z
                << "} illum={" << mean_i.x << "," << mean_i.y << ","
                << mean_i.z << "}\n";
      return EXIT_FAILURE;
    }
  }

  // Verify Blackbody GPU luminance Y == 1.0 and exact match to CIE Planck RGB.
  const double log_min_bb = std::log(static_cast<double>(blackbody_temp_min));
  const double inv_log_span_bb =
      1.0 / (std::log(static_cast<double>(blackbody_temp_max)) - log_min_bb);
  for (std::uint32_t t = 0; t < 4u; ++t) {
    const float T = bb_temps[t];
    const double pos =
        std::clamp((std::log(static_cast<double>(T)) - log_min_bb) *
                       inv_log_span_bb * (blackbody_lut_count - 1u),
                   0.0, static_cast<double>(blackbody_lut_count - 1u));
    const auto i0 = std::min(static_cast<std::size_t>(pos),
                             static_cast<std::size_t>(blackbody_lut_count - 2u));
    const double frac = pos - static_cast<double>(i0);
    const double norm =
        std::exp(std::log(static_cast<double>(host_tables.blackbody_norm[i0])) *
                     (1.0 - frac) +
                 std::log(static_cast<double>(
                     host_tables.blackbody_norm[i0 + 1u])) *
                     frac);
    double ex_x = 0.0, ex_y = 0.0, ex_z = 0.0;
    for (std::size_t i = 0; i < spectral_cie_sample_count; ++i) {
      const double wl = static_cast<double>(spectral_lambda_min + i);
      const double w =
          (i == 0 || i + 1 == spectral_cie_sample_count) ? 0.5 : 1.0;
      const double s = eval_planck_rel_ref(wl, T) * norm;
      ex_x += w * host_tables.cie_illuminant[i].x * s;
      ex_y += w * host_tables.cie_illuminant[i].y * s;
      ex_z += w * host_tables.cie_illuminant[i].z * s;
    }
    const luisa::float3 expected_bb{
        static_cast<float>(host_tables.spec_xyz_to_r.x * ex_x +
                           host_tables.spec_xyz_to_r.y * ex_y +
                           host_tables.spec_xyz_to_r.z * ex_z),
        static_cast<float>(host_tables.spec_xyz_to_g.x * ex_x +
                           host_tables.spec_xyz_to_g.y * ex_y +
                           host_tables.spec_xyz_to_g.z * ex_z),
        static_cast<float>(host_tables.spec_xyz_to_b.x * ex_x +
                           host_tables.spec_xyz_to_b.y * ex_y +
                           host_tables.spec_xyz_to_b.z * ex_z)};

    double sum_lum = 0.0;
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    for (std::uint32_t s = 0; s < sample_count; ++s) {
      const auto &v = host_bb[t * sample_count + s];
      if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
          !std::isfinite(v.w)) {
        std::cerr << "FAIL non-finite blackbody output on " << backend << "\n";
        return EXIT_FAILURE;
      }
      sum_r += v.x;
      sum_g += v.y;
      sum_b += v.z;
      sum_lum += v.w;
    }
    const float mean_lum = static_cast<float>(sum_lum / sample_count);
    const luisa::float3 mean_rgb{static_cast<float>(sum_r / sample_count),
                                 static_cast<float>(sum_g / sample_count),
                                 static_cast<float>(sum_b / sample_count)};
    if (!approx(mean_lum, 1.0f, 3.0e-3f) ||
        !approx3(mean_rgb, expected_bb, 3.0e-3f)) {
      std::cerr << "FAIL GPU blackbody at T=" << T << " on " << backend
                << " Y=" << mean_lum << " actual={" << mean_rgb.x << ","
                << mean_rgb.y << "," << mean_rgb.z << "} expected={"
                << expected_bb.x << "," << expected_bb.y << ","
                << expected_bb.z << "}\n";
      return EXIT_FAILURE;
    }
  }

  // Verify Cauchy entering/exiting identity & chromatic dispersion ordering.
  const auto enter = host_misc[0];
  const auto prod = host_misc[1];
  if (!(enter.x > enter.y && enter.y > enter.z && enter.z > enter.w) ||
      !approx(enter.z, 1.5f, 1.0e-5f) ||
      !approx(prod.x, 1.0f, 1.0e-5f) || !approx(prod.y, 1.0f, 1.0e-5f) ||
      !approx(prod.z, 1.0f, 1.0e-5f) || !approx(prod.w, 1.0f, 1.0e-5f)) {
    std::cerr << "FAIL Cauchy dispersion enter={" << enter.x << "," << enter.y
              << "," << enter.z << "," << enter.w << "} prod={" << prod.x
              << "," << prod.y << "," << prod.z << "," << prod.w << "}\n";
    return EXIT_FAILURE;
  }

  // Verify Monochromatic Airy quarter-wave & half-wave & energy conservation.
  const auto qw = host_misc[2];
  const auto hw = host_misc[3];
  const auto f82 = host_misc[4];
  const auto cond = host_misc[5];
  if (!approx(qw.x, 0.012598f, 1.0e-4f) ||
      !approx(hw.x, 0.042580f, 1.0e-4f)) {
    std::cerr << "FAIL Airy thin-film qw=" << qw.x << " hw=" << hw.x << "\n";
    return EXIT_FAILURE;
  }
  for (const auto &v : {f82, cond}) {
    for (float c : {v.x, v.y, v.z, v.w}) {
      if (!std::isfinite(c) || c < 0.0f || c > 1.0f) {
        std::cerr << "FAIL thin-film spectral energy conservation c=" << c << "\n";
        return EXIT_FAILURE;
      }
    }
  }

  std::cout << "test_luisa_cycles_spectrum PASSED on " << backend << "\n";
  return EXIT_SUCCESS;
}
