#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_spectrum.h> through the Psycles::luisa target."
#endif

#include <psycles/contract/scene.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>

#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

namespace psycles::luisa_backend::cycles_spectrum {

inline constexpr std::uint32_t spectral_wavelength_count = 4u;
inline constexpr float spectral_lambda_min = 360.0f;
inline constexpr float spectral_lambda_max = 830.0f;
inline constexpr float spectral_lambda_span =
    spectral_lambda_max - spectral_lambda_min;
inline constexpr std::uint32_t spectral_cie_sample_count = 471u;
inline constexpr std::uint32_t rgb2spec_table_resolution = 96u;
inline constexpr std::uint32_t blackbody_lut_count = 256u;
inline constexpr float blackbody_temp_min = 600.0f;
inline constexpr float blackbody_temp_max = 20000.0f;

// PBRT-v4 SampleVisibleWavelengths sech^2 importance-sampling parameters
// centered at 538 nm over [360, 830] nm.
inline constexpr double wavelength_sample_a = 0.0072;
inline constexpr double wavelength_sample_center = 538.0;
inline constexpr double wavelength_sample_t_lo = -0.8569189982493341;
inline constexpr double wavelength_sample_t_hi = 0.9706040286366544;
inline constexpr double wavelength_sample_span = 1.8275230268859885;
inline constexpr double wavelength_sample_pdf_scale = 0.003939758845096658;

struct SampledWavelengths {
  luisa::compute::Float4 lambdas;
  luisa::compute::Float4 pdfs;
};

struct SpectralTablesHost {
  // .xyz = CMF(lambda) / Y_int, .w = D65_norm(lambda)
  luisa::vector<luisa::float4> cie_illuminant;
  // 3 faces x res^3 entries: .xyz = (c0, c1, c2), .w = 0.0f
  luisa::vector<luisa::float4> rgb2spec_coeffs;
  // 256 entries over log-space T in [600K, 20000K], scaling peak-normalized
  // Planck radiance so integral(spd * y_bar) / Y_int == 1.0.
  luisa::vector<float> blackbody_norm;
  luisa::float3 spec_xyz_to_r{};
  luisa::float3 spec_xyz_to_g{};
  luisa::float3 spec_xyz_to_b{};
  luisa::float3 rgb_to_y{};
};

struct SpectralTablesGpu {
  luisa::compute::Buffer<luisa::float4> cie_illuminant_buffer;
  luisa::compute::Buffer<luisa::float4> rgb2spec_buffer;
  luisa::compute::Buffer<float> blackbody_norm_buffer;
  luisa::float3 spec_xyz_to_r{};
  luisa::float3 spec_xyz_to_g{};
  luisa::float3 spec_xyz_to_b{};
  luisa::float3 rgb_to_y{};
};

// Host builders & test/CPU reference utilities.
[[nodiscard]] SpectralTablesHost build_spectral_tables_host(
    const contract::ShaderColorSpace &color_space);

[[nodiscard]] std::shared_ptr<SpectralTablesGpu> upload_spectral_tables(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const contract::ShaderColorSpace &color_space);

[[nodiscard]] float sample_visible_wavelength_host(float u) noexcept;
[[nodiscard]] float visible_wavelength_pdf_host(float lambda_nm) noexcept;

[[nodiscard]] std::array<float, 3u> eval_rgb2spec_coeffs_host(
    const SpectralTablesHost &tables, luisa::float3 rgb) noexcept;

[[nodiscard]] luisa::float4 eval_rgb2spec_reflectance_host(
    const SpectralTablesHost &tables, luisa::float3 rgb,
    luisa::float4 lambdas) noexcept;

[[nodiscard]] luisa::float3 integrate_reflectance_xyz_host(
    const SpectralTablesHost &tables,
    luisa::float3 rgb) noexcept;

// Device DSL wavelength sampling & spectral color conversions.
[[nodiscard]] SampledWavelengths sample_visible_wavelengths(
    luisa::compute::Expr<float> u_lambda) noexcept;

[[nodiscard]] luisa::compute::Float4 visible_wavelength_pdfs(
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float4 rgb_to_spectrum_reflectance(
    const SpectralTablesGpu &tables, luisa::compute::Expr<luisa::float3> rgb,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float4 rgb_to_spectrum_illuminant(
    const SpectralTablesGpu &tables, luisa::compute::Expr<luisa::float3> rgb,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float4 sample_illuminant_4(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float3 spectrum_to_rgb(
    const SpectralTablesGpu &tables, luisa::compute::Expr<luisa::float4> radiance,
    const SampledWavelengths &wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept;

[[nodiscard]] luisa::compute::Float3 reflectance_to_rgb(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float4> reflectance,
    const SampledWavelengths &wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept;

[[nodiscard]] luisa::compute::Float spectrum_luminance(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float4> reflectance,
    const SampledWavelengths &wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept;

[[nodiscard]] luisa::compute::Float4 blackbody_spd(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<float> temperature_kelvin,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float4 wavelength_spd(
    luisa::compute::Expr<float> center_nm,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

[[nodiscard]] luisa::compute::Float4 cauchy_ior(
    luisa::compute::Expr<float> ior_d,
    luisa::compute::Expr<float> abbe_v,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept;

// Scale-invariant RGB -> reflectance spectrum lookup for quantities such as
// subsurface scattering radius that span [1e-5, 1e3] where raw rgb2spec dark/HDR
// branches differ in LUT precision.
[[nodiscard]] inline luisa::compute::Float4 scale_invariant_rgb_to_spectrum(
    const SpectralTablesGpu &tables, luisa::compute::Expr<luisa::float3> rgb,
    luisa::compute::Expr<luisa::float4> lambdas) noexcept {
  const auto clamped =
      luisa::compute::max(rgb, luisa::compute::make_float3(0.0f));
  const auto m = luisa::compute::max(
      clamped.x, luisa::compute::max(clamped.y, clamped.z));
  const auto mn = luisa::compute::min(
      clamped.x, luisa::compute::min(clamped.y, clamped.z));
  const auto is_gray =
      (m - mn) <= 1.0e-6f * luisa::compute::max(m, 1.0f);
  luisa::compute::Float4 out = luisa::compute::make_float4(m);
  $if(!is_gray & (m > 0.0f)) {
    const auto normalized =
        clamped * (0.5f / luisa::compute::max(m, 1.0e-30f));
    out = (2.0f * m) * rgb_to_spectrum_reflectance(tables, normalized, lambdas);
  };
  return out;
}

[[nodiscard]] inline luisa::compute::Float avg_active(
    luisa::compute::Expr<luisa::float4> v,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  return luisa::compute::select(
      (v.x + v.y + v.z + v.w) * 0.25f, v.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float max_active(
    luisa::compute::Expr<luisa::float4> v,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  return luisa::compute::select(
      luisa::compute::max(luisa::compute::max(v.x, v.y),
                          luisa::compute::max(v.z, v.w)),
      v.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float max_abs_active(
    luisa::compute::Expr<luisa::float4> v,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  const auto a = luisa::compute::abs(v);
  return luisa::compute::select(
      luisa::compute::max(luisa::compute::max(a.x, a.y),
                          luisa::compute::max(a.z, a.w)),
      a.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float min_active(
    luisa::compute::Expr<luisa::float4> v,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  return luisa::compute::select(
      luisa::compute::min(luisa::compute::min(v.x, v.y),
                          luisa::compute::min(v.z, v.w)),
      v.x, secondary_terminated);
}

// Transport wrapper: holds Float3 in RGB mode and Float4 in Spectral mode.
// Copy constructor allocates a fresh DSL Var; copy assignment emits a DSL
// assignment to the underlying variable without changing the active variant.
class Spectrum final {
private:
  std::variant<luisa::compute::Float3, luisa::compute::Float4> _data;

  explicit Spectrum(luisa::compute::Float3 value) noexcept
      : _data{std::move(value)} {}
  explicit Spectrum(luisa::compute::Float4 value) noexcept
      : _data{std::move(value)} {}

public:
  // Default construction selects RGB transport with a zero-initialized lane
  // var. Host-side path-kernel contexts rely on this so that they stay
  // aggregate-initializable; spectral pipelines must build their transport
  // explicitly (zero(true) / constant(true, ...) / from_lanes(...)).
  Spectrum() noexcept : _data{luisa::compute::Float3{}} {}

  Spectrum(const Spectrum &other) noexcept
      : _data{other.is_spectral()
                  ? std::variant<luisa::compute::Float3, luisa::compute::Float4>{
                        luisa::compute::def<luisa::float4>(other.lanes())}
                  : std::variant<luisa::compute::Float3, luisa::compute::Float4>{
                        luisa::compute::def<luisa::float3>(other.rgb())}} {}

  Spectrum(Spectrum &&) noexcept = default;

  Spectrum &operator=(const Spectrum &other) noexcept {
    LUISA_ASSERT(is_spectral() == other.is_spectral(),
                 "Spectrum assignment requires matching spectral mode.");
    if (is_spectral()) {
      std::get<luisa::compute::Float4>(_data) = other.lanes();
    } else {
      std::get<luisa::compute::Float3>(_data) = other.rgb();
    }
    return *this;
  }

  Spectrum &operator=(Spectrum &&other) noexcept {
    return *this = static_cast<const Spectrum &>(other);
  }

  // Raw lane assignment keeps the active variant and only rebinds its value.
  // It mirrors from_rgb_raw()/from_lanes() for already-constructed transports.
  void operator=(luisa::compute::Expr<luisa::float3> value) noexcept {
    LUISA_ASSERT(!is_spectral(),
                 "Assigning RGB lanes to a spectral Spectrum; use from_lanes().");
    std::get<luisa::compute::Float3>(_data) = std::move(value);
  }

  void operator=(luisa::compute::Expr<luisa::float4> value) noexcept {
    LUISA_ASSERT(is_spectral(),
                 "Assigning spectral lanes to an RGB Spectrum; use from_rgb_raw().");
    std::get<luisa::compute::Float4>(_data) = std::move(value);
  }

  [[nodiscard]] bool is_spectral() const noexcept {
    return std::holds_alternative<luisa::compute::Float4>(_data);
  }

  [[nodiscard]] static Spectrum zero(bool spectral) noexcept {
    return constant(spectral, 0.0f);
  }

  [[nodiscard]] static Spectrum one(bool spectral) noexcept {
    return constant(spectral, 1.0f);
  }

  [[nodiscard]] static Spectrum constant(
      bool spectral, luisa::compute::Expr<float> scalar) noexcept {
    if (spectral) {
      return Spectrum{luisa::compute::make_float4(scalar)};
    }
    return Spectrum{luisa::compute::make_float3(scalar)};
  }

  [[nodiscard]] static Spectrum from_rgb_raw(
      luisa::compute::Expr<luisa::float3> value) noexcept {
    return Spectrum{luisa::compute::Float3{value}};
  }

  [[nodiscard]] static Spectrum from_lanes(
      luisa::compute::Expr<luisa::float4> value) noexcept {
    return Spectrum{luisa::compute::Float4{value}};
  }

  [[nodiscard]] static Spectrum from_rgb_reflectance(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SampledWavelengths *wavelengths, bool spectral) noexcept {
    if (!spectral) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
                 "Spectral conversion requires tables and wavelengths.");
    return from_lanes(
        rgb_to_spectrum_reflectance(*tables, value, wavelengths->lambdas));
  }

  [[nodiscard]] static Spectrum from_rgb_illuminant(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SampledWavelengths *wavelengths, bool spectral) noexcept {
    if (!spectral) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
                 "Spectral conversion requires tables and wavelengths.");
    return from_lanes(
        rgb_to_spectrum_illuminant(*tables, value, wavelengths->lambdas));
  }

  [[nodiscard]] const luisa::compute::Float3 &rgb() const noexcept {
    LUISA_ASSERT(!is_spectral(), "Accessed RGB lanes of spectral Spectrum.");
    return std::get<luisa::compute::Float3>(_data);
  }
  [[nodiscard]] luisa::compute::Float3 &rgb() noexcept {
    LUISA_ASSERT(!is_spectral(), "Accessed RGB lanes of spectral Spectrum.");
    return std::get<luisa::compute::Float3>(_data);
  }

  [[nodiscard]] const luisa::compute::Float4 &lanes() const noexcept {
    LUISA_ASSERT(is_spectral(), "Accessed 4-wavelength lanes of RGB Spectrum.");
    return std::get<luisa::compute::Float4>(_data);
  }
  [[nodiscard]] luisa::compute::Float4 &lanes() noexcept {
    LUISA_ASSERT(is_spectral(), "Accessed 4-wavelength lanes of RGB Spectrum.");
    return std::get<luisa::compute::Float4>(_data);
  }

  [[nodiscard]] luisa::compute::Float hero() const noexcept {
    return is_spectral() ? lanes().x : rgb().x;
  }

  void terminate_secondary() noexcept {
    if (is_spectral()) {
      lanes() = luisa::compute::make_float4(lanes().x, 0.0f, 0.0f, 0.0f);
    }
  }

  Spectrum &operator+=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(is_spectral() == rhs.is_spectral(), "Spectrum mode mismatch.");
    if (is_spectral()) {
      lanes() += rhs.lanes();
    } else {
      rgb() += rhs.rgb();
    }
    return *this;
  }

  Spectrum &operator-=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(is_spectral() == rhs.is_spectral(), "Spectrum mode mismatch.");
    if (is_spectral()) {
      lanes() -= rhs.lanes();
    } else {
      rgb() -= rhs.rgb();
    }
    return *this;
  }

  Spectrum &operator*=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(is_spectral() == rhs.is_spectral(), "Spectrum mode mismatch.");
    if (is_spectral()) {
      lanes() *= rhs.lanes();
    } else {
      rgb() *= rhs.rgb();
    }
    return *this;
  }

  Spectrum &operator/=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(is_spectral() == rhs.is_spectral(), "Spectrum mode mismatch.");
    if (is_spectral()) {
      lanes() /= rhs.lanes();
    } else {
      rgb() /= rhs.rgb();
    }
    return *this;
  }

  Spectrum &operator*=(luisa::compute::Expr<float> s) noexcept {
    if (is_spectral()) {
      lanes() *= s;
    } else {
      rgb() *= s;
    }
    return *this;
  }

  Spectrum &operator/=(luisa::compute::Expr<float> s) noexcept {
    if (is_spectral()) {
      lanes() /= s;
    } else {
      rgb() /= s;
    }
    return *this;
  }

  [[nodiscard]] friend Spectrum operator+(const Spectrum &a,
                                          const Spectrum &b) noexcept {
    auto out = a;
    out += b;
    return out;
  }

  [[nodiscard]] friend Spectrum operator-(const Spectrum &a,
                                          const Spectrum &b) noexcept {
    auto out = a;
    out -= b;
    return out;
  }

  [[nodiscard]] friend Spectrum operator*(const Spectrum &a,
                                          const Spectrum &b) noexcept {
    auto out = a;
    out *= b;
    return out;
  }

  [[nodiscard]] friend Spectrum operator/(const Spectrum &a,
                                          const Spectrum &b) noexcept {
    auto out = a;
    out /= b;
    return out;
  }

  [[nodiscard]] friend Spectrum operator*(const Spectrum &a,
                                          luisa::compute::Expr<float> s) noexcept {
    auto out = a;
    out *= s;
    return out;
  }

  [[nodiscard]] friend Spectrum operator*(luisa::compute::Expr<float> s,
                                          const Spectrum &a) noexcept {
    return a * s;
  }

  [[nodiscard]] friend Spectrum operator/(const Spectrum &a,
                                          luisa::compute::Expr<float> s) noexcept {
    auto out = a;
    out /= s;
    return out;
  }
};

[[nodiscard]] inline Spectrum select(
    const Spectrum &false_val, const Spectrum &true_val,
    luisa::compute::Expr<bool> predicate) noexcept {
  LUISA_ASSERT(false_val.is_spectral() == true_val.is_spectral(),
               "Spectrum select mode mismatch.");
  if (false_val.is_spectral()) {
    return Spectrum::from_lanes(
        luisa::compute::select(false_val.lanes(), true_val.lanes(), predicate));
  }
  return Spectrum::from_rgb_raw(
      luisa::compute::select(false_val.rgb(), true_val.rgb(), predicate));
}

[[nodiscard]] inline luisa::compute::Float max_component(
    const Spectrum &s,
    luisa::compute::Expr<bool> secondary_terminated = false) noexcept {
  if (!s.is_spectral()) {
    const auto &v = s.rgb();
    return luisa::compute::max(v.x, luisa::compute::max(v.y, v.z));
  }
  const auto &v = s.lanes();
  const auto all4 = luisa::compute::max(
      luisa::compute::max(v.x, v.y), luisa::compute::max(v.z, v.w));
  return luisa::compute::select(all4, v.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float max_abs_component(
    const Spectrum &s,
    luisa::compute::Expr<bool> secondary_terminated = false) noexcept {
  if (!s.is_spectral()) {
    const auto &v = s.rgb();
    return luisa::compute::max(
        luisa::compute::abs(v.x),
        luisa::compute::max(luisa::compute::abs(v.y), luisa::compute::abs(v.z)));
  }
  const auto &v = s.lanes();
  const auto a = luisa::compute::abs(v);
  const auto all4 = luisa::compute::max(
      luisa::compute::max(a.x, a.y), luisa::compute::max(a.z, a.w));
  return luisa::compute::select(all4, a.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float average(
    const Spectrum &s,
    luisa::compute::Expr<bool> secondary_terminated = false) noexcept {
  if (!s.is_spectral()) {
    const auto &v = s.rgb();
    return (v.x + v.y + v.z) * (1.0f / 3.0f);
  }
  const auto &v = s.lanes();
  const auto avg4 = (v.x + v.y + v.z + v.w) * 0.25f;
  return luisa::compute::select(avg4, v.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Bool any_nonzero(
    const Spectrum &s) noexcept {
  if (!s.is_spectral()) {
    return luisa::compute::any(s.rgb() != luisa::compute::make_float3(0.0f));
  }
  return luisa::compute::any(s.lanes() != luisa::compute::make_float4(0.0f));
}

[[nodiscard]] inline luisa::compute::Bool all_zero(const Spectrum &s) noexcept {
  return !any_nonzero(s);
}

[[nodiscard]] inline Spectrum clamp_nonnegative(const Spectrum &s) noexcept {
  if (!s.is_spectral()) {
    return Spectrum::from_rgb_raw(
        luisa::compute::max(s.rgb(), luisa::compute::make_float3(0.0f)));
  }
  return Spectrum::from_lanes(
      luisa::compute::max(s.lanes(), luisa::compute::make_float4(0.0f)));
}

inline void terminate_secondary(Spectrum &s) noexcept {
  if (s.is_spectral()) {
    const auto h = s.lanes().x;
    s.lanes() = luisa::compute::make_float4(h, 0.0f, 0.0f, 0.0f);
  }
}

[[nodiscard]] inline luisa::compute::Float3 to_rgb(
    const Spectrum &radiance, const SpectralTablesGpu *tables,
    const SampledWavelengths *wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  if (!radiance.is_spectral()) {
    return radiance.rgb();
  }
  LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
               "Spectral to_rgb requires tables and wavelengths.");
  return spectrum_to_rgb(*tables, radiance.lanes(), *wavelengths,
                         secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float3 reflectance_to_rgb(
    const Spectrum &reflectance, const SpectralTablesGpu *tables,
    const SampledWavelengths *wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  if (!reflectance.is_spectral()) {
    return reflectance.rgb();
  }
  LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
               "Spectral reflectance_to_rgb requires tables and wavelengths.");
  return reflectance_to_rgb(*tables, reflectance.lanes(), *wavelengths,
                            secondary_terminated);
}

struct SpectralContext {
  const SpectralTablesGpu *tables{nullptr};
  SampledWavelengths wavelengths;
  luisa::compute::Bool secondary_terminated;
};

} // namespace psycles::luisa_backend::cycles_spectrum
