#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_spectrum.h> through the Psycles::luisa target."
#endif

#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <utility>
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
inline constexpr std::uint32_t xyz3_cdfinv_lut_count = 2048u;
inline constexpr std::uint32_t xyz3_collapse_lut_count = 512u;
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
  // xyz3 CDF^-1 LUT: 2048 entries, .xyz = (F_X^-1(u), F_Y^-1(u), F_Z^-1(u)), .w = 0
  luisa::vector<luisa::float4> xyz3_cdfinv;
  // xyz3 collapse LUT: 512 entries over [360..830] nm, .xyz = W_raw(lambda), .w = p_mix(lambda)
  luisa::vector<luisa::float4> xyz3_collapse;
  luisa::float3 xyz3_integral_ratio{};
  luisa::float3 xyz3_to_r{};
  luisa::float3 xyz3_to_g{};
  luisa::float3 xyz3_to_b{};
};

struct SpectralTablesGpu {
  luisa::compute::Buffer<luisa::float4> cie_illuminant_buffer;
  luisa::compute::Buffer<luisa::float4> rgb2spec_buffer;
  luisa::compute::Buffer<float> blackbody_norm_buffer;
  luisa::compute::Buffer<luisa::float4> xyz3_cdfinv_buffer;
  luisa::compute::Buffer<luisa::float4> xyz3_collapse_buffer;
  luisa::float3 spec_xyz_to_r{};
  luisa::float3 spec_xyz_to_g{};
  luisa::float3 spec_xyz_to_b{};
  luisa::float3 rgb_to_y{};
  luisa::float3 xyz3_integral_ratio{};
  luisa::float3 xyz3_to_r{};
  luisa::float3 xyz3_to_g{};
  luisa::float3 xyz3_to_b{};
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

[[nodiscard]] luisa::float3 sample_xyz3_lambdas_host(
    const SpectralTablesHost &tables, float u) noexcept;

[[nodiscard]] luisa::float3 xyz3_collapse_weight_host(
    const SpectralTablesHost &tables, float lambda_nm) noexcept;

[[nodiscard]] luisa::float3 xyz3_pdf_host(
    const SpectralTablesHost &tables, float lambda_nm) noexcept;

[[nodiscard]] luisa::float3 xyz3_to_rgb_host(
    const SpectralTablesHost &tables, luisa::float3 raw) noexcept;

// Device DSL wavelength sampling & spectral color conversions.
struct Xyz3DecodedU {
  luisa::compute::UInt hero_index;
  luisa::compute::Float u;
};

struct SpectralContext {
  const SpectralTablesGpu *tables{nullptr};
  SampledWavelengths wavelengths;
  luisa::compute::Bool secondary_terminated;
  contract::SpectralMode mode{contract::SpectralMode::hero4};
  std::optional<luisa::compute::UInt> hero_index{};
  std::optional<luisa::compute::Float3> lambdas_xyz{};
};

[[nodiscard]] Xyz3DecodedU xyz3_decode_u(
    luisa::compute::Expr<float> u_raw) noexcept;

[[nodiscard]] luisa::compute::Float3 sample_xyz3_lambdas(
    const SpectralTablesGpu &tables, luisa::compute::Expr<float> u) noexcept;

[[nodiscard]] luisa::compute::Float3 xyz3_collapse_weight(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<float> lambda_nm) noexcept;

[[nodiscard]] luisa::compute::Float4 xyz3_rotate_eval(
    luisa::compute::Expr<luisa::float3> v,
    luisa::compute::Expr<std::uint32_t> h) noexcept;

[[nodiscard]] luisa::compute::Float3 xyz3_unrotate_eval(
    luisa::compute::Expr<luisa::float4> v4,
    luisa::compute::Expr<std::uint32_t> h) noexcept;

[[nodiscard]] luisa::compute::Float xyz3_hero(
    luisa::compute::Expr<luisa::float3> v,
    luisa::compute::Expr<std::uint32_t> h) noexcept;

[[nodiscard]] luisa::compute::Float3 xyz3_to_rgb(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float3> raw) noexcept;

[[nodiscard]] luisa::compute::Float3 xyz3_reflectance_to_rgb(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float3> reflectance,
    luisa::compute::Expr<luisa::float3> lambdas_xyz) noexcept;

[[nodiscard]] luisa::compute::Float xyz3_luminance(
    const SpectralTablesGpu &tables,
    luisa::compute::Expr<luisa::float3> reflectance,
    luisa::compute::Expr<luisa::float3> lambdas_xyz) noexcept;

[[nodiscard]] SpectralContext make_spectral_context(
    const SpectralTablesGpu *tables, contract::SpectralMode mode,
    luisa::compute::Expr<float> u_lambda,
    luisa::compute::Expr<bool> secondary_terminated) noexcept;
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
    luisa::compute::Expr<bool> secondary_terminated,
    std::uint32_t lane_count) noexcept {
  if (lane_count == 3u) {
    return luisa::compute::select(
        (v.x + v.y + v.z) * (1.0f / 3.0f), v.x, secondary_terminated);
  }
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

// Transport wrapper: holds Float3 in RGB and xyz3 modes, Float4 in hero4 mode.
// Copy constructor allocates a fresh DSL Var; copy assignment emits a DSL
// assignment to the underlying variable without changing the active variant.
class Spectrum final {
private:
  contract::SpectralMode _mode{contract::SpectralMode::none};
  std::variant<luisa::compute::Float3, luisa::compute::Float4> _data;

  explicit Spectrum(contract::SpectralMode mode,
                    luisa::compute::Float3 value) noexcept
      : _mode{mode}, _data{std::move(value)} {
    LUISA_ASSERT(mode != contract::SpectralMode::hero4,
                 "hero4 Spectrum requires float4 storage.");
  }
  explicit Spectrum(luisa::compute::Float4 value) noexcept
      : _mode{contract::SpectralMode::hero4}, _data{std::move(value)} {}

public:
  // Default construction selects RGB transport with a zero-initialized lane
  // var. Host-side path-kernel contexts rely on this so that they stay
  // aggregate-initializable; spectral pipelines must build their transport
  // explicitly (zero(mode) / constant(mode, ...) / from_lanes / from_xyz3).
  Spectrum() noexcept
      : _mode{contract::SpectralMode::none}, _data{luisa::compute::Float3{}} {}

  Spectrum(const Spectrum &other) noexcept
      : _mode{other._mode},
        _data{other.has_float4_storage()
                  ? std::variant<luisa::compute::Float3, luisa::compute::Float4>{
                        luisa::compute::def<luisa::float4>(other.lanes())}
                  : std::variant<luisa::compute::Float3, luisa::compute::Float4>{
                        luisa::compute::def<luisa::float3>(other.vec3())}} {}

  Spectrum(Spectrum &&) noexcept = default;

  Spectrum &operator=(const Spectrum &other) noexcept {
    LUISA_ASSERT(_mode == other._mode,
                 "Spectrum assignment requires matching spectral mode.");
    if (has_float4_storage()) {
      std::get<luisa::compute::Float4>(_data) = other.lanes();
    } else {
      std::get<luisa::compute::Float3>(_data) = other.vec3();
    }
    return *this;
  }

  Spectrum &operator=(Spectrum &&other) noexcept {
    return *this = static_cast<const Spectrum &>(other);
  }

  // Raw lane assignment keeps the active variant and only rebinds its value.
  void operator=(luisa::compute::Expr<luisa::float3> value) noexcept {
    LUISA_ASSERT(!has_float4_storage(),
                 "Assigning float3 lanes to a hero4 Spectrum; use from_lanes().");
    std::get<luisa::compute::Float3>(_data) = std::move(value);
  }

  void operator=(luisa::compute::Expr<luisa::float4> value) noexcept {
    LUISA_ASSERT(has_float4_storage(),
                 "Assigning float4 lanes to a non-hero4 Spectrum.");
    std::get<luisa::compute::Float4>(_data) = std::move(value);
  }

  [[nodiscard]] contract::SpectralMode mode() const noexcept { return _mode; }
  [[nodiscard]] bool is_spectral() const noexcept {
    return _mode != contract::SpectralMode::none;
  }
  [[nodiscard]] bool is_hero4() const noexcept {
    return _mode == contract::SpectralMode::hero4;
  }
  [[nodiscard]] bool is_xyz3() const noexcept {
    return _mode == contract::SpectralMode::xyz3;
  }
  [[nodiscard]] bool has_float4_storage() const noexcept {
    return std::holds_alternative<luisa::compute::Float4>(_data);
  }

  [[nodiscard]] static Spectrum zero(contract::SpectralMode mode) noexcept {
    return constant(mode, 0.0f);
  }

  [[nodiscard]] static Spectrum one(contract::SpectralMode mode) noexcept {
    return constant(mode, 1.0f);
  }

  [[nodiscard]] static Spectrum constant(
      contract::SpectralMode mode, luisa::compute::Expr<float> scalar) noexcept {
    if (mode == contract::SpectralMode::hero4) {
      return Spectrum{luisa::compute::make_float4(scalar)};
    }
    return Spectrum{mode, luisa::compute::make_float3(scalar)};
  }

  [[nodiscard]] static Spectrum from_rgb_raw(
      luisa::compute::Expr<luisa::float3> value,
      std::source_location loc = std::source_location::current()) noexcept {
    static_cast<void>(loc);
    return Spectrum{contract::SpectralMode::none,
                    luisa::compute::Float3{value}};
  }

  [[nodiscard]] static Spectrum from_xyz3(
      luisa::compute::Expr<luisa::float3> value) noexcept {
    return Spectrum{contract::SpectralMode::xyz3,
                    luisa::compute::Float3{value}};
  }

  [[nodiscard]] static Spectrum from_lanes(
      luisa::compute::Expr<luisa::float4> value,
      std::source_location loc = std::source_location::current()) noexcept {
    static_cast<void>(loc);
    return Spectrum{luisa::compute::Float4{value}};
  }

  [[nodiscard]] static Spectrum from_eval4(
      const SpectralContext *spectral,
      luisa::compute::Expr<luisa::float4> eval4) noexcept {
    if (spectral == nullptr || spectral->mode == contract::SpectralMode::none) {
      return from_rgb_raw(eval4.xyz());
    }
    if (spectral->mode == contract::SpectralMode::xyz3) {
      LUISA_ASSERT(spectral->hero_index.has_value(),
                   "xyz3 SpectralContext requires hero_index.");
      return from_xyz3(xyz3_unrotate_eval(eval4, *spectral->hero_index));
    }
    return from_lanes(eval4);
  }

  [[nodiscard]] static Spectrum from_rgb_reflectance(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SampledWavelengths *wavelengths,
      contract::SpectralMode mode) noexcept {
    if (mode == contract::SpectralMode::none) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
                 "Spectral conversion requires tables and wavelengths.");
    if (mode == contract::SpectralMode::xyz3) {
      LUISA_ERROR_WITH_LOCATION(
          "from_rgb_reflectance in xyz3 requires the SpectralContext overload.");
    }
    return from_lanes(
        rgb_to_spectrum_reflectance(*tables, value, wavelengths->lambdas));
  }

  [[nodiscard]] static Spectrum from_rgb_reflectance(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SpectralContext *spectral) noexcept {
    if (spectral == nullptr || spectral->mode == contract::SpectralMode::none) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr,
                 "Spectral conversion requires tables.");
    if (spectral->mode == contract::SpectralMode::xyz3) {
      LUISA_ASSERT(spectral->lambdas_xyz.has_value(),
                   "xyz3 SpectralContext requires lambdas_xyz.");
      const auto l3 = *spectral->lambdas_xyz;
      const auto spec = rgb_to_spectrum_reflectance(
          *tables, value, luisa::compute::make_float4(l3, l3.x));
      return from_xyz3(spec.xyz());
    }
    return from_lanes(rgb_to_spectrum_reflectance(
        *tables, value, spectral->wavelengths.lambdas));
  }

  [[nodiscard]] static Spectrum from_rgb_illuminant(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SampledWavelengths *wavelengths,
      contract::SpectralMode mode) noexcept {
    if (mode == contract::SpectralMode::none) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr && wavelengths != nullptr,
                 "Spectral conversion requires tables and wavelengths.");
    if (mode == contract::SpectralMode::xyz3) {
      LUISA_ERROR_WITH_LOCATION(
          "from_rgb_illuminant in xyz3 requires the SpectralContext overload.");
    }
    return from_lanes(
        rgb_to_spectrum_illuminant(*tables, value, wavelengths->lambdas));
  }

  [[nodiscard]] static Spectrum from_rgb_illuminant(
      const SpectralTablesGpu *tables, luisa::compute::Expr<luisa::float3> value,
      const SpectralContext *spectral) noexcept {
    if (spectral == nullptr || spectral->mode == contract::SpectralMode::none) {
      return from_rgb_raw(value);
    }
    LUISA_ASSERT(tables != nullptr,
                 "Spectral conversion requires tables.");
    if (spectral->mode == contract::SpectralMode::xyz3) {
      LUISA_ASSERT(spectral->lambdas_xyz.has_value(),
                   "xyz3 SpectralContext requires lambdas_xyz.");
      const auto l3 = *spectral->lambdas_xyz;
      const auto spec = rgb_to_spectrum_illuminant(
          *tables, value, luisa::compute::make_float4(l3, l3.x));
      return from_xyz3(spec.xyz());
    }
    return from_lanes(rgb_to_spectrum_illuminant(
        *tables, value, spectral->wavelengths.lambdas));
  }

  [[nodiscard]] const luisa::compute::Float3 &rgb(
      std::source_location loc = std::source_location::current()) const noexcept {
    LUISA_ASSERT(_mode == contract::SpectralMode::none,
                 "Accessed rgb() of non-RGB Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }
  [[nodiscard]] luisa::compute::Float3 &rgb(
      std::source_location loc = std::source_location::current()) noexcept {
    LUISA_ASSERT(_mode == contract::SpectralMode::none,
                 "Accessed rgb() of non-RGB Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }

  [[nodiscard]] const luisa::compute::Float3 &xyz3(
      std::source_location loc = std::source_location::current()) const noexcept {
    LUISA_ASSERT(_mode == contract::SpectralMode::xyz3,
                 "Accessed xyz3() of non-xyz3 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }
  [[nodiscard]] luisa::compute::Float3 &xyz3(
      std::source_location loc = std::source_location::current()) noexcept {
    LUISA_ASSERT(_mode == contract::SpectralMode::xyz3,
                 "Accessed xyz3() of non-xyz3 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }

  [[nodiscard]] const luisa::compute::Float3 &vec3(
      std::source_location loc = std::source_location::current()) const noexcept {
    LUISA_ASSERT(!has_float4_storage(),
                 "Accessed vec3() of hero4 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }
  [[nodiscard]] luisa::compute::Float3 &vec3(
      std::source_location loc = std::source_location::current()) noexcept {
    LUISA_ASSERT(!has_float4_storage(),
                 "Accessed vec3() of hero4 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float3>(_data);
  }

  [[nodiscard]] const luisa::compute::Float4 &lanes(
      std::source_location loc = std::source_location::current()) const noexcept {
    LUISA_ASSERT(is_hero4(),
                 "Accessed lanes() of non-hero4 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float4>(_data);
  }
  [[nodiscard]] luisa::compute::Float4 &lanes(
      std::source_location loc = std::source_location::current()) noexcept {
    LUISA_ASSERT(is_hero4(),
                 "Accessed lanes() of non-hero4 Spectrum at {}:{}.",
                 loc.file_name(), loc.line());
    return std::get<luisa::compute::Float4>(_data);
  }

  [[nodiscard]] luisa::compute::Float hero() const noexcept {
    return has_float4_storage() ? lanes().x : vec3().x;
  }

  void terminate_secondary() noexcept {
    if (has_float4_storage()) {
      lanes() = luisa::compute::make_float4(lanes().x, 0.0f, 0.0f, 0.0f);
    }
  }

  Spectrum &operator+=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(_mode == rhs._mode, "Spectrum mode mismatch.");
    if (has_float4_storage()) {
      lanes() += rhs.lanes();
    } else {
      vec3() += rhs.vec3();
    }
    return *this;
  }

  Spectrum &operator-=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(_mode == rhs._mode, "Spectrum mode mismatch.");
    if (has_float4_storage()) {
      lanes() -= rhs.lanes();
    } else {
      vec3() -= rhs.vec3();
    }
    return *this;
  }

  Spectrum &operator*=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(_mode == rhs._mode, "Spectrum mode mismatch.");
    if (has_float4_storage()) {
      lanes() *= rhs.lanes();
    } else {
      vec3() *= rhs.vec3();
    }
    return *this;
  }

  Spectrum &operator/=(const Spectrum &rhs) noexcept {
    LUISA_ASSERT(_mode == rhs._mode, "Spectrum mode mismatch.");
    if (has_float4_storage()) {
      lanes() /= rhs.lanes();
    } else {
      vec3() /= rhs.vec3();
    }
    return *this;
  }

  Spectrum &operator*=(luisa::compute::Expr<float> s) noexcept {
    if (has_float4_storage()) {
      lanes() *= s;
    } else {
      vec3() *= s;
    }
    return *this;
  }

  Spectrum &operator/=(luisa::compute::Expr<float> s) noexcept {
    if (has_float4_storage()) {
      lanes() /= s;
    } else {
      vec3() /= s;
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
  LUISA_ASSERT(false_val.mode() == true_val.mode(),
               "Spectrum select mode mismatch.");
  if (false_val.has_float4_storage()) {
    return Spectrum::from_lanes(
        luisa::compute::select(false_val.lanes(), true_val.lanes(), predicate));
  }
  if (false_val.is_xyz3()) {
    return Spectrum::from_xyz3(
        luisa::compute::select(false_val.vec3(), true_val.vec3(), predicate));
  }
  return Spectrum::from_rgb_raw(
      luisa::compute::select(false_val.vec3(), true_val.vec3(), predicate));
}

[[nodiscard]] inline luisa::compute::Float max_component(
    const Spectrum &s,
    luisa::compute::Expr<bool> secondary_terminated = false) noexcept {
  if (!s.has_float4_storage()) {
    const auto &v = s.vec3();
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
  if (!s.has_float4_storage()) {
    const auto &v = s.vec3();
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
  if (!s.has_float4_storage()) {
    const auto &v = s.vec3();
    return (v.x + v.y + v.z) * (1.0f / 3.0f);
  }
  const auto &v = s.lanes();
  const auto avg4 = (v.x + v.y + v.z + v.w) * 0.25f;
  return luisa::compute::select(avg4, v.x, secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Bool any_nonzero(
    const Spectrum &s) noexcept {
  if (!s.has_float4_storage()) {
    return luisa::compute::any(s.vec3() != luisa::compute::make_float3(0.0f));
  }
  return luisa::compute::any(s.lanes() != luisa::compute::make_float4(0.0f));
}

[[nodiscard]] inline luisa::compute::Bool all_zero(const Spectrum &s) noexcept {
  return !any_nonzero(s);
}

[[nodiscard]] inline Spectrum clamp_nonnegative(const Spectrum &s) noexcept {
  if (!s.has_float4_storage()) {
    const auto clamped =
        luisa::compute::max(s.vec3(), luisa::compute::make_float3(0.0f));
    return s.is_xyz3() ? Spectrum::from_xyz3(clamped)
                       : Spectrum::from_rgb_raw(clamped);
  }
  return Spectrum::from_lanes(
      luisa::compute::max(s.lanes(), luisa::compute::make_float4(0.0f)));
}

inline void terminate_secondary(Spectrum &s) noexcept {
  s.terminate_secondary();
}

[[nodiscard]] inline luisa::compute::Float3 to_rgb(
    const Spectrum &radiance, const SpectralContext &ctx) noexcept {
  if (!radiance.is_spectral()) {
    return radiance.rgb();
  }
  LUISA_ASSERT(ctx.tables != nullptr, "Spectral to_rgb requires tables.");
  if (radiance.is_xyz3()) {
    return xyz3_to_rgb(*ctx.tables, radiance.xyz3());
  }
  return spectrum_to_rgb(*ctx.tables, radiance.lanes(), ctx.wavelengths,
                         ctx.secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float3 to_rgb(
    const Spectrum &radiance, const SpectralTablesGpu *tables,
    const SampledWavelengths *wavelengths,
    luisa::compute::Expr<bool> secondary_terminated) noexcept {
  if (!radiance.is_spectral()) {
    return radiance.rgb();
  }
  LUISA_ASSERT(tables != nullptr, "Spectral to_rgb requires tables.");
  if (radiance.is_xyz3()) {
    return xyz3_to_rgb(*tables, radiance.xyz3());
  }
  LUISA_ASSERT(wavelengths != nullptr,
               "hero4 to_rgb requires wavelengths.");
  return spectrum_to_rgb(*tables, radiance.lanes(), *wavelengths,
                         secondary_terminated);
}

[[nodiscard]] inline luisa::compute::Float3 reflectance_to_rgb(
    const Spectrum &reflectance, const SpectralContext &ctx) noexcept {
  if (!reflectance.is_spectral()) {
    return reflectance.rgb();
  }
  LUISA_ASSERT(ctx.tables != nullptr,
               "Spectral reflectance_to_rgb requires tables.");
  if (reflectance.is_xyz3()) {
    LUISA_ASSERT(ctx.lambdas_xyz.has_value(),
                 "xyz3 reflectance_to_rgb requires lambdas_xyz.");
    return xyz3_reflectance_to_rgb(*ctx.tables, reflectance.xyz3(),
                                   *ctx.lambdas_xyz);
  }
  return reflectance_to_rgb(*ctx.tables, reflectance.lanes(), ctx.wavelengths,
                            ctx.secondary_terminated);
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
  LUISA_ASSERT(!reflectance.is_xyz3(),
               "xyz3 reflectance_to_rgb requires SpectralContext.");
  return reflectance_to_rgb(*tables, reflectance.lanes(), *wavelengths,
                            secondary_terminated);
}

} // namespace psycles::luisa_backend::cycles_spectrum
