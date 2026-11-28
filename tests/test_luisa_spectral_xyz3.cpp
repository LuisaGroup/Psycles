#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/path_tracer.h>

#include "cycles_svm_internal.h"
#include "cycles_svm_microfacet_scattering.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace c = psycles::contract;
namespace b = psycles::luisa_backend;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;
namespace spec = psycles::luisa_backend::cycles_spectrum;
namespace closure = psycles::luisa_backend::cycles_closure;

inline constexpr float kPi = 3.14159265358979323846f;

class Xyz3MicrofacetKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const spec::SpectralTablesGpu &_tables;
  float _abbe;

public:
  Xyz3MicrofacetKernelGlobals(const spec::SpectralTablesGpu &tables,
                              float abbe) noexcept
      : _tables{tables}, _abbe{abbe} {}

  [[nodiscard]] bool has_shadow_terminator_shading_offset()
      const noexcept override {
    return false;
  }
  [[nodiscard]] bool may_have_thin_film() const noexcept override {
    return false;
  }
  [[nodiscard]] bool is_spectral() const noexcept override { return true; }
  [[nodiscard]] c::SpectralMode spectral_mode() const noexcept override {
    return c::SpectralMode::xyz3;
  }
  [[nodiscard]] bool xyz3_rough_dispersion() const noexcept override {
    return true;
  }
  [[nodiscard]] bool has_dispersive_materials() const noexcept override {
    return true;
  }
  [[nodiscard]] const spec::SpectralTablesGpu *
  spectral_tables() const noexcept override {
    return &_tables;
  }
  [[nodiscard]] Float
  shader_dispersion_abbe(Expr<std::uint32_t>) const noexcept override {
    return _abbe;
  }
};

[[nodiscard]] c::MaterialDesc make_constant_bg(psycles::Vec3f color,
                                               float strength,
                                               std::uint32_t shader_idx = 0u) {
  c::ShaderGraph g;
  const auto bg = g.add_node(psycles::compiler::node_type::background, "bg");
  static_cast<void>(g.set_input(bg, "Color", c::SocketValue::color(color)));
  static_cast<void>(
      g.set_input(bg, "Strength", c::SocketValue::floating(strength)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{bg, "Closure"});
  return c::MaterialDesc{.name = "World",
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_diffuse_material(std::string name,
                                                    psycles::Vec3f color,
                                                    std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto diff =
      g.add_node(psycles::compiler::node_type::diffuse_bsdf, "diff");
  static_cast<void>(g.set_input(diff, "Color", c::SocketValue::color(color)));
  static_cast<void>(
      g.set_input(diff, "Roughness", c::SocketValue::floating(0.0f)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{diff, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_glass_material(std::string name,
                                                  psycles::Vec3f color,
                                                  float ior, float roughness,
                                                  float abbe,
                                                  std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto glass =
      g.add_node(psycles::compiler::node_type::glass_bsdf, "glass");
  static_cast<void>(g.set_input(glass, "Color", c::SocketValue::color(color)));
  static_cast<void>(
      g.set_input(glass, "Roughness", c::SocketValue::floating(roughness)));
  static_cast<void>(g.set_input(glass, "IOR", c::SocketValue::floating(ior)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{glass, "Closure"});
  c::MaterialDesc mat{.name = std::move(name),
                      .shader = std::move(g),
                      .cycles_shader_index = shader_idx};
  mat.dispersion_abbe = abbe;
  return mat;
}

[[nodiscard]] c::MaterialDesc make_wavelength_emission_material(
    std::string name, float wavelength_nm, float strength,
    std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto wl =
      g.add_node(psycles::compiler::node_type::wavelength, "wl");
  const auto em =
      g.add_node(psycles::compiler::node_type::emission, "em");
  static_cast<void>(
      g.set_input(wl, "Wavelength", c::SocketValue::floating(wavelength_nm)));
  static_cast<void>(
      g.set_input(em, "Strength", c::SocketValue::floating(strength)));
  static_cast<void>(g.connect(c::OutputRef{wl, "Color"}, em, "Color"));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{em, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::TriangleMeshDesc make_quad_plane(float half_extent, float z,
                                                  c::MaterialId mat,
                                                  bool flip_down = false) {
  c::TriangleMeshDesc m;
  m.positions = {{-half_extent, -half_extent, z},
                 {half_extent, -half_extent, z},
                 {half_extent, half_extent, z},
                 {-half_extent, half_extent, z}};
  if (flip_down) {
    m.triangles = {{0u, 2u, 1u}, {0u, 3u, 2u}};
  } else {
    m.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
  }
  m.material_slots = {mat};
  m.triangle_material_slots = {0u, 0u};
  return m;
}

[[nodiscard]] c::TriangleMeshDesc make_box_mesh(float hx, float hy, float z0,
                                                float z1, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.positions = {
      {-hx, -hy, z0}, {hx, -hy, z0}, {hx, hy, z0}, {-hx, hy, z0},
      {-hx, -hy, z1}, {hx, -hy, z1}, {hx, hy, z1}, {-hx, hy, z1}};
  m.triangles = {
      {0u, 2u, 1u}, {0u, 3u, 2u}, // bottom
      {4u, 5u, 6u}, {4u, 6u, 7u}, // top
      {0u, 1u, 5u}, {0u, 5u, 4u}, // -Y
      {2u, 3u, 7u}, {2u, 7u, 6u}, // +Y
      {1u, 2u, 6u}, {1u, 6u, 5u}, // +X
      {3u, 0u, 4u}, {3u, 4u, 7u}  // -X
  };
  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  return m;
}

[[nodiscard]] c::TriangleMeshDesc make_smooth_uv_sphere(
    psycles::Vec3f center, float radius, std::uint32_t rings,
    std::uint32_t segs, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.normals.domain = c::MeshAttributeDomain::point;
  for (std::uint32_t r = 0u; r <= rings; ++r) {
    const float v = static_cast<float>(r) / static_cast<float>(rings);
    const float theta = kPi * (1.0f - v);
    const float sin_t = std::sin(theta);
    const float cos_t = std::cos(theta);
    for (std::uint32_t s = 0u; s <= segs; ++s) {
      const float u = static_cast<float>(s) / static_cast<float>(segs);
      const float phi = 2.0f * kPi * u;
      const psycles::Vec3f n{sin_t * std::cos(phi), sin_t * std::sin(phi),
                             cos_t};
      m.normals.values.push_back(n);
      m.positions.push_back({center.x + radius * n.x, center.y + radius * n.y,
                             center.z + radius * n.z});
    }
  }
  const std::uint32_t stride = segs + 1u;
  for (std::uint32_t r = 0u; r < rings; ++r) {
    for (std::uint32_t s = 0u; s < segs; ++s) {
      const std::uint32_t i00 = r * stride + s;
      const std::uint32_t i10 = i00 + 1u;
      const std::uint32_t i01 = (r + 1u) * stride + s;
      const std::uint32_t i11 = i01 + 1u;
      if (r > 0u) {
        m.triangles.push_back({i00, i10, i11});
      }
      if (r + 1u < rings) {
        m.triangles.push_back({i00, i11, i01});
      }
    }
  }
  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  m.triangle_smooth.assign(m.triangles.size(), 1u);
  return m;
}

[[nodiscard]] psycles::Mat4f make_look_transform(psycles::Vec3f pos,
                                                 psycles::Vec3f target,
                                                 psycles::Vec3f world_up) {
  const auto normalize_v = [](psycles::Vec3f v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return psycles::Vec3f{v.x / len, v.y / len, v.z / len};
  };
  const auto cross_v = [](psycles::Vec3f a, psycles::Vec3f b) {
    return psycles::Vec3f{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                          a.x * b.y - a.y * b.x};
  };
  const auto forward = normalize_v(
      {target.x - pos.x, target.y - pos.y, target.z - pos.z});
  const auto right = normalize_v(cross_v(forward, world_up));
  const auto up = cross_v(right, forward);
  psycles::Mat4f m;
  m.elements = {right.x, right.y, right.z, 0.0f,       up.x,    up.y,
                up.z,    0.0f,    -forward.x, -forward.y, -forward.z, 0.0f,
                pos.x,   pos.y,   pos.z,   1.0f};
  return m;
}

[[nodiscard]] psycles::Mat4f make_translation(float x, float y, float z) {
  psycles::Mat4f m;
  m.elements[12] = x;
  m.elements[13] = y;
  m.elements[14] = z;
  return m;
}

[[nodiscard]] bool render_scene(
    Device &device, const c::SceneSnapshot &scene,
    b::LuisaPathScheduler scheduler, c::SpectralMode spectral_mode,
    c::CausticsMethod caustics_method, std::uint32_t width,
    std::uint32_t height, std::uint32_t spp, std::vector<float> &out_rgba,
    bool allow_pt_caustics = true, bool sms_unbiased = true,
    float filter_glossy = 0.0f, std::uint32_t max_bounces = 8u) {
  b::LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .scheduler = scheduler,
       .wavefront_frame_capacity =
           std::max<std::uint32_t>(width * height * 4u, 4096u),
       .max_samples_per_dispatch = spp}};
  auto compiled = renderer.compile_scene(scene);
  if (!compiled.ok()) {
    for (const auto &d : compiled.diagnostics) {
      std::cerr << d.message << '\n';
    }
    return false;
  }
  c::RenderSettings settings{.full_extent = {width, height},
                             .seed = 2026u,
                             .pixel_filter = c::PixelFilter::box,
                             .filter_width = 1.0f};
  settings.integrator.spectral_rendering =
      (spectral_mode != c::SpectralMode::none);
  settings.integrator.spectral_mode = spectral_mode;
  settings.integrator.caustics_method = caustics_method;
  settings.integrator.sms_unbiased = sms_unbiased;
  settings.integrator.sms_max_trials = 8u;
  settings.integrator.max_bounces = max_bounces;
  settings.integrator.diffuse_bounces = max_bounces;
  settings.integrator.glossy_bounces = max_bounces;
  settings.integrator.transmission_bounces = max_bounces;
  settings.integrator.transparent_max_bounces = 16u;
  settings.integrator.reflective_caustics = allow_pt_caustics;
  settings.integrator.refractive_caustics = allow_pt_caustics;
  settings.integrator.filter_glossy = filter_glossy;
  settings.integrator.sample_clamp_direct = 0.0f;
  settings.integrator.sample_clamp_indirect = 0.0f;
  settings.integrator.direct_light_sampling =
      c::DirectLightSampling::multiple_importance_sampling;
  settings.passes = {{.kind = c::PassKind::combined, .channels = 4u}};

  auto session = renderer.create_session(*compiled.scene, settings);
  if (!session) {
    std::cerr << "Failed to create render session\n";
    return false;
  }
  psycles::io::MemoryOutputSink output;
  if (!session->render_samples(
          {.first = 0u, .count = spp, .offset = 0u, .total = spp}, output)) {
    std::cerr << "render_samples failed\n";
    return false;
  }
  const auto image = output.find(c::PassKind::combined);
  if (!image || image->channels < 4u) {
    return false;
  }
  out_rgba = image->pixels;
  return true;
}

struct Stats {
  std::array<double, 3u> mean{0.0, 0.0, 0.0};
  double luma{0.0};
  bool finite{true};
};

[[nodiscard]] Stats image_stats(const std::vector<float> &rgba) noexcept {
  Stats s;
  const std::size_t n = rgba.size() / 4u;
  for (std::size_t i = 0u; i < n; ++i) {
    const double r = rgba[i * 4u + 0u];
    const double g = rgba[i * 4u + 1u];
    const double b = rgba[i * 4u + 2u];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
      s.finite = false;
    }
    s.mean[0u] += r;
    s.mean[1u] += g;
    s.mean[2u] += b;
  }
  if (n > 0u) {
    for (std::size_t c_idx = 0u; c_idx < 3u; ++c_idx) {
      s.mean[c_idx] /= static_cast<double>(n);
    }
    s.luma = 0.2126 * s.mean[0u] + 0.7152 * s.mean[1u] + 0.0722 * s.mean[2u];
  }
  return s;
}

[[nodiscard]] double rel_diff(double a, double b) noexcept {
  const double denom = std::max({std::abs(a), std::abs(b), 1.0e-5});
  return std::abs(a - b) / denom;
}

// 1. Host LUT & quadrature verification
[[nodiscard]] bool test_xyz3_luts_and_quadrature() {
  const c::ShaderColorSpace cs{};
  const auto tables = spec::build_spectral_tables_host(cs);
  if (tables.xyz3_cdfinv.size() != spec::xyz3_cdfinv_lut_count ||
      tables.xyz3_collapse.size() != spec::xyz3_collapse_lut_count) {
    std::cerr << "FAIL Test 1: unexpected xyz3 LUT sizes\n";
    return false;
  }

  auto prev = tables.xyz3_cdfinv.front();
  for (std::size_t i = 0u; i < tables.xyz3_cdfinv.size(); ++i) {
    const auto v = tables.xyz3_cdfinv[i];
    if (v.x < spec::spectral_lambda_min - 1e-4f ||
        v.x > spec::spectral_lambda_max + 1e-4f ||
        v.y < spec::spectral_lambda_min - 1e-4f ||
        v.y > spec::spectral_lambda_max + 1e-4f ||
        v.z < spec::spectral_lambda_min - 1e-4f ||
        v.z > spec::spectral_lambda_max + 1e-4f) {
      std::cerr << "FAIL Test 1: xyz3_cdfinv out of [360, 830] at i=" << i
                << '\n';
      return false;
    }
    if (v.x < prev.x || v.y < prev.y || v.z < prev.z) {
      std::cerr << "FAIL Test 1: non-monotonic xyz3_cdfinv at i=" << i << '\n';
      return false;
    }
    prev = v;
  }

  // Exact collapse weight identity & trapezoidal integral normalization
  constexpr int kSteps = 4700;
  const double d_lambda =
      static_cast<double>(spec::spectral_lambda_span) / kSteps;
  std::array<double, 3u> sum_w_pmix{0.0, 0.0, 0.0};
  std::array<double, 3u> sum_d65_w_pmix{0.0, 0.0, 0.0};
  double sum_pmix = 0.0;
  for (int i = 0; i <= kSteps; ++i) {
    const float lam = spec::spectral_lambda_min +
                      static_cast<float>(i * d_lambda);
    const double wt = (i == 0 || i == kSteps) ? 0.5 * d_lambda : d_lambda;
    const auto w_raw = spec::xyz3_collapse_weight_host(tables, lam);
    const auto p_xyz = spec::xyz3_pdf_host(tables, lam);
    const double p_mix =
        (static_cast<double>(p_xyz.x) + static_cast<double>(p_xyz.y) +
         static_cast<double>(p_xyz.z)) /
        3.0;
    const double cie_pos = std::clamp(
        static_cast<double>(lam - spec::spectral_lambda_min), 0.0,
        static_cast<double>(spec::spectral_cie_sample_count - 1u));
    const std::size_t c0 = std::min<std::size_t>(
        static_cast<std::size_t>(cie_pos),
        static_cast<std::size_t>(spec::spectral_cie_sample_count) - 2u);
    const double cf = cie_pos - static_cast<double>(c0);
    const double d65 =
        static_cast<double>(tables.cie_illuminant[c0].w) * (1.0 - cf) +
        static_cast<double>(tables.cie_illuminant[c0 + 1u].w) * cf;
    sum_pmix += wt * p_mix;
    sum_w_pmix[0u] += wt * static_cast<double>(w_raw.x) * p_mix;
    sum_w_pmix[1u] += wt * static_cast<double>(w_raw.y) * p_mix;
    sum_w_pmix[2u] += wt * static_cast<double>(w_raw.z) * p_mix;
    sum_d65_w_pmix[0u] += wt * d65 * static_cast<double>(w_raw.x) * p_mix;
    sum_d65_w_pmix[1u] += wt * d65 * static_cast<double>(w_raw.y) * p_mix;
    sum_d65_w_pmix[2u] += wt * d65 * static_cast<double>(w_raw.z) * p_mix;

    // Verify point identity W_raw(lambda) * p_mix(lambda) == p_xyz(lambda)
    if (p_mix > 1.0e-5) {
      if (std::abs(w_raw.x * p_mix - p_xyz.x) > 2.0e-3 ||
          std::abs(w_raw.y * p_mix - p_xyz.y) > 2.0e-3 ||
          std::abs(w_raw.z * p_mix - p_xyz.z) > 2.0e-3) {
        std::cerr << "FAIL Test 1: collapse point identity mismatch at lam="
                  << lam << '\n';
        return false;
      }
    }
  }
  const auto rgb_white = spec::xyz3_to_rgb_host(
      tables, {static_cast<float>(sum_d65_w_pmix[0u]),
               static_cast<float>(sum_d65_w_pmix[1u]),
               static_cast<float>(sum_d65_w_pmix[2u])});
  if (std::abs(sum_pmix - 1.0) > 1.0e-3 ||
      std::abs(sum_w_pmix[0u] - 1.0) > 2.0e-3 ||
      std::abs(sum_w_pmix[1u] - 1.0) > 2.0e-3 ||
      std::abs(sum_w_pmix[2u] - 1.0) > 2.0e-3 ||
      std::abs(rgb_white.x - 1.0f) > 2.0e-3f ||
      std::abs(rgb_white.y - 1.0f) > 2.0e-3f ||
      std::abs(rgb_white.z - 1.0f) > 2.0e-3f) {
    std::cerr << "FAIL Test 1: quadrature mismatch pmix=" << sum_pmix
              << " E_xyz=(" << sum_w_pmix[0u] << ", " << sum_w_pmix[1u]
              << ", " << sum_w_pmix[2u] << ") D65_rgb=(" << rgb_white.x
              << ", " << rgb_white.y << ", " << rgb_white.z << ")\n";
    return false;
  }

  // Verify signed negative Rec.709 R at monochromatic cyan-green 500 nm
  const auto w500 = spec::xyz3_collapse_weight_host(tables, 500.0f);
  const auto rgb500 = spec::xyz3_to_rgb_host(tables, w500);
  if (!(rgb500.x < -0.1f && rgb500.y > 0.2f)) {
    std::cerr << "FAIL Test 1: expected negative Rec.709 R at 500 nm, got ("
              << rgb500.x << ", " << rgb500.y << ", " << rgb500.z << ")\n";
    return false;
  }
  std::cout << "  [PASS] Test 1: xyz3 LUT monotonicity & quadrature (white=("
            << rgb_white.x << "," << rgb_white.y << "," << rgb_white.z
            << "), rgb500.r=" << rgb500.x << ")\n";
  return true;
}

// 2. Sampled vs Eval lane_pdf consistency on rough dispersive GGX glass
[[nodiscard]] bool test_xyz3_lane_pdf_consistency(Device &device, Stream &stream,
                                                  std::string_view backend) {
  const c::ShaderColorSpace cs{};
  const auto tables = spec::upload_spectral_tables(device, stream, cs);

  constexpr std::uint32_t kN = 16u;
  auto out_sample_eval = device.create_buffer<luisa::float4>(kN);
  auto out_eval_eval = device.create_buffer<luisa::float4>(kN);
  auto out_sample_lpdf = device.create_buffer<luisa::float4>(kN);
  auto out_eval_lpdf = device.create_buffer<luisa::float4>(kN);
  auto out_meta = device.create_buffer<luisa::float4>(kN); // spdf, epdf, label, 0

  Kernel1D kernel = [&](BufferFloat4 s_val, BufferFloat4 e_val,
                        BufferFloat4 s_lpdf, BufferFloat4 e_lpdf,
                        BufferFloat4 meta) noexcept {
    const UInt idx = dispatch_x();
    const Float u_lambda = ((idx % 4u).cast<float>() + 0.17f) * 0.25f;
    const Float u0 = ((idx / 4u).cast<float>() + 0.31f) * 0.25f;
    const Float u1 = (((idx * 3u) % 4u).cast<float>() + 0.23f) * 0.25f;

    const auto ctx = spec::make_spectral_context(
        tables.get(), c::SpectralMode::xyz3, u_lambda, false);
    const Xyz3MicrofacetKernelGlobals kg{*tables, 15.0f};
    svm::MicrofacetClosure cl{};
    cl.common.weight = make_float3(1.0f);
    cl.common.type = closure::type_microfacet_ggx_glass;
    cl.common.sample_weight = 1.0f;
    cl.common.N = make_float3(0.0f, 0.0f, 1.0f);
    cl.param.alpha_x = 0.22f;
    cl.param.alpha_y = 0.22f;
    cl.param.ior = 1.52f;
    cl.param.energy_scale = 1.0f;
    cl.param.fresnel_type =
        static_cast<std::uint32_t>(svm::MicrofacetFresnel::dielectric);
    cl.param.T = make_float3(1.0f, 0.0f, 0.0f);

    const Float3 wi = normalize(make_float3(0.25f, 0.12f, 0.96f));
    const Float3 Ng = make_float3(0.0f, 0.0f, 1.0f);
    const Float3 rnd = make_float3(u0, u1, 0.92f);

    const auto samp = svm_detail::bsdf_microfacet_ggx_sample_spectral(
        kg, cl, Ng, wi, rnd, ctx, 15.0f);
    const auto eval = svm_detail::bsdf_microfacet_ggx_eval_spectral(
        kg, cl, wi, samp.wo, ctx, 15.0f);

    s_val.write(idx, samp.value);
    e_val.write(idx, eval.value);
    s_lpdf.write(idx, samp.lane_pdf.value_or(make_float4(samp.pdf)));
    e_lpdf.write(idx, eval.lane_pdf.value_or(make_float4(eval.pdf)));
    meta.write(idx, make_float4(samp.pdf, eval.pdf, samp.label.cast<float>(),
                                0.0f));
  };

  auto shader = device.compile(
      kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::vector<luisa::float4> sv(kN), ev(kN), sl(kN), el(kN), meta(kN);
  stream << shader(out_sample_eval, out_eval_eval, out_sample_lpdf,
                   out_eval_lpdf, out_meta)
                .dispatch(kN)
         << out_sample_eval.copy_to(sv.data())
         << out_eval_eval.copy_to(ev.data())
         << out_sample_lpdf.copy_to(sl.data())
         << out_eval_lpdf.copy_to(el.data()) << out_meta.copy_to(meta.data())
         << synchronize();

  std::uint32_t checked = 0u;
  for (std::uint32_t i = 0u; i < kN; ++i) {
    if (meta[i].x <= 0.0f) {
      continue;
    }
    ++checked;
    if (rel_diff(meta[i].x, meta[i].y) > 1.0e-3 ||
        rel_diff(sl[i].x, el[i].x) > 1.0e-3 ||
        rel_diff(sl[i].y, el[i].y) > 1.0e-3 ||
        rel_diff(sl[i].z, el[i].z) > 1.0e-3 ||
        rel_diff(sv[i].x, ev[i].x) > 1.0e-3 ||
        rel_diff(sv[i].y, ev[i].y) > 1.0e-3 ||
        rel_diff(sv[i].z, ev[i].z) > 1.0e-3) {
      std::cerr << "FAIL Test 2 lane_pdf mismatch at i=" << i << " on "
                << backend << ": spdf=" << meta[i].x << " epdf=" << meta[i].y
                << " sl=(" << sl[i].x << "," << sl[i].y << "," << sl[i].z
                << ") el=(" << el[i].x << "," << el[i].y << "," << el[i].z
                << ")\n";
      return false;
    }
  }
  if (checked < 8u) {
    std::cerr << "FAIL Test 2 too few valid refracted samples: " << checked
              << '\n';
    return false;
  }
  std::cout << "  [PASS] Test 2: sampled vs eval lane_pdf parity (" << checked
            << " directions) on " << backend << '\n';
  return true;
}

// 3. White furnace unbiasedness (xyz3 megakernel & wavefront_staged)
[[nodiscard]] bool test_xyz3_white_furnace(Device &device,
                                           std::string_view backend) {
  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{10u},
                          make_constant_bg({1.0f, 1.0f, 1.0f}, 1.0f, 0u));
  scene.world_shader = c::MaterialId{10u};
  scene.world_sampling = c::WorldSampling::automatic;

  scene.materials.emplace(c::MaterialId{1u},
                          make_diffuse_material("WhiteDiff", {1.0f, 1.0f, 1.0f}, 1u));
  scene.materials.emplace(c::MaterialId{2u},
                          make_glass_material("WhiteDispGlass", {1.0f, 1.0f, 1.0f},
                                              1.5f, 0.2f, 20.0f, 2u));

  scene.geometries.emplace(
      c::GeometryId{1u},
      make_smooth_uv_sphere({-0.6f, 0.0f, 0.0f}, 0.5f, 16u, 32u,
                            c::MaterialId{1u}));
  scene.geometries.emplace(
      c::GeometryId{2u},
      make_smooth_uv_sphere({0.6f, 0.0f, 0.0f}, 0.5f, 16u, 32u,
                            c::MaterialId{2u}));
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});

  psycles::Mat4f cam_tfm;
  cam_tfm.elements[14] = 3.0f;
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 2.6f});
  scene.active_camera = c::CameraId{1u};

  constexpr std::uint32_t kSize = 16u;
  constexpr std::uint32_t kSpp = 96u;
  for (const auto sched : {b::LuisaPathScheduler::megakernel,
                           b::LuisaPathScheduler::wavefront_staged}) {
    std::vector<float> rgba;
    if (!render_scene(device, scene, sched, c::SpectralMode::xyz3,
                      c::CausticsMethod::none, kSize, kSize, kSpp, rgba, true,
                      true, 0.0f, 16u)) {
      return false;
    }
    const auto st = image_stats(rgba);
    if (!st.finite || std::abs(st.mean[0u] - 1.0) > 0.02 ||
        std::abs(st.mean[1u] - 1.0) > 0.02 ||
        std::abs(st.mean[2u] - 1.0) > 0.02) {
      std::cerr << "FAIL Test 3 white furnace xyz3 ("
                << (sched == b::LuisaPathScheduler::megakernel ? "mk" : "wf")
                << ") on " << backend << ": rgb=(" << st.mean[0u] << ", "
                << st.mean[1u] << ", " << st.mean[2u] << ")\n";
      return false;
    }
    std::cout << "  [PASS] Test 3: xyz3 white furnace ("
              << (sched == b::LuisaPathScheduler::megakernel ? "mk" : "wf")
              << ") rgb=(" << st.mean[0u] << ", " << st.mean[1u] << ", "
              << st.mean[2u] << ")\n";
  }
  return true;
}

// 4. Rough dispersive glass parity (xyz3 vs hero4, mk vs wf)
[[nodiscard]] bool test_xyz3_rough_dispersive_glass_parity(
    Device &device, std::string_view backend) {
  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{10u},
                          make_constant_bg({0.35f, 0.35f, 0.35f}, 1.0f, 0u));
  scene.world_shader = c::MaterialId{10u};
  scene.world_sampling = c::WorldSampling::automatic;

  scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("Floor", {0.75f, 0.72f, 0.68f}, 1u));
  scene.materials.emplace(
      c::MaterialId{2u},
      make_glass_material("RoughDispGlass", {0.95f, 0.98f, 1.0f}, 1.55f, 0.22f,
                          20.0f, 2u));

  scene.geometries.emplace(c::GeometryId{1u},
                           make_quad_plane(4.0f, 0.0f, c::MaterialId{1u}));
  scene.geometries.emplace(
      c::GeometryId{2u},
      make_smooth_uv_sphere({0.0f, 0.0f, 0.65f}, 0.55f, 24u, 48u,
                            c::MaterialId{2u}));
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::perspective,
                    .transform = make_look_transform(
                        {0.0f, -2.4f, 1.5f}, {0.0f, 0.0f, 0.55f},
                        {0.0f, 0.0f, 1.0f}),
                    .field_of_view = 0.7f,
                    .horizontal_field_of_view = 0.7f,
                    .sensor_fit = c::CameraSensorFit::vertical});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "KeyArea",
                   .type = c::LightType::point,
                   .transform = make_translation(0.8f, -0.8f, 2.5f),
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 90.0f,
                   .size = 0.6f,
                   .normalize = true,
                   .is_sphere = true,
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .cycles_object_index = 2u});

  constexpr std::uint32_t kSize = 20u;
  constexpr std::uint32_t kSpp = 128u;
  std::vector<float> h_wf, x_mk, x_wf;
  if (!render_scene(device, scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::hero4, c::CausticsMethod::none, kSize,
                    kSize, kSpp, h_wf) ||
      !render_scene(device, scene, b::LuisaPathScheduler::megakernel,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, x_mk) ||
      !render_scene(device, scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, x_wf)) {
    return false;
  }
  const auto sh = image_stats(h_wf);
  const auto sm = image_stats(x_mk);
  const auto sw = image_stats(x_wf);
  if (!sh.finite || !sm.finite || !sw.finite ||
      rel_diff(sh.luma, sw.luma) > 0.02 || rel_diff(sm.luma, sw.luma) > 0.02 ||
      rel_diff(sh.mean[0u], sw.mean[0u]) > 0.025 ||
      rel_diff(sh.mean[1u], sw.mean[1u]) > 0.025 ||
      rel_diff(sh.mean[2u], sw.mean[2u]) > 0.025) {
    std::cerr << "FAIL Test 4 rough dispersive glass parity on " << backend
              << ": hero4_lum=" << sh.luma << " xyz3_mk=" << sm.luma
              << " xyz3_wf=" << sw.luma << '\n';
    return false;
  }
  std::cout << "  [PASS] Test 4: rough dispersive glass parity hero4="
            << sh.luma << " xyz3_mk=" << sm.luma << " xyz3_wf=" << sw.luma
            << '\n';
  return true;
}

// 5. Delayed collapse prevents dispersive reflection corruption
[[nodiscard]] bool test_xyz3_delayed_collapse_reflection(
    Device &device, std::string_view backend) {
  auto make_refl_scene = [](float abbe) {
    c::SceneSnapshot scene;
    scene.materials.emplace(c::MaterialId{10u},
                            make_constant_bg({0.0f, 0.0f, 0.0f}, 0.0f, 0u));
    scene.world_shader = c::MaterialId{10u};
    scene.world_sampling = c::WorldSampling::none;

    scene.materials.emplace(
        c::MaterialId{1u},
        make_glass_material("Slab", {1.0f, 1.0f, 1.0f}, 1.55f, 0.0f, abbe, 1u));
    scene.materials.emplace(
        c::MaterialId{2u},
        make_diffuse_material("BlackTrap", {0.0f, 0.0f, 0.0f}, 2u));

    scene.geometries.emplace(
        c::GeometryId{1u}, make_quad_plane(2.0f, 0.0f, c::MaterialId{1u}, false));
    scene.geometries.emplace(
        c::GeometryId{2u}, make_quad_plane(4.0f, -0.5f, c::MaterialId{2u}, false));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.instances.emplace(
        c::InstanceId{2u},
        c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});

    // Camera looks at specular reflection of large overhead light source
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::perspective,
                      .transform = make_look_transform(
                          {0.0f, -1.2f, 0.8f}, {0.0f, 0.0f, 0.0f},
                          {0.0f, 0.0f, 1.0f}),
                      .field_of_view = 0.55f,
                      .horizontal_field_of_view = 0.55f,
                      .sensor_fit = c::CameraSensorFit::vertical});
    scene.active_camera = c::CameraId{1u};

    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "SpecPanel",
                     .type = c::LightType::point,
                     .transform = make_translation(0.0f, 1.2f, 0.8f),
                     .color = {0.95f, 0.55f, 0.25f},
                     .power = 60.0f,
                     .size = 0.9f,
                     .normalize = true,
                     .is_sphere = true,
                     .use_mis = true,
                     .cast_shadow = false,
                     .visibility_mask = c::all_ray_visibility,
                     .cycles_object_index = 2u});
    return scene;
  };

  constexpr std::uint32_t kSize = 16u;
  constexpr std::uint32_t kSpp = 96u;
  std::vector<float> nodisp, disp_x, disp_h;
  if (!render_scene(device, make_refl_scene(0.0f),
                    b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, nodisp) ||
      !render_scene(device, make_refl_scene(15.0f),
                    b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, disp_x) ||
      !render_scene(device, make_refl_scene(15.0f),
                    b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::hero4, c::CausticsMethod::none, kSize,
                    kSize, kSpp, disp_h)) {
    return false;
  }
  const auto sn = image_stats(nodisp);
  const auto sx = image_stats(disp_x);
  const auto sh = image_stats(disp_h);
  if (!sn.finite || !sx.finite || !sh.finite || sx.luma <= 1.0e-3 ||
      rel_diff(sx.luma, sn.luma) > 0.02 || rel_diff(sx.luma, sh.luma) > 0.02) {
    std::cerr << "FAIL Test 5 delayed collapse reflection on " << backend
              << ": nodisp=" << sn.luma << " disp_xyz3=" << sx.luma
              << " disp_hero4=" << sh.luma << '\n';
    return false;
  }
  std::cout << "  [PASS] Test 5: delayed collapse reflection nodisp=" << sn.luma
            << " disp_xyz3=" << sx.luma << " disp_hero4=" << sh.luma << '\n';
  return true;
}

// 6. Nested dispersive media compatibility (xyz3 vs hero4, mk vs wf)
[[nodiscard]] bool test_xyz3_nested_dispersive_media(Device &device,
                                                     std::string_view backend) {
  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{10u},
                          make_constant_bg({0.45f, 0.45f, 0.45f}, 1.0f, 0u));
  scene.world_shader = c::MaterialId{10u};
  scene.world_sampling = c::WorldSampling::automatic;

  scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("Floor", {0.7f, 0.7f, 0.7f}, 1u));
  scene.materials.emplace(
      c::MaterialId{2u},
      make_glass_material("OuterGlass", {0.98f, 0.99f, 1.0f}, 1.40f, 0.0f,
                          25.0f, 2u));
  scene.materials.emplace(
      c::MaterialId{3u},
      make_glass_material("InnerGlass", {0.96f, 0.98f, 1.0f}, 1.65f, 0.02f,
                          18.0f, 3u));

  scene.geometries.emplace(c::GeometryId{1u},
                           make_quad_plane(4.0f, 0.0f, c::MaterialId{1u}));
  scene.geometries.emplace(
      c::GeometryId{2u}, make_box_mesh(0.85f, 0.85f, 0.05f, 1.55f, c::MaterialId{2u}));
  scene.geometries.emplace(
      c::GeometryId{3u},
      make_smooth_uv_sphere({0.0f, 0.0f, 0.80f}, 0.45f, 20u, 40u,
                            c::MaterialId{3u}));
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});
  scene.instances.emplace(
      c::InstanceId{3u},
      c::InstanceDesc{.geometry = c::GeometryId{3u}, .cycles_object_index = 2u});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::perspective,
                    .transform = make_look_transform(
                        {0.0f, -2.8f, 1.7f}, {0.0f, 0.0f, 0.75f},
                        {0.0f, 0.0f, 1.0f}),
                    .field_of_view = 0.75f,
                    .horizontal_field_of_view = 0.75f,
                    .sensor_fit = c::CameraSensorFit::vertical});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "Overhead",
                   .type = c::LightType::point,
                   .transform = make_translation(0.3f, -0.5f, 2.8f),
                   .color = {1.0f, 0.98f, 0.95f},
                   .power = 120.0f,
                   .size = 0.5f,
                   .normalize = true,
                   .is_sphere = true,
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .cycles_object_index = 3u});

  constexpr std::uint32_t kSize = 20u;
  constexpr std::uint32_t kSpp = 96u;
  std::vector<float> h_wf, x_mk, x_wf;
  if (!render_scene(device, scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::hero4, c::CausticsMethod::none, kSize,
                    kSize, kSpp, h_wf, true, true, 0.0f, 12u) ||
      !render_scene(device, scene, b::LuisaPathScheduler::megakernel,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, x_mk, true, true, 0.0f, 12u) ||
      !render_scene(device, scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, kSize,
                    kSize, kSpp, x_wf, true, true, 0.0f, 12u)) {
    return false;
  }
  const auto sh = image_stats(h_wf);
  const auto sm = image_stats(x_mk);
  const auto sw = image_stats(x_wf);
  if (!sh.finite || !sm.finite || !sw.finite ||
      rel_diff(sh.luma, sw.luma) > 0.03 || rel_diff(sm.luma, sw.luma) > 0.02) {
    std::cerr << "FAIL Test 6 nested dispersive media on " << backend
              << ": hero4=" << sh.luma << " xyz3_mk=" << sm.luma
              << " xyz3_wf=" << sw.luma << '\n';
    return false;
  }
  std::cout << "  [PASS] Test 6: nested dispersive media hero4=" << sh.luma
            << " xyz3_mk=" << sm.luma << " xyz3_wf=" << sw.luma << '\n';
  return true;
}

// 7. MNEE & SMS caustics parity + radial dispersion + negative-RGB preservation
[[nodiscard]] bool test_xyz3_caustics_and_neg_rgb(Device &device,
                                                  std::string_view backend) {
  auto make_caustic_scene = [](float abbe) {
    c::SceneSnapshot scene;
    scene.world_sampling = c::WorldSampling::none;
    scene.materials.emplace(c::MaterialId{10u},
                            make_constant_bg({0.0f, 0.0f, 0.0f}, 0.0f, 0u));
    scene.world_shader = c::MaterialId{10u};
    scene.materials.emplace(
        c::MaterialId{1u},
        make_diffuse_material("Floor", {0.75f, 0.75f, 0.75f}, 1u));
    scene.materials.emplace(
        c::MaterialId{2u},
        make_glass_material("SphereGlass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, abbe,
                            2u));
    scene.geometries.emplace(c::GeometryId{1u},
                             make_quad_plane(4.0f, 0.0f, c::MaterialId{1u}));
    scene.geometries.emplace(
        c::GeometryId{2u},
        make_smooth_uv_sphere({0.0f, 0.0f, 0.9f}, 0.5f, 32u, 64u,
                              c::MaterialId{2u}));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u},
                        .cycles_object_index = 0u,
                        .is_caustics_receiver = true});
    scene.instances.emplace(
        c::InstanceId{2u},
        c::InstanceDesc{.geometry = c::GeometryId{2u},
                        .cycles_object_index = 1u,
                        .is_caustics_caster = true});

    psycles::Mat4f cam_tfm;
    cam_tfm.elements[14] = 0.25f;
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = cam_tfm,
                      .orthographic_scale = 0.9f});
    scene.active_camera = c::CameraId{1u};
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "Sun",
                     .type = c::LightType::distant,
                     .transform = psycles::Mat4f{},
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 4.0f,
                     .angle = 0.005f,
                     .normalize = true,
                     .use_mis = false,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .is_caustics_light = true,
                     .cycles_object_index = 2u});
    return scene;
  };

  constexpr std::uint32_t kSize = 24u;
  constexpr std::uint32_t kSpp = 48u;
  const auto non_disp_scene = make_caustic_scene(0.0f);
  for (const auto method : {c::CausticsMethod::mnee, c::CausticsMethod::sms}) {
    std::vector<float> h_wf, x_mk, x_wf;
    if (!render_scene(device, non_disp_scene,
                      b::LuisaPathScheduler::wavefront_staged,
                      c::SpectralMode::hero4, method, kSize, kSize, kSpp, h_wf,
                      false) ||
        !render_scene(device, non_disp_scene, b::LuisaPathScheduler::megakernel,
                      c::SpectralMode::xyz3, method, kSize, kSize, kSpp, x_mk,
                      false) ||
        !render_scene(device, non_disp_scene,
                      b::LuisaPathScheduler::wavefront_staged,
                      c::SpectralMode::xyz3, method, kSize, kSize, kSpp, x_wf,
                      false)) {
      return false;
    }
    const auto sh = image_stats(h_wf);
    const auto sm = image_stats(x_mk);
    const auto sw = image_stats(x_wf);
    if (!sh.finite || !sm.finite || !sw.finite || sh.luma <= 1.0e-3 ||
        rel_diff(sh.luma, sw.luma) > 0.025 ||
        rel_diff(sm.luma, sw.luma) > 0.025) {
      std::cerr << "FAIL Test 7a non-dispersive "
                << (method == c::CausticsMethod::mnee ? "MNEE" : "SMS")
                << " on " << backend << ": hero4=" << sh.luma
                << " xyz3_mk=" << sm.luma << " xyz3_wf=" << sw.luma << '\n';
      return false;
    }
    std::cout << "  [PASS] Test 7a: non-disp "
              << (method == c::CausticsMethod::mnee ? "MNEE" : "SMS")
              << " hero4=" << sh.luma << " xyz3_mk=" << sm.luma
              << " xyz3_wf=" << sw.luma << '\n';
  }

  const auto disp_scene = make_caustic_scene(15.0f);
  for (const auto method : {c::CausticsMethod::mnee, c::CausticsMethod::sms}) {
    std::vector<float> x_mk, x_wf;
    if (!render_scene(device, disp_scene, b::LuisaPathScheduler::megakernel,
                      c::SpectralMode::xyz3, method, kSize, kSize, 64u, x_mk,
                      false) ||
        !render_scene(device, disp_scene,
                      b::LuisaPathScheduler::wavefront_staged,
                      c::SpectralMode::xyz3, method, kSize, kSize, 64u, x_wf,
                      false)) {
      return false;
    }
    auto radial_exp = [&](const std::vector<float> &rgba) {
      double wr = 0.0, rr = 0.0, wb = 0.0, rb = 0.0;
      for (std::uint32_t y = 0u; y < kSize; ++y) {
        const float py =
            (((static_cast<float>(y) + 0.5f) / kSize) - 0.5f) * 0.9f;
        for (std::uint32_t x = 0u; x < kSize; ++x) {
          const float px =
              (((static_cast<float>(x) + 0.5f) / kSize) - 0.5f) * 0.9f;
          const float rad = std::sqrt(px * px + py * py);
          const std::size_t idx = (static_cast<std::size_t>(y) * kSize + x) * 4u;
          if (rad < 0.40f) {
            wr += std::max(0.0f, rgba[idx + 0u]);
            rr += std::max(0.0f, rgba[idx + 0u]) * rad;
            wb += std::max(0.0f, rgba[idx + 2u]);
            rb += std::max(0.0f, rgba[idx + 2u]) * rad;
          }
        }
      }
      const double mr = wr > 0.0 ? rr / wr : 0.0;
      const double mb = wb > 0.0 ? rb / wb : 0.0;
      return mr > 0.0 ? (mb - mr) / mr : 0.0;
    };
    const auto sm = image_stats(x_mk);
    const auto sw = image_stats(x_wf);
    const double exp_m = radial_exp(x_mk);
    const double exp_w = radial_exp(x_wf);
    if (!sm.finite || !sw.finite || exp_m <= 0.04 || exp_w <= 0.04 ||
        rel_diff(sm.luma, sw.luma) > 0.025) {
      std::cerr << "FAIL Test 7b dispersive xyz3 "
                << (method == c::CausticsMethod::mnee ? "MNEE" : "SMS")
                << " on " << backend << ": mk=" << sm.luma << " wf=" << sw.luma
                << " exp_m=" << exp_m << " exp_w=" << exp_w << '\n';
      return false;
    }
    std::cout << "  [PASS] Test 7b: disp xyz3 "
              << (method == c::CausticsMethod::mnee ? "MNEE" : "SMS")
              << " mk=" << sm.luma << " wf=" << sw.luma << " exp=(" << exp_m
              << ", " << exp_w << ")\n";
  }

  // 7c. Negative Rec.709 R preservation under narrowband 500 nm illumination
  c::SceneSnapshot nb_scene;
  nb_scene.materials.emplace(c::MaterialId{10u},
                             make_constant_bg({0.0f, 0.0f, 0.0f}, 0.0f, 0u));
  nb_scene.world_shader = c::MaterialId{10u};
  nb_scene.world_sampling = c::WorldSampling::none;
  nb_scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("WhiteRecv", {1.0f, 1.0f, 1.0f}, 1u));
  nb_scene.materials.emplace(
      c::MaterialId{2u},
      make_wavelength_emission_material("Narrow500", 500.0f, 20.0f, 2u));
  nb_scene.geometries.emplace(
      c::GeometryId{1u}, make_quad_plane(2.0f, 0.0f, c::MaterialId{1u}, false));
  nb_scene.geometries.emplace(
      c::GeometryId{2u}, make_quad_plane(2.0f, 2.0f, c::MaterialId{2u}, true));
  nb_scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
  nb_scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});
  psycles::Mat4f nb_cam;
  nb_cam.elements[14] = 1.0f;
  nb_scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = nb_cam,
                    .orthographic_scale = 1.6f});
  nb_scene.active_camera = c::CameraId{1u};

  std::vector<float> nb_h, nb_x;
  if (!render_scene(device, nb_scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::hero4, c::CausticsMethod::none, 16u, 16u,
                    64u, nb_h) ||
      !render_scene(device, nb_scene, b::LuisaPathScheduler::wavefront_staged,
                    c::SpectralMode::xyz3, c::CausticsMethod::none, 16u, 16u,
                    64u, nb_x)) {
    return false;
  }
  const auto sh_nb = image_stats(nb_h);
  const auto sx_nb = image_stats(nb_x);
  if (!sh_nb.finite || !sx_nb.finite || sx_nb.mean[0u] >= -0.02 ||
      std::abs(sx_nb.mean[0u] - sh_nb.mean[0u]) /
              std::max(std::abs(sh_nb.mean[1u]), 1.0e-4) >
          0.03 ||
      rel_diff(sx_nb.mean[1u], sh_nb.mean[1u]) > 0.03) {
    std::cerr << "FAIL Test 7c narrowband 500 nm negative RGB on " << backend
              << ": hero4=(" << sh_nb.mean[0u] << "," << sh_nb.mean[1u] << ","
              << sh_nb.mean[2u] << ") xyz3=(" << sx_nb.mean[0u] << ","
              << sx_nb.mean[1u] << "," << sx_nb.mean[2u] << ")\n";
    return false;
  }
  std::cout << "  [PASS] Test 7c: narrowband 500 nm signed RGB xyz3=("
            << sx_nb.mean[0u] << "," << sx_nb.mean[1u] << "," << sx_nb.mean[2u]
            << ") hero4=(" << sh_nb.mean[0u] << "," << sh_nb.mean[1u] << ","
            << sh_nb.mean[2u] << ")\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running test_luisa_spectral_xyz3 on " << backend << "...\n";
    if (!test_xyz3_luts_and_quadrature() ||
        !test_xyz3_lane_pdf_consistency(device, stream, backend) ||
        !test_xyz3_white_furnace(device, backend) ||
        !test_xyz3_rough_dispersive_glass_parity(device, backend) ||
        !test_xyz3_delayed_collapse_reflection(device, backend) ||
        !test_xyz3_nested_dispersive_media(device, backend) ||
        !test_xyz3_caustics_and_neg_rgb(device, backend)) {
      return EXIT_FAILURE;
    }
    std::cout << "test_luisa_spectral_xyz3 PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Exception in test_luisa_spectral_xyz3: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
