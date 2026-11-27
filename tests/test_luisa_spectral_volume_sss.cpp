#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>
#include <psycles/luisa/heterogeneous_volume_tracking.h>
#include <psycles/luisa/path_tracer.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace abi = psycles::compiler::cycles_svm;
namespace c = psycles::contract;
namespace b = psycles::luisa_backend;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace spec = psycles::luisa_backend::cycles_spectrum;

class SpectralVolumeTestKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const spec::SpectralTablesGpu &_tables;

public:
  explicit SpectralVolumeTestKernelGlobals(
      const spec::SpectralTablesGpu &tables) noexcept
      : _tables{tables} {}

  [[nodiscard]] bool is_spectral() const noexcept override { return true; }

  [[nodiscard]] const spec::SpectralTablesGpu *
  spectral_tables() const noexcept override {
    return &_tables;
  }
};

[[nodiscard]] psycles::Mat4f translated(float x, float y, float z) noexcept {
  psycles::Mat4f m{};
  m.elements[12u] = x;
  m.elements[13u] = y;
  m.elements[14u] = z;
  return m;
}

[[nodiscard]] c::MaterialDesc make_constant_world(psycles::Vec3f color,
                                                  float strength,
                                                  std::uint32_t shader_idx) {
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

[[nodiscard]] c::TriangleMeshDesc make_box_mesh(c::MaterialId material,
                                                float hx, float hy, float hz) {
  c::TriangleMeshDesc mesh;
  mesh.name = "BoxMesh";
  mesh.positions = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
      {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz},
  };
  mesh.triangles = {
      {0u, 3u, 2u}, {0u, 2u, 1u}, {4u, 5u, 6u}, {4u, 6u, 7u},
      {0u, 1u, 5u}, {0u, 5u, 4u}, {3u, 7u, 6u}, {3u, 6u, 2u},
      {0u, 4u, 7u}, {0u, 7u, 3u}, {1u, 2u, 6u}, {1u, 6u, 5u},
  };
  mesh.material_slots = {material};
  mesh.triangle_material_slots.assign(mesh.triangles.size(), 0u);
  mesh.triangle_smooth.assign(mesh.triangles.size(), 0u);
  return mesh;
}

[[nodiscard]] c::TriangleMeshDesc make_uv_sphere_mesh(c::MaterialId material,
                                                      float radius,
                                                      std::uint32_t slices,
                                                      std::uint32_t stacks) {
  c::TriangleMeshDesc mesh;
  mesh.name = "SphereMesh";
  mesh.normals.domain = c::MeshAttributeDomain::point;
  constexpr float kPi = 3.14159265358979323846f;
  mesh.positions.push_back({0.0f, 0.0f, radius});
  mesh.normals.values.push_back({0.0f, 0.0f, 1.0f});
  for (std::uint32_t st = 1u; st < stacks; ++st) {
    const float theta = kPi * static_cast<float>(st) / static_cast<float>(stacks);
    const float sin_t = std::sin(theta);
    const float cos_t = std::cos(theta);
    for (std::uint32_t sl = 0u; sl < slices; ++sl) {
      const float phi =
          2.0f * kPi * static_cast<float>(sl) / static_cast<float>(slices);
      const float nx = sin_t * std::cos(phi);
      const float ny = sin_t * std::sin(phi);
      const float nz = cos_t;
      mesh.positions.push_back({radius * nx, radius * ny, radius * nz});
      mesh.normals.values.push_back({nx, ny, nz});
    }
  }
  const auto south_idx = static_cast<std::uint32_t>(mesh.positions.size());
  mesh.positions.push_back({0.0f, 0.0f, -radius});
  mesh.normals.values.push_back({0.0f, 0.0f, -1.0f});

  for (std::uint32_t sl = 0u; sl < slices; ++sl) {
    const std::uint32_t next = (sl + 1u) % slices;
    mesh.triangles.push_back({0u, 1u + sl, 1u + next});
  }
  for (std::uint32_t st = 1u; st + 1u < stacks; ++st) {
    const std::uint32_t row0 = 1u + (st - 1u) * slices;
    const std::uint32_t row1 = 1u + st * slices;
    for (std::uint32_t sl = 0u; sl < slices; ++sl) {
      const std::uint32_t next = (sl + 1u) % slices;
      mesh.triangles.push_back({row0 + sl, row1 + sl, row1 + next});
      mesh.triangles.push_back({row0 + sl, row1 + next, row0 + next});
    }
  }
  const std::uint32_t last_row = 1u + (stacks - 2u) * slices;
  for (std::uint32_t sl = 0u; sl < slices; ++sl) {
    const std::uint32_t next = (sl + 1u) % slices;
    mesh.triangles.push_back({last_row + sl, south_idx, last_row + next});
  }
  mesh.material_slots = {material};
  mesh.triangle_material_slots.assign(mesh.triangles.size(), 0u);
  mesh.triangle_smooth.assign(mesh.triangles.size(), 1u);
  return mesh;
}

[[nodiscard]] c::RenderSettings make_render_settings(std::uint32_t width,
                                                     std::uint32_t height,
                                                     bool spectral,
                                                     std::uint32_t max_bounces,
                                                     std::uint32_t vol_bounces,
                                                     c::DirectLightSampling dls) {
  c::RenderSettings settings{.full_extent = {width, height},
                             .seed = 4242u,
                             .pixel_filter = c::PixelFilter::box,
                             .filter_width = 1.0f};
  settings.integrator.spectral_rendering = spectral;
  settings.integrator.caustics_method = c::CausticsMethod::none;
  settings.integrator.max_bounces = max_bounces;
  settings.integrator.diffuse_bounces = max_bounces;
  settings.integrator.glossy_bounces = max_bounces;
  settings.integrator.transmission_bounces = max_bounces;
  settings.integrator.volume_bounces = vol_bounces;
  settings.integrator.transparent_max_bounces = 16u;
  settings.integrator.sample_clamp_direct = 0.0f;
  settings.integrator.sample_clamp_indirect = 0.0f;
  settings.integrator.direct_light_sampling = dls;
  settings.passes = {{.kind = c::PassKind::combined, .channels = 4u}};
  return settings;
}

struct ImageStats {
  std::array<float, 3u> mean{};
  float max_luma{0.0f};
  bool finite{true};
};

[[nodiscard]] bool render_mean_rgb(Device &device, const c::SceneSnapshot &scene,
                                   const c::RenderSettings &settings,
                                   b::LuisaPathScheduler scheduler,
                                   bool nee_enabled, std::uint32_t spp,
                                   ImageStats &out_stats,
                                   std::string_view label,
                                   std::string_view backend) {
  b::LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = nee_enabled,
       .scheduler = scheduler,
       .wavefront_frame_capacity = 2048u,
       .max_samples_per_dispatch = spp}};
  auto compiled = renderer.compile_scene(scene);
  if (!compiled.ok()) {
    std::cerr << "FAIL compile_scene for " << label << " on " << backend << ":\n";
    for (const auto &d : compiled.diagnostics) {
      std::cerr << "  " << d.message << '\n';
    }
    return false;
  }
  auto session = renderer.create_session(*compiled.scene, settings);
  if (!session) {
    std::cerr << "FAIL create_session for " << label << " on " << backend << '\n';
    return false;
  }
  psycles::io::MemoryOutputSink output;
  if (!session->render_samples(
          {.first = 0u, .count = spp, .offset = 0u, .total = spp}, output)) {
    std::cerr << "FAIL render_samples for " << label << " on " << backend << '\n';
    return false;
  }
  const auto image = output.find(c::PassKind::combined);
  if (!image || image->extent.width == 0u || image->extent.height == 0u) {
    std::cerr << "FAIL missing Combined pass for " << label << '\n';
    return false;
  }
  double sr = 0.0, sg = 0.0, sb = 0.0;
  float max_lum = 0.0f;
  const std::size_t count =
      static_cast<std::size_t>(image->extent.width) * image->extent.height;
  for (std::size_t i = 0u; i < count; ++i) {
    const float r = image->pixels[i * image->channels + 0u];
    const float g = image->pixels[i * image->channels + 1u];
    const float bl = image->pixels[i * image->channels + 2u];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bl)) {
      std::cerr << "FAIL non-finite pixel at i=" << i << " in " << label
                << " on " << backend << '\n';
      return false;
    }
    sr += r;
    sg += g;
    sb += bl;
    const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * bl;
    max_lum = std::max(max_lum, lum);
  }
  out_stats = {
      .mean = {static_cast<float>(sr / static_cast<double>(count)),
               static_cast<float>(sg / static_cast<double>(count)),
               static_cast<float>(sb / static_cast<double>(count))},
      .max_luma = max_lum,
      .finite = true};
  return true;
}

[[nodiscard]] float luma_of(const std::array<float, 3u> &rgb) noexcept {
  return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
}

// Test 1: Volume SVM BytecodeVm vs DirectJit parity + PrincipledVolume 2500K Blackbody
[[nodiscard]] bool test_volume_svm_parity_and_blackbody(Device &device,
                                                        Stream &stream,
                                                        std::string_view backend) {
  const c::ShaderColorSpace cs{};
  const auto tables = spec::upload_spectral_tables(device, stream, cs);
  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};

  c::ShaderGraph g0;
  const auto pv =
      g0.add_node(psycles::compiler::node_type::principled_volume, "pv");
  static_cast<void>(
      g0.set_input(pv, "Color", c::SocketValue::color({0.8f, 0.4f, 0.2f})));
  static_cast<void>(g0.set_input(pv, "Density", c::SocketValue::floating(0.5f)));
  static_cast<void>(
      g0.set_input(pv, "Anisotropy", c::SocketValue::floating(0.3f)));
  static_cast<void>(g0.set_input(
      pv, "AbsorptionColor", c::SocketValue::color({0.2f, 0.6f, 0.9f})));
  static_cast<void>(
      g0.set_input(pv, "BlackbodyIntensity", c::SocketValue::floating(1.0f)));
  static_cast<void>(
      g0.set_input(pv, "Temperature", c::SocketValue::floating(2500.0f)));
  static_cast<void>(
      g0.set_input(pv, "BlackbodyTint", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
  g0.set_root(c::ShaderDomain::volume, c::OutputRef{pv, "Volume"});

  c::ShaderGraph g1;
  const auto va =
      g1.add_node(psycles::compiler::node_type::volume_absorption, "va");
  const auto vs =
      g1.add_node(psycles::compiler::node_type::volume_scatter, "vs");
  const auto vadd = g1.add_node(psycles::compiler::node_type::add_volume, "vadd");
  static_cast<void>(
      g1.set_input(va, "Color", c::SocketValue::color({0.25f, 0.75f, 0.90f})));
  static_cast<void>(g1.set_input(va, "Density", c::SocketValue::floating(0.6f)));
  static_cast<void>(
      g1.set_input(vs, "Color", c::SocketValue::color({0.85f, 0.55f, 0.30f})));
  static_cast<void>(g1.set_input(vs, "Density", c::SocketValue::floating(0.4f)));
  static_cast<void>(
      g1.set_input(vs, "Anisotropy", c::SocketValue::floating(0.15f)));
  static_cast<void>(g1.connect(c::OutputRef{va, "Volume"}, vadd, "A"));
  static_cast<void>(g1.connect(c::OutputRef{vs, "Volume"}, vadd, "B"));
  g1.set_root(c::ShaderDomain::volume, c::OutputRef{vadd, "Volume"});

  const auto p0 = frontend.compile(g0);
  const auto p1 = frontend.compile(g1);
  if (!p0.ok() || !p1.ok()) {
    std::cerr << "FAIL volume SVM frontend compilation\n";
    return false;
  }

  std::array<abi::ShaderTableCompileUnit, 2u> units{{
      {.shader_index = 0u, .shader = p0.program.get()},
      {.shader_index = 1u, .shader = p1.program.get()},
  }};
  const auto compiled = abi::compile_shader_table(units);
  if (!compiled.table.valid) {
    std::cerr << "FAIL compile_shader_table: " << compiled.table.diagnostic << '\n';
    return false;
  }

  auto words = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  stream << words.copy_from(compiled.table.words.data());

  constexpr std::uint32_t kCount = 16u;
  static constexpr std::uint32_t kStride = 4u;
  std::array<std::vector<luisa::float4>, 2u> outputs;
  const auto volume_usage =
      compiled.table.usage_for(abi::SHADER_TYPE_VOLUME);

  for (std::size_t s = 0u; s < 2u; ++s) {
    const auto strategy = s == 0u ? svm::SvmBackendStrategy::bytecode_vm
                                  : svm::SvmBackendStrategy::direct_jit;
    auto out_buf = device.create_buffer<luisa::float4>(kCount * kStride);
    Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>> kernel =
        [&tables, &compiled, volume_usage, strategy](
            BufferUInt word_buf, BufferFloat4 out) noexcept {
          const UInt idx = dispatch_x();
          const Float u_lambda =
              ((idx % 8u).cast<float>() + 0.5f) * (1.0f / 8.0f);
          const auto wavelengths = spec::sample_visible_wavelengths(u_lambda);
          const auto identity = make_float4x4(1.0f);
          const auto normal = make_float3(0.0f, 0.0f, 1.0f);
          svm::ClosurePool pool{4u, {.spectral = true}};
          svm::ShaderData sd{
              make_float3(0.0f), normal, normal, normal,
              static_cast<unsigned>(abi::PRIMITIVE_VOLUME), idx / 8u,
              svm::shader_data_is_volume_shader_eval, 0u,
              svm::primitive_none, 0.0f, 0.0f, svm::object_none, 0.0f, 0.0f,
              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
              make_float3(1.0f, 0.0f, 0.0f),
              make_float3(0.0f, 1.0f, 0.0f), identity, identity, 0u, &pool};
          sd.shader = idx / 8u;
          sd.object = svm::object_none;
          sd.prim = svm::primitive_none;
          sd.type = static_cast<unsigned>(abi::PRIMITIVE_VOLUME);
          sd.flag = svm::shader_data_is_volume_shader_eval;
          sd.spectral = spec::SpectralContext{
              .tables = tables.get(),
              .wavelengths = wavelengths,
              .secondary_terminated = false};
          sd.spectral_extinction.emplace(make_float4(0.0f));
          sd.spectral_scatter.emplace(make_float4(0.0f));
          sd.spectral_emission_background.emplace(make_float4(0.0f));
          const svm::TransformState transforms{
              identity, identity, identity, identity};
          const svm::PathState path{svm::path_ray_visibility_camera, 0u};
          const SpectralVolumeTestKernelGlobals globals{*tables};
          svm::svm_translator_for(strategy).translate(svm::SvmTranslationRequest{
              .kernel_globals = globals,
              .table = compiled.table,
              .words = word_buf,
              .shader_type = abi::SHADER_TYPE_VOLUME,
              .kernel_features = compiled.kernel_features | svm::kernel_feature_volume,
              .node_feature_mask = svm::kernel_feature_node_mask_volume,
              .node_types_used = volume_usage.node_types_used,
              .transform_state = transforms,
              .shader_data = sd,
              .path_state = path,
              .stack_size =
                  std::max<std::size_t>(1u, volume_usage.peak_stack_usage),
              .noise_usage = volume_usage.noise_usage});
          out.write(idx * kStride + 0u, *sd.spectral_extinction);
          out.write(idx * kStride + 1u, *sd.spectral_scatter);
          out.write(idx * kStride + 2u, *sd.spectral_emission_background);
          const auto bb_rgb = spec::spectrum_to_rgb(
              *tables, *sd.spectral_emission_background, wavelengths, false);
          out.write(idx * kStride + 3u, make_float4(bb_rgb, 1.0f));
        };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    outputs[s].resize(kCount * kStride);
    stream << shader(words, out_buf).dispatch(kCount)
           << out_buf.copy_to(outputs[s].data()) << synchronize();
  }

  double bb_r = 0.0, bb_g = 0.0, bb_b = 0.0;
  for (std::uint32_t i = 0u; i < kCount; ++i) {
    for (std::uint32_t slot = 0u; slot < 3u; ++slot) {
      const auto a = outputs[0][i * kStride + slot];
      const auto d = outputs[1][i * kStride + slot];
      const std::array<float, 4u> va{a.x, a.y, a.z, a.w};
      const std::array<float, 4u> vd{d.x, d.y, d.z, d.w};
      for (std::size_t k = 0u; k < 4u; ++k) {
        if (!std::isfinite(va[k]) || !std::isfinite(vd[k])) {
          std::cerr << "FAIL volume SVM non-finite idx=" << i << " slot=" << slot
                    << " on " << backend << '\n';
          return false;
        }
        const float diff = std::abs(va[k] - vd[k]);
        const float scale =
            std::max(1.0f, std::max(std::abs(va[k]), std::abs(vd[k])));
        if (diff > 1.0e-5f * scale) {
          std::cerr << "FAIL volume SVM VM vs JIT mismatch idx=" << i
                    << " slot=" << slot << " lane=" << k << " vm=" << va[k]
                    << " jit=" << vd[k] << " on " << backend << '\n';
          return false;
        }
      }
    }
    if (i < 8u) {
      const auto c = outputs[1][i * kStride + 3u];
      bb_r += c.x;
      bb_g += c.y;
      bb_b += c.z;
    }
  }
  bb_r /= 8.0;
  bb_g /= 8.0;
  bb_b /= 8.0;
  if (!(bb_r > bb_g && bb_g > bb_b && bb_b > 0.0)) {
    std::cerr << "FAIL 2500K Blackbody ordering expected R > G > B > 0, got ("
              << bb_r << ", " << bb_g << ", " << bb_b << ")\n";
    return false;
  }
  std::cout << "  [PASS] Volume SVM VM vs JIT parity + 2500K Blackbody ("
            << bb_r << ", " << bb_g << ", " << bb_b << ") on " << backend
            << std::endl;
  return true;
}

// Test 2: Homogeneous Chromatic Volume (Beer-Lambert + Furnace + Emission)
[[nodiscard]] bool test_homogeneous_chromatic_volume(Device &device,
                                                     std::string_view backend) {
  // 2a: Beer-Lambert chromatic absorption vs analytic host spectral integral
  {
    const c::ShaderColorSpace cs{};
    const auto host_tables = spec::build_spectral_tables_host(cs);
    const luisa::float3 abs_color{0.20f, 0.85f, 0.95f};
    constexpr std::uint32_t kStrata = 256u;
    double ref_x = 0.0, ref_y = 0.0, ref_z = 0.0;
    for (std::uint32_t s = 0u; s < kStrata; ++s) {
      const float u0 = (static_cast<float>(s) + 0.5f) / static_cast<float>(kStrata);
      const float h = u0 * 0.25f;
      const luisa::float4 lambdas{
          spec::sample_visible_wavelength_host(h),
          spec::sample_visible_wavelength_host(h + 0.25f),
          spec::sample_visible_wavelength_host(h + 0.50f),
          spec::sample_visible_wavelength_host(h + 0.75f)};
      const luisa::float4 refl =
          spec::eval_rgb2spec_reflectance_host(host_tables, abs_color, lambdas);
      const float *ls = &lambdas.x;
      const float *rs = &refl.x;
      for (std::size_t k = 0u; k < 4u; ++k) {
        const float sigma_a = std::max(1.0f - rs[k], 0.0f);
        const float trans = std::exp(-sigma_a * 1.0f);
        const float pdf = spec::visible_wavelength_pdf_host(ls[k]);
        const float pos = std::clamp(
            ls[k] - spec::spectral_lambda_min, 0.0f,
            static_cast<float>(spec::spectral_lambda_max -
                               spec::spectral_lambda_min));
        const auto i0 = std::min(static_cast<std::size_t>(pos),
                                 host_tables.cie_illuminant.size() - 2u);
        const float f = pos - static_cast<float>(i0);
        const auto c0 = host_tables.cie_illuminant[i0];
        const auto c1 = host_tables.cie_illuminant[i0 + 1u];
        const float cx = c0.x + (c1.x - c0.x) * f;
        const float cy = c0.y + (c1.y - c0.y) * f;
        const float cz = c0.z + (c1.z - c0.z) * f;
        const float il = c0.w + (c1.w - c0.w) * f;
        const float w = (trans * il) / (4.0f * pdf);
        ref_x += w * cx;
        ref_y += w * cy;
        ref_z += w * cz;
      }
    }
    ref_x /= static_cast<double>(kStrata);
    ref_y /= static_cast<double>(kStrata);
    ref_z /= static_cast<double>(kStrata);
    const std::array<float, 3u> ref_rgb{
        static_cast<float>(ref_x * host_tables.spec_xyz_to_r.x +
                           ref_y * host_tables.spec_xyz_to_r.y +
                           ref_z * host_tables.spec_xyz_to_r.z),
        static_cast<float>(ref_x * host_tables.spec_xyz_to_g.x +
                           ref_y * host_tables.spec_xyz_to_g.y +
                           ref_z * host_tables.spec_xyz_to_g.z),
        static_cast<float>(ref_x * host_tables.spec_xyz_to_b.x +
                           ref_y * host_tables.spec_xyz_to_b.y +
                           ref_z * host_tables.spec_xyz_to_b.z)};

    c::ShaderGraph vol_g;
    const auto transp = vol_g.add_node(
        psycles::compiler::node_type::transparent_bsdf, "transparent");
    const auto vabs = vol_g.add_node(
        psycles::compiler::node_type::volume_absorption, "vabs");
    static_cast<void>(vol_g.set_input(
        transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(vol_g.set_input(
        vabs, "Color",
        c::SocketValue::color({abs_color.x, abs_color.y, abs_color.z})));
    static_cast<void>(
        vol_g.set_input(vabs, "Density", c::SocketValue::floating(1.0f)));
    vol_g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});
    vol_g.set_root(c::ShaderDomain::volume, c::OutputRef{vabs, "Volume"});

    c::SceneSnapshot scene;
    scene.materials.emplace(c::MaterialId{1u},
                            c::MaterialDesc{.name = "AbsBox",
                                            .shader = std::move(vol_g),
                                            .cycles_shader_index = 0u});
    scene.materials.emplace(c::MaterialId{2u},
                            make_constant_world({1.0f, 1.0f, 1.0f}, 1.0f, 1u));
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 1u;
    scene.world_sampling = c::WorldSampling::none;
    scene.geometries.emplace(c::GeometryId{1u},
                             make_box_mesh(c::MaterialId{1u}, 2.0f, 2.0f, 0.5f));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 3.0f),
                      .orthographic_scale = 1.5f});
    scene.active_camera = c::CameraId{1u};

    const auto settings = make_render_settings(
        16u, 16u, true, 8u, 8u, c::DirectLightSampling::forward_path_tracing);
    ImageStats mega_stats{}, wf_stats{};
    if (!render_mean_rgb(device, scene, settings, b::LuisaPathScheduler::megakernel,
                         false, 64u, mega_stats, "HomogeneousBeerLambertMega",
                         backend) ||
        !render_mean_rgb(device, scene, settings,
                         b::LuisaPathScheduler::wavefront_staged, false, 64u,
                         wf_stats, "HomogeneousBeerLambertWF", backend)) {
      return false;
    }
    for (const auto *st : {&mega_stats, &wf_stats}) {
      if (!(st->mean[2] > st->mean[1] && st->mean[1] > st->mean[0] &&
            st->mean[0] > 0.05f)) {
        std::cerr << "FAIL Beer-Lambert chromatic ordering on " << backend
                  << ": got (" << st->mean[0] << ", " << st->mean[1] << ", "
                  << st->mean[2] << ")\n";
        return false;
      }
      for (std::size_t ch = 0u; ch < 3u; ++ch) {
        const float rel = std::abs(st->mean[ch] - ref_rgb[ch]) /
                          std::max(ref_rgb[ch], 1.0e-4f);
        if (rel > 0.03f) {
          std::cerr << "FAIL Beer-Lambert analytic mismatch ch=" << ch
                    << " got=" << st->mean[ch] << " ref=" << ref_rgb[ch]
                    << " rel=" << rel << " on " << backend << '\n';
          return false;
        }
      }
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float rel = std::abs(mega_stats.mean[ch] - wf_stats.mean[ch]) /
                        std::max(wf_stats.mean[ch], 1.0e-4f);
      if (rel > 0.02f) {
        std::cerr << "FAIL Beer-Lambert Mega vs WF ch=" << ch << '\n';
        return false;
      }
    }
    std::cout << "  [PASS] Homogeneous Beer-Lambert analytic=("
              << ref_rgb[0] << ", " << ref_rgb[1] << ", " << ref_rgb[2]
              << ") wf=(" << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
              << wf_stats.mean[2] << ") on " << backend << std::endl;
  }

  // 2b: Homogeneous White Volume Scatter Furnace
  {
    c::ShaderGraph vol_g;
    const auto transp = vol_g.add_node(
        psycles::compiler::node_type::transparent_bsdf, "transparent");
    const auto vscat = vol_g.add_node(
        psycles::compiler::node_type::volume_scatter, "vscat");
    static_cast<void>(vol_g.set_input(
        transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(vol_g.set_input(
        vscat, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        vol_g.set_input(vscat, "Density", c::SocketValue::floating(0.75f)));
    static_cast<void>(
        vol_g.set_input(vscat, "Anisotropy", c::SocketValue::floating(0.0f)));
    vol_g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});
    vol_g.set_root(c::ShaderDomain::volume, c::OutputRef{vscat, "Volume"});

    c::SceneSnapshot scene;
    scene.materials.emplace(c::MaterialId{1u},
                            c::MaterialDesc{.name = "ScatterFurnaceBox",
                                            .shader = std::move(vol_g),
                                            .cycles_shader_index = 0u});
    scene.materials.emplace(c::MaterialId{2u},
                            make_constant_world({1.0f, 1.0f, 1.0f}, 1.0f, 1u));
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 1u;
    scene.world_sampling = c::WorldSampling::automatic;
    scene.geometries.emplace(c::GeometryId{1u},
                             make_box_mesh(c::MaterialId{1u}, 2.0f, 2.0f, 2.0f));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 5.0f),
                      .orthographic_scale = 2.0f});
    scene.active_camera = c::CameraId{1u};

    const auto settings = make_render_settings(
        16u, 16u, true, 64u, 64u,
        c::DirectLightSampling::multiple_importance_sampling);
    ImageStats mega_stats{}, wf_stats{};
    if (!render_mean_rgb(device, scene, settings, b::LuisaPathScheduler::megakernel,
                         true, 64u, mega_stats, "HomogeneousFurnaceMega",
                         backend) ||
        !render_mean_rgb(device, scene, settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 64u,
                         wf_stats, "HomogeneousFurnaceWF", backend)) {
      return false;
    }
    for (const auto *st : {&mega_stats, &wf_stats}) {
      for (std::size_t ch = 0u; ch < 3u; ++ch) {
        if (st->mean[ch] < 0.93f || st->mean[ch] > 1.07f) {
          std::cerr << "FAIL Homogeneous White Furnace ch=" << ch
                    << " val=" << st->mean[ch] << " on " << backend << '\n';
          return false;
        }
      }
    }
    std::cout << "  [PASS] Homogeneous Volume White Furnace wf=("
              << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
              << wf_stats.mean[2] << ") on " << backend << std::endl;
  }

  // 2c: Homogeneous 2500K Blackbody Volume Emission
  {
    c::ShaderGraph vol_g;
    const auto transp = vol_g.add_node(
        psycles::compiler::node_type::transparent_bsdf, "transparent");
    const auto pv = vol_g.add_node(
        psycles::compiler::node_type::principled_volume, "pv");
    static_cast<void>(vol_g.set_input(
        transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        vol_g.set_input(pv, "Color", c::SocketValue::color({0.2f, 0.2f, 0.2f})));
    static_cast<void>(
        vol_g.set_input(pv, "Density", c::SocketValue::floating(0.5f)));
    static_cast<void>(vol_g.set_input(
        pv, "AbsorptionColor", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(vol_g.set_input(
        pv, "BlackbodyIntensity", c::SocketValue::floating(1.0f)));
    static_cast<void>(
        vol_g.set_input(pv, "Temperature", c::SocketValue::floating(2500.0f)));
    vol_g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});
    vol_g.set_root(c::ShaderDomain::volume, c::OutputRef{pv, "Volume"});

    c::SceneSnapshot scene;
    scene.materials.emplace(c::MaterialId{1u},
                            c::MaterialDesc{.name = "EmissionBox",
                                            .shader = std::move(vol_g),
                                            .cycles_shader_index = 0u});
    scene.materials.emplace(c::MaterialId{2u},
                            make_constant_world({0.0f, 0.0f, 0.0f}, 0.0f, 1u));
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 1u;
    scene.world_sampling = c::WorldSampling::none;
    scene.geometries.emplace(c::GeometryId{1u},
                             make_box_mesh(c::MaterialId{1u}, 2.0f, 2.0f, 0.5f));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 3.0f),
                      .orthographic_scale = 1.5f});
    scene.active_camera = c::CameraId{1u};

    const auto settings = make_render_settings(
        16u, 16u, true, 4u, 4u, c::DirectLightSampling::forward_path_tracing);
    ImageStats mega_stats{}, wf_stats{};
    if (!render_mean_rgb(device, scene, settings, b::LuisaPathScheduler::megakernel,
                         false, 64u, mega_stats, "HomogeneousBlackbodyMega",
                         backend) ||
        !render_mean_rgb(device, scene, settings,
                         b::LuisaPathScheduler::wavefront_staged, false, 64u,
                         wf_stats, "HomogeneousBlackbodyWF", backend)) {
      return false;
    }
    if (!(wf_stats.mean[0] > wf_stats.mean[1] &&
          wf_stats.mean[1] > wf_stats.mean[2] && wf_stats.mean[2] > 0.01f)) {
      std::cerr << "FAIL homogeneous 2500K Blackbody emission expected R>G>B>0.01, "
                   "got ("
                << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
                << wf_stats.mean[2] << ")\n";
      return false;
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float rel = std::abs(mega_stats.mean[ch] - wf_stats.mean[ch]) /
                        std::max(wf_stats.mean[ch], 1.0e-4f);
      if (rel > 0.04f) {
        std::cerr << "FAIL homogeneous blackbody Mega vs WF ch=" << ch << '\n';
        return false;
      }
    }
    std::cout << "  [PASS] Homogeneous 2500K Emission wf=("
              << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
              << wf_stats.mean[2] << ") on " << backend << std::endl;
  }
  return true;
}

// Test 3: Heterogeneous Spatial Chromatic Volume (Mega/WF + NEE ON/OFF + Het/Hom)
[[nodiscard]] bool test_heterogeneous_spatial_volume(Device &device,
                                                     std::string_view backend) {
  auto make_het_volume_graph = [](bool constant_density) {
    c::ShaderGraph g;
    const auto transp = g.add_node(
        psycles::compiler::node_type::transparent_bsdf, "transparent");
    const auto coords = g.add_node(
        psycles::compiler::node_type::texture_coordinate, "coords");
    const auto p2v = g.add_node(
        psycles::compiler::node_type::point_to_vector, "p2v");
    const auto v2s = g.add_node(
        psycles::compiler::node_type::vector_to_scalar, "v2s");
    const auto mul = g.add_node(
        psycles::compiler::node_type::multiply_float, "mul");
    const auto add = g.add_node(
        psycles::compiler::node_type::add_float, "add");
    const auto vscat = g.add_node(
        psycles::compiler::node_type::volume_scatter, "vscat");
    const auto vabs = g.add_node(
        psycles::compiler::node_type::volume_absorption, "vabs");
    const auto vadd = g.add_node(
        psycles::compiler::node_type::add_volume, "vadd");
    static_cast<void>(g.set_input(
        transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        g.connect(c::OutputRef{coords, "Generated"}, p2v, "Point"));
    static_cast<void>(g.connect(c::OutputRef{p2v, "Vector"}, v2s, "Vector"));
    static_cast<void>(g.connect(c::OutputRef{v2s, "Value"}, mul, "A"));
    static_cast<void>(g.set_input(
        mul, "B", c::SocketValue::floating(constant_density ? 0.0f : 0.8f)));
    static_cast<void>(g.connect(c::OutputRef{mul, "Value"}, add, "A"));
    static_cast<void>(g.set_input(
        add, "B", c::SocketValue::floating(constant_density ? 0.5f : 0.2f)));
    static_cast<void>(g.connect(c::OutputRef{add, "Value"}, vscat, "Density"));
    static_cast<void>(g.set_input(
        vscat, "Color", c::SocketValue::color({0.85f, 0.45f, 0.20f})));
    static_cast<void>(
        g.set_input(vscat, "Anisotropy", c::SocketValue::floating(0.25f)));
    static_cast<void>(g.connect(c::OutputRef{add, "Value"}, vabs, "Density"));
    static_cast<void>(g.set_input(
        vabs, "Color", c::SocketValue::color({0.35f, 0.70f, 0.90f})));
    static_cast<void>(g.connect(c::OutputRef{vscat, "Volume"}, vadd, "A"));
    static_cast<void>(g.connect(c::OutputRef{vabs, "Volume"}, vadd, "B"));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});
    g.set_root(c::ShaderDomain::volume, c::OutputRef{vadd, "Volume"});
    return g;
  };

  auto make_hom_ref_graph = []() {
    c::ShaderGraph g;
    const auto transp = g.add_node(
        psycles::compiler::node_type::transparent_bsdf, "transparent");
    const auto vscat = g.add_node(
        psycles::compiler::node_type::volume_scatter, "vscat");
    const auto vabs = g.add_node(
        psycles::compiler::node_type::volume_absorption, "vabs");
    const auto vadd = g.add_node(
        psycles::compiler::node_type::add_volume, "vadd");
    static_cast<void>(g.set_input(
        transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        g.set_input(vscat, "Density", c::SocketValue::floating(0.5f)));
    static_cast<void>(g.set_input(
        vscat, "Color", c::SocketValue::color({0.85f, 0.45f, 0.20f})));
    static_cast<void>(
        g.set_input(vscat, "Anisotropy", c::SocketValue::floating(0.25f)));
    static_cast<void>(
        g.set_input(vabs, "Density", c::SocketValue::floating(0.5f)));
    static_cast<void>(g.set_input(
        vabs, "Color", c::SocketValue::color({0.35f, 0.70f, 0.90f})));
    static_cast<void>(g.connect(c::OutputRef{vscat, "Volume"}, vadd, "A"));
    static_cast<void>(g.connect(c::OutputRef{vabs, "Volume"}, vadd, "B"));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});
    g.set_root(c::ShaderDomain::volume, c::OutputRef{vadd, "Volume"});
    return g;
  };

  // 3a: Spatial heterogeneous chromatic volume + emissive quad light
  // Check: Mega vs WF parity, NEE ON vs NEE OFF unbiasedness, firefly guard
  {
    c::ShaderGraph emit_g;
    const auto em =
        emit_g.add_node(psycles::compiler::node_type::emission, "em");
    static_cast<void>(
        emit_g.set_input(em, "Color", c::SocketValue::color({1.0f, 0.9f, 0.8f})));
    static_cast<void>(
        emit_g.set_input(em, "Strength", c::SocketValue::floating(24.0f)));
    emit_g.set_root(c::ShaderDomain::surface, c::OutputRef{em, "Closure"});

    c::TriangleMeshDesc quad;
    quad.name = "MeshLightQuad";
    quad.positions = {
        {1.3f, -0.5f, -0.5f},
        {1.3f, 0.5f, -0.5f},
        {1.3f, 0.5f, 0.5f},
        {1.3f, -0.5f, 0.5f},
    };
    quad.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
    quad.material_slots = {c::MaterialId{3u}};
    quad.triangle_material_slots = {0u, 0u};
    quad.triangle_smooth = {0u, 0u};

    c::SceneSnapshot scene;
    scene.materials.emplace(
        c::MaterialId{1u},
        c::MaterialDesc{
            .name = "HetVol",
            .shader = make_het_volume_graph(false),
            .volume_sampling = c::VolumeSampling::multiple_importance,
            .cycles_shader_index = 0u});
    scene.materials.emplace(
        c::MaterialId{2u}, make_constant_world({0.0f, 0.0f, 0.0f}, 0.0f, 1u));
    scene.materials.emplace(
        c::MaterialId{3u},
        c::MaterialDesc{
            .name = "Emitter",
            .shader = std::move(emit_g),
            .emission_sampling = c::EmissionSampling::front_back,
            .cycles_shader_index = 2u});
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 2u;
    scene.world_sampling = c::WorldSampling::none;
    scene.geometries.emplace(
        c::GeometryId{1u}, make_box_mesh(c::MaterialId{1u}, 1.0f, 1.0f, 1.0f));
    scene.geometries.emplace(c::GeometryId{2u}, std::move(quad));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.instances.emplace(
        c::InstanceId{2u},
        c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 3.0f),
                      .orthographic_scale = 1.4f});
    scene.active_camera = c::CameraId{1u};

    const auto mis_settings = make_render_settings(
        16u, 16u, true, 8u, 4u,
        c::DirectLightSampling::multiple_importance_sampling);
    const auto pt_settings = make_render_settings(
        16u, 16u, true, 8u, 4u, c::DirectLightSampling::forward_path_tracing);

    ImageStats mega_mis{}, wf_mis{}, wf_pt{};
    if (!render_mean_rgb(device, scene, mis_settings,
                         b::LuisaPathScheduler::megakernel, true, 128u,
                         mega_mis, "HetVolMIS_Mega", backend) ||
        !render_mean_rgb(device, scene, mis_settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 128u,
                         wf_mis, "HetVolMIS_WF", backend) ||
        !render_mean_rgb(device, scene, pt_settings,
                         b::LuisaPathScheduler::wavefront_staged, false, 512u,
                         wf_pt, "HetVolPT_WF", backend)) {
      return false;
    }

    if (!(wf_mis.mean[0] > wf_mis.mean[2] && wf_mis.mean[2] > 1.0e-3f)) {
      std::cerr << "FAIL HetVol chromatic ordering on " << backend << ": ("
                << wf_mis.mean[0] << ", " << wf_mis.mean[1] << ", "
                << wf_mis.mean[2] << ")\n";
      return false;
    }
    const float mis_mean_lum = luma_of(wf_mis.mean);
    if (wf_mis.max_luma > 25.0f * std::max(mis_mean_lum, 1.0e-5f)) {
      std::cerr << "FAIL HetVol firefly spike max_luma=" << wf_mis.max_luma
                << " mean_luma=" << mis_mean_lum << " on " << backend << '\n';
      return false;
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float mega_wf_rel =
          std::abs(mega_mis.mean[ch] - wf_mis.mean[ch]) /
          std::max(wf_mis.mean[ch], 1.0e-4f);
      if (mega_wf_rel > 0.06f) {
        std::cerr << "FAIL HetVol Mega vs WF mismatch ch=" << ch
                  << " mega=" << mega_mis.mean[ch]
                  << " wf=" << wf_mis.mean[ch] << '\n';
        return false;
      }
      const float nee_rel = std::abs(wf_mis.mean[ch] - wf_pt.mean[ch]) /
                            std::max(wf_pt.mean[ch], 1.0e-4f);
      if (nee_rel > 0.08f) {
        std::cerr << "FAIL HetVol NEE ON vs NEE OFF mismatch ch=" << ch
                  << " mis=" << wf_mis.mean[ch] << " pt=" << wf_pt.mean[ch]
                  << " rel=" << nee_rel << '\n';
        return false;
      }
    }
    std::cout << "  [PASS] Heterogeneous Chromatic Volume MIS=("
              << wf_mis.mean[0] << ", " << wf_mis.mean[1] << ", "
              << wf_mis.mean[2] << ") PT=(" << wf_pt.mean[0] << ", "
              << wf_pt.mean[1] << ", " << wf_pt.mean[2] << ") on " << backend
              << std::endl;
  }

  // 3b: Constant-density heterogeneous matches homogeneous reference under point light
  {
    auto build_scene = [&](c::ShaderGraph vol_g) {
      c::SceneSnapshot scene;
      scene.materials.emplace(
          c::MaterialId{1u},
          c::MaterialDesc{
              .name = "VolBox",
              .shader = std::move(vol_g),
              .volume_sampling = c::VolumeSampling::multiple_importance,
              .cycles_shader_index = 0u});
      scene.materials.emplace(
          c::MaterialId{2u}, make_constant_world({0.0f, 0.0f, 0.0f}, 0.0f, 1u));
      scene.world_shader = c::MaterialId{2u};
      scene.cycles_background_object_index = 2u;
      scene.world_sampling = c::WorldSampling::none;
      scene.geometries.emplace(
          c::GeometryId{1u}, make_box_mesh(c::MaterialId{1u}, 2.0f, 2.0f, 2.0f));
      scene.instances.emplace(
          c::InstanceId{1u},
          c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
      scene.lights.emplace(
          c::LightId{1u},
          c::LightDesc{.name = "VolPoint",
                       .type = c::LightType::point,
                       .transform = translated(0.4f, 0.2f, -0.5f),
                       .color = {1.0f, 0.9f, 0.8f},
                       .power = 20.0f,
                       .size = 0.0f,
                       .normalize = true,
                       .is_sphere = true,
                       .use_mis = true,
                       .cast_shadow = true,
                       .visibility_mask = c::all_ray_visibility,
                       .cycles_shader_index = 1u,
                       .cycles_object_index = 1u});
      scene.cameras.emplace(
          c::CameraId{1u},
          c::CameraDesc{.projection = c::CameraProjection::orthographic,
                        .transform = translated(0.0f, 0.0f, 3.5f),
                        .orthographic_scale = 1.6f});
      scene.active_camera = c::CameraId{1u};
      return scene;
    };

    const auto het_const = build_scene(make_het_volume_graph(true));
    const auto hom_ref = build_scene(make_hom_ref_graph());
    const auto settings = make_render_settings(
        16u, 16u, true, 4u, 2u,
        c::DirectLightSampling::multiple_importance_sampling);
    ImageStats het_stats{}, hom_stats{};
    if (!render_mean_rgb(device, het_const, settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 128u,
                         het_stats, "HetConstVol", backend) ||
        !render_mean_rgb(device, hom_ref, settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 128u,
                         hom_stats, "HomRefVol", backend)) {
      return false;
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float rel = std::abs(het_stats.mean[ch] - hom_stats.mean[ch]) /
                        std::max(hom_stats.mean[ch], 1.0e-4f);
      if (rel > 0.05f) {
        std::cerr << "FAIL const heterogeneous vs homogeneous mismatch ch="
                  << ch << " het=" << het_stats.mean[ch]
                  << " hom=" << hom_stats.mean[ch] << " rel=" << rel << '\n';
        return false;
      }
    }
    std::cout << "  [PASS] Const-Het vs Homogeneous ref het=("
              << het_stats.mean[0] << ", " << het_stats.mean[1] << ", "
              << het_stats.mean[2] << ") hom=(" << hom_stats.mean[0] << ", "
              << hom_stats.mean[1] << ", " << hom_stats.mean[2] << ") on "
              << backend << std::endl;
  }
  return true;
}

// Test 4: Spectral Random-Walk SSS (White Furnace + Chromatic Backlit Slab)
[[nodiscard]] bool test_spectral_random_walk_sss(Device &device,
                                                 std::string_view backend) {
  auto make_sss_mat = [](psycles::Vec3f color, float scale,
                         psycles::Vec3f radius, std::uint32_t shader_idx) {
    c::ShaderGraph g;
    const auto sss = g.add_node(
        psycles::compiler::node_type::subsurface_scattering, "sss");
    static_cast<void>(g.set_input(sss, "Color", c::SocketValue::color(color)));
    static_cast<void>(
        g.set_input(sss, "Scale", c::SocketValue::floating(scale)));
    static_cast<void>(
        g.set_input(sss, "Radius", c::SocketValue::vector(radius)));
    static_cast<void>(g.set_input(sss, "IOR", c::SocketValue::floating(1.4f)));
    static_cast<void>(
        g.set_input(sss, "Roughness", c::SocketValue::floating(0.0f)));
    static_cast<void>(
        g.set_input(sss, "Anisotropy", c::SocketValue::floating(0.0f)));
    static_cast<void>(g.set_property(sss, "Method",
                                     c::SocketValue::string("RANDOM_WALK")));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{sss, "Closure"});
    return c::MaterialDesc{
        .name = "SSS", .shader = std::move(g), .cycles_shader_index = shader_idx};
  };

  // 4a: SSS White Furnace (1,1,1) sphere inside white (1,1,1) environment
  {
    c::SceneSnapshot scene;
    scene.materials.emplace(
        c::MaterialId{1u},
        make_sss_mat({1.0f, 1.0f, 1.0f}, 0.2f, {1.0f, 1.0f, 1.0f}, 0u));
    scene.materials.emplace(
        c::MaterialId{2u}, make_constant_world({1.0f, 1.0f, 1.0f}, 1.0f, 1u));
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 1u;
    scene.world_sampling = c::WorldSampling::automatic;
    scene.geometries.emplace(
        c::GeometryId{1u},
        make_uv_sphere_mesh(c::MaterialId{1u}, 1.0f, 32u, 20u));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 3.0f),
                      .orthographic_scale = 1.5f});
    scene.active_camera = c::CameraId{1u};

    const auto settings = make_render_settings(
        16u, 16u, true, 32u, 4u,
        c::DirectLightSampling::multiple_importance_sampling);
    ImageStats mega_stats{}, wf_stats{};
    if (!render_mean_rgb(device, scene, settings, b::LuisaPathScheduler::megakernel,
                         true, 64u, mega_stats, "SSSWhiteFurnaceMega", backend) ||
        !render_mean_rgb(device, scene, settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 64u,
                         wf_stats, "SSSWhiteFurnaceWF", backend)) {
      return false;
    }
    for (const auto *st : {&mega_stats, &wf_stats}) {
      for (std::size_t ch = 0u; ch < 3u; ++ch) {
        if (st->mean[ch] < 0.92f || st->mean[ch] > 1.08f) {
          std::cerr << "FAIL SSS White Furnace ch=" << ch
                    << " val=" << st->mean[ch] << " on " << backend << '\n';
          return false;
        }
      }
    }
    std::cout << "  [PASS] Spectral RW-SSS White Furnace wf=("
              << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
              << wf_stats.mean[2] << ") mega=(" << mega_stats.mean[0] << ", "
              << mega_stats.mean[1] << ", " << mega_stats.mean[2] << ") on "
              << backend << std::endl;
  }

  // 4b: Chromatic Backlit Slab radius=(0.80, 0.15, 0.04) + Mega vs WF parity
  {
    c::SceneSnapshot scene;
    scene.materials.emplace(
        c::MaterialId{1u},
        make_sss_mat({0.95f, 0.90f, 0.85f}, 1.0f, {0.80f, 0.15f, 0.04f}, 0u));
    scene.materials.emplace(
        c::MaterialId{2u}, make_constant_world({0.0f, 0.0f, 0.0f}, 0.0f, 1u));
    scene.world_shader = c::MaterialId{2u};
    scene.cycles_background_object_index = 2u;
    scene.world_sampling = c::WorldSampling::none;
    scene.geometries.emplace(
        c::GeometryId{1u},
        make_box_mesh(c::MaterialId{1u}, 3.0f, 3.0f, 0.075f));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "BacklightPoint",
                     .type = c::LightType::point,
                     .transform = translated(0.0f, 0.0f, -0.45f),
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 60.0f,
                     .size = 0.0f,
                     .normalize = true,
                     .is_sphere = true,
                     .use_mis = true,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .cycles_shader_index = 1u,
                     .cycles_object_index = 1u});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = translated(0.0f, 0.0f, 2.0f),
                      .orthographic_scale = 1.6f});
    scene.active_camera = c::CameraId{1u};

    const auto settings = make_render_settings(
        16u, 16u, true, 16u, 4u,
        c::DirectLightSampling::multiple_importance_sampling);
    ImageStats mega_stats{}, wf_stats{};
    if (!render_mean_rgb(device, scene, settings, b::LuisaPathScheduler::megakernel,
                         true, 128u, mega_stats, "SSSBacklitSlabMega",
                         backend) ||
        !render_mean_rgb(device, scene, settings,
                         b::LuisaPathScheduler::wavefront_staged, true, 128u,
                         wf_stats, "SSSBacklitSlabWF", backend)) {
      return false;
    }
    for (const auto *st : {&mega_stats, &wf_stats}) {
      if (!(st->mean[0] > 2.0f * st->mean[2] &&
            st->mean[0] > st->mean[1] && st->mean[1] > st->mean[2] &&
            st->mean[2] > 1.0e-4f)) {
        std::cerr << "FAIL chromatic backlit SSS ordering expected R>2B & "
                     "R>G>B>1e-4, got ("
                  << st->mean[0] << ", " << st->mean[1] << ", " << st->mean[2]
                  << ") on " << backend << '\n';
        return false;
      }
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float rel = std::abs(mega_stats.mean[ch] - wf_stats.mean[ch]) /
                        std::max(wf_stats.mean[ch], 1.0e-4f);
      if (rel > 0.06f) {
        std::cerr << "FAIL backlit SSS Mega vs WF mismatch ch=" << ch
                  << " mega=" << mega_stats.mean[ch]
                  << " wf=" << wf_stats.mean[ch] << " rel=" << rel << '\n';
        return false;
      }
    }
    std::cout << "  [PASS] Spectral Backlit Chromatic RW-SSS wf=("
              << wf_stats.mean[0] << ", " << wf_stats.mean[1] << ", "
              << wf_stats.mean[2] << ") mega=(" << mega_stats.mean[0] << ", "
              << mega_stats.mean[1] << ", " << mega_stats.mean[2] << ") on "
              << backend << std::endl;
  }
  return true;
}

[[nodiscard]] bool test_spectral_majorant_exceeded_defense(
    Device &device, Stream &stream, std::string_view backend) {
  auto out = device.create_buffer<luisa::float4>(4u);
  Kernel1D k = [](BufferVar<luisa::float4> buf) noexcept {
    psycles::luisa_backend::HeterogeneousVolumeTracking tracking;
    auto coeff = psycles::luisa_backend::VolumeCoefficients::zero(true);
    coeff.sigma_t = make_float3(0.5f, 2.5f, 1.2f);
    coeff.sigma_s = make_float3(0.3f, 1.8f, 0.7f);
    coeff.emission = make_float3(0.2f, 0.4f, 0.1f);
    coeff.has_extinction = true;
    coeff.has_scatter = true;
    coeff.has_emission = true;
    coeff.spectral_sigma_t = make_float4(0.5f, 2.5f, 1.2f, 0.8f);
    coeff.spectral_sigma_s = make_float4(0.3f, 1.8f, 0.7f, 0.4f);
    coeff.spectral_emission = make_float4(0.2f, 0.4f, 0.1f, 0.05f);
    const auto col = tracking.evaluate_collision_spectral(
        coeff, 1.0f, 0.2f, make_float4(1.0f), make_float4(1.0f), Bool{false});
    buf.write(0u, make_float4(select(0.0f, 1.0f, col.majorant_exceeded),
                              col.majorant, col.emission.x, col.emission.y));
    buf.write(1u, col.sigma_n);
    buf.write(2u, col.null_beta_u);
    buf.write(3u, col.scatter_throughput);
  };
  auto shader = device.compile(k, ShaderOption{.enable_cache = false});
  std::array<luisa::float4, 4u> actual{};
  stream << shader(out).dispatch(1u) << out.copy_to(actual.data())
         << synchronize();
  const auto non_negative = [](const luisa::float4 &v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) &&
           std::isfinite(v.w) && v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f &&
           v.w >= 0.0f;
  };
  if (actual[0u].x != 1.0f || std::abs(actual[0u].y - 2.5f) > 1.0e-5f ||
      !non_negative(actual[1u]) || !non_negative(actual[2u]) ||
      !non_negative(actual[3u])) {
    std::cerr << "FAIL spectral majorant_exceeded defense on " << backend
              << ": exceeded=" << actual[0u].x << " maj=" << actual[0u].y
              << " sigma_n=(" << actual[1u].x << "," << actual[1u].y << ","
              << actual[1u].z << "," << actual[1u].w << ") null_beta=("
              << actual[2u].x << "," << actual[2u].y << "," << actual[2u].z
              << "," << actual[2u].w << ")\n";
    return false;
  }
  std::cout << "  [PASS] Spectral defensive majorant_exceeded maj="
            << actual[0u].y << " on " << backend << std::endl;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running test_luisa_spectral_volume_sss on " << backend
              << "...\n";
    if (!test_spectral_majorant_exceeded_defense(device, stream, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_volume_svm_parity_and_blackbody(device, stream, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_homogeneous_chromatic_volume(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_heterogeneous_spatial_volume(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_spectral_random_walk_sss(device, backend)) {
      return EXIT_FAILURE;
    }
    std::cout << "test_luisa_spectral_volume_sss PASSED on " << backend
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Exception in test_luisa_spectral_volume_sss: " << e.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
