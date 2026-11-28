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
#include <psycles/luisa/path_tracer.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_kernel_direct_light_task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace abi = psycles::compiler::cycles_svm;
namespace c = psycles::contract;
namespace b = psycles::luisa_backend;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace spec = psycles::luisa_backend::cycles_spectrum;

static_assert(sizeof(b::detail::DirectLightTaskCall) == 224u,
              "DirectLightTaskCall must remain exactly 224 bytes.");

class SpectralTestKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const spec::SpectralTablesGpu &_tables;

public:
  explicit SpectralTestKernelGlobals(
      const spec::SpectralTablesGpu &tables) noexcept
      : _tables{tables} {}

  [[nodiscard]] bool is_spectral() const noexcept override { return true; }

  [[nodiscard]] const spec::SpectralTablesGpu *
  spectral_tables() const noexcept override {
    return &_tables;
  }
};

[[nodiscard]] bool test_translator_parity(Device &device, Stream &stream,
                                          std::string_view backend) {
  const c::ShaderColorSpace cs{};
  const auto tables = spec::upload_spectral_tables(device, stream, cs);

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};

  c::ShaderGraph blackbody_graph;
  const auto bb = blackbody_graph.add_node(psycles::compiler::node_type::blackbody,
                                           "bb");
  const auto bb_emit = blackbody_graph.add_node(
      psycles::compiler::node_type::emission, "bb_emit");
  if (!blackbody_graph.set_input(bb, "Temperature",
                                 c::SocketValue::floating(3315.0f)) ||
      !blackbody_graph.set_input(bb_emit, "Strength",
                                 c::SocketValue::floating(1.0f)) ||
      !blackbody_graph.connect(c::OutputRef{bb, "Color"}, bb_emit, "Color")) {
    std::cerr << "Failed constructing Blackbody graph\n";
    return false;
  }
  blackbody_graph.set_root(c::ShaderDomain::surface,
                           c::OutputRef{bb_emit, "Closure"});

  c::ShaderGraph wavelength_graph;
  const auto wl = wavelength_graph.add_node(
      psycles::compiler::node_type::wavelength, "wl");
  const auto wl_emit = wavelength_graph.add_node(
      psycles::compiler::node_type::emission, "wl_emit");
  if (!wavelength_graph.set_input(wl, "Wavelength",
                                  c::SocketValue::floating(530.0f)) ||
      !wavelength_graph.set_input(wl_emit, "Strength",
                                  c::SocketValue::floating(1.0f)) ||
      !wavelength_graph.connect(c::OutputRef{wl, "Color"}, wl_emit, "Color")) {
    std::cerr << "Failed constructing Wavelength graph\n";
    return false;
  }
  wavelength_graph.set_root(c::ShaderDomain::surface,
                            c::OutputRef{wl_emit, "Closure"});

  const auto prog_bb = frontend.compile(blackbody_graph);
  const auto prog_wl = frontend.compile(wavelength_graph);
  if (!prog_bb.ok() || !prog_wl.ok()) {
    std::cerr << "Failed frontend compilation for translator parity graphs\n";
    return false;
  }

  std::array<abi::ShaderTableCompileUnit, 2u> units{{
      {.shader_index = 0u, .shader = prog_bb.program.get()},
      {.shader_index = 1u, .shader = prog_wl.program.get()},
  }};
  const auto table = abi::compile_shader_table(units);
  if (!table.table.valid) {
    std::cerr << "compile_shader_table failed: " << table.table.diagnostic << '\n';
    return false;
  }

  auto words = device.create_buffer<std::uint32_t>(table.table.words.size());
  stream << words.copy_from(table.table.words.data());

  constexpr std::uint32_t kCount = 16u;
  std::array<std::vector<luisa::float4>, 2u> outputs;
  const auto surface_usage = table.table.usage_for(abi::SHADER_TYPE_SURFACE);

  for (std::size_t s = 0u; s < 2u; ++s) {
    const auto strategy = s == 0u ? svm::SvmBackendStrategy::bytecode_vm
                                  : svm::SvmBackendStrategy::direct_jit;
    auto out_buf = device.create_buffer<luisa::float4>(kCount);
    Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>> kernel =
        [&tables, &table, surface_usage, strategy](
            BufferUInt word_buf, BufferFloat4 out) noexcept {
          const UInt idx = dispatch_x();
          const Float u_lambda =
              ((idx % 8u).cast<float>() + 0.5f) * (1.0f / 8.0f);
          const auto wavelengths = spec::sample_visible_wavelengths(u_lambda);
          const auto identity = make_float4x4(1.0f);
          const auto normal = make_float3(0.0f, 0.0f, 1.0f);
          svm::ClosurePool pool{1u, {.spectral = true}};
          svm::ShaderData sd{
              make_float3(0.0f), normal, normal, normal,
              svm::primitive_triangle, idx / 8u, 0u, 0u,
              svm::primitive_none, 0.0f, 0.0f, svm::object_none, 0.0f, 0.0f,
              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
              make_float3(1.0f, 0.0f, 0.0f),
              make_float3(0.0f, 1.0f, 0.0f), identity, identity, 0u, &pool};
          sd.spectral = spec::SpectralContext{
              .tables = tables.get(),
              .wavelengths = wavelengths,
              .secondary_terminated = false};
          sd.spectral_emission_background.emplace(make_float4(0.0f));
          const svm::TransformState transforms{
              identity, identity, identity, identity};
          const svm::PathState path{svm::path_ray_visibility_camera, 0u};
          const SpectralTestKernelGlobals globals{*tables};
          svm::svm_translator_for(strategy).translate(svm::SvmTranslationRequest{
              .kernel_globals = globals,
              .table = table.table,
              .words = word_buf,
              .shader_type = abi::SHADER_TYPE_SURFACE,
              .kernel_features = table.kernel_features,
              .node_feature_mask = svm::kernel_feature_node_mask_surface,
              .node_types_used = surface_usage.node_types_used,
              .transform_state = transforms,
              .shader_data = sd,
              .path_state = path,
              .stack_size =
                  std::max<std::size_t>(1u, surface_usage.peak_stack_usage),
              .noise_usage = surface_usage.noise_usage});
          out.write(idx, *sd.spectral_emission_background);
        };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    outputs[s].resize(kCount);
    stream << shader(words, out_buf).dispatch(kCount)
           << out_buf.copy_to(outputs[s].data()) << synchronize();
  }

  for (std::uint32_t i = 0u; i < kCount; ++i) {
    const auto a = outputs[0][i];
    const auto d = outputs[1][i];
    const std::array<float, 4u> va{a.x, a.y, a.z, a.w};
    const std::array<float, 4u> vd{d.x, d.y, d.z, d.w};
    float sum = 0.0f;
    for (std::size_t k = 0u; k < 4u; ++k) {
      if (!std::isfinite(va[k]) || !std::isfinite(vd[k])) {
        std::cerr << "FAIL spectral translator parity non-finite at idx=" << i
                  << " on " << backend << '\n';
        return false;
      }
      const float diff = std::abs(va[k] - vd[k]);
      const float scale = std::max(1.0f, std::max(std::abs(va[k]), std::abs(vd[k])));
      if (diff > 1.0e-5f * scale) {
        std::cerr << "FAIL spectral translator parity mismatch idx=" << i
                  << " lane=" << k << " vm=" << va[k] << " jit=" << vd[k]
                  << " on " << backend << '\n';
        return false;
      }
      sum += vd[k];
    }
    if (i < 8u && sum <= 0.0f) {
      std::cerr << "FAIL blackbody spectral emission zero at idx=" << i << '\n';
      return false;
    }
  }
  std::cout << "  [PASS] BytecodeVm vs DirectJit spectral parity on " << backend
            << std::endl;
  return true;
}

c::MaterialDesc make_constant_bg(psycles::Vec3f color, float strength) {
  c::ShaderGraph g;
  const auto bg = g.add_node(psycles::compiler::node_type::background, "bg");
  static_cast<void>(g.set_input(bg, "Color", c::SocketValue::color(color)));
  static_cast<void>(g.set_input(bg, "Strength", c::SocketValue::floating(strength)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{bg, "Closure"});
  return c::MaterialDesc{.name = "World", .shader = std::move(g), .cycles_shader_index = 0u};
}

c::MaterialDesc make_furnace_material(int kind, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  c::NodeId out_node{};
  float dispersion_abbe = 0.0f;
  std::string name;
  switch (kind) {
    case 0: {
      name = "DiffuseWhite";
      out_node = g.add_node(psycles::compiler::node_type::diffuse_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.0f)));
      break;
    }
    case 1: {
      name = "GGXMetallicWhite";
      out_node = g.add_node(psycles::compiler::node_type::principled_bsdf, name);
      static_cast<void>(g.set_input(out_node, "BaseColor", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Metallic", c::SocketValue::floating(1.0f)));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.25f)));
      break;
    }
    case 2: {
      name = "SmoothGlassWhite";
      out_node = g.add_node(psycles::compiler::node_type::glass_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.0f)));
      static_cast<void>(g.set_input(out_node, "IOR", c::SocketValue::floating(1.5f)));
      break;
    }
    case 3: {
      name = "RoughGlassWhite";
      out_node = g.add_node(psycles::compiler::node_type::glass_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.2f)));
      static_cast<void>(g.set_input(out_node, "IOR", c::SocketValue::floating(1.5f)));
      break;
    }
    case 4: {
      name = "SmoothDispersiveGlass";
      out_node = g.add_node(psycles::compiler::node_type::glass_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.0f)));
      static_cast<void>(g.set_input(out_node, "IOR", c::SocketValue::floating(1.5f)));
      dispersion_abbe = 25.0f;
      break;
    }
    case 5: {
      name = "RoughDispersiveGlass";
      out_node = g.add_node(psycles::compiler::node_type::glass_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.2f)));
      static_cast<void>(g.set_input(out_node, "IOR", c::SocketValue::floating(1.5f)));
      dispersion_abbe = 25.0f;
      break;
    }
    case 6: {
      name = "ThinFilmDielectric";
      out_node = g.add_node(psycles::compiler::node_type::glass_bsdf, name);
      static_cast<void>(g.set_input(out_node, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.05f)));
      static_cast<void>(g.set_input(out_node, "IOR", c::SocketValue::floating(1.5f)));
      static_cast<void>(g.set_input(out_node, "ThinFilmThickness", c::SocketValue::floating(350.0f)));
      static_cast<void>(g.set_input(out_node, "ThinFilmIOR", c::SocketValue::floating(1.38f)));
      break;
    }
    default: {
      name = "ThinFilmConductor";
      out_node = g.add_node(psycles::compiler::node_type::metallic_bsdf, name);
      static_cast<void>(g.set_input(out_node, "BaseColor", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "EdgeTint", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
      static_cast<void>(g.set_input(out_node, "Roughness", c::SocketValue::floating(0.08f)));
      static_cast<void>(g.set_input(out_node, "ThinFilmThickness", c::SocketValue::floating(400.0f)));
      static_cast<void>(g.set_input(out_node, "ThinFilmIOR", c::SocketValue::floating(1.45f)));
      break;
    }
  }
  g.set_root(c::ShaderDomain::surface, c::OutputRef{out_node, "Closure"});
  c::MaterialDesc mat{.name = name, .shader = std::move(g)};
  mat.dispersion_abbe = dispersion_abbe;
  mat.cycles_shader_index = shader_idx;
  return mat;
}

[[nodiscard]] bool test_white_furnace(Device &device, std::string_view backend) {
  constexpr int kTiles = 8;
  constexpr std::uint32_t kTileWidth = 10u;
  constexpr std::uint32_t kWidth = kTiles * kTileWidth;
  constexpr std::uint32_t kHeight = 16u;
  constexpr std::uint32_t kSpp = 64u;
  constexpr float kSlope = 0.57735027f; // tan(30 deg), coplanar across tiles

  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{100u},
                          make_constant_bg({1.0f, 1.0f, 1.0f}, 1.0f));
  scene.world_shader = c::MaterialId{100u};
  scene.world_sampling = c::WorldSampling::automatic;

  c::TriangleMeshDesc mesh;
  mesh.name = "FurnaceStrips";
  for (int k = 0; k < kTiles; ++k) {
    scene.materials.emplace(
        c::MaterialId{static_cast<std::uint64_t>(k + 1)},
        make_furnace_material(k, static_cast<std::uint32_t>(k + 1)));
    const float x0 = -4.0f + static_cast<float>(k) + 0.02f;
    const float x1 = -3.0f + static_cast<float>(k) - 0.02f;
    const float y0 = -3.0f;
    const float y1 = 3.0f;
    const auto base = static_cast<std::uint32_t>(mesh.positions.size());
    mesh.positions.push_back({x0, y0, y0 * kSlope});
    mesh.positions.push_back({x1, y0, y0 * kSlope});
    mesh.positions.push_back({x1, y1, y1 * kSlope});
    mesh.positions.push_back({x0, y1, y1 * kSlope});
    mesh.triangles.push_back({base + 0u, base + 1u, base + 2u});
    mesh.triangles.push_back({base + 0u, base + 2u, base + 3u});
    mesh.material_slots.push_back(c::MaterialId{static_cast<std::uint64_t>(k + 1)});
    mesh.triangle_material_slots.push_back(static_cast<std::uint32_t>(k));
    mesh.triangle_material_slots.push_back(static_cast<std::uint32_t>(k));
  }
  scene.geometries.emplace(c::GeometryId{1u}, std::move(mesh));
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});

  psycles::Mat4f cam_tfm;
  cam_tfm.elements[14] = 5.0f;
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 8.0f});
  scene.active_camera = c::CameraId{1u};

  constexpr std::array<const char *, kTiles> kNames{{
      "DiffuseWhite",
      "GGXMetallicWhite",
      "SmoothGlassWhite",
      "RoughGlassWhite",
      "SmoothDispersiveGlass",
      "RoughDispersiveGlass",
      "ThinFilmDielectric",
      "ThinFilmConductor",
  }};
  constexpr std::array<float, kTiles> kTols{
      0.03f, 0.05f, 0.02f, 0.05f, 0.05f, 0.06f, 0.04f, 0.05f};

  for (const auto mode : {c::SpectralMode::hero4, c::SpectralMode::xyz3}) {
    for (const auto scheduler : {b::LuisaPathScheduler::megakernel,
                                 b::LuisaPathScheduler::wavefront_staged}) {
      b::LuisaPathTracerBackend renderer{
          Device{device.impl_shared()},
          {.next_event_estimation = true,
           .scheduler = scheduler,
           .wavefront_frame_capacity = 2048u,
           .max_samples_per_dispatch = kSpp}};
      auto compiled = renderer.compile_scene(scene);
      if (!compiled.ok()) {
        for (const auto &d : compiled.diagnostics) {
          std::cerr << d.message << '\n';
        }
        return false;
      }

      c::RenderSettings settings{.full_extent = {kWidth, kHeight},
                                 .pixel_filter = c::PixelFilter::box,
                                 .filter_width = 1.0f};
      settings.integrator.spectral_rendering = true;
      settings.integrator.spectral_mode = mode;
      settings.integrator.max_bounces = 16;
      settings.integrator.diffuse_bounces = 16;
      settings.integrator.glossy_bounces = 16;
      settings.integrator.transmission_bounces = 16;
      settings.integrator.transparent_max_bounces = 16;
      settings.integrator.sample_clamp_direct = 0.0f;
      settings.integrator.sample_clamp_indirect = 0.0f;
      settings.integrator.direct_light_sampling =
          c::DirectLightSampling::multiple_importance_sampling;
      settings.passes = {{.kind = c::PassKind::combined, .channels = 4}};

      auto session = renderer.create_session(*compiled.scene, settings);
      if (!session) {
        std::cerr << "Failed creating spectral white furnace session\n";
        return false;
      }
      psycles::io::MemoryOutputSink output;
      if (!session->render_samples(
              {.first = 0, .count = kSpp, .offset = 0, .total = kSpp}, output)) {
        std::cerr << "render_samples failed on white furnace\n";
        return false;
      }
      const auto image = output.find(c::PassKind::combined);
      if (!image) {
        std::cerr << "Missing combined output pass\n";
        return false;
      }

      for (std::size_t i = 0u; i < image->pixels.size(); ++i) {
        if (!std::isfinite(image->pixels[i])) {
          std::cerr << "FAIL non-finite pixel in spectral white furnace on "
                    << backend << '\n';
          return false;
        }
      }

      const char *mode_name = mode == c::SpectralMode::hero4 ? "hero4" : "xyz3";
      for (int tile = 0; tile < kTiles; ++tile) {
        double sr = 0.0, sg = 0.0, sb = 0.0;
        std::uint32_t count = 0u;
        for (std::uint32_t y = 2u; y + 2u < kHeight; ++y) {
          for (std::uint32_t x = static_cast<std::uint32_t>(tile) * kTileWidth + 2u;
               x + 2u < static_cast<std::uint32_t>(tile + 1) * kTileWidth; ++x) {
            const auto idx = (y * kWidth + x) * image->channels;
            sr += image->pixels[idx + 0u];
            sg += image->pixels[idx + 1u];
            sb += image->pixels[idx + 2u];
            ++count;
          }
        }
        const float r = static_cast<float>(sr / count);
        const float g = static_cast<float>(sg / count);
        const float bch = static_cast<float>(sb / count);
        const float tol = kTols[static_cast<std::size_t>(tile)];
        if (std::abs(r - 1.0f) > tol || std::abs(g - 1.0f) > tol ||
            std::abs(bch - 1.0f) > tol) {
          std::cerr << "FAIL furnace [" << kNames[tile] << "][" << mode_name
                    << "] sched="
                    << (scheduler == b::LuisaPathScheduler::megakernel
                            ? "megakernel"
                            : "wavefront_staged")
                    << " on " << backend << ": rgb=(" << r << ", " << g << ", "
                    << bch << ") tol=" << tol << '\n';
          return false;
        }
        std::cout << "  [PASS] Furnace " << kNames[tile] << " (" << mode_name
                  << ", "
                  << (scheduler == b::LuisaPathScheduler::megakernel
                          ? "megakernel"
                          : "wavefront_staged")
                  << "): rgb=(" << r << ", " << g << ", " << bch << ")"
                  << std::endl;
      }
    }
  }
  return true;
}

[[nodiscard]] bool test_spectral_nee_transparent_shadow(Device &device,
                                                        std::string_view backend) {
  constexpr std::uint32_t kSize = 16u;
  constexpr std::uint32_t kSpp = 32u;

  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{10u},
                          make_constant_bg({0.0f, 0.0f, 0.0f}, 0.0f));
  scene.world_shader = c::MaterialId{10u};
  scene.world_sampling = c::WorldSampling::none;

  // 1: Diffuse white receiver at z=0
  {
    c::ShaderGraph g;
    const auto n = g.add_node(psycles::compiler::node_type::diffuse_bsdf, "recv");
    static_cast<void>(g.set_input(n, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{n, "Closure"});
    scene.materials.emplace(
        c::MaterialId{1u},
        c::MaterialDesc{.name = "Receiver", .shader = std::move(g), .cycles_shader_index = 1u});
  }
  // 2: Tinted transparent shadow blocker at z=2
  {
    c::ShaderGraph g;
    const auto n = g.add_node(psycles::compiler::node_type::transparent_bsdf, "trans");
    static_cast<void>(g.set_input(n, "Color", c::SocketValue::color({0.8f, 0.45f, 0.25f})));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{n, "Closure"});
    scene.materials.emplace(
        c::MaterialId{2u},
        c::MaterialDesc{.name = "Blocker", .shader = std::move(g), .cycles_shader_index = 2u});
  }
  // 3: Blackbody (4500K) mesh emitter at z=4
  {
    c::ShaderGraph g;
    const auto bb = g.add_node(psycles::compiler::node_type::blackbody, "bb");
    const auto em = g.add_node(psycles::compiler::node_type::emission, "em");
    static_cast<void>(g.set_input(bb, "Temperature", c::SocketValue::floating(4500.0f)));
    static_cast<void>(g.set_input(em, "Strength", c::SocketValue::floating(12.0f)));
    static_cast<void>(g.connect(c::OutputRef{bb, "Color"}, em, "Color"));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{em, "Closure"});
    scene.materials.emplace(
        c::MaterialId{3u},
        c::MaterialDesc{.name = "Emitter", .shader = std::move(g), .cycles_shader_index = 3u});
  }

  auto make_quad = [](float z, c::MaterialId mat, bool flip_down) {
    c::TriangleMeshDesc m;
    m.positions = {{-2.0f, -2.0f, z}, {2.0f, -2.0f, z},
                   {2.0f, 2.0f, z}, {-2.0f, 2.0f, z}};
    if (flip_down) {
      m.triangles = {{0u, 2u, 1u}, {0u, 3u, 2u}};
    } else {
      m.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
    }
    m.material_slots = {mat};
    m.triangle_material_slots = {0u, 0u};
    return m;
  };
  scene.geometries.emplace(c::GeometryId{1u}, make_quad(0.0f, c::MaterialId{1u}, false));
  scene.geometries.emplace(c::GeometryId{2u}, make_quad(2.0f, c::MaterialId{2u}, false));
  scene.geometries.emplace(c::GeometryId{3u}, make_quad(4.0f, c::MaterialId{3u}, true));
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = c::GeometryId{2u}, .cycles_object_index = 1u});
  scene.instances.emplace(
      c::InstanceId{3u},
      c::InstanceDesc{.geometry = c::GeometryId{3u}, .cycles_object_index = 2u});

  psycles::Mat4f cam_tfm;
  cam_tfm.elements[14] = 1.5f; // Between receiver (z=0) and blocker (z=2), looking -Z
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 2.0f});
  scene.active_camera = c::CameraId{1u};

  std::array<std::array<float, 3u>, 2u> means{};
  std::size_t s_idx = 0u;
  for (const auto scheduler : {b::LuisaPathScheduler::megakernel,
                               b::LuisaPathScheduler::wavefront_staged}) {
    b::LuisaPathTracerBackend renderer{
        Device{device.impl_shared()},
        {.next_event_estimation = true,
         .scheduler = scheduler,
         .wavefront_frame_capacity = 2048u,
         .max_samples_per_dispatch = kSpp}};
    auto compiled = renderer.compile_scene(scene);
    if (!compiled.ok()) {
      for (const auto &d : compiled.diagnostics) {
        std::cerr << d.message << '\n';
      }
      return false;
    }
    c::RenderSettings settings{.full_extent = {kSize, kSize},
                               .pixel_filter = c::PixelFilter::box,
                               .filter_width = 1.0f};
    settings.integrator.spectral_rendering = true;
    settings.integrator.max_bounces = 4;
    settings.integrator.diffuse_bounces = 2;
    settings.integrator.transparent_max_bounces = 8;
    settings.integrator.sample_clamp_direct = 0.0f;
    settings.integrator.sample_clamp_indirect = 0.0f;
    settings.integrator.direct_light_sampling =
        c::DirectLightSampling::multiple_importance_sampling;
    settings.passes = {{.kind = c::PassKind::combined, .channels = 4}};
    auto session = renderer.create_session(*compiled.scene, settings);
    if (!session) {
      std::cerr << "Failed creating transparent shadow spectral session\n";
      return false;
    }
    psycles::io::MemoryOutputSink output;
    if (!session->render_samples(
            {.first = 0, .count = kSpp, .offset = 0, .total = kSpp}, output)) {
      return false;
    }
    const auto image = output.find(c::PassKind::combined);
    if (!image) {
      return false;
    }
    double sr = 0.0, sg = 0.0, sb = 0.0;
    const auto count = kSize * kSize;
    for (std::uint32_t i = 0u; i < count; ++i) {
      const float r = image->pixels[i * image->channels + 0u];
      const float g = image->pixels[i * image->channels + 1u];
      const float bl = image->pixels[i * image->channels + 2u];
      if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bl)) {
        std::cerr << "FAIL non-finite transparent shadow pixel on " << backend << '\n';
        return false;
      }
      sr += r;
      sg += g;
      sb += bl;
    }
    means[s_idx] = {static_cast<float>(sr / count),
                    static_cast<float>(sg / count),
                    static_cast<float>(sb / count)};
    const auto &m = means[s_idx];
    if (!(m[0] > m[1] && m[1] > m[2] && m[2] > 0.005f)) {
      std::cerr << "FAIL transparent shadow channel ordering on " << backend
                << ": rgb=(" << m[0] << ", " << m[1] << ", " << m[2] << ")\n";
      return false;
    }
    std::cout << "  [PASS] Blackbody NEE + transparent shadow ("
              << (scheduler == b::LuisaPathScheduler::megakernel
                      ? "megakernel"
                      : "wavefront_staged")
              << "): rgb=(" << m[0] << ", " << m[1] << ", " << m[2] << ")"
              << std::endl;
    ++s_idx;
  }
  for (std::size_t ch = 0u; ch < 3u; ++ch) {
    const float diff = std::abs(means[0][ch] - means[1][ch]);
    const float scale = std::max(means[0][ch], means[1][ch]);
    if (diff > 0.06f * scale) {
      std::cerr << "FAIL megakernel vs wavefront_staged mismatch in NEE shadow ch="
                << ch << " mega=" << means[0][ch] << " wf=" << means[1][ch]
                << " on " << backend << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_spectral_nee_nonconstant_point_light(
    Device &device, std::string_view backend) {
  constexpr std::uint32_t kSize = 16u;
  constexpr std::uint32_t kSpp = 64u;

  c::SceneSnapshot scene;
  scene.materials.emplace(c::MaterialId{10u},
                          make_constant_bg({0.0f, 0.0f, 0.0f}, 0.0f));
  scene.world_shader = c::MaterialId{10u};
  scene.world_sampling = c::WorldSampling::none;

  // 1: Diffuse white receiver at z=0
  {
    c::ShaderGraph g;
    const auto n = g.add_node(psycles::compiler::node_type::diffuse_bsdf, "recv");
    static_cast<void>(
        g.set_input(n, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
    g.set_root(c::ShaderDomain::surface, c::OutputRef{n, "Closure"});
    scene.materials.emplace(
        c::MaterialId{1u},
        c::MaterialDesc{.name = "Receiver",
                        .shader = std::move(g),
                        .cycles_shader_index = 1u});
  }
  // 2: Non-constant point light shader (Blackbody + LightFalloff Quadratic)
  {
    c::ShaderGraph g;
    const auto bb = g.add_node(psycles::compiler::node_type::blackbody, "bb");
    const auto lf =
        g.add_node(psycles::compiler::node_type::light_falloff, "lf");
    const auto em = g.add_node(psycles::compiler::node_type::emission, "em");
    const bool ok =
        g.set_input(bb, "Temperature", c::SocketValue::floating(4000.0f)) &&
        g.set_input(lf, "Strength", c::SocketValue::floating(80.0f)) &&
        g.set_input(lf, "Smooth", c::SocketValue::floating(0.0f)) &&
        g.connect(c::OutputRef{bb, "Color"}, em, "Color") &&
        g.connect(c::OutputRef{lf, "Quadratic"}, em, "Strength");
    if (!ok) {
      std::cerr << "FAIL building non-constant point light shader graph\n";
      return false;
    }
    g.set_root(c::ShaderDomain::surface, c::OutputRef{em, "Closure"});
    scene.materials.emplace(
        c::MaterialId{2u},
        c::MaterialDesc{.name = "PointLightShader",
                        .shader = std::move(g),
                        .cycles_shader_index = 2u});
  }

  {
    c::TriangleMeshDesc m;
    m.positions = {
        {-2.0f, -2.0f, 0.0f}, {2.0f, -2.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {-2.0f, 2.0f, 0.0f}};
    m.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
    m.material_slots = {c::MaterialId{1u}};
    m.triangle_material_slots = {0u, 0u};
    scene.geometries.emplace(c::GeometryId{1u}, std::move(m));
  }
  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = c::GeometryId{1u}, .cycles_object_index = 0u});

  psycles::Mat4f light_tfm;
  light_tfm.elements[14] = 2.0f;
  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "NonConstantPointLight",
                   .type = c::LightType::point,
                   .transform = light_tfm,
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 1.0f,
                   .size = 0.05f,
                   .normalize = true,
                   .is_sphere = true,
                   .shader = c::MaterialId{2u},
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .cycles_shader_index = 2u,
                   .cycles_object_index = 1u});

  psycles::Mat4f cam_tfm;
  cam_tfm.elements[14] = 1.5f;
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 2.0f});
  scene.active_camera = c::CameraId{1u};

  // [0]=rgb mega, [1]=rgb wf, [2]=hero4 mega, [3]=hero4 wf, [4]=xyz3 mega, [5]=xyz3 wf
  std::array<std::array<float, 3u>, 6u> means{};
  std::size_t idx_out = 0u;
  for (const auto mode :
       {c::SpectralMode::none, c::SpectralMode::hero4, c::SpectralMode::xyz3}) {
    const bool spectral = mode != c::SpectralMode::none;
    for (const auto scheduler : {b::LuisaPathScheduler::megakernel,
                                 b::LuisaPathScheduler::wavefront_staged}) {
      b::LuisaPathTracerBackend renderer{
          Device{device.impl_shared()},
          {.next_event_estimation = true,
           .scheduler = scheduler,
           .wavefront_frame_capacity = 2048u,
           .max_samples_per_dispatch = kSpp}};
      auto compiled = renderer.compile_scene(scene);
      if (!compiled.ok()) {
        for (const auto &d : compiled.diagnostics) {
          std::cerr << d.message << '\n';
        }
        return false;
      }
      c::RenderSettings settings{.full_extent = {kSize, kSize},
                                 .pixel_filter = c::PixelFilter::box,
                                 .filter_width = 1.0f};
      settings.integrator.spectral_rendering = spectral;
      settings.integrator.spectral_mode = mode;
      settings.integrator.max_bounces = 2;
      settings.integrator.diffuse_bounces = 0; // Forces 100% NEE path
      settings.integrator.sample_clamp_direct = 0.0f;
      settings.integrator.sample_clamp_indirect = 0.0f;
      settings.integrator.direct_light_sampling =
          c::DirectLightSampling::multiple_importance_sampling;
      settings.passes = {{.kind = c::PassKind::combined, .channels = 4}};
      auto session = renderer.create_session(*compiled.scene, settings);
      if (!session) {
        std::cerr << "Failed creating non-constant point light session\n";
        return false;
      }
      psycles::io::MemoryOutputSink output;
      if (!session->render_samples(
              {.first = 0, .count = kSpp, .offset = 0, .total = kSpp}, output)) {
        return false;
      }
      const auto image = output.find(c::PassKind::combined);
      if (!image) {
        return false;
      }
      double sr = 0.0, sg = 0.0, sb = 0.0;
      const auto count = kSize * kSize;
      for (std::uint32_t i = 0u; i < count; ++i) {
        const float r = image->pixels[i * image->channels + 0u];
        const float g = image->pixels[i * image->channels + 1u];
        const float bl = image->pixels[i * image->channels + 2u];
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bl)) {
          std::cerr << "FAIL non-finite point light NEE pixel on " << backend << '\n';
          return false;
        }
        sr += r;
        sg += g;
        sb += bl;
      }
      means[idx_out] = {static_cast<float>(sr / count),
                        static_cast<float>(sg / count),
                        static_cast<float>(sb / count)};
      const auto &m = means[idx_out];
      const char *mode_name = mode == c::SpectralMode::none ? "none"
                            : (mode == c::SpectralMode::hero4 ? "hero4" : "xyz3");
      if (!(m[0] > m[1] && m[1] > m[2] && m[2] > 0.01f)) {
        std::cerr << "FAIL non-constant point light NEE ordering (mode="
                  << mode_name << ") on " << backend << ": rgb=(" << m[0] << ", "
                  << m[1] << ", " << m[2] << ")\n";
        return false;
      }
      std::cout << "  [PASS] Non-constant point NEE (mode=" << mode_name
                << ", "
                << (scheduler == b::LuisaPathScheduler::megakernel
                        ? "megakernel"
                        : "wavefront_staged")
                << "): rgb=(" << m[0] << ", " << m[1] << ", " << m[2] << ")"
                << std::endl;
      ++idx_out;
    }
  }
  const auto luma = [](const std::array<float, 3u> &rgb) noexcept {
    return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
  };
  for (std::size_t base : {std::size_t{0u}, std::size_t{2u}, std::size_t{4u}}) {
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const float diff = std::abs(means[base][ch] - means[base + 1u][ch]);
      const float scale = std::max(means[base][ch], means[base + 1u][ch]);
      if (diff > 0.06f * scale) {
        std::cerr << "FAIL mega vs wf mismatch in point light NEE ch=" << ch
                  << " mega=" << means[base][ch] << " wf=" << means[base + 1u][ch]
                  << " on " << backend << '\n';
        return false;
      }
    }
  }
  const float lum_rgb = luma(means[0]);
  for (const std::size_t spec_idx : {std::size_t{2u}, std::size_t{4u}}) {
    const float lum_spec = luma(means[spec_idx]);
    if (std::abs(lum_spec - lum_rgb) > 0.15f * std::max(lum_rgb, 1.0e-5f)) {
      std::cerr << "FAIL RGB vs Spectral point light NEE lum mismatch rgb="
                << lum_rgb << " spec=" << lum_spec << " idx=" << spec_idx
                << " on " << backend << '\n';
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running test_luisa_spectral_furnace on " << backend << "...\n";
    if (!test_translator_parity(device, stream, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_white_furnace(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_spectral_nee_transparent_shadow(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_spectral_nee_nonconstant_point_light(device, backend)) {
      return EXIT_FAILURE;
    }
    std::cout << "test_luisa_spectral_furnace PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Exception in test_luisa_spectral_furnace: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
