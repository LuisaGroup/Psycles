#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <luisa/luisa-compute.h>

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

namespace {

namespace c = psycles::contract;
namespace b = psycles::luisa_backend;
using luisa::compute::Context;
using luisa::compute::Device;

inline constexpr float kPi = 3.14159265358979323846f;

[[nodiscard]] c::MaterialDesc make_black_world(std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto bg = g.add_node(psycles::compiler::node_type::background, "bg");
  static_cast<void>(
      g.set_input(bg, "Color", c::SocketValue::color({0.0f, 0.0f, 0.0f})));
  static_cast<void>(g.set_input(bg, "Strength", c::SocketValue::floating(0.0f)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{bg, "Closure"});
  return c::MaterialDesc{
      .name = "World", .shader = std::move(g), .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_diffuse_material(
    std::string name, psycles::Vec3f color, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto diff = g.add_node(psycles::compiler::node_type::diffuse_bsdf, "diff");
  static_cast<void>(g.set_input(diff, "Color", c::SocketValue::color(color)));
  static_cast<void>(g.set_input(diff, "Roughness", c::SocketValue::floating(0.0f)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{diff, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_glass_material(
    std::string name, psycles::Vec3f color, float ior, float roughness,
    float abbe, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto glass = g.add_node(psycles::compiler::node_type::glass_bsdf, "glass");
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

[[nodiscard]] c::TriangleMeshDesc make_receiver_plane(
    float half_extent, float z, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.positions = {{-half_extent, -half_extent, z},
                 {half_extent, -half_extent, z},
                 {half_extent, half_extent, z},
                 {-half_extent, half_extent, z}};
  m.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
  m.material_slots = {mat};
  m.triangle_material_slots = {0u, 0u};
  return m;
}

[[nodiscard]] c::TriangleMeshDesc make_box_slab(
    float half_extent, float z0, float z1, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  const float x = half_extent;
  m.positions = {
      {-x, -x, z0}, {x, -x, z0}, {x, x, z0}, {-x, x, z0},
      {-x, -x, z1}, {x, -x, z1}, {x, x, z1}, {-x, x, z1}};
  // Outward-facing CCW triangles for a closed box in [z0, z1].
  m.triangles = {
      {0u, 2u, 1u}, {0u, 3u, 2u}, // bottom (-Z)
      {4u, 5u, 6u}, {4u, 6u, 7u}, // top (+Z)
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
    const float theta = kPi * (1.0f - v); // south pole (-Z) to north pole (+Z)
    const float sin_t = std::sin(theta);
    const float cos_t = std::cos(theta);
    for (std::uint32_t s = 0u; s <= segs; ++s) {
      const float u = static_cast<float>(s) / static_cast<float>(segs);
      const float phi = 2.0f * kPi * u;
      const psycles::Vec3f n{sin_t * std::cos(phi), sin_t * std::sin(phi), cos_t};
      m.normals.values.push_back(n);
      m.positions.push_back({center.x + radius * n.x,
                             center.y + radius * n.y,
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

[[nodiscard]] c::TriangleMeshDesc make_plano_convex_lens(
    float half_extent, float z_bot, float z_top_center, float curvature,
    std::uint32_t grid, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.normals.domain = c::MeshAttributeDomain::point;
  const auto top_z = [&](float x, float y) noexcept {
    return z_top_center - curvature * (x * x + y * y);
  };
  const std::uint32_t stride = grid + 1u;
  for (std::uint32_t j = 0u; j <= grid; ++j) {
    const float y = -half_extent +
                    (2.0f * half_extent * static_cast<float>(j)) /
                        static_cast<float>(grid);
    for (std::uint32_t i = 0u; i <= grid; ++i) {
      const float x = -half_extent +
                      (2.0f * half_extent * static_cast<float>(i)) /
                          static_cast<float>(grid);
      const float nx_r = 2.0f * curvature * x;
      const float ny_r = 2.0f * curvature * y;
      const float nz_r = 1.0f;
      const float inv =
          1.0f / std::sqrt(nx_r * nx_r + ny_r * ny_r + nz_r * nz_r);
      m.positions.push_back({x, y, top_z(x, y)});
      m.normals.values.push_back({nx_r * inv, ny_r * inv, nz_r * inv});
    }
  }
  for (std::uint32_t j = 0u; j < grid; ++j) {
    for (std::uint32_t i = 0u; i < grid; ++i) {
      const std::uint32_t i00 = j * stride + i;
      const std::uint32_t i10 = i00 + 1u;
      const std::uint32_t i01 = (j + 1u) * stride + i;
      const std::uint32_t i11 = i01 + 1u;
      m.triangles.push_back({i00, i10, i11});
      m.triangles.push_back({i00, i11, i01});
      m.triangle_smooth.push_back(1u);
      m.triangle_smooth.push_back(1u);
    }
  }

  // Bottom flat face (separate vertices so smooth normals never bleed).
  const auto b_base = static_cast<std::uint32_t>(m.positions.size());
  for (const auto &p :
       std::array<psycles::Vec3f, 4u>{{{-half_extent, -half_extent, z_bot},
                                       {half_extent, -half_extent, z_bot},
                                       {half_extent, half_extent, z_bot},
                                       {-half_extent, half_extent, z_bot}}}) {
    m.positions.push_back(p);
    m.normals.values.push_back({0.0f, 0.0f, -1.0f});
  }
  m.triangles.push_back({b_base + 0u, b_base + 2u, b_base + 1u});
  m.triangles.push_back({b_base + 0u, b_base + 3u, b_base + 2u});
  m.triangle_smooth.push_back(0u);
  m.triangle_smooth.push_back(0u);

  // Side walls subdivided along the perimeter to match the top grid seamlessly.
  const auto emit_wall_quad = [&](float xa, float ya, float xb, float yb,
                                  float wnx, float wny) {
    const auto base = static_cast<std::uint32_t>(m.positions.size());
    const psycles::Vec3f wn{wnx, wny, 0.0f};
    m.positions.push_back({xa, ya, z_bot});
    m.positions.push_back({xb, yb, z_bot});
    m.positions.push_back({xb, yb, top_z(xb, yb)});
    m.positions.push_back({xa, ya, top_z(xa, ya)});
    for (int k = 0; k < 4; ++k) {
      m.normals.values.push_back(wn);
    }
    m.triangles.push_back({base + 0u, base + 1u, base + 2u});
    m.triangles.push_back({base + 0u, base + 2u, base + 3u});
    m.triangle_smooth.push_back(0u);
    m.triangle_smooth.push_back(0u);
  };
  for (std::uint32_t k = 0u; k < grid; ++k) {
    const float t0 = -half_extent +
                     (2.0f * half_extent * static_cast<float>(k)) /
                         static_cast<float>(grid);
    const float t1 = -half_extent +
                     (2.0f * half_extent * static_cast<float>(k + 1u)) /
                         static_cast<float>(grid);
    emit_wall_quad(t0, -half_extent, t1, -half_extent, 0.0f, -1.0f);
    emit_wall_quad(half_extent, t0, half_extent, t1, 1.0f, 0.0f);
    emit_wall_quad(-t0, half_extent, -t1, half_extent, 0.0f, 1.0f);
    emit_wall_quad(-half_extent, -t0, -half_extent, -t1, -1.0f, 0.0f);
  }

  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  return m;
}

[[nodiscard]] bool render_rgba(
    Device &device, const c::SceneSnapshot &scene,
    b::LuisaPathScheduler scheduler, bool spectral,
    c::CausticsMethod caustics_method, std::uint32_t size, std::uint32_t spp,
    std::vector<float> &out_rgba, bool allow_pt_caustics = false,
    c::SpectralMode spectral_mode = c::SpectralMode::hero4) {
  b::LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .scheduler = scheduler,
       .wavefront_frame_capacity = std::max<std::uint32_t>(size * size * 4u, 4096u),
       .max_samples_per_dispatch = spp}};
  auto compiled = renderer.compile_scene(scene);
  if (!compiled.ok()) {
    for (const auto &d : compiled.diagnostics) {
      std::cerr << d.message << '\n';
    }
    return false;
  }
  c::RenderSettings settings{
      .full_extent = {size, size},
      .seed = 1337u,
      .pixel_filter = c::PixelFilter::box,
      .filter_width = 1.0f};
  settings.integrator.max_bounces = 8u;
  settings.integrator.diffuse_bounces = 4u;
  settings.integrator.glossy_bounces = 8u;
  settings.integrator.transmission_bounces = 8u;
  settings.integrator.reflective_caustics = allow_pt_caustics;
  settings.integrator.refractive_caustics = allow_pt_caustics;
  settings.integrator.sample_clamp_direct = 0.0f;
  settings.integrator.sample_clamp_indirect = 0.0f;
  settings.integrator.direct_light_sampling =
      c::DirectLightSampling::multiple_importance_sampling;
  settings.integrator.spectral_rendering = spectral;
  settings.integrator.spectral_mode =
      spectral ? spectral_mode : c::SpectralMode::none;
  settings.integrator.caustics_method = caustics_method;
  settings.passes = {{.kind = c::PassKind::combined, .channels = 4}};

  auto session = renderer.create_session(*compiled.scene, settings);
  if (!session) {
    std::cerr << "Failed to create render session\n";
    return false;
  }
  psycles::io::MemoryOutputSink output;
  if (!session->render_samples(
          {.first = 0, .count = spp, .offset = 0, .total = spp}, output)) {
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

[[nodiscard]] std::array<float, 3u> mean_center_patch(
    const std::vector<float> &rgba, std::uint32_t size,
    float max_norm_radius, bool &all_finite) {
  all_finite = true;
  double sr = 0.0, sg = 0.0, sb = 0.0;
  std::size_t n = 0u;
  for (std::uint32_t y = 0u; y < size; ++y) {
    const float ny = ((static_cast<float>(y) + 0.5f) / size) * 2.0f - 1.0f;
    for (std::uint32_t x = 0u; x < size; ++x) {
      const float nx = ((static_cast<float>(x) + 0.5f) / size) * 2.0f - 1.0f;
      const std::size_t idx = (static_cast<std::size_t>(y) * size + x) * 4u;
      const float r = rgba[idx + 0u];
      const float g = rgba[idx + 1u];
      const float bl = rgba[idx + 2u];
      if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bl)) {
        all_finite = false;
      }
      if (nx * nx + ny * ny <= max_norm_radius * max_norm_radius) {
        sr += r;
        sg += g;
        sb += bl;
        ++n;
      }
    }
  }
  return {static_cast<float>(sr / n), static_cast<float>(sg / n),
          static_cast<float>(sb / n)};
}

// Test 1: Analytical flat glass slab under delta distant sun & delta point light.
[[nodiscard]] bool test_analytical_flat_slab(Device &device,
                                             std::string_view backend) {
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  scene.materials.emplace(c::MaterialId{10u}, make_black_world(0u));
  scene.world_shader = c::MaterialId{10u};
  scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      c::MaterialId{2u},
      make_glass_material("Slab", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));

  scene.geometries.emplace(c::GeometryId{1u},
                           make_receiver_plane(4.0f, 0.0f, c::MaterialId{1u}));
  scene.geometries.emplace(c::GeometryId{2u},
                           make_box_slab(6.0f, 1.0f, 1.5f, c::MaterialId{2u}));
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
  cam_tfm.elements[14] = 0.8f; // Below slab (z=1.0), viewing floor (z=0)
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 0.25f});
  scene.active_camera = c::CameraId{1u};

  constexpr std::uint32_t kSize = 16u;
  constexpr std::uint32_t kSpp = 16u;
  const float f0 = ((1.5f - 1.0f) / (1.5f + 1.0f)) *
                   ((1.5f - 1.0f) / (1.5f + 1.0f)); // 0.04
  const float t_slab = (1.0f - f0) * (1.0f - f0);   // 0.9216

  // (1a) Delta distant light straight down (angle=0, power=10).
  {
    scene.lights.clear();
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "SunDelta",
                     .type = c::LightType::distant,
                     .transform = psycles::Mat4f{},
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 10.0f,
                     .angle = 0.0f,
                     .normalize = true,
                     .use_mis = false,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .is_caustics_light = true,
                     .cycles_object_index = 2u});

    const float expected = (0.8f / kPi) * 10.0f * t_slab; // 2.34684
    for (const bool spectral : {false, true}) {
      for (const auto sched : {b::LuisaPathScheduler::megakernel,
                               b::LuisaPathScheduler::wavefront_staged}) {
        std::vector<float> rgba;
        if (!render_rgba(device, scene, sched, spectral, c::CausticsMethod::mnee,
                         kSize, kSpp, rgba)) {
          return false;
        }
        bool finite = false;
        const auto m = mean_center_patch(rgba, kSize, 0.5f, finite);
        const float rel_err = std::abs(m[0] - expected) / expected;
        if (!finite || rel_err > 0.03f) {
          std::cerr << "FAIL Test 1a distant slab (spectral=" << spectral
                    << ", sched="
                    << (sched == b::LuisaPathScheduler::megakernel ? "mega"
                                                                   : "wf")
                    << ") on " << backend << ": got=(" << m[0] << ", " << m[1]
                    << ", " << m[2] << ") expected=" << expected
                    << " rel_err=" << rel_err << '\n';
          return false;
        }
        std::cout << "  [PASS] Slab distant MNEE (spectral=" << spectral
                  << ", "
                  << (sched == b::LuisaPathScheduler::megakernel ? "mega" : "wf")
                  << "): got=" << m[0] << " expected=" << expected << '\n';
      }
    }

    // Test 4: caustics_method = none guard (same delta sun -> must stay dark).
    std::vector<float> rgba_off;
    if (!render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                     false, c::CausticsMethod::none, kSize, kSpp, rgba_off,
                     true)) {
      return false;
    }
    bool finite_off = false;
    const auto m_off = mean_center_patch(rgba_off, kSize, 0.8f, finite_off);
    if (!finite_off || m_off[0] > 1.0e-5f) {
      std::cerr << "FAIL Test 4 caustics=none guard on " << backend
                << ": got=" << m_off[0] << '\n';
      return false;
    }
    std::cout << "  [PASS] caustics=none guard: mean=" << m_off[0] << '\n';
  }

  // (1b) Delta point light at z=2.5 (radius=0, power=100).
  {
    psycles::Mat4f pt_tfm;
    pt_tfm.elements[14] = 2.5f;
    scene.lights.clear();
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "PointDelta",
                     .type = c::LightType::point,
                     .transform = pt_tfm,
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 100.0f,
                     .size = 0.0f,
                     .normalize = true,
                     .is_sphere = true,
                     .use_mis = false,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .is_caustics_light = true,
                     .cycles_object_index = 2u});

    const float d_app = 1.0f + 0.5f / 1.5f + 1.0f; // 7/3
    const float expected =
        (0.8f / kPi) * (100.0f / (4.0f * kPi * d_app * d_app)) * t_slab;
    for (const auto sched : {b::LuisaPathScheduler::megakernel,
                             b::LuisaPathScheduler::wavefront_staged}) {
      std::vector<float> rgba;
      if (!render_rgba(device, scene, sched, false, c::CausticsMethod::mnee,
                       kSize, kSpp, rgba)) {
        return false;
      }
      bool finite = false;
      const auto m = mean_center_patch(rgba, kSize, 0.25f, finite);
      const float rel_err = std::abs(m[0] - expected) / expected;
      if (!finite || rel_err > 0.03f) {
        std::cerr << "FAIL Test 1b point slab ("
                  << (sched == b::LuisaPathScheduler::megakernel ? "mega"
                                                                 : "wf")
                  << ") on " << backend << ": got=" << m[0]
                  << " expected=" << expected << " rel_err=" << rel_err << '\n';
        return false;
      }
      std::cout << "  [PASS] Slab point MNEE ("
                << (sched == b::LuisaPathScheduler::megakernel ? "mega" : "wf")
                << "): got=" << m[0] << " expected=" << expected << '\n';
    }
  }
  return true;
}

// Test 2: Smooth glass sphere caustic focus + area light energy conservation.
[[nodiscard]] bool test_sphere_focus_and_energy(Device &device,
                                                std::string_view backend) {
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  scene.materials.emplace(c::MaterialId{10u}, make_black_world(0u));
  scene.world_shader = c::MaterialId{10u};
  scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      c::MaterialId{2u},
      make_glass_material("SphereGlass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));

  scene.geometries.emplace(c::GeometryId{1u},
                           make_receiver_plane(4.0f, 0.0f, c::MaterialId{1u}));
  scene.geometries.emplace(
      c::GeometryId{2u},
      make_smooth_uv_sphere({0.0f, 0.0f, 1.0f}, 0.5f, 32u, 64u,
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
  cam_tfm.elements[14] = 0.3f;
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 2.0f}); // spans [-1, 1] on floor
  scene.active_camera = c::CameraId{1u};

  // (2a) Sun focus concentration + mega vs wf parity.
  {
    scene.lights.clear();
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "Sun",
                     .type = c::LightType::distant,
                     .transform = psycles::Mat4f{},
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 5.0f,
                     .angle = 0.01f,
                     .normalize = true,
                     .use_mis = false,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .is_caustics_light = true,
                     .cycles_object_index = 2u});
    constexpr std::uint32_t kSize = 32u;
    constexpr std::uint32_t kSpp = 16u;
    std::array<float, 2u> focus_vals{};
    std::size_t idx = 0u;
    for (const auto sched : {b::LuisaPathScheduler::megakernel,
                             b::LuisaPathScheduler::wavefront_staged}) {
      std::vector<float> rgba;
      if (!render_rgba(device, scene, sched, false, c::CausticsMethod::mnee,
                       kSize, kSpp, rgba)) {
        return false;
      }
      bool finite = false;
      const auto center = mean_center_patch(rgba, kSize, 0.15f, finite);
      // Compare with outer unrefracted floor annulus.
      double outer_sum = 0.0;
      std::size_t outer_n = 0u;
      for (std::uint32_t y = 0u; y < kSize; ++y) {
        const float ny = ((static_cast<float>(y) + 0.5f) / kSize) * 2.0f - 1.0f;
        for (std::uint32_t x = 0u; x < kSize; ++x) {
          const float nx = ((static_cast<float>(x) + 0.5f) / kSize) * 2.0f - 1.0f;
          const float r2 = nx * nx + ny * ny;
          if (r2 >= 0.7f * 0.7f && r2 <= 0.95f * 0.95f) {
            outer_sum += rgba[(static_cast<std::size_t>(y) * kSize + x) * 4u];
            ++outer_n;
          }
        }
      }
      const float outer = static_cast<float>(outer_sum / outer_n);
      if (!finite || center[0] <= 2.5f * outer) {
        std::cerr << "FAIL Test 2a sphere caustic focus on " << backend
                  << ": center=" << center[0] << " outer=" << outer << '\n';
        return false;
      }
      focus_vals[idx++] = center[0];
      std::cout << "  [PASS] Sphere focus ("
                << (sched == b::LuisaPathScheduler::megakernel ? "mega" : "wf")
                << "): center=" << center[0] << " outer=" << outer << '\n';
    }
    const float diff = std::abs(focus_vals[0] - focus_vals[1]);
    if (diff > 0.05f * std::max(focus_vals[0], focus_vals[1])) {
      std::cerr << "FAIL Test 2a mega vs wf focus mismatch: " << focus_vals[0]
                << " vs " << focus_vals[1] << '\n';
      return false;
    }
  }

  // (2b) Area light energy conservation: MNEE (with PT caustics allowed on
  // reflections) vs pure brute-force PT (512 spp) on single-solution refractors.
  {
    // (2b-i) Flat glass slab (IOR=1.5, single-solution) under 0.6x0.6 Area light.
    {
      c::SceneSnapshot slab_scene;
      slab_scene.world_sampling = c::WorldSampling::none;
      slab_scene.materials.emplace(c::MaterialId{10u}, make_black_world(0u));
      slab_scene.world_shader = c::MaterialId{10u};
      slab_scene.materials.emplace(
          c::MaterialId{1u},
          make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
      slab_scene.materials.emplace(
          c::MaterialId{2u},
          make_glass_material("SlabGlass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));

      slab_scene.geometries.emplace(
          c::GeometryId{1u}, make_receiver_plane(4.0f, 0.0f, c::MaterialId{1u}));
      slab_scene.geometries.emplace(
          c::GeometryId{2u},
          make_box_slab(6.0f, 1.0f, 1.5f, c::MaterialId{2u}));
      slab_scene.instances.emplace(
          c::InstanceId{1u},
          c::InstanceDesc{.geometry = c::GeometryId{1u},
                          .cycles_object_index = 0u,
                          .is_caustics_receiver = true});
      slab_scene.instances.emplace(
          c::InstanceId{2u},
          c::InstanceDesc{.geometry = c::GeometryId{2u},
                          .cycles_object_index = 1u,
                          .is_caustics_caster = true});

      psycles::Mat4f slab_cam;
      slab_cam.elements[14] = 0.8f;
      slab_scene.cameras.emplace(
          c::CameraId{1u},
          c::CameraDesc{.projection = c::CameraProjection::orthographic,
                        .transform = slab_cam,
                        .orthographic_scale = 1.0f});
      slab_scene.active_camera = c::CameraId{1u};

      psycles::Mat4f area_tfm;
      area_tfm.elements[14] = 2.2f;
      slab_scene.lights.emplace(
          c::LightId{1u},
          c::LightDesc{.name = "Area",
                       .type = c::LightType::area,
                       .transform = area_tfm,
                       .color = {1.0f, 1.0f, 1.0f},
                       .power = 25.0f,
                       .size = 0.6f,
                       .size_y = 0.6f,
                       .spread = kPi,
                       .normalize = true,
                       .use_mis = true,
                       .cast_shadow = true,
                       .visibility_mask = c::all_ray_visibility,
                       .is_caustics_light = true,
                       .cycles_object_index = 2u});

      constexpr std::uint32_t kSize = 32u;
      std::vector<float> rgba_ref, rgba_mnee;
      if (!render_rgba(device, slab_scene, b::LuisaPathScheduler::wavefront_staged,
                       false, c::CausticsMethod::none, kSize, 512u, rgba_ref,
                       true)) {
        return false;
      }
      if (!render_rgba(device, slab_scene, b::LuisaPathScheduler::wavefront_staged,
                       false, c::CausticsMethod::mnee, kSize, 32u, rgba_mnee,
                       true)) {
        return false;
      }
      bool f0 = false, f1 = false;
      const auto ref_m = mean_center_patch(rgba_ref, kSize, 0.5f, f0);
      const auto mnee_m = mean_center_patch(rgba_mnee, kSize, 0.5f, f1);
      const float rel =
          std::abs(mnee_m[0] - ref_m[0]) / std::max(ref_m[0], 1.0e-5f);
      if (!f0 || !f1 || rel > 0.05f) {
        std::cerr << "FAIL Test 2b-i area slab energy match on " << backend
                  << ": ref_pt512=" << ref_m[0] << " mnee=" << mnee_m[0]
                  << " rel=" << rel << '\n';
        return false;
      }
      std::cout << "  [PASS] Area light slab MNEE energy match (wf): ref_pt512="
                << ref_m[0] << " mnee32=" << mnee_m[0] << " rel=" << rel << '\n';
    }

    // (2b-ii) Smooth plano-convex glass lens (IOR=1.5, strong curvature,
    // single-solution) under 0.6x0.6 Area light: tests curved dn_dx/dn_dy.
    {
      c::SceneSnapshot lens_scene;
      lens_scene.world_sampling = c::WorldSampling::none;
      lens_scene.materials.emplace(c::MaterialId{10u}, make_black_world(0u));
      lens_scene.world_shader = c::MaterialId{10u};
      lens_scene.materials.emplace(
          c::MaterialId{1u},
          make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
      lens_scene.materials.emplace(
          c::MaterialId{2u},
          make_glass_material("LensGlass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));

      lens_scene.geometries.emplace(
          c::GeometryId{1u}, make_receiver_plane(4.0f, 0.0f, c::MaterialId{1u}));
      lens_scene.geometries.emplace(
          c::GeometryId{2u},
          make_plano_convex_lens(1.2f, 0.8f, 1.5f, 0.20f, 16u, c::MaterialId{2u}));
      lens_scene.instances.emplace(
          c::InstanceId{1u},
          c::InstanceDesc{.geometry = c::GeometryId{1u},
                          .cycles_object_index = 0u,
                          .is_caustics_receiver = true});
      lens_scene.instances.emplace(
          c::InstanceId{2u},
          c::InstanceDesc{.geometry = c::GeometryId{2u},
                          .cycles_object_index = 1u,
                          .is_caustics_caster = true});

      psycles::Mat4f lens_cam;
      lens_cam.elements[14] = 0.6f;
      lens_scene.cameras.emplace(
          c::CameraId{1u},
          c::CameraDesc{.projection = c::CameraProjection::orthographic,
                        .transform = lens_cam,
                        .orthographic_scale = 0.6f});
      lens_scene.active_camera = c::CameraId{1u};

      psycles::Mat4f area_tfm;
      area_tfm.elements[14] = 2.2f;
      lens_scene.lights.emplace(
          c::LightId{1u},
          c::LightDesc{.name = "Area",
                       .type = c::LightType::area,
                       .transform = area_tfm,
                       .color = {1.0f, 1.0f, 1.0f},
                       .power = 25.0f,
                       .size = 0.6f,
                       .size_y = 0.6f,
                       .spread = kPi,
                       .normalize = true,
                       .use_mis = true,
                       .cast_shadow = true,
                       .visibility_mask = c::all_ray_visibility,
                       .is_caustics_light = true,
                       .cycles_object_index = 2u});

      constexpr std::uint32_t kSize = 32u;
      std::vector<float> rgba_ref, rgba_mnee;
      if (!render_rgba(device, lens_scene, b::LuisaPathScheduler::wavefront_staged,
                       false, c::CausticsMethod::none, kSize, 512u, rgba_ref,
                       true)) {
        return false;
      }
      if (!render_rgba(device, lens_scene, b::LuisaPathScheduler::wavefront_staged,
                       false, c::CausticsMethod::mnee, kSize, 32u, rgba_mnee,
                       true)) {
        return false;
      }
      bool f0 = false, f1 = false;
      const auto ref_m = mean_center_patch(rgba_ref, kSize, 0.5f, f0);
      const auto mnee_m = mean_center_patch(rgba_mnee, kSize, 0.5f, f1);
      const float rel =
          std::abs(mnee_m[0] - ref_m[0]) / std::max(ref_m[0], 1.0e-5f);
      if (!f0 || !f1 || rel > 0.05f) {
        std::cerr << "FAIL Test 2b-ii area plano-convex lens energy match on "
                  << backend << ": ref_pt512=" << ref_m[0]
                  << " mnee=" << mnee_m[0] << " rel=" << rel << '\n';
        return false;
      }
      std::cout << "  [PASS] Area light plano-convex lens MNEE energy match (wf): ref_pt512="
                << ref_m[0] << " mnee32=" << mnee_m[0] << " rel=" << rel << '\n';
    }
  }
  return true;
}

// Test 3: Spectral dispersive glass sphere rainbow caustic (hero dispersion).
[[nodiscard]] bool test_spectral_dispersive_caustic(Device &device,
                                                    std::string_view backend) {
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  scene.materials.emplace(c::MaterialId{10u}, make_black_world(0u));
  scene.world_shader = c::MaterialId{10u};
  scene.materials.emplace(
      c::MaterialId{1u},
      make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      c::MaterialId{2u},
      make_glass_material("DispGlass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 15.0f, 2u));

  scene.geometries.emplace(c::GeometryId{1u},
                           make_receiver_plane(4.0f, 0.0f, c::MaterialId{1u}));
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
  cam_tfm.elements[14] = 0.3f;
  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = cam_tfm,
                    .orthographic_scale = 1.2f});
  scene.active_camera = c::CameraId{1u};

  // Straight-down sun focuses blue higher above floor than red, expanding blue ring radially.
  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "Sun",
                   .type = c::LightType::distant,
                   .transform = psycles::Mat4f{},
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 6.0f,
                   .angle = 0.005f,
                   .normalize = true,
                   .use_mis = false,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .is_caustics_light = true,
                   .cycles_object_index = 2u});

  constexpr std::uint32_t kSize = 32u;
  constexpr std::uint32_t kSpp = 64u;
  for (const auto mode : {c::SpectralMode::hero4, c::SpectralMode::xyz3}) {
    std::array<double, 2u> expansions{};
    std::size_t idx = 0u;
    const char *mode_name = mode == c::SpectralMode::hero4 ? "hero4" : "xyz3";
    for (const auto sched : {b::LuisaPathScheduler::megakernel,
                             b::LuisaPathScheduler::wavefront_staged}) {
      std::vector<float> rgba;
      if (!render_rgba(device, scene, sched, true, c::CausticsMethod::mnee,
                       kSize, kSpp, rgba, false, mode)) {
        return false;
      }
      double sum_r = 0.0, sum_b = 0.0;
      double weighted_r_red = 0.0, weighted_r_blue = 0.0;
      for (std::uint32_t y = 0u; y < kSize; ++y) {
        for (std::uint32_t x = 0u; x < kSize; ++x) {
          const float nx =
              ((static_cast<float>(x) + 0.5f) / kSize) * 2.0f - 1.0f;
          const float ny =
              ((static_cast<float>(y) + 0.5f) / kSize) * 2.0f - 1.0f;
          const float x_w = nx * 0.6f;
          const float y_w = ny * 0.6f;
          const float r_rad = std::sqrt(x_w * x_w + y_w * y_w);
          const std::size_t p =
              (static_cast<std::size_t>(y) * kSize + x) * 4u;
          const float r = rgba[p + 0u];
          const float g = rgba[p + 1u];
          const float bl = rgba[p + 2u];
          if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bl)) {
            std::cerr << "FAIL Test 3 (" << mode_name << ") non-finite pixel on "
                      << backend << '\n';
            return false;
          }
          if (r_rad <= 0.45f) {
            const double r_pos = std::max(r, 0.0f);
            const double b_pos = std::max(bl, 0.0f);
            sum_r += r_pos;
            sum_b += b_pos;
            weighted_r_red += r_rad * r_pos;
            weighted_r_blue += r_rad * b_pos;
          }
        }
      }
      if (sum_r <= 0.0 || sum_b <= 0.0) {
        std::cerr << "FAIL Test 3 (" << mode_name
                  << ") empty dispersive caustic on " << backend << '\n';
        return false;
      }
      const double mean_r_red = weighted_r_red / sum_r;
      const double mean_r_blue = weighted_r_blue / sum_b;
      const double rel_expansion = (mean_r_blue - mean_r_red) / mean_r_red;
      if (rel_expansion <= 0.04) {
        std::cerr << "FAIL Test 3 (" << mode_name
                  << ") insufficient radial dispersion expansion="
                  << rel_expansion << " mean_r_red=" << mean_r_red
                  << " mean_r_blue=" << mean_r_blue << " on " << backend
                  << '\n';
        return false;
      }
      expansions[idx++] = rel_expansion;
      std::cout << "  [PASS] Dispersive spectral caustic (" << mode_name << ", "
                << (sched == b::LuisaPathScheduler::megakernel ? "mega" : "wf")
                << "): r_red=" << mean_r_red << " r_blue=" << mean_r_blue
                << " expansion=" << rel_expansion << '\n';
    }
    const double rel_sched =
        std::abs(expansions[0] - expansions[1]) /
        std::max(std::abs(expansions[0]), std::abs(expansions[1]));
    if (rel_sched > 0.15) {
      std::cerr << "FAIL Test 3 (" << mode_name
                << ") mega vs wf dispersion expansion mismatch on " << backend
                << ": " << expansions[0] << " vs " << expansions[1] << '\n';
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
    std::cout << "Running test_luisa_caustics_mnee on " << backend << "...\n";
    if (!test_analytical_flat_slab(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_sphere_focus_and_energy(device, backend)) {
      return EXIT_FAILURE;
    }
    if (!test_spectral_dispersive_caustic(device, backend)) {
      return EXIT_FAILURE;
    }
    std::cout << "test_luisa_caustics_mnee PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Exception in test_luisa_caustics_mnee: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
