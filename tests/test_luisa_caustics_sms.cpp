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
#include <tuple>
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
  static_cast<void>(
      g.set_input(bg, "Strength", c::SocketValue::floating(0.0f)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{bg, "Closure"});
  return c::MaterialDesc{
      .name = "World", .shader = std::move(g), .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_diffuse_material(
    std::string name, psycles::Vec3f color, std::uint32_t shader_idx) {
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

[[nodiscard]] c::MaterialDesc make_glass_material(
    std::string name, psycles::Vec3f color, float ior, float roughness,
    float abbe, std::uint32_t shader_idx) {
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

[[nodiscard]] c::MaterialDesc make_glossy_material(
    std::string name, psycles::Vec3f color, float roughness,
    std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto glossy =
      g.add_node(psycles::compiler::node_type::glossy_bsdf, "glossy");
  static_cast<void>(
      g.set_input(glossy, "Color", c::SocketValue::color(color)));
  static_cast<void>(
      g.set_input(glossy, "Roughness", c::SocketValue::floating(roughness)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{glossy, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
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

// Open smooth concave cylindrical mirror arc centered at (0, cy), radius R,
// facing -Y (inward), standing from z0 to z1.
[[nodiscard]] c::TriangleMeshDesc make_concave_cylinder_arc(
    float cy, float radius, float half_angle_rad, float z0, float z1,
    std::uint32_t segs, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.normals.domain = c::MeshAttributeDomain::point;
  for (std::uint32_t i = 0u; i <= segs; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(segs);
    const float theta = -half_angle_rad + 2.0f * half_angle_rad * t;
    const float sx = std::sin(theta);
    const float cy_dir = std::cos(theta);
    const float px = radius * sx;
    const float py = cy + radius * cy_dir;
    const psycles::Vec3f n{-sx, -cy_dir, 0.0f};
    m.positions.push_back({px, py, z0});
    m.normals.values.push_back(n);
    m.positions.push_back({px, py, z1});
    m.normals.values.push_back(n);
  }
  for (std::uint32_t i = 0u; i < segs; ++i) {
    const std::uint32_t b0 = 2u * i;
    const std::uint32_t t0 = b0 + 1u;
    const std::uint32_t b1 = b0 + 2u;
    const std::uint32_t t1 = b0 + 3u;
    m.triangles.push_back({b0, b1, t1});
    m.triangles.push_back({b0, t1, t0});
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

// Smooth curved Fresnel biprism-lens half-block (sign = -1 left, +1 right),
// watertight with outward-pointing normals and zero fold/Einstein-ring
// singularities onto a central area light.
[[nodiscard]] c::TriangleMeshDesc make_biprism_lens_half(
    float sign, float x_inner, float x_outer, float y_half, float z_bot,
    float z0, float slope, float curv, std::uint32_t grid, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.normals.domain = c::MeshAttributeDomain::point;
  const float x_lo = (sign > 0.0f) ? x_inner : -x_outer;
  const float x_hi = (sign > 0.0f) ? x_outer : -x_inner;
  const float y_lo = -y_half;
  const float y_hi = y_half;
  const auto top_z = [&](float x, float y) noexcept {
    const float sx = sign * x;
    const float dx = sx - 0.45f;
    return z0 - slope * sx - curv * (dx * dx + y * y);
  };
  const auto tx = [&](std::uint32_t k) noexcept {
    return x_lo + ((x_hi - x_lo) * static_cast<float>(k)) /
                      static_cast<float>(grid);
  };
  const auto ty = [&](std::uint32_t k) noexcept {
    return y_lo + ((y_hi - y_lo) * static_cast<float>(k)) /
                      static_cast<float>(grid);
  };

  const std::uint32_t stride = grid + 1u;
  for (std::uint32_t j = 0u; j <= grid; ++j) {
    const float y = ty(j);
    for (std::uint32_t i = 0u; i <= grid; ++i) {
      const float x = tx(i);
      const float sx = sign * x;
      const float nx_r = sign * (slope + 2.0f * curv * (sx - 0.45f));
      const float ny_r = 2.0f * curv * y;
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

  const auto b_base = static_cast<std::uint32_t>(m.positions.size());
  for (const auto &p :
       std::array<psycles::Vec3f, 4u>{{{x_lo, y_lo, z_bot},
                                       {x_hi, y_lo, z_bot},
                                       {x_hi, y_hi, z_bot},
                                       {x_lo, y_hi, z_bot}}}) {
    m.positions.push_back(p);
    m.normals.values.push_back({0.0f, 0.0f, -1.0f});
  }
  m.triangles.push_back({b_base + 0u, b_base + 2u, b_base + 1u});
  m.triangles.push_back({b_base + 0u, b_base + 3u, b_base + 2u});
  m.triangle_smooth.push_back(0u);
  m.triangle_smooth.push_back(0u);

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
    emit_wall_quad(tx(k), y_lo, tx(k + 1u), y_lo, 0.0f, -1.0f);
    emit_wall_quad(x_hi, ty(k), x_hi, ty(k + 1u), 1.0f, 0.0f);
    emit_wall_quad(tx(grid - k), y_hi, tx(grid - k - 1u), y_hi, 0.0f, 1.0f);
    emit_wall_quad(x_lo, ty(grid - k), x_lo, ty(grid - k - 1u), -1.0f, 0.0f);
  }

  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  return m;
}

[[nodiscard]] bool render_rgba(
    Device &device, const c::SceneSnapshot &scene,
    b::LuisaPathScheduler scheduler, bool spectral,
    c::CausticsMethod caustics_method, bool sms_unbiased,
    std::uint32_t sms_max_trials, std::uint32_t size, std::uint32_t spp,
    std::vector<float> &out_rgba, bool allow_pt_caustics = false,
    float filter_glossy = 0.0f) {
  b::LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .scheduler = scheduler,
       .wavefront_frame_capacity =
           std::max<std::uint32_t>(size * size * 4u, 4096u),
       .max_samples_per_dispatch = spp}};
  auto compiled = renderer.compile_scene(scene);
  if (!compiled.ok()) {
    for (const auto &d : compiled.diagnostics) {
      std::cerr << d.message << '\n';
    }
    return false;
  }
  c::RenderSettings settings{.full_extent = {size, size},
                             .seed = 2026u,
                             .pixel_filter = c::PixelFilter::box,
                             .filter_width = 1.0f};
  settings.integrator.max_bounces = 8u;
  settings.integrator.diffuse_bounces = 4u;
  settings.integrator.glossy_bounces = 8u;
  settings.integrator.transmission_bounces = 8u;
  settings.integrator.reflective_caustics = allow_pt_caustics;
  settings.integrator.refractive_caustics = allow_pt_caustics;
  settings.integrator.filter_glossy = filter_glossy;
  settings.integrator.sample_clamp_direct = 0.0f;
  settings.integrator.sample_clamp_indirect = 0.0f;
  settings.integrator.direct_light_sampling =
      c::DirectLightSampling::multiple_importance_sampling;
  settings.integrator.spectral_rendering = spectral;
  settings.integrator.caustics_method = caustics_method;
  settings.integrator.sms_unbiased = sms_unbiased;
  settings.integrator.sms_max_trials = sms_max_trials;
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
      for (int c_idx = 0; c_idx < 3; ++c_idx) {
        if (!std::isfinite(rgba[idx + c_idx])) {
          all_finite = false;
        }
      }
      if (nx * nx + ny * ny <= max_norm_radius * max_norm_radius) {
        sr += rgba[idx + 0u];
        sg += rgba[idx + 1u];
        sb += rgba[idx + 2u];
        ++n;
      }
    }
  }
  if (n == 0u) {
    return {0.0f, 0.0f, 0.0f};
  }
  return {static_cast<float>(sr / n), static_cast<float>(sg / n),
          static_cast<float>(sb / n)};
}

// Column-major Mat4f where -axis_z points from pos toward target.
[[nodiscard]] psycles::Mat4f make_look_transform(
    psycles::Vec3f pos, psycles::Vec3f target, psycles::Vec3f world_up) {
  const auto normalize_v = [](psycles::Vec3f v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return psycles::Vec3f{v.x / len, v.y / len, v.z / len};
  };
  const auto cross_v = [](psycles::Vec3f a, psycles::Vec3f b) {
    return psycles::Vec3f{a.y * b.z - a.z * b.y,
                          a.z * b.x - a.x * b.z,
                          a.x * b.y - a.y * b.x};
  };
  const auto forward = normalize_v(
      {target.x - pos.x, target.y - pos.y, target.z - pos.z});
  const auto right = normalize_v(cross_v(forward, world_up));
  const auto up = cross_v(right, forward);
  psycles::Mat4f m;
  m.elements = {
      right.x,    right.y,    right.z,    0.0f,
      up.x,       up.y,       up.z,       0.0f,
      -forward.x, -forward.y, -forward.z, 0.0f,
      pos.x,      pos.y,      pos.z,      1.0f};
  return m;
}

[[nodiscard]] psycles::Mat4f make_translation(float x, float y, float z) {
  psycles::Mat4f m;
  m.elements[12] = x;
  m.elements[13] = y;
  m.elements[14] = z;
  return m;
}

[[nodiscard]] bool test_dual_lens_multisolution_area_sms(Device &device) {
  std::cout << "[SMS Test 1] Multi-solution dual biprism lens under Area light "
               "(Unbiased & Biased SMS vs MNEE vs PT 512 spp)...\n";
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  const c::MaterialId world_mat{10u}, diff_mat{1u}, glass_mat{2u};
  scene.materials.emplace(world_mat, make_black_world(0u));
  scene.materials.emplace(
      diff_mat, make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      glass_mat,
      make_glass_material("Glass", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));
  scene.world_shader = world_mat;

  const c::GeometryId floor_geo{1u}, left_geo{2u}, right_geo{3u};
  scene.geometries.emplace(floor_geo, make_receiver_plane(4.0f, 0.0f, diff_mat));
  scene.geometries.emplace(
      left_geo, make_biprism_lens_half(-1.0f, 0.002f, 0.90f, 0.85f, 0.65f,
                                       1.45f, 0.68f, 0.08f, 24u, glass_mat));
  scene.geometries.emplace(
      right_geo, make_biprism_lens_half(1.0f, 0.002f, 0.90f, 0.85f, 0.65f,
                                        1.45f, 0.68f, 0.08f, 24u, glass_mat));

  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = floor_geo,
                      .cycles_object_index = 0u,
                      .is_caustics_receiver = true});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = left_geo,
                      .cycles_object_index = 1u,
                      .is_caustics_caster = true});
  scene.instances.emplace(
      c::InstanceId{3u},
      c::InstanceDesc{.geometry = right_geo,
                      .cycles_object_index = 2u,
                      .is_caustics_caster = true});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = make_translation(0.0f, 0.0f, 0.35f),
                    .orthographic_scale = 0.50f});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "AreaLight",
                   .type = c::LightType::area,
                   .transform = make_translation(0.0f, 0.0f, 2.4f),
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 25.0f,
                   .size = 0.70f,
                   .size_y = 0.70f,
                   .spread = kPi,
                   .normalize = true,
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .is_caustics_light = true,
                   .cycles_object_index = 3u});

  constexpr std::uint32_t kSize = 24u;
  std::vector<float> base_none, ref_pt, out_mnee, out_bia, out_mega, out_wf;
  if (!render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::none, true, 8u, kSize, 64u, base_none,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::none, true, 8u, kSize, 512u, ref_pt,
                   true) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::mnee, true, 8u, kSize, 32u, out_mnee,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::sms, false, 16u, kSize, 64u, out_bia,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::megakernel, false,
                   c::CausticsMethod::sms, true, 8u, kSize, 64u, out_mega,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                   false, c::CausticsMethod::sms, true, 8u, kSize, 64u, out_wf,
                   false)) {
    return false;
  }

  bool fin0 = true, fin_pt = true, fin_mnee = true, fin_bia = true,
       fin_mega = true, fin_wf = true;
  const auto m_base = mean_center_patch(base_none, kSize, 0.65f, fin0);
  const auto m_pt = mean_center_patch(ref_pt, kSize, 0.65f, fin_pt);
  const auto m_mnee = mean_center_patch(out_mnee, kSize, 0.65f, fin_mnee);
  const auto m_bia = mean_center_patch(out_bia, kSize, 0.65f, fin_bia);
  const auto m_mega = mean_center_patch(out_mega, kSize, 0.65f, fin_mega);
  const auto m_wf = mean_center_patch(out_wf, kSize, 0.65f, fin_wf);
  const float rel_bia =
      std::abs(m_bia[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_mega =
      std::abs(m_mega[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_wf =
      std::abs(m_wf[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_sched =
      std::abs(m_mega[0] - m_wf[0]) / std::max(m_pt[0], 1.0e-5f);
  std::cout << "  base=" << m_base[0] << " pt512=" << m_pt[0]
            << " mnee=" << m_mnee[0] << " bia16=" << m_bia[0]
            << " (rel=" << (rel_bia * 100.0f) << "%)"
            << " sms_mega=" << m_mega[0]
            << " (rel=" << (rel_mega * 100.0f) << "%)"
            << " sms_wf=" << m_wf[0] << " (rel=" << (rel_wf * 100.0f) << "%)"
            << " mega_vs_wf=" << (rel_sched * 100.0f) << "%\n";
  return fin0 && fin_pt && fin_mnee && fin_bia && fin_mega && fin_wf &&
         (m_pt[0] > 0.25f) && (m_base[0] < 0.10f * m_pt[0]) &&
         (m_mnee[0] < 0.75f * m_pt[0]) && (rel_bia < 0.06f) &&
         (m_bia[0] <= 1.06f * m_pt[0]) && (rel_mega < 0.06f) &&
         (rel_wf < 0.06f) && (rel_sched < 0.04f);
}

[[nodiscard]] bool test_concave_metal_mirror_caustic(
    Device &device, float roughness, const char *label) {
  std::cout << "[SMS Test 2 " << label
            << "] Concave metallic mirror ring (roughness=" << roughness
            << ") vs PT 512 spp...\n";
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  const c::MaterialId world_mat{10u}, diff_mat{1u}, metal_mat{2u};
  scene.materials.emplace(world_mat, make_black_world(0u));
  scene.materials.emplace(
      diff_mat, make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      metal_mat,
      make_glossy_material("GoldMirror", {0.95f, 0.82f, 0.55f}, roughness, 2u));
  scene.world_shader = world_mat;

  const c::GeometryId floor_geo{1u}, mirror_geo{2u};
  scene.geometries.emplace(floor_geo, make_receiver_plane(4.0f, 0.0f, diff_mat));
  scene.geometries.emplace(
      mirror_geo,
      make_concave_cylinder_arc(0.0f, 1.0f, 1.35f, 0.0f, 0.7f, 96u, metal_mat));

  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = floor_geo,
                      .cycles_object_index = 0u,
                      .is_caustics_receiver = true});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = mirror_geo,
                      .cycles_object_index = 1u,
                      .is_caustics_caster = true});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = make_translation(0.0f, 0.15f, 0.6f),
                    .orthographic_scale = 0.7f});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "AreaLight",
                   .type = c::LightType::area,
                   .transform = make_look_transform(
                       {0.0f, -1.3f, 1.4f}, {0.0f, 0.5f, 0.2f}, {0.0f, 0.0f, 1.0f}),
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 25.0f,
                   .size = 0.8f,
                   .size_y = 0.8f,
                   .spread = kPi,
                   .normalize = true,
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .is_caustics_light = true,
                   .cycles_object_index = 2u});

  constexpr std::uint32_t kSize = 24u;
  std::vector<float> base_none, ref_pt, out_mega, out_wf;
  if (!render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::none, true, 8u, kSize, 64u, base_none,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::none, true, 8u, kSize, 512u, ref_pt,
                   true) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::megakernel, false,
                   c::CausticsMethod::sms, true, 8u, kSize, 64u, out_mega,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                   false, c::CausticsMethod::sms, true, 8u, kSize, 64u, out_wf,
                   roughness > 0.0f, roughness > 0.0f ? 1.0f : 0.0f)) {
    return false;
  }

  bool fin0 = true, fin1 = true, fin2 = true, fin3 = true;
  const auto m_base = mean_center_patch(base_none, kSize, 0.7f, fin0);
  const auto m_pt = mean_center_patch(ref_pt, kSize, 0.7f, fin1);
  const auto m_mega = mean_center_patch(out_mega, kSize, 0.7f, fin2);
  const auto m_wf = mean_center_patch(out_wf, kSize, 0.7f, fin3);

  const float exc_pt = m_pt[0] - m_base[0];
  const float exc_mega = m_mega[0] - m_base[0];
  const float exc_wf = m_wf[0] - m_base[0];
  const float rel_total_mega =
      std::abs(m_mega[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_total_wf =
      std::abs(m_wf[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_exc_mega =
      std::abs(exc_mega - exc_pt) / std::max(exc_pt, 1.0e-5f);
  const float rel_exc_wf =
      std::abs(exc_wf - exc_pt) / std::max(exc_pt, 1.0e-5f);
  const float rel_sched =
      std::abs(m_mega[0] - m_wf[0]) / std::max(m_pt[0], 1.0e-5f);
  const float exc_r = m_wf[0] - m_base[0];
  const float exc_b = m_wf[2] - m_base[2];
  std::cout << "  base=" << m_base[0] << " pt512=" << m_pt[0]
            << " (exc=" << exc_pt << ") mega=" << m_mega[0]
            << " (exc=" << exc_mega << ", exc_rel=" << (rel_exc_mega * 100.0f)
            << "%) wf=" << m_wf[0] << " (exc=" << exc_wf
            << ", exc_rel=" << (rel_exc_wf * 100.0f)
            << "%) gold(exc_R/B=" << exc_r << "/" << exc_b << ")\n";
  return fin0 && fin1 && fin2 && fin3 && (exc_pt > 0.15f) &&
         (exc_r > 1.25f * exc_b) && (rel_total_mega < 0.06f) &&
         (rel_total_wf < 0.06f) && (rel_exc_mega < 0.12f) &&
         (rel_exc_wf < 0.12f) && (rel_sched < 0.04f);
}

[[nodiscard]] bool test_glossy_refractive_lens_sms(Device &device) {
  std::cout << "[SMS Test 3] Glossy Glass Lens (roughness=0.20, alpha=0.04) "
               "vs PT 512 spp...\n";
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  const c::MaterialId world_mat{10u}, diff_mat{1u}, glass_mat{2u};
  scene.materials.emplace(world_mat, make_black_world(0u));
  scene.materials.emplace(
      diff_mat, make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
  scene.materials.emplace(
      glass_mat,
      make_glass_material("GlossyLens", {1.0f, 1.0f, 1.0f}, 1.5f, 0.20f, 0.0f, 2u));
  scene.world_shader = world_mat;

  const c::GeometryId floor_geo{1u}, lens_geo{2u};
  scene.geometries.emplace(floor_geo, make_receiver_plane(4.0f, 0.0f, diff_mat));
  scene.geometries.emplace(
      lens_geo, make_plano_convex_lens(1.2f, 0.8f, 1.5f, 0.20f, 16u, glass_mat));

  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = floor_geo,
                      .cycles_object_index = 0u,
                      .is_caustics_receiver = true});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = lens_geo,
                      .cycles_object_index = 1u,
                      .is_caustics_caster = true});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = make_translation(0.0f, 0.0f, 0.6f),
                    .orthographic_scale = 0.6f});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "AreaLight",
                   .type = c::LightType::area,
                   .transform = make_translation(0.0f, 0.0f, 2.2f),
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

  constexpr std::uint32_t kSize = 24u;
  std::vector<float> ref_pt, out_mega, out_wf;
  if (!render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged, false,
                   c::CausticsMethod::none, true, 8u, kSize, 512u, ref_pt,
                   true) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::megakernel, false,
                   c::CausticsMethod::sms, true, 8u, kSize, 64u, out_mega,
                   false) ||
      !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                   false, c::CausticsMethod::sms, true, 8u, kSize, 64u, out_wf,
                   false)) {
    return false;
  }

  bool fin_pt = true, fin_mega = true, fin_wf = true;
  const auto m_pt = mean_center_patch(ref_pt, kSize, 0.5f, fin_pt);
  const auto m_mega = mean_center_patch(out_mega, kSize, 0.5f, fin_mega);
  const auto m_wf = mean_center_patch(out_wf, kSize, 0.5f, fin_wf);
  const float rel_mega =
      std::abs(m_mega[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_wf =
      std::abs(m_wf[0] - m_pt[0]) / std::max(m_pt[0], 1.0e-5f);
  const float rel_sched =
      std::abs(m_mega[0] - m_wf[0]) / std::max(m_pt[0], 1.0e-5f);
  std::cout << "  pt512=" << m_pt[0] << " glossy_mega=" << m_mega[0]
            << " (rel=" << (rel_mega * 100.0f) << "%)"
            << " glossy_wf=" << m_wf[0] << " (rel=" << (rel_wf * 100.0f) << "%)"
            << " mega_vs_wf=" << (rel_sched * 100.0f) << "%\n";
  return fin_pt && fin_mega && fin_wf && (m_pt[0] > 0.30f) &&
         (rel_mega < 0.07f) && (rel_wf < 0.07f) && (rel_sched < 0.04f);
}

[[nodiscard]] bool test_biased_sms_and_spectral_dispersion(Device &device) {
  std::cout << "[SMS Test 4] Biased vs Unbiased SMS & Spectral Dispersion "
               "(Abbe=15)...\n";
  {
    c::SceneSnapshot scene;
    scene.world_sampling = c::WorldSampling::none;
    const c::MaterialId world_mat{10u}, diff_mat{1u}, glass_mat{2u};
    scene.materials.emplace(world_mat, make_black_world(0u));
    scene.materials.emplace(
        diff_mat, make_diffuse_material("Floor", {0.8f, 0.8f, 0.8f}, 1u));
    scene.materials.emplace(
        glass_mat,
        make_glass_material("Lens", {1.0f, 1.0f, 1.0f}, 1.5f, 0.0f, 0.0f, 2u));
    scene.world_shader = world_mat;
    const c::GeometryId floor_geo{1u}, lens_geo{2u};
    scene.geometries.emplace(floor_geo, make_receiver_plane(4.0f, 0.0f, diff_mat));
    scene.geometries.emplace(
        lens_geo,
        make_plano_convex_lens(1.2f, 0.8f, 1.5f, 0.20f, 16u, glass_mat));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = floor_geo,
                        .cycles_object_index = 0u,
                        .is_caustics_receiver = true});
    scene.instances.emplace(
        c::InstanceId{2u},
        c::InstanceDesc{.geometry = lens_geo,
                        .cycles_object_index = 1u,
                        .is_caustics_caster = true});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = make_translation(0.0f, 0.0f, 0.6f),
                      .orthographic_scale = 0.6f});
    scene.active_camera = c::CameraId{1u};
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "AreaLight",
                     .type = c::LightType::area,
                     .transform = make_translation(0.0f, 0.0f, 2.2f),
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

    constexpr std::uint32_t kSize = 24u;
    std::vector<float> unb_wf, b_mega, b_wf;
    if (!render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                     false, c::CausticsMethod::sms, true, 8u, kSize, 64u,
                     unb_wf, false) ||
        !render_rgba(device, scene, b::LuisaPathScheduler::megakernel, false,
                     c::CausticsMethod::sms, false, 8u, kSize, 32u, b_mega,
                     false) ||
        !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                     false, c::CausticsMethod::sms, false, 8u, kSize, 32u, b_wf,
                     false)) {
      return false;
    }
    bool fu = true, fm = true, fw = true;
    const auto mu = mean_center_patch(unb_wf, kSize, 0.5f, fu);
    const auto mbm = mean_center_patch(b_mega, kSize, 0.5f, fm);
    const auto mbw = mean_center_patch(b_wf, kSize, 0.5f, fw);
    const float rel_bm = std::abs(mbm[0] - mu[0]) / std::max(mu[0], 1.0e-5f);
    const float rel_bw = std::abs(mbw[0] - mu[0]) / std::max(mu[0], 1.0e-5f);
    const float rel_sched =
        std::abs(mbm[0] - mbw[0]) / std::max(mu[0], 1.0e-5f);
    std::cout << "  4a unbiased=" << mu[0] << " biased_mega=" << mbm[0]
              << " (" << (rel_bm * 100.0f) << "%) biased_wf=" << mbw[0] << " ("
              << (rel_bw * 100.0f) << "%)\n";
    if (!fu || !fm || !fw || rel_bm > 0.04f || rel_bw > 0.04f ||
        rel_sched > 0.03f) {
      return false;
    }
  }

  {
    c::SceneSnapshot scene;
    scene.world_sampling = c::WorldSampling::none;
    const c::MaterialId world_mat{10u}, diff_mat{1u}, disp_mat{2u};
    scene.materials.emplace(world_mat, make_black_world(0u));
    scene.materials.emplace(
        diff_mat, make_diffuse_material("Floor", {0.75f, 0.75f, 0.75f}, 1u));
    scene.materials.emplace(
        disp_mat, make_glass_material("DispGlass", {1.0f, 1.0f, 1.0f}, 1.5f,
                                      0.0f, 15.0f, 2u));
    scene.world_shader = world_mat;
    const c::GeometryId floor_geo{1u}, sphere_geo{2u};
    scene.geometries.emplace(floor_geo, make_receiver_plane(4.0f, 0.0f, diff_mat));
    scene.geometries.emplace(
        sphere_geo,
        make_smooth_uv_sphere({0.0f, 0.0f, 0.9f}, 0.5f, 48u, 96u, disp_mat));
    scene.instances.emplace(
        c::InstanceId{1u},
        c::InstanceDesc{.geometry = floor_geo,
                        .cycles_object_index = 0u,
                        .is_caustics_receiver = true});
    scene.instances.emplace(
        c::InstanceId{2u},
        c::InstanceDesc{.geometry = sphere_geo,
                        .cycles_object_index = 1u,
                        .is_caustics_caster = true});
    scene.cameras.emplace(
        c::CameraId{1u},
        c::CameraDesc{.projection = c::CameraProjection::orthographic,
                      .transform = make_translation(0.0f, 0.0f, 0.25f),
                      .orthographic_scale = 0.8f});
    scene.active_camera = c::CameraId{1u};
    scene.lights.emplace(
        c::LightId{1u},
        c::LightDesc{.name = "Sun",
                     .type = c::LightType::distant,
                     .transform = psycles::Mat4f{},
                     .color = {1.0f, 1.0f, 1.0f},
                     .power = 3.0f,
                     .angle = 0.0f,
                     .normalize = true,
                     .use_mis = false,
                     .cast_shadow = true,
                     .visibility_mask = c::all_ray_visibility,
                     .is_caustics_light = true,
                     .cycles_object_index = 2u});

    constexpr std::uint32_t kSize = 32u;
    std::vector<float> out_mega, out_wf;
    if (!render_rgba(device, scene, b::LuisaPathScheduler::megakernel, true,
                     c::CausticsMethod::sms, true, 8u, kSize, 32u, out_mega,
                     false) ||
        !render_rgba(device, scene, b::LuisaPathScheduler::wavefront_staged,
                     true, c::CausticsMethod::sms, true, 8u, kSize, 32u, out_wf,
                     false)) {
      return false;
    }

    auto radial_stats = [&](const std::vector<float> &rgba) {
      double wr = 0.0, rr = 0.0, wb = 0.0, rb = 0.0, sum_y = 0.0;
      bool finite = true;
      for (std::uint32_t y = 0u; y < kSize; ++y) {
        const float py =
            (((static_cast<float>(y) + 0.5f) / kSize) - 0.5f) * 0.8f;
        for (std::uint32_t x = 0u; x < kSize; ++x) {
          const float px =
              (((static_cast<float>(x) + 0.5f) / kSize) - 0.5f) * 0.8f;
          const float rad = std::sqrt(px * px + py * py);
          const std::size_t idx =
              (static_cast<std::size_t>(y) * kSize + x) * 4u;
          const float r = rgba[idx + 0u];
          const float g = rgba[idx + 1u];
          const float bv = rgba[idx + 2u];
          if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bv)) {
            finite = false;
          }
          sum_y += g;
          if (rad < 0.35f) {
            wr += r;
            rr += r * rad;
            wb += bv;
            rb += bv * rad;
          }
        }
      }
      const double mean_rr = wr > 0.0 ? rr / wr : 0.0;
      const double mean_rb = wb > 0.0 ? rb / wb : 0.0;
      const double exp_frac = mean_rr > 0.0 ? (mean_rb - mean_rr) / mean_rr : 0.0;
      return std::tuple{finite, mean_rr, mean_rb, exp_frac,
                        sum_y / (kSize * kSize)};
    };

    const auto [fin_m, rr_m, rb_m, exp_m, mean_m] = radial_stats(out_mega);
    const auto [fin_w, rr_w, rb_w, exp_w, mean_w] = radial_stats(out_wf);
    const double rel_sched =
        std::abs(mean_m - mean_w) / std::max(mean_w, 1.0e-5);
    std::cout << "  4b disp_mega r_R=" << rr_m << " r_B=" << rb_m
              << " exp=" << (exp_m * 100.0) << "% mean=" << mean_m << '\n';
    std::cout << "  4b disp_wf   r_R=" << rr_w << " r_B=" << rb_w
              << " exp=" << (exp_w * 100.0) << "% mean=" << mean_w
              << " mega_vs_wf=" << (rel_sched * 100.0) << "%\n";
    if (!fin_m || !fin_w || exp_m <= 0.04 || exp_w <= 0.04 ||
        rel_sched > 0.04) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const std::string backend = argc > 1 ? argv[1] : "fallback";
  Context ctx{argv[0]};
  Device device = ctx.create_device(backend);

  bool all_ok = true;
  if (!test_dual_lens_multisolution_area_sms(device)) {
    std::cerr << "FAILED: test_dual_lens_multisolution_area_sms\n";
    all_ok = false;
  }
  if (!test_concave_metal_mirror_caustic(device, 0.0f, "a_smooth")) {
    std::cerr << "FAILED: test_concave_metal_mirror_caustic (smooth)\n";
    all_ok = false;
  }
  if (!test_concave_metal_mirror_caustic(device, 0.25f, "b_glossy")) {
    std::cerr << "FAILED: test_concave_metal_mirror_caustic (glossy)\n";
    all_ok = false;
  }
  if (!test_glossy_refractive_lens_sms(device)) {
    std::cerr << "FAILED: test_glossy_refractive_lens_sms\n";
    all_ok = false;
  }
  if (!test_biased_sms_and_spectral_dispersion(device)) {
    std::cerr << "FAILED: test_biased_sms_and_spectral_dispersion\n";
    all_ok = false;
  }
  if (!all_ok) {
    return 1;
  }

  std::cout << "ALL Phase 2B SMS caustics tests PASSED on backend [" << backend
            << "]\n";
  return 0;
}
