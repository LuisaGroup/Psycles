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

[[nodiscard]] psycles::Mat4f make_translation(float x, float y, float z) {
  psycles::Mat4f m;
  m.elements[12] = x;
  m.elements[13] = y;
  m.elements[14] = z;
  return m;
}

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

[[nodiscard]] c::MaterialDesc make_world_material(std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto bg = g.add_node(psycles::compiler::node_type::background, "bg");
  static_cast<void>(g.set_input(
      bg, "Color", c::SocketValue::color({0.70f, 0.80f, 1.0f})));
  static_cast<void>(
      g.set_input(bg, "Strength", c::SocketValue::floating(0.03f)));
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
  static_cast<void>(g.set_property(
      glossy, "Distribution", c::SocketValue::string("GGX")));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{glossy, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_thin_film_material(
    std::string name, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto bsdf =
      g.add_node(psycles::compiler::node_type::principled_bsdf, "thin_film");
  static_cast<void>(g.set_input(
      bsdf, "BaseColor", c::SocketValue::color({0.15f, 0.18f, 0.22f})));
  static_cast<void>(
      g.set_input(bsdf, "Roughness", c::SocketValue::floating(0.06f)));
  static_cast<void>(g.set_input(bsdf, "IOR", c::SocketValue::floating(1.50f)));
  static_cast<void>(g.set_input(
      bsdf, "ThinFilmThickness", c::SocketValue::floating(480.0f)));
  static_cast<void>(
      g.set_input(bsdf, "ThinFilmIOR", c::SocketValue::floating(1.38f)));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{bsdf, "Closure"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_het_blackbody_volume_material(
    std::string name, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto transp =
      g.add_node(psycles::compiler::node_type::transparent_bsdf, "transp");
  static_cast<void>(g.set_input(
      transp, "Color", c::SocketValue::color({1.0f, 1.0f, 1.0f})));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{transp, "Closure"});

  const auto tc =
      g.add_node(psycles::compiler::node_type::texture_coordinate, "tc");
  const auto p2v =
      g.add_node(psycles::compiler::node_type::point_to_vector, "p2v");
  const auto v2s =
      g.add_node(psycles::compiler::node_type::vector_to_scalar, "v2s");
  const auto mul =
      g.add_node(psycles::compiler::node_type::multiply_float, "mul");
  const auto add = g.add_node(psycles::compiler::node_type::add_float, "add");
  const auto pv =
      g.add_node(psycles::compiler::node_type::principled_volume, "pv");

  static_cast<void>(g.connect(c::OutputRef{tc, "Generated"}, p2v, "Point"));
  static_cast<void>(g.connect(c::OutputRef{p2v, "Vector"}, v2s, "Vector"));
  static_cast<void>(g.connect(c::OutputRef{v2s, "Value"}, mul, "A"));
  static_cast<void>(g.set_input(mul, "B", c::SocketValue::floating(1.8f)));
  static_cast<void>(g.connect(c::OutputRef{mul, "Value"}, add, "A"));
  static_cast<void>(g.set_input(add, "B", c::SocketValue::floating(0.4f)));
  static_cast<void>(g.connect(c::OutputRef{add, "Value"}, pv, "Density"));

  static_cast<void>(g.set_input(
      pv, "Color", c::SocketValue::color({0.28f, 0.78f, 0.95f})));
  static_cast<void>(g.set_input(
      pv, "AbsorptionColor", c::SocketValue::color({0.90f, 0.35f, 0.18f})));
  static_cast<void>(
      g.set_input(pv, "Anisotropy", c::SocketValue::floating(0.25f)));
  static_cast<void>(
      g.set_input(pv, "EmissionStrength", c::SocketValue::floating(0.0f)));
  static_cast<void>(
      g.set_input(pv, "BlackbodyIntensity", c::SocketValue::floating(1.5f)));
  static_cast<void>(
      g.set_input(pv, "Temperature", c::SocketValue::floating(2500.0f)));
  static_cast<void>(g.set_input(
      pv, "BlackbodyTint", c::SocketValue::color({1.0f, 1.0f, 1.0f})));

  g.set_root(c::ShaderDomain::volume, c::OutputRef{pv, "Volume"});
  return c::MaterialDesc{.name = std::move(name),
                         .shader = std::move(g),
                         .cycles_shader_index = shader_idx};
}

[[nodiscard]] c::MaterialDesc make_rw_sss_material(
    std::string name, std::uint32_t shader_idx) {
  c::ShaderGraph g;
  const auto sss =
      g.add_node(psycles::compiler::node_type::subsurface_scattering, "sss");
  static_cast<void>(g.set_input(
      sss, "Color", c::SocketValue::color({0.85f, 0.72f, 0.62f})));
  static_cast<void>(
      g.set_input(sss, "Scale", c::SocketValue::floating(0.25f)));
  static_cast<void>(g.set_input(
      sss, "Radius", c::SocketValue::vector({0.80f, 0.15f, 0.04f})));
  static_cast<void>(g.set_input(sss, "IOR", c::SocketValue::floating(1.40f)));
  static_cast<void>(
      g.set_input(sss, "Roughness", c::SocketValue::floating(0.0f)));
  static_cast<void>(
      g.set_input(sss, "Anisotropy", c::SocketValue::floating(0.0f)));
  static_cast<void>(
      g.set_property(sss, "Method", c::SocketValue::string("RANDOM_WALK")));
  g.set_root(c::ShaderDomain::surface, c::OutputRef{sss, "Closure"});
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
  m.normals.domain = c::MeshAttributeDomain::point;
  m.normals.values = {{0.0f, 0.0f, 1.0f},
                      {0.0f, 0.0f, 1.0f},
                      {0.0f, 0.0f, 1.0f},
                      {0.0f, 0.0f, 1.0f}};
  m.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
  m.triangle_smooth = {0u, 0u};
  m.material_slots = {mat};
  m.triangle_material_slots = {0u, 0u};
  return m;
}

[[nodiscard]] c::TriangleMeshDesc make_box_mesh(
    float hx, float hy, float hz, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.positions = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
      {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz}};
  m.triangles = {
      {0u, 2u, 1u}, {0u, 3u, 2u},
      {4u, 5u, 6u}, {4u, 6u, 7u},
      {0u, 1u, 5u}, {0u, 5u, 4u},
      {2u, 3u, 7u}, {2u, 7u, 6u},
      {1u, 2u, 6u}, {1u, 6u, 5u},
      {3u, 0u, 4u}, {3u, 4u, 7u}};
  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  m.triangle_smooth.assign(m.triangles.size(), 0u);
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

[[nodiscard]] c::TriangleMeshDesc make_concave_cylinder_arc(
    float cx, float cy, float radius, float half_angle, float z0, float z1,
    std::uint32_t segs, c::MaterialId mat) {
  c::TriangleMeshDesc m;
  m.normals.domain = c::MeshAttributeDomain::point;
  for (std::uint32_t i = 0u; i <= segs; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(segs);
    const float theta = -half_angle + 2.0f * half_angle * t;
    const float sx = std::sin(theta);
    const float cy_cos = std::cos(theta);
    const float px = cx + radius * sx;
    const float py = cy + radius * cy_cos;
    const psycles::Vec3f inward_n{-sx, -cy_cos, 0.0f};
    m.positions.push_back({px, py, z0});
    m.normals.values.push_back(inward_n);
    m.positions.push_back({px, py, z1});
    m.normals.values.push_back(inward_n);
  }
  for (std::uint32_t i = 0u; i < segs; ++i) {
    const std::uint32_t b0 = i * 2u;
    const std::uint32_t t0 = b0 + 1u;
    const std::uint32_t b1 = b0 + 2u;
    const std::uint32_t t1 = b0 + 3u;
    // CCW when viewed from the concave interior (-Y side).
    m.triangles.push_back({b0, b1, t1});
    m.triangles.push_back({b0, t1, t0});
  }
  m.material_slots = {mat};
  m.triangle_material_slots.assign(m.triangles.size(), 0u);
  m.triangle_smooth.assign(m.triangles.size(), 1u);
  return m;
}

struct ImageMetrics {
  std::array<double, 3u> mean{};
  double max_luma{};
  std::size_t nan_inf_count{};
};

[[nodiscard]] ImageMetrics compute_metrics(const std::vector<float> &rgba,
                                           std::uint32_t width,
                                           std::uint32_t height) {
  ImageMetrics out{};
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::array<double, 3u> sums{};
  for (std::size_t p = 0u; p < pixels; ++p) {
    const std::size_t base = p * 4u;
    const float r = rgba[base + 0u];
    const float g = rgba[base + 1u];
    const float bv = rgba[base + 2u];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(bv)) {
      ++out.nan_inf_count;
      continue;
    }
    sums[0u] += r;
    sums[1u] += g;
    sums[2u] += bv;
    const double luma = 0.2126 * r + 0.7152 * g + 0.0722 * bv;
    out.max_luma = std::max(out.max_luma, luma);
  }
  if (pixels > 0u) {
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      out.mean[ch] = sums[ch] / static_cast<double>(pixels);
    }
  }
  return out;
}

[[nodiscard]] bool render_kitchen_sink(
    Device &device, const c::SceneSnapshot &scene,
    b::LuisaPathScheduler scheduler, c::CausticsMethod caustics_method,
    std::uint32_t size, std::uint32_t spp, std::vector<float> &out_rgba) {
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
                             .seed = 4242u,
                             .pixel_filter = c::PixelFilter::box,
                             .filter_width = 1.0f};
  settings.integrator.spectral_rendering = true;
  settings.integrator.caustics_method = caustics_method;
  settings.integrator.sms_unbiased = true;
  settings.integrator.sms_max_trials = 8u;
  settings.integrator.max_bounces = 8u;
  settings.integrator.diffuse_bounces = 4u;
  settings.integrator.glossy_bounces = 8u;
  settings.integrator.transmission_bounces = 8u;
  settings.integrator.volume_bounces = 4u;
  settings.integrator.transparent_max_bounces = 16u;
  settings.integrator.reflective_caustics = true;
  settings.integrator.refractive_caustics = true;
  settings.integrator.sample_clamp_direct = 0.0f;
  settings.integrator.sample_clamp_indirect = 0.0f;
  settings.integrator.direct_light_sampling =
      c::DirectLightSampling::multiple_importance_sampling;
  settings.passes = {{.kind = c::PassKind::combined, .channels = 4u}};

  auto session = renderer.create_session(*compiled.scene, settings);
  if (!session) {
    std::cerr << "Failed to create kitchen-sink render session\n";
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

[[nodiscard]] bool test_kitchen_sink(Device &device,
                                     std::string_view backend) {
  c::SceneSnapshot scene;
  scene.world_sampling = c::WorldSampling::none;
  scene.cycles_background_object_index = 7u;

  const c::MaterialId world_mat{10u}, floor_mat{1u}, glass_mat{2u}, gold_mat{3u},
      thin_film_mat{4u}, vol_mat{5u}, sss_mat{6u};
  scene.materials.emplace(world_mat, make_world_material(0u));
  scene.materials.emplace(
      floor_mat, make_diffuse_material("Floor", {0.55f, 0.55f, 0.55f}, 1u));
  scene.materials.emplace(
      glass_mat,
      make_glass_material("GlassDispersive", {0.98f, 0.99f, 1.0f}, 1.52f, 0.0f,
                          15.0f, 2u));
  scene.materials.emplace(
      gold_mat,
      make_glossy_material("GoldMirror", {0.96f, 0.78f, 0.42f}, 0.04f, 3u));
  scene.materials.emplace(thin_film_mat,
                          make_thin_film_material("ThinFilmDielectric", 4u));
  scene.materials.emplace(
      vol_mat, make_het_blackbody_volume_material("HetBlackbodyVol", 5u));
  scene.materials.emplace(sss_mat, make_rw_sss_material("ChromaticSSS", 6u));
  scene.world_shader = world_mat;

  const c::GeometryId floor_geo{1u}, glass_geo{2u}, gold_geo{3u}, tf_geo{4u},
      vol_geo{5u}, sss_geo{6u};
  scene.geometries.emplace(floor_geo,
                           make_receiver_plane(16.0f, 0.0f, floor_mat));
  scene.geometries.emplace(
      glass_geo,
      make_smooth_uv_sphere({-0.9f, -0.3f, 0.52f}, 0.50f, 24u, 48u, glass_mat));
  scene.geometries.emplace(
      gold_geo, make_concave_cylinder_arc(0.9f, -0.2f, 0.75f, 1.45f, 0.0f,
                                          0.65f, 48u, gold_mat));
  scene.geometries.emplace(
      tf_geo,
      make_smooth_uv_sphere({0.0f, 0.25f, 0.34f}, 0.34f, 20u, 40u,
                            thin_film_mat));
  scene.geometries.emplace(vol_geo,
                           make_box_mesh(0.50f, 0.50f, 0.50f, vol_mat));
  scene.geometries.emplace(
      sss_geo,
      make_smooth_uv_sphere({1.15f, 1.10f, 0.55f}, 0.48f, 24u, 48u, sss_mat));

  scene.instances.emplace(
      c::InstanceId{1u},
      c::InstanceDesc{.geometry = floor_geo,
                      .cycles_object_index = 0u,
                      .is_caustics_receiver = true});
  scene.instances.emplace(
      c::InstanceId{2u},
      c::InstanceDesc{.geometry = glass_geo,
                      .cycles_object_index = 1u,
                      .is_caustics_caster = true});
  scene.instances.emplace(
      c::InstanceId{3u},
      c::InstanceDesc{.geometry = gold_geo,
                      .cycles_object_index = 2u,
                      .is_caustics_caster = true});
  scene.instances.emplace(
      c::InstanceId{4u},
      c::InstanceDesc{.geometry = tf_geo, .cycles_object_index = 3u});
  scene.instances.emplace(
      c::InstanceId{5u},
      c::InstanceDesc{.geometry = vol_geo,
                      .transform = make_translation(-1.15f, 1.15f, 0.65f),
                      .cycles_object_index = 4u});
  scene.instances.emplace(
      c::InstanceId{6u},
      c::InstanceDesc{.geometry = sss_geo, .cycles_object_index = 5u});

  scene.cameras.emplace(
      c::CameraId{1u},
      c::CameraDesc{.projection = c::CameraProjection::perspective,
                    .transform = make_look_transform(
                        {0.0f, -4.4f, 2.3f}, {0.0f, 0.2f, 0.45f},
                        {0.0f, 0.0f, 1.0f}),
                    .field_of_view = 0.78539816f,
                    .horizontal_field_of_view = 0.78539816f,
                    .sensor_fit = c::CameraSensorFit::vertical});
  scene.active_camera = c::CameraId{1u};

  scene.lights.emplace(
      c::LightId{1u},
      c::LightDesc{.name = "CausticPoint",
                   .type = c::LightType::point,
                   .transform = make_translation(0.05f, -2.1f, 2.2f),
                   .color = {1.0f, 1.0f, 1.0f},
                   .power = 120.0f,
                   .size = 0.10f,
                   .normalize = true,
                   .is_sphere = true,
                   .use_mis = true,
                   .cast_shadow = true,
                   .visibility_mask = c::all_ray_visibility,
                   .is_caustics_light = true,
                   .cycles_object_index = 6u});

  constexpr std::uint32_t kSize = 24u;
  constexpr std::uint32_t kSpp = 32u;

  struct ModeSpec {
    const char *label;
    c::CausticsMethod caustics;
    b::LuisaPathScheduler scheduler;
  };
  const std::array<ModeSpec, 4u> modes{{
      {"mnee_mega", c::CausticsMethod::mnee, b::LuisaPathScheduler::megakernel},
      {"mnee_wf", c::CausticsMethod::mnee,
       b::LuisaPathScheduler::wavefront_staged},
      {"sms_mega", c::CausticsMethod::sms, b::LuisaPathScheduler::megakernel},
      {"sms_wf", c::CausticsMethod::sms,
       b::LuisaPathScheduler::wavefront_staged},
  }};

  std::array<ImageMetrics, 4u> stats{};
  for (std::size_t i = 0u; i < modes.size(); ++i) {
    std::vector<float> rgba;
    if (!render_kitchen_sink(device, scene, modes[i].scheduler,
                             modes[i].caustics, kSize, kSpp, rgba)) {
      std::cerr << "FAIL render mode=" << modes[i].label << " on " << backend
                << '\n';
      return false;
    }
    stats[i] = compute_metrics(rgba, kSize, kSize);
    std::cout << "  [" << backend << " | " << modes[i].label << "] mean=("
              << stats[i].mean[0] << ", " << stats[i].mean[1] << ", "
              << stats[i].mean[2] << ") max_luma=" << stats[i].max_luma
              << " nan_inf=" << stats[i].nan_inf_count << '\n';
    if (stats[i].nan_inf_count != 0u) {
      std::cerr << "FAIL non-finite pixels in " << modes[i].label << " on "
                << backend << '\n';
      return false;
    }
    if (stats[i].max_luma >= 500.0) {
      std::cerr << "FAIL firefly max_luma=" << stats[i].max_luma << " in "
                << modes[i].label << " on " << backend << '\n';
      return false;
    }
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      if (stats[i].mean[ch] <= 1.0e-4) {
        std::cerr << "FAIL channel " << ch << " too dark (" << stats[i].mean[ch]
                  << ") in " << modes[i].label << " on " << backend << '\n';
        return false;
      }
    }
  }

  const auto check_parity = [&](const char *pair_name, const ImageMetrics &mega,
                                const ImageMetrics &wf) {
    for (std::size_t ch = 0u; ch < 3u; ++ch) {
      const double denom = std::max(wf.mean[ch], 1.0e-4);
      const double rel = std::abs(mega.mean[ch] - wf.mean[ch]) / denom;
      if (rel > 0.06) {
        std::cerr << "FAIL " << pair_name << " mega_vs_wf ch=" << ch
                  << " mega=" << mega.mean[ch] << " wf=" << wf.mean[ch]
                  << " rel=" << (rel * 100.0) << "% on " << backend << '\n';
        return false;
      }
    }
    std::cout << "  [PASS parity " << pair_name << " on " << backend
              << "] rel=("
              << (std::abs(mega.mean[0] - wf.mean[0]) /
                  std::max(wf.mean[0], 1.0e-4) * 100.0)
              << "%, "
              << (std::abs(mega.mean[1] - wf.mean[1]) /
                  std::max(wf.mean[1], 1.0e-4) * 100.0)
              << "%, "
              << (std::abs(mega.mean[2] - wf.mean[2]) /
                  std::max(wf.mean[2], 1.0e-4) * 100.0)
              << "%)\n";
    return true;
  };

  return check_parity("MNEE", stats[0], stats[1]) &&
         check_parity("SMS", stats[2], stats[3]);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string backend = argc > 1 ? argv[1] : "fallback";
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    std::cout << "Running test_luisa_spectral_caustics_kitchen_sink on ["
              << backend << "]...\n";
    if (!test_kitchen_sink(device, backend)) {
      return EXIT_FAILURE;
    }
    std::cout << "test_luisa_spectral_caustics_kitchen_sink PASSED on ["
              << backend << "]\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Exception in test_luisa_spectral_caustics_kitchen_sink: "
              << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
