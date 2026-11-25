#include "path_tracer_internal.h"
#include "virtual_texture_cache.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#if defined(PSYCLES_WITH_OPENIMAGEIO)
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>
#endif

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

class ScopedEnvVar final {
private:
  std::string _name;
  std::string _prev;
  bool _had_prev{false};

public:
  ScopedEnvVar(std::string name, std::string_view value)
      : _name{std::move(name)} {
    if (const char *old = std::getenv(_name.c_str())) {
      _had_prev = true;
      _prev = old;
    }
#ifdef _WIN32
    _putenv_s(_name.c_str(), std::string{value}.c_str());
#else
    setenv(_name.c_str(), std::string{value}.c_str(), 1);
#endif
  }

  ~ScopedEnvVar() noexcept {
#ifdef _WIN32
    _putenv_s(_name.c_str(), _had_prev ? _prev.c_str() : "");
#else
    if (_had_prev) {
      setenv(_name.c_str(), _prev.c_str(), 1);
    } else {
      unsetenv(_name.c_str());
    }
#endif
  }

  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;
};

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
  Mat4f m{};
  m.elements[12u] = x;
  m.elements[13u] = y;
  m.elements[14u] = z;
  return m;
}

[[nodiscard]] std::vector<std::uint8_t>
encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                 const std::vector<std::uint8_t> &rgba8) {
  require(rgba8.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA8 buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::UINT8};
  if (auto out = OIIO::ImageOutput::create("vt_pf_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_pf_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_pf_tmp_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".png");
  {
    auto out = OIIO::ImageOutput::create(temp_path.string());
    require(bool(out), "OIIO create temp PNG failed");
    require(out->open(temp_path.string(), spec), "OIIO open temp PNG failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write temp PNG failed");
    require(out->close(), "OIIO close temp PNG failed");
  }
  std::ifstream ifs{temp_path, std::ios::binary};
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>()};
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  require(!bytes.empty(), "empty PNG file readback");
  return bytes;
#else
  std::ostringstream hdr;
  hdr << "P6\n" << width << ' ' << height << "\n255\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  bytes.reserve(bytes.size() + static_cast<std::size_t>(width) * height * 3u);
  for (std::size_t i = 0u; i < rgba8.size(); i += 4u) {
    bytes.push_back(rgba8[i]);
    bytes.push_back(rgba8[i + 1u]);
    bytes.push_back(rgba8[i + 2u]);
  }
  return bytes;
#endif
}

[[nodiscard]] std::vector<std::uint8_t>
make_pattern_png(std::uint32_t s, std::uint8_t r_bias, std::uint8_t g_bias,
                 std::uint8_t b_bias) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(s) * s * 4u);
  for (std::uint32_t y = 0u; y < s; ++y) {
    for (std::uint32_t x = 0u; x < s; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * s + x) * 4u;
      const auto tx = x / 60u;
      const auto ty = y / 60u;
      px[i + 0u] = static_cast<std::uint8_t>(r_bias + ((tx * 23u + x / 3u) & 0x7fu));
      px[i + 1u] = static_cast<std::uint8_t>(g_bias + ((ty * 29u + y / 3u) & 0x7fu));
      px[i + 2u] = static_cast<std::uint8_t>(b_bias + (((tx + ty) * 19u) & 0x7fu));
      px[i + 3u] = 255u;
    }
  }
  return encode_png_rgba8(s, s, px);
}

[[nodiscard]] TriangleMeshDesc make_quad_mesh(
    float min_x, float max_x, float min_y, float max_y, float z,
    MaterialId material, bool flip_winding = false) {
  TriangleMeshDesc mesh;
  mesh.positions = {{min_x, min_y, z},
                    {max_x, min_y, z},
                    {max_x, max_y, z},
                    {min_x, max_y, z}};
  const Vec3f n = flip_winding ? Vec3f{0.0f, 0.0f, -1.0f}
                               : Vec3f{0.0f, 0.0f, 1.0f};
  mesh.normals = {MeshAttributeDomain::point, {n, n, n, n}};
  mesh.uv = {MeshAttributeDomain::corner,
             {{0.0f, 0.0f},
              {1.0f, 0.0f},
              {1.0f, 1.0f},
              {0.0f, 0.0f},
              {1.0f, 1.0f},
              {0.0f, 1.0f}}};
  mesh.default_uv_available = true;
  mesh.material_slots = {material};
  if (flip_winding) {
    mesh.triangles = {{0u, 2u, 1u}, {0u, 3u, 2u}};
  } else {
    mesh.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
  }
  mesh.triangle_material_slots = {0u, 0u};
  return mesh;
}

[[nodiscard]] NodeId add_uv_image_node(ShaderGraph &graph, ImageId image_id) {
  const auto tc = graph.add_node(node_type::texture_coordinate, "TexCoord");
  const auto p2v = graph.add_node(node_type::point_to_vector, "UvToVec");
  const auto tex = graph.add_node(node_type::image_texture, "Image");
  require(graph.set_property(tex, "Image",
                             SocketValue::unsigned_integer(image_id.value)) &&
              graph.set_property(tex, "Interpolation",
                                 SocketValue::string("Linear")) &&
              graph.set_property(tex, "Extension",
                                 SocketValue::string("REPEAT")) &&
              graph.set_property(tex, "Projection",
                                 SocketValue::string("FLAT")) &&
              graph.set_property(tex, "ProjectionBlend",
                                 SocketValue::floating(0.0f)) &&
              graph.set_property(tex, "ColorSpace",
                                 SocketValue::string("Non-Color")) &&
              graph.set_property(tex, "UnassociateAlpha",
                                 SocketValue::boolean(false)) &&
              graph.connect({tc, "UV"}, p2v, "Point") &&
              graph.connect({p2v, "Vector"}, tex, "Vector"),
          "failed to configure image texture node");
  return tex;
}

[[nodiscard]] ShaderGraph make_textured_emission_shader(ImageId image_id,
                                                        float strength = 1.0f) {
  ShaderGraph graph;
  const auto tex = add_uv_image_node(graph, image_id);
  const auto em = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({tex, "Color"}, em, "Color") &&
              graph.set_input(em, "Strength", SocketValue::floating(strength)),
          "failed to connect textured emission shader");
  graph.set_root(ShaderDomain::surface, OutputRef{em, "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_textured_diffuse_shader(ImageId image_id) {
  ShaderGraph graph;
  const auto tex = add_uv_image_node(graph, image_id);
  const auto diff = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
  require(graph.connect({tex, "Color"}, diff, "Color") &&
              graph.set_input(diff, "Roughness", SocketValue::floating(0.0f)),
          "failed to connect textured diffuse shader");
  graph.set_root(ShaderDomain::surface, OutputRef{diff, "Closure"});
  return graph;
}

struct ImageDiffMetrics {
  double rmse{};
  double max_abs{};
  std::size_t invalid_pixels{};
};

[[nodiscard]] ImageDiffMetrics compare_combined_pass(
    const io::MemoryOutputSink &a_sink, const io::MemoryOutputSink &b_sink) {
  const auto *a = a_sink.find(PassKind::combined);
  const auto *b = b_sink.find(PassKind::combined);
  require(a != nullptr && b != nullptr, "missing Combined pass");
  require(a->channels == 4u && b->channels == 4u &&
              a->pixels.size() == b->pixels.size(),
          "Combined pass dimension/channel mismatch");
  ImageDiffMetrics metrics;
  double sum_sq = 0.0;
  std::size_t count = 0u;
  for (std::size_t i = 0u; i < a->pixels.size(); ++i) {
    if (!std::isfinite(a->pixels[i]) || !std::isfinite(b->pixels[i])) {
      ++metrics.invalid_pixels;
      continue;
    }
    const double diff =
        std::abs(static_cast<double>(a->pixels[i]) - static_cast<double>(b->pixels[i]));
    sum_sq += diff * diff;
    metrics.max_abs = std::max(metrics.max_abs, diff);
    ++count;
  }
  metrics.rmse = count > 0u ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
  return metrics;
}

struct RenderOutcome {
  io::MemoryOutputSink sink;
  VirtualTextureCacheStats post_init_stats{};
  VirtualTextureCacheStats final_stats{};
  VirtualTextureCacheStats second_init_stats{};
  VirtualTextureCacheStats second_final_stats{};
};

[[nodiscard]] RenderOutcome run_scene_with_options(
    Device &device, const SceneSnapshot &scene,
    const LuisaVirtualTextureCacheOptions &vt_options,
    std::uint32_t width, std::uint32_t height, std::uint32_t spp,
    bool test_second_session = false) {
  LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .enable_fast_math = true,
       .scheduler = LuisaPathScheduler::wavefront_staged,
       .wavefront_frame_capacity = 4096u,
       .max_samples_per_dispatch = spp,
       .virtual_texture_cache = vt_options}};
  const auto compilation = renderer.compile_scene(scene);
  require(compilation.ok(), "scene compilation failed");
  const auto *native =
      dynamic_cast<const LuisaCompiledScene *>(compilation.scene.get());
  require(native != nullptr && native->data()->virtual_texture_cache != nullptr,
          "missing VT cache in compiled scene");

  RenderSettings settings;
  settings.full_extent = {width, height};
  settings.window = {.x = 0u, .y = 0u, .width = width, .height = height};
  settings.seed = 12345u;
  settings.integrator.max_bounces = 4u;
  settings.integrator.diffuse_bounces = 2u;
  settings.integrator.glossy_bounces = 2u;
  settings.integrator.transparent_max_bounces = 8u;
  settings.passes = {{.kind = PassKind::combined, .name = "Combined", .channels = 4u}};

  RenderOutcome outcome;
  auto session = renderer.create_session(*compilation.scene, settings);
  require(bool(session), "create_session failed");
  outcome.post_init_stats = native->data()->virtual_texture_cache->stats();

  require(session->render_samples(
              {.first = 0u, .count = spp, .offset = 0u, .total = spp},
              outcome.sink),
          "render_samples failed");
  outcome.final_stats = native->data()->virtual_texture_cache->stats();

  if (test_second_session) {
    auto session2 = renderer.create_session(*compilation.scene, settings);
    require(bool(session2), "second create_session failed");
    outcome.second_init_stats = native->data()->virtual_texture_cache->stats();
    io::MemoryOutputSink sink2;
    require(session2->render_samples(
                {.first = 0u, .count = spp, .offset = 0u, .total = spp}, sink2),
            "second render_samples failed");
    outcome.second_final_stats = native->data()->virtual_texture_cache->stats();
    const auto d = compare_combined_pass(outcome.sink, sink2);
    require(d.invalid_pixels == 0u && d.rmse < 1e-5,
            "second session parity mismatch: rmse=" + std::to_string(d.rmse));
  }
  return outcome;
}

[[nodiscard]] SceneSnapshot make_primary_emissive_scene() {
  SceneSnapshot scene;
  scene.images.emplace(ImageId{1u},
                       ImageDesc{.name = "primary_512.png",
                                 .color_space = ImageColorSpace::data,
                                 .alpha_type = ImageAlphaType::straight,
                                 .width = 512u,
                                 .height = 512u,
                                 .encoded_data = make_pattern_png(512u, 30u, 50u, 70u)});
  scene.materials.emplace(
      MaterialId{1u},
      MaterialDesc{.name = "PrimaryMat",
                   .shader = make_textured_emission_shader(ImageId{1u})});
  scene.geometries.emplace(
      GeometryId{1u},
      make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{1u}));
  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "PrimaryQuad", .geometry = GeometryId{1u}});
  scene.cameras.emplace(CameraId{1u},
                        CameraDesc{.name = "Cam",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 2.0f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;
  return scene;
}

[[nodiscard]] SceneSnapshot make_twobounce_scene() {
  SceneSnapshot scene;
  scene.images.emplace(ImageId{1u},
                       ImageDesc{.name = "diffuse_512.png",
                                 .color_space = ImageColorSpace::data,
                                 .alpha_type = ImageAlphaType::straight,
                                 .width = 512u,
                                 .height = 512u,
                                 .encoded_data = make_pattern_png(512u, 40u, 60u, 80u)});
  scene.images.emplace(ImageId{2u},
                       ImageDesc{.name = "indirect_256.png",
                                 .color_space = ImageColorSpace::data,
                                 .alpha_type = ImageAlphaType::straight,
                                 .width = 256u,
                                 .height = 256u,
                                 .encoded_data = make_pattern_png(256u, 90u, 40u, 50u)});
  scene.materials.emplace(
      MaterialId{1u},
      MaterialDesc{.name = "DiffFront",
                   .shader = make_textured_diffuse_shader(ImageId{1u})});
  scene.materials.emplace(
      MaterialId{2u},
      MaterialDesc{.name = "EmitBack",
                   .shader = make_textured_emission_shader(ImageId{2u}, 2.0f)});
  scene.geometries.emplace(
      GeometryId{1u},
      make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{1u}, false));
  // Large back wall behind the camera (+Z) so camera rays never hit it directly,
  // while 1-bounce diffuse rays from z=0 (+Z hemisphere) do!
  scene.geometries.emplace(
      GeometryId{2u},
      make_quad_mesh(-40.0f, 40.0f, -40.0f, 40.0f, 5.0f, MaterialId{2u}, true));
  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "DiffQuad", .geometry = GeometryId{1u}});
  scene.instances.emplace(
      InstanceId{2u},
      InstanceDesc{.name = "BackWall", .geometry = GeometryId{2u}});
  scene.cameras.emplace(CameraId{1u},
                        CameraDesc{.name = "Cam",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 2.0f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;
  return scene;
}

void verify_json_braces_balanced(std::string_view json_text) {
  bool in_str = false;
  bool esc = false;
  int brace_depth = 0;
  int bracket_depth = 0;
  for (const char ch : json_text) {
    if (in_str) {
      if (esc) {
        esc = false;
      } else if (ch == '\\') {
        esc = true;
      } else if (ch == '"') {
        in_str = false;
      }
      continue;
    }
    if (ch == '"') {
      in_str = true;
    } else if (ch == '{') {
      ++brace_depth;
    } else if (ch == '}') {
      --brace_depth;
      require(brace_depth >= 0, "unbalanced } in vt_profile.json");
    } else if (ch == '[') {
      ++bracket_depth;
    } else if (ch == ']') {
      --bracket_depth;
      require(bracket_depth >= 0, "unbalanced ] in vt_profile.json");
    }
  }
  require(!in_str && brace_depth == 0 && bracket_depth == 0,
          "unclosed JSON structure in vt_profile.json");
}

void test_preflight_discovery_and_miss_elimination(Device &device) {
  std::cout << "[Preflight Test 1] Discovery & runtime miss elimination...\n";
  const auto prim_scene = make_primary_emissive_scene();
  LuisaVirtualTextureCacheOptions base_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u,
      .enable_async_worker = true};

  const auto off_res =
      run_scene_with_options(device, prim_scene, base_opts, 32u, 32u, 1u);

  LuisaVirtualTextureCacheOptions s1_opts = base_opts;
  s1_opts.enable_preflight = true;
  s1_opts.preflight_stride = 1u;
  s1_opts.preflight_bounces = 1u;
  s1_opts.preflight_overlap_pso = true;
  const auto s1_res =
      run_scene_with_options(device, prim_scene, s1_opts, 32u, 32u, 1u);

  const auto d1 = compare_combined_pass(off_res.sink, s1_res.sink);
  require(s1_res.post_init_stats.preflight_enabled, "preflight_enabled false");
  require(s1_res.post_init_stats.preflight_overlapped, "preflight_overlapped false");
  require(s1_res.post_init_stats.preflight_tiles_discovered == 81u,
          "expected 81 discovered primary tiles at stride=1, got " +
              std::to_string(s1_res.post_init_stats.preflight_tiles_discovered));
  require(s1_res.post_init_stats.preflight_tiles_loaded == 81u,
          "expected 81 preloaded tiles at stride=1, got " +
              std::to_string(s1_res.post_init_stats.preflight_tiles_loaded));
  require(s1_res.final_stats.tiles_evicted == 0u, "expected 0 evictions");
  require(s1_res.final_stats.tiles_loaded == 0u,
          "expected 0 runtime tiles_loaded at stride=1, got " +
              std::to_string(s1_res.final_stats.tiles_loaded));
  require(d1.invalid_pixels == 0u && d1.rmse < 1e-5,
          "stride=1 parity failed: rmse=" + std::to_string(d1.rmse));

  LuisaVirtualTextureCacheOptions s2_opts = s1_opts;
  s2_opts.preflight_stride = 2u;
  const auto s2_res =
      run_scene_with_options(device, prim_scene, s2_opts, 32u, 32u, 1u);
  const auto d2 = compare_combined_pass(off_res.sink, s2_res.sink);
  require(s2_res.post_init_stats.preflight_tiles_discovered > 0u &&
              s2_res.post_init_stats.preflight_tiles_loaded > 0u,
          "stride=2 preflight discovered/loaded 0 tiles");
  require(s2_res.final_stats.tiles_loaded < off_res.final_stats.tiles_loaded,
          "stride=2 preflight did not reduce runtime tiles_loaded");
  require(d2.invalid_pixels == 0u && d2.rmse < 1e-5,
          "stride=2 parity failed: rmse=" + std::to_string(d2.rmse));

  // Verify 1-bounce discovers indirect back-wall tiles in addition to bounce-0 tiles.
  const auto bounce_scene = make_twobounce_scene();
  LuisaVirtualTextureCacheOptions b0_opts = s1_opts;
  b0_opts.preflight_bounces = 0u;
  LuisaVirtualTextureCacheOptions b1_opts = s1_opts;
  b1_opts.preflight_bounces = 1u;
  const auto b_off_res =
      run_scene_with_options(device, bounce_scene, base_opts, 32u, 32u, 4u);
  const auto b0_res =
      run_scene_with_options(device, bounce_scene, b0_opts, 32u, 32u, 4u);
  const auto b1_res =
      run_scene_with_options(device, bounce_scene, b1_opts, 32u, 32u, 4u);
  const auto db1 = compare_combined_pass(b_off_res.sink, b1_res.sink);
  std::cout << "  b0_discovered=" << b0_res.post_init_stats.preflight_tiles_discovered
            << " b1_discovered=" << b1_res.post_init_stats.preflight_tiles_discovered
            << " runtime_loaded(off/b0/b1)=" << b_off_res.final_stats.tiles_loaded
            << "/" << b0_res.final_stats.tiles_loaded
            << "/" << b1_res.final_stats.tiles_loaded
            << " rmse=" << db1.rmse << '\n';
  require(b1_res.post_init_stats.preflight_tiles_discovered >
              b0_res.post_init_stats.preflight_tiles_discovered,
          "1-bounce preflight should discover more tiles than 0-bounce");
  require(b1_res.final_stats.tiles_loaded <= b0_res.final_stats.tiles_loaded,
          "1-bounce preflight should load <= runtime tiles than 0-bounce");
  require(db1.invalid_pixels == 0u && db1.rmse < 1e-5,
          "2-bounce scene parity failed: rmse=" + std::to_string(db1.rmse));
}

void test_preflight_tight_budget_cap_and_zero_eviction(Device &device) {
  std::cout << "[Preflight Test 2] Tight budget cap & zero eviction...\n";
  const auto bounce_scene = make_twobounce_scene();
  LuisaVirtualTextureCacheOptions base_tight{
      .enabled = true,
      .gpu_cache_budget_bytes = 256ull * 1024ull,
      .physical_tile_size = 64u,
      .enable_async_worker = true};
  const auto off_res =
      run_scene_with_options(device, bounce_scene, base_tight, 16u, 16u, 4u);

  for (const std::uint32_t bounces : {1u, 0u}) {
    LuisaVirtualTextureCacheOptions pf_tight = base_tight;
    pf_tight.enable_preflight = true;
    pf_tight.preflight_stride = 1u;
    pf_tight.preflight_bounces = bounces;
    pf_tight.preflight_overlap_pso = true;

    const auto pf_res =
        run_scene_with_options(device, bounce_scene, pf_tight, 16u, 16u, 4u);
    const auto d = compare_combined_pass(off_res.sink, pf_res.sink);
    const auto cap = pf_res.post_init_stats.byte4_cache_capacity;
    std::cout << "  bounces=" << bounces
              << " cap=" << cap
              << " discovered=" << pf_res.post_init_stats.preflight_tiles_discovered
              << " pf_loaded=" << pf_res.post_init_stats.preflight_tiles_loaded
              << " pf_evicted=" << pf_res.post_init_stats.tiles_evicted
              << " rmse=" << d.rmse << '\n';
    require(cap == 16u, "expected 16-tile capacity at 256 KiB");
    require(pf_res.post_init_stats.preflight_tiles_discovered > cap,
            "working set must exceed tight capacity");
    require(pf_res.post_init_stats.preflight_tiles_loaded <= cap &&
                pf_res.post_init_stats.preflight_tiles_loaded > 0u,
            "preflight_tiles_loaded must respect capacity");
    require(pf_res.post_init_stats.tiles_evicted == 0u,
            "preflight itself must perform zero evictions");
    require(pf_res.post_init_stats.byte4_peak_resident <= cap,
            "peak resident exceeded budget");
    require(d.invalid_pixels == 0u && d.rmse < 1e-4,
            "tight budget render parity failed: rmse=" + std::to_string(d.rmse));
  }
}

void test_preflight_pso_overlap_and_profile_json(Device &device) {
  std::cout << "[Preflight Test 3] PSO overlap equivalence, cache reuse & profile JSON...\n";
  const auto temp_dir =
      std::filesystem::temp_directory_path() / "psycles_vt_preflight_profile_test";
  std::error_code ec;
  std::filesystem::create_directories(temp_dir, ec);
  const auto prof_sync = (temp_dir / "vt_profile_sync.json").string();
  const auto prof_ovl = (temp_dir / "vt_profile_ovl.json").string();

  const auto bounce_scene = make_twobounce_scene();
  LuisaVirtualTextureCacheOptions opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u,
      .enable_async_worker = true,
      .enable_preflight = true,
      .preflight_stride = 2u,
      .preflight_bounces = 1u,
      .preflight_overlap_pso = false};

  RenderOutcome sync_res;
  {
    ScopedEnvVar env_prof{"PSYCLES_VT_PROFILE_JSON", prof_sync};
    sync_res = run_scene_with_options(device, bounce_scene, opts, 32u, 32u, 2u);
  }

  RenderOutcome ovl_res;
  opts.preflight_overlap_pso = true;
  {
    ScopedEnvVar env_prof{"PSYCLES_VT_PROFILE_JSON", prof_ovl};
    ovl_res = run_scene_with_options(device, bounce_scene, opts, 32u, 32u, 2u, true);
  }

  const auto d = compare_combined_pass(sync_res.sink, ovl_res.sink);
  std::cout << "  sync pso=" << sync_res.post_init_stats.pso_compile_ms
            << "ms pf=" << sync_res.post_init_stats.preflight_total_ms
            << "ms | ovl pso=" << ovl_res.post_init_stats.pso_compile_ms
            << "ms pf=" << ovl_res.post_init_stats.preflight_total_ms
            << "ms (2nd compile=" << ovl_res.second_init_stats.preflight_compile_ms
            << "ms) rmse=" << d.rmse << '\n';

  require(d.invalid_pixels == 0u && d.rmse < 1e-5,
          "sync vs overlap parity failed: rmse=" + std::to_string(d.rmse));
  require(ovl_res.post_init_stats.pso_compile_ms > 0.0,
          "expected pso_compile_ms > 0");
  require(ovl_res.post_init_stats.preflight_total_ms > 0.0,
          "expected preflight_total_ms > 0");
  require(ovl_res.second_init_stats.preflight_compile_ms <=
              std::max(1.0, ovl_res.post_init_stats.preflight_compile_ms),
          "expected cached preflight compile on second session");

  {
    std::ifstream ifs{prof_ovl, std::ios::binary};
    require(ifs.is_open(), "missing vt_profile_ovl.json");
    std::string text{std::istreambuf_iterator<char>(ifs),
                     std::istreambuf_iterator<char>()};
    verify_json_braces_balanced(text);
    const auto pf_pos = text.find("\"preflight_events\": [");
    const auto pso_pos = text.find("\"pso_compile_events\": [");
    require(pf_pos != std::string::npos && pso_pos != std::string::npos,
            "missing preflight_events or pso_compile_events array");
    require(text.find("\"cat\": \"preflight\"") != std::string::npos,
            "expected non-empty preflight_events entries");
    require(text.find("\"cat\": \"pso\"") != std::string::npos,
            "expected non-empty pso_compile_events entries");
  }
  std::filesystem::remove_all(temp_dir, ec);
}

[[nodiscard]] std::vector<std::uint8_t>
make_pattern_png_checker_alpha(std::uint32_t s, std::uint8_t r_bias,
                               std::uint8_t g_bias, std::uint8_t b_bias) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(s) * s * 4u);
  for (std::uint32_t y = 0u; y < s; ++y) {
    for (std::uint32_t x = 0u; x < s; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * s + x) * 4u;
      const auto tx = x / 60u;
      const auto ty = y / 60u;
      px[i + 0u] = static_cast<std::uint8_t>(r_bias + ((tx * 23u + x / 3u) & 0x7fu));
      px[i + 1u] = static_cast<std::uint8_t>(g_bias + ((ty * 29u + y / 3u) & 0x7fu));
      px[i + 2u] = static_cast<std::uint8_t>(b_bias + (((tx + ty) * 19u) & 0x7fu));
      px[i + 3u] = (((x / 4u) ^ (y / 4u)) & 1u) ? 255u : 0u;
    }
  }
  return encode_png_rgba8(s, s, px);
}

void test_preflight_transparent_mirror_mix_and_runtime_bounces(Device &device) {
  std::cout << "[Preflight Test 4] Transparent pass-through, glossy mirror, MixShader flattening, & runtime bounce reuse...\n";
  const LuisaVirtualTextureCacheOptions base_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u,
      .enable_async_worker = true};

  // 1. Transparent quad in front of a textured emission quad: b=0 discovers behind-glass tiles.
  {
    SceneSnapshot scene;
    scene.images.emplace(ImageId{1u},
                         ImageDesc{.name = "behind_glass_256.png",
                                   .color_space = ImageColorSpace::data,
                                   .alpha_type = ImageAlphaType::straight,
                                   .width = 256u,
                                   .height = 256u,
                                   .encoded_data = make_pattern_png(256u, 50u, 80u, 30u)});
    {
      ShaderGraph glass;
      const auto tr = glass.add_node(node_type::transparent_bsdf, "Transparent");
      require(glass.set_input(tr, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})),
              "failed to set Transparent color");
      glass.set_root(ShaderDomain::surface, OutputRef{tr, "Closure"});
      scene.materials.emplace(MaterialId{1u},
                              MaterialDesc{.name = "GlassPane", .shader = std::move(glass)});
    }
    scene.materials.emplace(
        MaterialId{2u},
        MaterialDesc{.name = "BehindEmit",
                     .shader = make_textured_emission_shader(ImageId{1u}, 1.0f)});
    scene.geometries.emplace(
        GeometryId{1u}, make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, MaterialId{1u}));
    scene.geometries.emplace(
        GeometryId{2u}, make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{2u}));
    scene.instances.emplace(InstanceId{1u}, InstanceDesc{.name = "GlassQuad", .geometry = GeometryId{1u}});
    scene.instances.emplace(InstanceId{2u}, InstanceDesc{.name = "BackQuad", .geometry = GeometryId{2u}});
    scene.cameras.emplace(CameraId{1u},
                          CameraDesc{.name = "Cam",
                                     .projection = CameraProjection::orthographic,
                                     .transform = translated(0.0f, 0.0f, 3.0f),
                                     .orthographic_scale = 2.0f});
    scene.active_camera = CameraId{1u};
    scene.world_sampling = WorldSampling::none;

    const auto off_res = run_scene_with_options(device, scene, base_opts, 32u, 32u, 1u);
    LuisaVirtualTextureCacheOptions pf_b0 = base_opts;
    pf_b0.enable_preflight = true;
    pf_b0.preflight_stride = 1u;
    pf_b0.preflight_bounces = 0u;
    pf_b0.preflight_overlap_pso = true;
    const auto pf_res = run_scene_with_options(device, scene, pf_b0, 32u, 32u, 1u);
    const auto d = compare_combined_pass(off_res.sink, pf_res.sink);
    std::cout << "  [transparent b0] discovered=" << pf_res.post_init_stats.preflight_tiles_discovered
              << " rt_loaded=" << pf_res.final_stats.tiles_loaded << " rmse=" << d.rmse << '\n';
    require(pf_res.post_init_stats.preflight_tiles_discovered == 25u,
            "expected all 25 behind-glass tiles discovered at b=0, got " +
                std::to_string(pf_res.post_init_stats.preflight_tiles_discovered));
    require(pf_res.final_stats.tiles_loaded == 0u,
            "expected 0 runtime tiles_loaded behind transparent quad, got " +
                std::to_string(pf_res.final_stats.tiles_loaded));
    require(d.invalid_pixels == 0u && d.rmse < 1e-5, "transparent pass parity failed");
  }

  // 2. Pure glossy mirror facing camera reflects to a textured wall behind the camera.
  {
    SceneSnapshot scene;
    scene.images.emplace(ImageId{1u},
                         ImageDesc{.name = "mirror_wall_256.png",
                                   .color_space = ImageColorSpace::data,
                                   .alpha_type = ImageAlphaType::straight,
                                   .width = 256u,
                                   .height = 256u,
                                   .encoded_data = make_pattern_png(256u, 90u, 30u, 70u)});
    {
      ShaderGraph mirror;
      const auto gl = mirror.add_node(node_type::glossy_bsdf, "MirrorGlossy");
      require(mirror.set_property(gl, "Distribution", SocketValue::string("GGX")) &&
                  mirror.set_input(gl, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})) &&
                  mirror.set_input(gl, "Roughness", SocketValue::floating(0.0f)),
              "failed to configure Glossy mirror");
      mirror.set_root(ShaderDomain::surface, OutputRef{gl, "Closure"});
      scene.materials.emplace(MaterialId{1u},
                              MaterialDesc{.name = "MirrorMat", .shader = std::move(mirror)});
    }
    scene.materials.emplace(
        MaterialId{2u},
        MaterialDesc{.name = "BackEmit",
                     .shader = make_textured_emission_shader(ImageId{1u}, 1.0f)});
    scene.geometries.emplace(
        GeometryId{1u}, make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{1u}, false));
    scene.geometries.emplace(
        GeometryId{2u}, make_quad_mesh(-4.0f, 4.0f, -4.0f, 4.0f, 5.0f, MaterialId{2u}, true));
    scene.instances.emplace(InstanceId{1u}, InstanceDesc{.name = "MirrorQuad", .geometry = GeometryId{1u}});
    scene.instances.emplace(InstanceId{2u}, InstanceDesc{.name = "ReflWall", .geometry = GeometryId{2u}});
    scene.cameras.emplace(CameraId{1u},
                          CameraDesc{.name = "Cam",
                                     .projection = CameraProjection::orthographic,
                                     .transform = translated(0.0f, 0.0f, 3.0f),
                                     .orthographic_scale = 2.0f});
    scene.active_camera = CameraId{1u};
    scene.world_sampling = WorldSampling::none;

    LuisaVirtualTextureCacheOptions b0 = base_opts;
    b0.enable_preflight = true;
    b0.preflight_stride = 1u;
    b0.preflight_bounces = 0u;
    LuisaVirtualTextureCacheOptions b1 = b0;
    b1.preflight_bounces = 1u;
    const auto off_res = run_scene_with_options(device, scene, base_opts, 32u, 32u, 2u);
    const auto b0_res = run_scene_with_options(device, scene, b0, 32u, 32u, 2u);
    const auto b1_res = run_scene_with_options(device, scene, b1, 32u, 32u, 2u);
    const auto d = compare_combined_pass(off_res.sink, b1_res.sink);
    std::cout << "  [mirror] b0_disc=" << b0_res.post_init_stats.preflight_tiles_discovered
              << " b1_disc=" << b1_res.post_init_stats.preflight_tiles_discovered
              << " b1_rt_ld=" << b1_res.final_stats.tiles_loaded << " rmse=" << d.rmse << '\n';
    require(b0_res.post_init_stats.preflight_tiles_discovered == 0u,
            "b0 should discover 0 tiles on untextured mirror");
    require(b1_res.post_init_stats.preflight_tiles_discovered > 0u,
            "b1 mirror reflection should discover reflected back-wall tiles");
    require(b1_res.final_stats.tiles_loaded < b0_res.final_stats.tiles_loaded,
            "b1 mirror preflight should reduce runtime tiles_loaded vs b0");
    require(d.invalid_pixels == 0u && d.rmse < 1e-5, "mirror reflection parity failed");
  }

  // 3. MixShader(Factor=texA.Alpha, Diffuse(texB), Diffuse(texC)) flattening + runtime bounce reuse.
  {
    SceneSnapshot scene;
    scene.images.emplace(ImageId{1u},
                         ImageDesc{.name = "mask_256.png",
                                   .color_space = ImageColorSpace::data,
                                   .alpha_type = ImageAlphaType::straight,
                                   .width = 256u,
                                   .height = 256u,
                                   .encoded_data = make_pattern_png_checker_alpha(256u, 10u, 20u, 30u)});
    scene.images.emplace(ImageId{2u},
                         ImageDesc{.name = "branchA_256.png",
                                   .color_space = ImageColorSpace::data,
                                   .alpha_type = ImageAlphaType::straight,
                                   .width = 256u,
                                   .height = 256u,
                                   .encoded_data = make_pattern_png(256u, 40u, 70u, 90u)});
    scene.images.emplace(ImageId{3u},
                         ImageDesc{.name = "branchB_256.png",
                                   .color_space = ImageColorSpace::data,
                                   .alpha_type = ImageAlphaType::straight,
                                   .width = 256u,
                                   .height = 256u,
                                   .encoded_data = make_pattern_png(256u, 80u, 50u, 20u)});
    {
      ShaderGraph mixg;
      const auto tA = add_uv_image_node(mixg, ImageId{1u});
      const auto tB = add_uv_image_node(mixg, ImageId{2u});
      const auto tC = add_uv_image_node(mixg, ImageId{3u});
      const auto e1 = mixg.add_node(node_type::emission, "EmitA");
      const auto e2 = mixg.add_node(node_type::emission, "EmitB");
      const auto mx = mixg.add_node(node_type::mix_closure, "MixMat");
      require(mixg.set_input(e1, "Strength", SocketValue::floating(1.0f)) &&
                  mixg.set_input(e2, "Strength", SocketValue::floating(1.0f)) &&
                  mixg.connect({tB, "Color"}, e1, "Color") &&
                  mixg.connect({tC, "Color"}, e2, "Color") &&
                  mixg.connect({tA, "Alpha"}, mx, "Factor") &&
                  mixg.connect({e1, "Closure"}, mx, "A") &&
                  mixg.connect({e2, "Closure"}, mx, "B"),
              "failed to wire MixShader test graph");
      mixg.set_root(ShaderDomain::surface, OutputRef{mx, "Closure"});
      scene.materials.emplace(MaterialId{1u},
                              MaterialDesc{.name = "MixedMat", .shader = std::move(mixg)});
    }
    scene.geometries.emplace(
        GeometryId{1u}, make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{1u}));
    scene.instances.emplace(InstanceId{1u}, InstanceDesc{.name = "MixQuad", .geometry = GeometryId{1u}});
    scene.cameras.emplace(CameraId{1u},
                          CameraDesc{.name = "Cam",
                                     .projection = CameraProjection::orthographic,
                                     .transform = translated(0.0f, 0.0f, 3.0f),
                                     .orthographic_scale = 2.0f});
    scene.active_camera = CameraId{1u};
    scene.world_sampling = WorldSampling::none;

    // Reference render on its own CompiledScene so its VT cache stays isolated.
    const auto off_res = run_scene_with_options(device, scene, base_opts, 32u, 32u, 1u);
    require(off_res.final_stats.tiles_loaded == 75u,
            "expected baseline MixShader render to load 75 tiles, got " +
                std::to_string(off_res.final_stats.tiles_loaded));

    // Reuse one clean CompiledScene across preflight bounces=0, 1, 2.
    LuisaPathTracerOptions pf_opt{
        .next_event_estimation = true,
        .enable_fast_math = true,
        .scheduler = LuisaPathScheduler::wavefront_staged,
        .wavefront_frame_capacity = 4096u,
        .max_samples_per_dispatch = 2u,
        .virtual_texture_cache = base_opts};
    pf_opt.virtual_texture_cache.enable_preflight = true;
    pf_opt.virtual_texture_cache.preflight_stride = 1u;
    pf_opt.virtual_texture_cache.preflight_overlap_pso = true;

    RenderSettings s;
    s.full_extent = {32u, 32u};
    s.window = {.x = 0u, .y = 0u, .width = 32u, .height = 32u};
    s.seed = 42u;
    s.passes = {{.kind = PassKind::combined, .name = "Combined", .channels = 4u}};

    LuisaPathTracerBackend init_backend{Device{device.impl_shared()}, pf_opt};
    const auto compilation = init_backend.compile_scene(scene);
    require(compilation.ok(), "MixShader scene compilation failed");
    const auto *native =
        dynamic_cast<const LuisaCompiledScene *>(compilation.scene.get());
    require(native != nullptr && native->data()->virtual_texture_cache != nullptr,
            "missing VT cache");

    std::array<double, 3> compile_ms{};
    for (std::uint32_t b = 0u; b < 3u; ++b) {
      pf_opt.virtual_texture_cache.preflight_bounces = b;
      LuisaPathTracerBackend pf_renderer{Device{device.impl_shared()}, pf_opt};
      auto sess = pf_renderer.create_session(*compilation.scene, s);
      require(bool(sess), "preflight reuse session failed");
      const auto init_st = native->data()->virtual_texture_cache->stats();
      compile_ms[b] = init_st.preflight_compile_ms;
      io::MemoryOutputSink pf_sink;
      require(sess->render_samples({.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, pf_sink),
              "preflight reuse render failed");
      const auto final_st = native->data()->virtual_texture_cache->stats();
      const auto d = compare_combined_pass(off_res.sink, pf_sink);
      std::cout << "  [mix+reuse b=" << b << "] compile_ms=" << compile_ms[b]
                << " discovered=" << init_st.preflight_tiles_discovered
                << " rt_ld=" << final_st.tiles_loaded
                << " rmse=" << d.rmse << '\n';
      if (b == 0u) {
        require(init_st.preflight_tiles_discovered == 75u,
                "MixShader flattening should discover all 75 tiles across 3 textures, got " +
                    std::to_string(init_st.preflight_tiles_discovered));
      } else {
        require(compile_ms[b] < 5.0,
                "expected preflight shader cache hit across runtime bounces (< 5.0 ms), got " +
                    std::to_string(compile_ms[b]));
      }
      require(final_st.tiles_loaded == 0u,
              "expected 0 runtime tiles_loaded after MixShader preflight, got " +
                  std::to_string(final_st.tiles_loaded));
      require(d.invalid_pixels == 0u && d.rmse < 1e-5, "MixShader parity failed");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "vk"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    std::cout << "Running Psycles VT Preflight tests on backend: " << backend << '\n';
    test_preflight_discovery_and_miss_elimination(device);
    test_preflight_tight_budget_cap_and_zero_eviction(device);
    test_preflight_pso_overlap_and_profile_json(device);
    test_preflight_transparent_mirror_mix_and_runtime_bounces(device);
    std::cout << "ALL 4 VT PREFLIGHT TESTS PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
