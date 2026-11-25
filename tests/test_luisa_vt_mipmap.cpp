#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#if defined(PSYCLES_WITH_OPENIMAGEIO)
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>
#endif

#include "path_tracer_internal.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_cache_lod.h"

#include <luisa/dsl/syntax.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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

[[nodiscard]] Mat4f translated(float tx, float ty, float tz) noexcept {
  Mat4f m{};
  m.elements[12u] = tx;
  m.elements[13u] = ty;
  m.elements[14u] = tz;
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
  if (auto out = OIIO::ImageOutput::create("vt_mip_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_mip_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() / "psycles_vt_mip_test_tmp.png";
  auto out = OIIO::ImageOutput::create(temp_path.string());
  require(out != nullptr, "OIIO PNG output create failed");
  require(out->open(temp_path.string(), spec), "OIIO PNG open failed");
  require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
          "OIIO PNG write failed");
  require(out->close(), "OIIO PNG close failed");
  std::ifstream ifs{temp_path, std::ios::binary | std::ios::ate};
  require(ifs.is_open(), "temp PNG reopen failed");
  const auto sz = static_cast<std::size_t>(ifs.tellg());
  std::vector<std::uint8_t> bytes(sz);
  ifs.seekg(0);
  ifs.read(reinterpret_cast<char *>(bytes.data()),
           static_cast<std::streamsize>(sz));
  ifs.close();
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  return bytes;
#else
  (void)width;
  (void)height;
  throw std::runtime_error{"PSYCLES_WITH_OPENIMAGEIO is required for test PNG"};
#endif
}

// High-frequency checker/stripe pattern whose mip 0 averages cleanly to smoother
// intermediate colors at coarser mip levels so mip 0 vs isotropic are distinctly
// separable while remaining exact against direct multi-mip reference.
[[nodiscard]] std::vector<std::uint8_t> make_mipmap_pattern_png(
    std::uint32_t width, std::uint32_t height,
    std::uint8_t r_bias = 20u, std::uint8_t g_bias = 30u,
    std::uint8_t b_bias = 40u) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(width) * height * 4u);
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      const std::size_t o = (static_cast<std::size_t>(y) * width + x) * 4u;
      const bool fine_chk = ((x ^ y) & 1u) != 0u;
      const bool coarse_chk = (((x >> 4u) ^ (y >> 4u)) & 1u) != 0u;
      const float u_grad = static_cast<float>(x) / static_cast<float>(std::max(width, 1u));
      const float v_grad = static_cast<float>(y) / static_cast<float>(std::max(height, 1u));
      const std::uint32_t r =
          (fine_chk ? 220u : 20u) + static_cast<std::uint32_t>(u_grad * 30.0f) + r_bias;
      const std::uint32_t g =
          (coarse_chk ? 180u : 45u) + static_cast<std::uint32_t>(v_grad * 40.0f) + g_bias;
      const std::uint32_t b =
          ((x + y) % 3u == 0u ? 200u : 40u) + b_bias;
      px[o + 0u] = static_cast<std::uint8_t>(std::min(r, 255u));
      px[o + 1u] = static_cast<std::uint8_t>(std::min(g, 255u));
      px[o + 2u] = static_cast<std::uint8_t>(std::min(b, 255u));
      px[o + 3u] = 255u;
    }
  }
  return encode_png_rgba8(width, height, px);
}

[[nodiscard]] TriangleMeshDesc make_tilted_foreshortened_quad(
    MaterialId material) {
  // Large quad sloping away from camera (-Z), creating magnification near
  // bottom edge and strong minification toward the far top edge.
  TriangleMeshDesc mesh;
  mesh.positions = {
      {-2.2f, -1.6f, 0.2f},
      { 2.2f, -1.6f, 0.2f},
      { 2.2f,  4.0f, -18.0f},
      {-2.2f,  4.0f, -18.0f}};
  const Vec3f n{0.0f, 0.96f, 0.28f};
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
  mesh.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
  mesh.triangle_material_slots = {0u, 0u};
  return mesh;
}

[[nodiscard]] ShaderGraph make_mapped_image_emission_shader(
    ImageId image_id,
    Vec3f location,
    Vec3f rotation,
    Vec3f scale,
    std::string_view interpolation,
    std::string_view extension) {
  ShaderGraph graph;
  const auto tc = graph.add_node(node_type::texture_coordinate, "TexCoord");
  const auto p2v = graph.add_node(node_type::point_to_vector, "UvToVec");
  const auto map = graph.add_node(node_type::mapping, "Mapping");
  const auto tex = graph.add_node(node_type::image_texture, "Image");
  const auto em = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_property(map, "VectorType", SocketValue::string("POINT")) &&
          graph.set_input(map, "Location", SocketValue::vector(location)) &&
          graph.set_input(map, "Rotation", SocketValue::vector(rotation)) &&
          graph.set_input(map, "Scale", SocketValue::vector(scale)) &&
          graph.set_property(tex, "Image",
                             SocketValue::unsigned_integer(image_id.value)) &&
          graph.set_property(tex, "Interpolation",
                             SocketValue::string(std::string{interpolation})) &&
          graph.set_property(tex, "Extension",
                             SocketValue::string(std::string{extension})) &&
          graph.set_property(tex, "Projection", SocketValue::string("FLAT")) &&
          graph.set_property(tex, "ProjectionBlend",
                             SocketValue::floating(0.0f)) &&
          graph.set_property(tex, "ColorSpace",
                             SocketValue::string("Non-Color")) &&
          graph.set_property(tex, "UnassociateAlpha",
                             SocketValue::boolean(false)) &&
          graph.connect({tc, "UV"}, p2v, "Point") &&
          graph.connect({p2v, "Vector"}, map, "Vector") &&
          graph.connect({map, "Vector"}, tex, "Vector") &&
          graph.connect({tex, "Color"}, em, "Color") &&
          graph.set_input(em, "Strength", SocketValue::floating(1.0f)),
      "failed to configure mapped image emission shader");
  graph.set_root(ShaderDomain::surface, OutputRef{em, "Closure"});
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
    const double diff = std::abs(
        static_cast<double>(a->pixels[i]) - static_cast<double>(b->pixels[i]));
    sum_sq += diff * diff;
    metrics.max_abs = std::max(metrics.max_abs, diff);
    ++count;
  }
  metrics.rmse = count > 0u ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
  return metrics;
}

struct RenderOutcome {
  io::MemoryOutputSink sink;
  VirtualTextureCacheStats post_init_stats;
  VirtualTextureCacheStats final_stats;
  bool has_vt{false};
};

[[nodiscard]] RenderOutcome run_scene(
    Device &device,
    const SceneSnapshot &scene,
    const LuisaVirtualTextureCacheOptions &vt_options,
    LuisaTextureMipMode direct_mip_mode,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t spp) {
  LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .enable_fast_math = true,
       .scheduler = LuisaPathScheduler::wavefront_staged,
       .wavefront_frame_capacity = 8192u,
       .virtual_texture_cache = vt_options,
       .direct_texture_mip_mode = direct_mip_mode}};
  const auto compilation = renderer.compile_scene(scene);
  if (!compilation.ok()) {
    for (const auto &d : compilation.diagnostics) {
      std::cerr << "Compile error: " << d.message << '\n';
    }
  }
  require(compilation.ok(), "compile_scene failed");

  RenderSettings settings;
  settings.full_extent = {width, height};
  settings.window = {.x = 0u, .y = 0u, .width = width, .height = height};
  settings.integrator.max_bounces = 4u;
  settings.integrator.direct_light_sampling =
      DirectLightSampling::multiple_importance_sampling;
  settings.seed = 42u;
  settings.passes = {{.kind = PassKind::combined, .name = "Combined", .channels = 4u}};

  auto session = renderer.create_session(*compilation.scene, settings);
  require(bool(session), "create_session returned null");

  RenderOutcome out;
  const auto *native =
      dynamic_cast<const LuisaCompiledScene *>(compilation.scene.get());
  if (native != nullptr && native->data()->virtual_texture_cache != nullptr) {
    out.has_vt = true;
    out.post_init_stats = native->data()->virtual_texture_cache->stats();
  }
  require(session->render_samples(
              {.first = 0u, .count = spp, .offset = 0u, .total = spp}, out.sink),
          "render_samples failed");
  if (native != nullptr && native->data()->virtual_texture_cache != nullptr) {
    out.final_stats = native->data()->virtual_texture_cache->stats();
  }
  return out;
}

void test_lod_math_and_invalid_gradient_fallback(Device &device) {
  std::cout << "[Mip Test 1] LOD math & invalid gradient fallback...\n";
  struct Case {
    luisa::float4 grads; // dux, duy, dvx, dvy
    luisa::uint width;
    luisa::uint height;
    luisa::uint mip_count;
    luisa::uint is_closest;
    float expected_lod;
    luisa::uint expected_mip0;
    luisa::uint expected_mip1;
    float expected_frac;
  };
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<Case> cases{
      {{qnan, 0.0f, 0.0f, 0.0f}, 512u, 128u, 10u, 0u, 0.0f, 0u, 0u, 0.0f},
      {{inf, 0.0f, 0.0f, 0.0f}, 512u, 128u, 10u, 0u, 0.0f, 0u, 0u, 0.0f},
      {{0.0f, -inf, 0.0f, 0.0f}, 512u, 128u, 10u, 0u, 0.0f, 0u, 0u, 0.0f},
      {{0.0f, 0.0f, 0.0f, 0.0f}, 512u, 128u, 10u, 0u, 0.0f, 0u, 0u, 0.0f},
      // Magnification -> clamped to mip 0
      {{0.5f / 512.0f, 0.0f, 0.0f, 0.25f / 128.0f}, 512u, 128u, 10u, 0u, 0.0f, 0u, 0u, 0.0f},
      // Separate width vs height scaling on 512x128:
      {{8.0f / 512.0f, 0.0f, 0.0f, 1.0f / 128.0f}, 512u, 128u, 10u, 0u, 3.0f, 3u, 3u, 0.0f},
      {{1.0f / 512.0f, 0.0f, 0.0f, 8.0f / 128.0f}, 512u, 128u, 10u, 0u, 3.0f, 3u, 3u, 0.0f},
      // Fractional LOD = 1.5 (max_len2 = 8 -> log2(8)*0.5 = 1.5)
      {{std::sqrt(8.0f) / 512.0f, 0.0f, 0.0f, 1.0f / 128.0f}, 512u, 128u, 10u, 0u, 1.5f, 1u, 2u, 0.5f},
      // Closest rounds 1.6 -> 2
      {{std::pow(2.0f, 1.6f) / 512.0f, 0.0f, 0.0f, 0.0f}, 512u, 128u, 10u, 1u, 1.6f, 2u, 2u, 0.0f},
      // Clamp to max_mip (mip_count - 1)
      {{4096.0f / 512.0f, 0.0f, 0.0f, 0.0f}, 512u, 128u, 10u, 0u, 9.0f, 9u, 9u, 0.0f},
  };
  const std::uint32_t n = static_cast<std::uint32_t>(cases.size());
  auto grad_buf = device.create_buffer<luisa::float4>(n);
  auto meta_buf = device.create_buffer<luisa::uint4>(n);
  auto out_buf = device.create_buffer<luisa::float4>(n);
  std::vector<luisa::float4> host_g(n);
  std::vector<luisa::uint4> host_m(n);
  for (std::uint32_t i = 0u; i < n; ++i) {
    host_g[i] = cases[i].grads;
    host_m[i] = luisa::make_uint4(
        cases[i].width, cases[i].height, cases[i].mip_count, cases[i].is_closest);
  }
  auto stream = device.create_stream();
  auto kernel = device.compile<1>([&](BufferFloat4 gbuf, BufferUInt4 mbuf, BufferFloat4 obuf) noexcept {
    const auto i = dispatch_x();
    const auto g = gbuf.read(i);
    const auto m = mbuf.read(i);
    const auto duv_dx = make_float2(g.x, g.z);
    const auto duv_dy = make_float2(g.y, g.w);
    const auto lod = vt_compute_isotropic_lod(duv_dx, duv_dy, m.x, m.y, m.z);
    const auto sel = vt_select_mip_levels(lod, m.z, m.w != 0u);
    obuf.write(i, make_float4(lod, cast<float>(sel.mip0), cast<float>(sel.mip1), sel.frac));
  });
  std::vector<luisa::float4> host_out(n);
  stream << grad_buf.copy_from(host_g.data())
         << meta_buf.copy_from(host_m.data())
         << kernel(grad_buf, meta_buf, out_buf).dispatch(n)
         << out_buf.copy_to(host_out.data())
         << synchronize();
  for (std::uint32_t i = 0u; i < n; ++i) {
    const auto &c = cases[i];
    const auto got = host_out[i];
    require(std::abs(got.x - c.expected_lod) < 1e-4f,
            "LOD mismatch at case " + std::to_string(i));
    require(static_cast<std::uint32_t>(std::lround(got.y)) == c.expected_mip0,
            "mip0 mismatch at case " + std::to_string(i));
    require(static_cast<std::uint32_t>(std::lround(got.z)) == c.expected_mip1,
            "mip1 mismatch at case " + std::to_string(i));
    require(std::abs(got.w - c.expected_frac) < 1e-4f,
            "frac mismatch at case " + std::to_string(i));
  }
}

void test_nonsquare_and_npot_with_transforms(Device &device) {
  std::cout << "[Mip Test 2] Non-square 512x128 & NPOT 257x129 with transforms vs direct reference...\n";
  struct ShapeCase {
    std::string name;
    std::uint32_t w;
    std::uint32_t h;
  };
  const std::vector<ShapeCase> shapes{
      {"nonsq_512x128.png", 512u, 128u},
      {"npot_257x129.png", 257u, 129u}};
  for (const auto &sh : shapes) {
    SceneSnapshot scene;
    scene.images.emplace(
        ImageId{1u},
        ImageDesc{.name = sh.name,
                  .color_space = ImageColorSpace::srgb,
                  .alpha_type = ImageAlphaType::straight,
                  .width = sh.w,
                  .height = sh.h,
                  .encoded_data = make_mipmap_pattern_png(sh.w, sh.h)});
    scene.materials.emplace(
        MaterialId{1u},
        MaterialDesc{
            .name = "MappedEmissive",
            .shader = make_mapped_image_emission_shader(
                ImageId{1u}, {0.1f, -0.2f, 0.0f}, {0.0f, 0.0f, 0.35f},
                {3.5f, 2.0f, 1.0f}, "Linear", "REPEAT")});
    scene.geometries.emplace(
        GeometryId{1u}, make_tilted_foreshortened_quad(MaterialId{1u}));
    scene.instances.emplace(
        InstanceId{1u},
        InstanceDesc{.name = "TiltedQuad", .geometry = GeometryId{1u}});
    scene.cameras.emplace(
        CameraId{1u},
        CameraDesc{.name = "PerspCam",
                   .projection = CameraProjection::perspective,
                   .transform = translated(0.0f, 0.0f, 2.5f),
                   .field_of_view = 0.85f});
    scene.active_camera = CameraId{1u};
    scene.world_sampling = WorldSampling::none;

    LuisaVirtualTextureCacheOptions vt_off{.enabled = false};
    LuisaVirtualTextureCacheOptions vt_m0{
        .enabled = true,
        .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
        .physical_tile_size = 64u,
        .enable_async_worker = true,
        .mip_mode = LuisaTextureMipMode::mip0};
    LuisaVirtualTextureCacheOptions vt_iso = vt_m0;
    vt_iso.mip_mode = LuisaTextureMipMode::isotropic;

    const auto dir_m0 = run_scene(
        device, scene, vt_off, LuisaTextureMipMode::mip0, 128u, 128u, 8u);
    const auto dir_iso = run_scene(
        device, scene, vt_off, LuisaTextureMipMode::isotropic, 128u, 128u, 8u);
    const auto vt_m0_res = run_scene(
        device, scene, vt_m0, LuisaTextureMipMode::mip0, 128u, 128u, 8u);
    const auto vt_iso_res = run_scene(
        device, scene, vt_iso, LuisaTextureMipMode::isotropic, 128u, 128u, 8u);

    const auto d_m0 = compare_combined_pass(dir_m0.sink, vt_m0_res.sink);
    const auto d_iso = compare_combined_pass(dir_iso.sink, vt_iso_res.sink);
    const auto d_m0_vs_iso = compare_combined_pass(dir_m0.sink, dir_iso.sink);
    std::cout << "  [" << sh.name << "] rmse(vt_m0/dir_m0)=" << d_m0.rmse
              << " rmse(vt_iso/dir_iso)=" << d_iso.rmse
              << " diff(m0 vs iso)=" << d_m0_vs_iso.rmse
              << " unique(m0/iso)=" << vt_m0_res.final_stats.unique_pages_loaded
              << "/" << vt_iso_res.final_stats.unique_pages_loaded << '\n';
    require(d_m0.invalid_pixels == 0u && d_m0.rmse < 1e-4,
            "mip0 VT vs direct reference failed on " + sh.name);
    require(d_iso.invalid_pixels == 0u && d_iso.rmse < 0.005,
            "isotropic VT vs direct reference failed on " + sh.name);
    require(d_m0_vs_iso.rmse > 0.03,
            "expected distinct filtering between mip0 and isotropic on " + sh.name);
    require(vt_iso_res.final_stats.active_mip_mode == LuisaTextureMipMode::isotropic,
            "active_mip_mode mismatch");
    std::uint64_t coarse_unique = 0ull;
    for (std::size_t m = 1u; m < vt_iso_res.final_stats.unique_pages_per_mip.size(); ++m) {
      coarse_unique += vt_iso_res.final_stats.unique_pages_per_mip[m];
    }
    require(coarse_unique > 0ull,
            "expected coarse mips (>0) loaded on foreshortened " + sh.name);
  }
}

void test_tile_boundaries_extensions_and_filters(Device &device) {
  std::cout << "[Mip Test 3] Tile boundaries, extensions, and interpolation filters across mips...\n";
  SceneSnapshot base_scene;
  base_scene.images.emplace(
      ImageId{1u},
      ImageDesc{.name = "multi_tile_256x192.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 256u,
                .height = 192u,
                .encoded_data = make_mipmap_pattern_png(256u, 192u, 15u, 25u, 35u)});
  base_scene.geometries.emplace(
      GeometryId{1u}, make_tilted_foreshortened_quad(MaterialId{1u}));
  base_scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "Quad", .geometry = GeometryId{1u}});
  base_scene.cameras.emplace(
      CameraId{1u},
      CameraDesc{.name = "Cam",
                 .projection = CameraProjection::perspective,
                 .transform = translated(0.0f, 0.0f, 2.4f),
                 .field_of_view = 0.85f});
  base_scene.active_camera = CameraId{1u};
  base_scene.world_sampling = WorldSampling::none;

  const std::vector<std::string_view> extensions{"REPEAT", "EXTEND", "CLIP", "MIRROR"};
  const std::vector<std::string_view> interps{"Closest", "Linear", "Cubic"};
  for (const auto ext : extensions) {
    for (const auto interp : interps) {
      SceneSnapshot s = base_scene;
      s.materials.emplace(
          MaterialId{1u},
          MaterialDesc{
              .name = "TestMat",
              .shader = make_mapped_image_emission_shader(
                  ImageId{1u}, {-0.35f, -0.25f, 0.0f}, {0.0f, 0.0f, 0.15f},
                  {2.8f, 2.4f, 1.0f}, interp, ext)});
      LuisaVirtualTextureCacheOptions vt_off{.enabled = false};
      LuisaVirtualTextureCacheOptions vt_iso{
          .enabled = true,
          .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
          .physical_tile_size = 64u,
          .enable_async_worker = true,
          .mip_mode = LuisaTextureMipMode::isotropic};
      const auto ref = run_scene(
          device, s, vt_off, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);
      const auto vt = run_scene(
          device, s, vt_iso, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);
      const auto d = compare_combined_pass(ref.sink, vt.sink);
      std::cout << "  [" << ext << "/" << interp << "] rmse=" << d.rmse
                << " max_abs=" << d.max_abs << '\n';
      require(d.invalid_pixels == 0u && d.rmse < 0.015,
              std::string{"extension/filter mismatch: "} + std::string{ext} + "/" +
                  std::string{interp});
    }
  }
}

void test_tight_eviction_dual_mip_misses(Device &device) {
  std::cout << "[Mip Test 4] Dual-mip misses under tight 5-slot eviction pressure...\n";
  SceneSnapshot scene;
  scene.images.emplace(
      ImageId{1u},
      ImageDesc{.name = "tight_512x512.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 512u,
                .height = 512u,
                .encoded_data = make_mipmap_pattern_png(512u, 512u)});
  scene.materials.emplace(
      MaterialId{1u},
      MaterialDesc{
          .name = "TightMat",
          .shader = make_mapped_image_emission_shader(
              ImageId{1u}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.25f},
              {2.5f, 2.5f, 1.0f}, "Linear", "REPEAT")});
  scene.geometries.emplace(
      GeometryId{1u}, make_tilted_foreshortened_quad(MaterialId{1u}));
  scene.instances.emplace(
      InstanceId{1u}, InstanceDesc{.name = "Quad", .geometry = GeometryId{1u}});
  scene.cameras.emplace(
      CameraId{1u},
      CameraDesc{.name = "Cam",
                 .projection = CameraProjection::perspective,
                 .transform = translated(0.0f, 0.0f, 2.5f),
                 .field_of_view = 0.85f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;

  const auto ref = run_scene(
      device, scene, {.enabled = false}, LuisaTextureMipMode::isotropic,
      96u, 96u, 4u);

  const bool supports_bc_vt = (device.impl()->query("vk_batch_copy_cmd") == "1");
  for (const auto comp : {LuisaVirtualTextureCompressionMode::off,
                          LuisaVirtualTextureCompressionMode::auto_hq}) {
    if (comp != LuisaVirtualTextureCompressionMode::off && !supports_bc_vt) {
      continue;
    }
    for (const std::uint32_t slots : {1u, 2u}) {
      // physical_tile_size=128: RGBA8 tile=64 KiB -> 320 KiB = 5 slots;
      // BC7 tile=16 KiB -> 80 KiB = 5 slots (both >= 64 KiB minimum).
      const std::uint64_t budget =
          (comp == LuisaVirtualTextureCompressionMode::off)
              ? 320ull * 1024ull
              : 80ull * 1024ull;
      LuisaVirtualTextureCacheOptions opts{
          .enabled = true,
          .gpu_cache_budget_bytes = budget,
          .physical_tile_size = 128u,
          .enable_async_worker = true,
          .async_batch_slots = slots,
          .compression_mode = comp,
          .mip_mode = LuisaTextureMipMode::isotropic};
      const auto res = run_scene(
          device, scene, opts, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);
      const auto d = compare_combined_pass(ref.sink, res.sink);
      std::cout << "  [comp=" << luisa_virtual_texture_compression_mode_name(comp)
                << ", slots=" << slots << "] loaded=" << res.final_stats.tiles_loaded
                << " evicted=" << res.final_stats.tiles_evicted
                << " reloaded=" << res.final_stats.tiles_reloaded
                << " faults=" << res.final_stats.fault_frames_total
                << " rmse=" << d.rmse << '\n';
      require(d.invalid_pixels == 0u, "tight eviction produced NaN/Inf");
      require(res.final_stats.tiles_evicted > 0u,
              "expected tiles_evicted > 0 under 5-slot pressure");
      require(res.final_stats.tiles_reloaded > 0u,
              "expected tiles_reloaded > 0 under 5-slot pressure");
      require(res.final_stats.fault_frames_total > 0u,
              "expected fault_frames_total > 0 under 5-slot pressure");
      const double max_rmse =
          (comp == LuisaVirtualTextureCompressionMode::off) ? 0.005 : 0.05;
      require(d.rmse < max_rmse,
              "tight eviction dual-mip render deviated from direct reference");
    }
  }
}

void test_preflight_isotropic_shared_lod_consistency(Device &device) {
  std::cout << "[Mip Test 5] Preflight + isotropic shared LOD consistency...\n";
  SceneSnapshot scene;
  scene.images.emplace(
      ImageId{1u},
      ImageDesc{.name = "pf_512x256.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 512u,
                .height = 256u,
                .encoded_data = make_mipmap_pattern_png(512u, 256u)});
  scene.materials.emplace(
      MaterialId{1u},
      MaterialDesc{
          .name = "PfMat",
          .shader = make_mapped_image_emission_shader(
              ImageId{1u}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.3f},
              {3.0f, 2.0f, 1.0f}, "Linear", "REPEAT")});
  scene.geometries.emplace(
      GeometryId{1u}, make_tilted_foreshortened_quad(MaterialId{1u}));
  scene.instances.emplace(
      InstanceId{1u}, InstanceDesc{.name = "Quad", .geometry = GeometryId{1u}});
  scene.cameras.emplace(
      CameraId{1u},
      CameraDesc{.name = "Cam",
                 .projection = CameraProjection::perspective,
                 .transform = translated(0.0f, 0.0f, 2.5f),
                 .field_of_view = 0.85f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;

  const auto ref = run_scene(
      device, scene, {.enabled = false}, LuisaTextureMipMode::isotropic,
      96u, 96u, 4u);

  LuisaVirtualTextureCacheOptions base_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u,
      .enable_async_worker = true,
      .mip_mode = LuisaTextureMipMode::isotropic};
  LuisaVirtualTextureCacheOptions pf_s2 = base_opts;
  pf_s2.enable_preflight = true;
  pf_s2.preflight_stride = 2u;
  pf_s2.preflight_bounces = 0u;
  LuisaVirtualTextureCacheOptions pf_s1 = pf_s2;
  pf_s1.preflight_stride = 1u;

  const auto no_pf = run_scene(
      device, scene, base_opts, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);
  const auto s2_res = run_scene(
      device, scene, pf_s2, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);
  const auto s1_res = run_scene(
      device, scene, pf_s1, LuisaTextureMipMode::isotropic, 96u, 96u, 4u);

  const auto d_s1 = compare_combined_pass(ref.sink, s1_res.sink);
  const auto d_s2 = compare_combined_pass(ref.sink, s2_res.sink);
  std::cout << "  [pf] no_pf_rt_ld=" << no_pf.final_stats.tiles_loaded
            << " s2(disc/pf_ld/rt_ld)="
            << s2_res.post_init_stats.preflight_tiles_discovered << "/"
            << s2_res.post_init_stats.preflight_tiles_loaded << "/"
            << s2_res.final_stats.tiles_loaded
            << " s1(pf_ld/pf_bytes/rt_ld)="
            << s1_res.post_init_stats.preflight_tiles_loaded << "/"
            << s1_res.post_init_stats.preflight_bytes_loaded << "/"
            << s1_res.final_stats.tiles_loaded
            << " rmse(s1/s2)=" << d_s1.rmse << "/" << d_s2.rmse << '\n';
  require(s1_res.post_init_stats.preflight_tiles_discovered > 0u,
          "expected preflight_tiles_discovered > 0");
  require(s1_res.post_init_stats.preflight_tiles_loaded > 0u,
          "expected preflight_tiles_loaded > 0");
  require(s1_res.post_init_stats.preflight_bytes_loaded > 0ull,
          "expected preflight_bytes_loaded > 0");
  require(s1_res.final_stats.tiles_loaded < no_pf.final_stats.tiles_loaded,
          "stride=1 preflight did not reduce runtime tiles_loaded");
  require(s2_res.final_stats.tiles_loaded < no_pf.final_stats.tiles_loaded,
          "stride=2 preflight did not reduce runtime tiles_loaded");
  require(d_s1.invalid_pixels == 0u && d_s1.rmse < 0.005,
          "stride=1 preflight parity failed");
  require(d_s2.invalid_pixels == 0u && d_s2.rmse < 0.005,
          "stride=2 preflight parity failed");
}

void test_srgb_lut_and_npot_mipmap_parity() {
  std::cout << "[VT Mipmap Test 6] Fast float-bit sRGB LUT + NPOT parity...\n";
  // Full 256-value sRGB ramp + NPOT 137x89 image to exercise both 1:1 ramp
  // values, fractional linear averages, and odd NPOT 3-tap vertical rows.
  constexpr std::uint32_t kW = 137u;
  constexpr std::uint32_t kH = 89u;
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(kW) * kH * 4u);
  for (std::uint32_t y = 0u; y < kH; ++y) {
    for (std::uint32_t x = 0u; x < kW; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * kW + x;
      const std::size_t off = i * 4u;
      rgba[off + 0u] = static_cast<std::uint8_t>(i & 255u);
      rgba[off + 1u] = static_cast<std::uint8_t>((x * 255u) / (kW - 1u));
      rgba[off + 2u] = static_cast<std::uint8_t>((y * 255u) / (kH - 1u));
      rgba[off + 3u] = static_cast<std::uint8_t>((i * 7u) & 255u);
    }
  }
  const auto mips = vt_build_unorm8_tail_mips(rgba, kW, kH, true);
  require(!mips.empty(), "Expected non-empty NPOT sRGB mip pyramid");

  auto ref_srgb_to_lin = [](std::uint8_t v) noexcept -> float {
    const float s = static_cast<float>(v) / 255.0f;
    return (s <= 0.04045f) ? (s / 12.92f)
                           : std::pow((s + 0.055f) / 1.055f, 2.4f);
  };
  auto ref_lin_to_srgb = [](float v) noexcept -> std::uint8_t {
    const float c = std::clamp(v, 0.0f, 1.0f);
    const float s = (c <= 0.0031308f)
                        ? (c * 12.92f)
                        : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
    return static_cast<std::uint8_t>(
        std::clamp(std::lround(std::clamp(s, 0.0f, 1.0f) * 255.0f), 0l, 255l));
  };
  auto ref_downsample_line = [](const float *src, std::size_t s_stride,
                                std::uint32_t s_len, float *dst,
                                std::size_t d_stride,
                                std::uint32_t d_len) noexcept {
    if (s_len == d_len) {
      for (std::uint32_t i = 0u; i < d_len; ++i) {
        for (int c = 0; c < 4; ++c) {
          dst[i * d_stride + c] = src[i * s_stride + c];
        }
      }
      return;
    }
    if ((s_len & 1u) == 0u && s_len == d_len * 2u) {
      for (std::uint32_t i = 0u; i < d_len; ++i) {
        const float *s0 = src + static_cast<std::size_t>(i * 2u) * s_stride;
        const float *s1 = s0 + s_stride;
        float *d = dst + static_cast<std::size_t>(i) * d_stride;
        for (int c = 0; c < 4; ++c) {
          d[c] = 0.5f * (s0[c] + s1[c]);
        }
      }
      return;
    }
    const float inv = 1.0f / static_cast<float>(s_len);
    for (std::uint32_t i = 0u; i < d_len; ++i) {
      const float w0 = static_cast<float>(d_len - i) * inv;
      const float w1 = static_cast<float>(d_len) * inv;
      const float w2 = static_cast<float>(i + 1u) * inv;
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      const float *s0 = src + base * s_stride;
      const float *s1 = s0 + s_stride;
      const float *s2 =
          src + std::min<std::size_t>(base + 2u, s_len - 1u) * s_stride;
      float *d = dst + static_cast<std::size_t>(i) * d_stride;
      for (int c = 0; c < 4; ++c) {
        d[c] = w0 * s0[c] + w1 * s1[c] + w2 * s2[c];
      }
    }
  };

  std::vector<float> cur_lin(static_cast<std::size_t>(kW) * kH * 4u);
  for (std::size_t i = 0u; i < static_cast<std::size_t>(kW) * kH; ++i) {
    cur_lin[i * 4u + 0u] = ref_srgb_to_lin(rgba[i * 4u + 0u]);
    cur_lin[i * 4u + 1u] = ref_srgb_to_lin(rgba[i * 4u + 1u]);
    cur_lin[i * 4u + 2u] = ref_srgb_to_lin(rgba[i * 4u + 2u]);
    cur_lin[i * 4u + 3u] = static_cast<float>(rgba[i * 4u + 3u]) / 255.0f;
  }
  std::uint32_t cw = kW;
  std::uint32_t ch = kH;
  int max_ulp_srgb = 0;
  int max_ulp_alpha = 0;
  for (std::size_t m = 0u; m < mips.size(); ++m) {
    const std::uint32_t dw = mips[m].width;
    const std::uint32_t dh = mips[m].height;
    std::vector<float> horiz(static_cast<std::size_t>(dw) * ch * 4u);
    std::vector<float> next_lin(static_cast<std::size_t>(dw) * dh * 4u);
    for (std::uint32_t y = 0u; y < ch; ++y) {
      ref_downsample_line(cur_lin.data() + static_cast<std::size_t>(y) * cw * 4u,
                          4u, cw,
                          horiz.data() + static_cast<std::size_t>(y) * dw * 4u,
                          4u, dw);
    }
    for (std::uint32_t x = 0u; x < dw; ++x) {
      ref_downsample_line(horiz.data() + static_cast<std::size_t>(x) * 4u,
                          static_cast<std::size_t>(dw) * 4u, ch,
                          next_lin.data() + static_cast<std::size_t>(x) * 4u,
                          static_cast<std::size_t>(dw) * 4u, dh);
    }
    const std::size_t np = static_cast<std::size_t>(dw) * dh;
    for (std::size_t i = 0u; i < np; ++i) {
      for (int c = 0; c < 3; ++c) {
        const std::uint8_t exp_u8 = ref_lin_to_srgb(next_lin[i * 4u + c]);
        const int diff = std::abs(static_cast<int>(mips[m].rgba[i * 4u + c]) -
                                  static_cast<int>(exp_u8));
        max_ulp_srgb = std::max(max_ulp_srgb, diff);
      }
      const std::uint8_t exp_a = static_cast<std::uint8_t>(std::clamp(
          std::lround(std::clamp(next_lin[i * 4u + 3u], 0.0f, 1.0f) * 255.0f),
          0l, 255l));
      max_ulp_alpha = std::max(
          max_ulp_alpha, std::abs(static_cast<int>(mips[m].rgba[i * 4u + 3u]) -
                                  static_cast<int>(exp_a)));
    }
    cur_lin.swap(next_lin);
    cw = dw;
    ch = dh;
  }
  std::cout << "  NPOT 137x89 mip pyramid max_ulp(srgb/alpha)="
            << max_ulp_srgb << "/" << max_ulp_alpha << '\n';
  require(max_ulp_srgb <= 1, "sRGB LUT mip pyramid exceeded 1 ULP vs std::pow");
  require(max_ulp_alpha == 0, "Alpha mip pyramid differed from reference");
}

} // namespace

int main(int argc, char **argv) {
  try {
    test_srgb_lut_and_npot_mipmap_parity();
    const std::string_view backend{argc > 1 ? argv[1] : "vk"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    std::cout << "Running Psycles VT Mipmap tests on backend: " << backend << '\n';
    test_lod_math_and_invalid_gradient_fallback(device);
    test_nonsquare_and_npot_with_transforms(device);
    test_tile_boundaries_extensions_and_filters(device);
    test_tight_eviction_dual_mip_misses(device);
    test_preflight_isotropic_shared_lod_consistency(device);
    std::cout << "ALL 5 VT MIPMAP TESTS PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
