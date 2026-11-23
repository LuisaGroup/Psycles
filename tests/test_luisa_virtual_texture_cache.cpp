#include "cycles_svm_scene_image.h"
#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>

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

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
  Mat4f m{};
  m.elements[12u] = x;
  m.elements[13u] = y;
  m.elements[14u] = z;
  return m;
}

[[nodiscard]] Mat4f rotated_y_translated(float radians, float x, float y,
                                         float z) noexcept {
  const float c = std::cos(radians);
  const float s = std::sin(radians);
  Mat4f m{};
  // Column-major: col0 = (c, 0, -s, 0), col1 = (0, 1, 0, 0), col2 = (s, 0, c, 0).
  // Note local -Z maps to (-s, 0, -c).
  m.elements[0u] = c;
  m.elements[1u] = 0.0f;
  m.elements[2u] = -s;
  m.elements[8u] = s;
  m.elements[9u] = 0.0f;
  m.elements[10u] = c;
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
  if (auto out = OIIO::ImageOutput::create("vt_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_tmp_" +
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
encode_exr_rgba32f(std::uint32_t width, std::uint32_t height,
                   const std::vector<float> &rgba32f) {
  require(rgba32f.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA32F buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::FLOAT};
  if (auto out = OIIO::ImageOutput::create("vt_test.exr", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_test.exr", spec), "OIIO open EXR mem failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write EXR mem failed");
    require(out->close(), "OIIO close EXR mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_tmp_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".exr");
  {
    auto out = OIIO::ImageOutput::create(temp_path.string());
    require(bool(out), "OIIO create temp EXR failed");
    require(out->open(temp_path.string(), spec), "OIIO open temp EXR failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write temp EXR failed");
    require(out->close(), "OIIO close temp EXR failed");
  }
  std::ifstream ifs{temp_path, std::ios::binary};
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>()};
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  require(!bytes.empty(), "empty EXR file readback");
  return bytes;
#else
  std::ostringstream hdr;
  hdr << "PF\n" << width << ' ' << height << "\n-1.0\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  for (std::uint32_t row = height; row > 0u; --row) {
    const std::size_t y = row - 1u;
    for (std::uint32_t x = 0u; x < width; ++x) {
      const float *px = &rgba32f[(y * width + x) * 4u];
      const std::uint8_t *raw = reinterpret_cast<const std::uint8_t *>(px);
      bytes.insert(bytes.end(), raw, raw + 3u * sizeof(float));
    }
  }
  return bytes;
#endif
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

[[nodiscard]] NodeId add_uv_image_node(
    ShaderGraph &graph, ImageId image_id,
    std::string_view interpolation = "Linear",
    std::string_view extension = "REPEAT",
    std::string_view color_space = "Non-Color") {
  const auto tc = graph.add_node(node_type::texture_coordinate, "TexCoord");
  const auto p2v = graph.add_node(node_type::point_to_vector, "UvToVec");
  const auto tex = graph.add_node(node_type::image_texture, "Image");
  require(
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
                             SocketValue::string(std::string{color_space})) &&
          graph.set_property(tex, "UnassociateAlpha",
                             SocketValue::boolean(false)) &&
          graph.connect({tc, "UV"}, p2v, "Point") &&
          graph.connect({p2v, "Vector"}, tex, "Vector"),
      "failed to configure image texture node");
  return tex;
}

[[nodiscard]] ShaderGraph make_textured_emission_shader(
    ImageId image_id, float strength = 1.0f,
    std::string_view interpolation = "Linear") {
  ShaderGraph graph;
  const auto tex = add_uv_image_node(graph, image_id, interpolation);
  const auto em = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({tex, "Color"}, em, "Color") &&
              graph.set_input(em, "Strength", SocketValue::floating(strength)),
          "failed to connect textured emission shader");
  graph.set_root(ShaderDomain::surface, OutputRef{em, "Closure"});
  return graph;
}

struct SampleQuery {
  luisa::float2 uv;
  std::uint32_t binding_id{};
  std::uint32_t pad{};
};

void test_byte4_and_float4_roundtrip(Device &device, Stream &stream) {
  std::cout << "[VT Test 1] BYTE4 0-ULP + FLOAT4 HDR parity...\n";
  constexpr std::uint32_t w = 131u;
  constexpr std::uint32_t h = 97u;

  std::vector<std::uint8_t> b4_pixels(static_cast<std::size_t>(w) * h * 4u);
  std::vector<float> f4_pixels(static_cast<std::size_t>(w) * h * 4u);
  for (std::uint32_t y = 0u; y < h; ++y) {
    for (std::uint32_t x = 0u; x < w; ++x) {
      const std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4u;
      const std::uint32_t linear_idx = y * w + x;
      // Cover every 0..255 value across R, G, B, A.
      b4_pixels[idx + 0u] = static_cast<std::uint8_t>(linear_idx & 0xffu);
      b4_pixels[idx + 1u] = static_cast<std::uint8_t>((linear_idx * 3u + 17u) & 0xffu);
      b4_pixels[idx + 2u] = static_cast<std::uint8_t>((linear_idx * 7u + 91u) & 0xffu);
      b4_pixels[idx + 3u] = static_cast<std::uint8_t>((linear_idx * 13u + 205u) & 0xffu);

      // Cover negative, zero, sub-unit, and >1 HDR floats.
      const float u = static_cast<float>(x) / static_cast<float>(w - 1u);
      const float v = static_cast<float>(y) / static_cast<float>(h - 1u);
      f4_pixels[idx + 0u] = -0.25f + 12.75f * u;
      f4_pixels[idx + 1u] = 12.5f * (1.0f - v) - 0.125f;
      f4_pixels[idx + 2u] = 0.0f + 6.25f * (u * v);
      f4_pixels[idx + 3u] = 0.5f + 0.5f * std::sin(u * 9.0f + v * 5.0f);
    }
  }

  const auto png_bytes = encode_png_rgba8(w, h, b4_pixels);
  const auto exr_bytes = encode_exr_rgba32f(w, h, f4_pixels);
  const auto b4_probe = probe_scene_image_info(png_bytes, "b4.png");
  const auto f4_probe = probe_scene_image_info(exr_bytes, "f4.exr");
  require(b4_probe.has_value() && b4_probe->width == w && b4_probe->height == h &&
              b4_probe->storage == io::DecodedImageStorage::unorm8,
          "PNG probe mismatch");
  require(f4_probe.has_value() && f4_probe->width == w && f4_probe->height == h &&
              f4_probe->storage == io::DecodedImageStorage::float32,
          "EXR probe mismatch");

  const std::vector<VtProbedSceneImage> probed_images{
      VtProbedSceneImage{.id = ImageId{1u},
                         .name = "b4.png",
                         .encoded_data = png_bytes,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .load_failed = false,
                         .force_direct = false,
                         .probed = *b4_probe},
      VtProbedSceneImage{.id = ImageId{2u},
                         .name = "f4.exr",
                         .encoded_data = exr_bytes,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .load_failed = false,
                         .force_direct = false,
                         .probed = *f4_probe}};

  auto decoded_b4 = decode_scene_image(png_bytes, "b4.png");
  auto decoded_f4 = decode_scene_image(exr_bytes, "f4.exr");
  require(decoded_b4.has_value() && decoded_b4->storage == io::DecodedImageStorage::unorm8,
          "decoded PNG mismatch");
  require(decoded_f4.has_value() && decoded_f4->storage == io::DecodedImageStorage::float32,
          "decoded EXR mismatch");

  auto tex_b4 = device.create_image<float>(PixelStorage::BYTE4, w, h);
  auto tex_f4 = device.create_image<float>(PixelStorage::FLOAT4, w, h);
  BindlessArray direct_heap = device.create_bindless_array(8u);
  direct_heap.emplace_on_update(0u, tex_b4, Sampler::point_repeat());
  direct_heap.emplace_on_update(1u, tex_b4, Sampler::linear_point_repeat());
  direct_heap.emplace_on_update(2u, tex_b4, Sampler::linear_point_repeat());
  direct_heap.emplace_on_update(3u, tex_f4, Sampler::point_repeat());
  direct_heap.emplace_on_update(4u, tex_f4, Sampler::linear_point_repeat());
  direct_heap.emplace_on_update(5u, tex_f4, Sampler::linear_point_repeat());
  stream << tex_b4.copy_from(decoded_b4->unorm8_pixels.data())
         << tex_f4.copy_from(decoded_f4->float_pixels.data())
         << direct_heap.update() << synchronize();

  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  const std::vector<ImageBinding> handles{
      {.resource_id = 1u, .interpolation = ImageInterpolation::closest, .extension = ImageExtension::repeat},
      {.resource_id = 1u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 1u, .interpolation = ImageInterpolation::cubic, .extension = ImageExtension::repeat},
      {.resource_id = 2u, .interpolation = ImageInterpolation::closest, .extension = ImageExtension::repeat},
      {.resource_id = 2u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 2u, .interpolation = ImageInterpolation::cubic, .extension = ImageExtension::repeat}};
  std::vector<CyclesSvmImageBindingGpu> gpu_bindings;
  for (std::uint32_t i = 0u; i < handles.size(); ++i) {
    gpu_bindings.push_back(make_cycles_svm_image_binding(
        i, handles[i].interpolation, handles[i].extension));
  }

  LuisaVirtualTextureCacheOptions vt_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u};
  auto vt = VirtualTextureCache::create(device, stream, vt_opts, probed_images,
                                        handles, gpu_bindings);

  std::vector<luisa::float2> test_uvs;
  test_uvs.reserve(static_cast<std::size_t>(w) * h * 2u);
  for (std::uint32_t y = 0u; y < h; ++y) {
    for (std::uint32_t x = 0u; x < w; ++x) {
      test_uvs.push_back(luisa::make_float2((static_cast<float>(x) + 0.5f) / w,
                                            (static_cast<float>(y) + 0.5f) / h));
    }
  }
  for (std::uint32_t y = 0u; y < h; ++y) {
    for (std::uint32_t x = 0u; x < w; ++x) {
      test_uvs.push_back(luisa::make_float2((static_cast<float>(x) + 0.37f) / w,
                                            (static_cast<float>(y) + 0.61f) / h));
    }
  }

  const auto n = static_cast<std::uint32_t>(test_uvs.size());
  auto uv_buf = device.create_buffer<luisa::float2>(n);
  auto vt_out = device.create_buffer<luisa::float4>(n * 6u);
  auto direct_out = device.create_buffer<luisa::float4>(n * 6u);
  auto miss_count = device.create_buffer<luisa::uint>(1u);
  stream << uv_buf.copy_from(luisa::span{test_uvs}) << synchronize();

  auto test_shader = device.compile<1>(
      [&](BufferVar<luisa::float2> uvs, BufferVar<luisa::float4> vt_res,
          BufferVar<luisa::float4> dir_res, BufferVar<luisa::uint> misses,
          UInt count) noexcept {
        const auto i = dispatch_x();
        $if(i >= count) { $return(); };
        const auto uv = uvs.read(i);
        for (std::uint32_t b = 0u; b < 6u; ++b) {
          UInt miss = vt_invalid_page;
          const auto vt_val = vt->sample_2d_const(static_cast<std::int32_t>(b), uv, &miss);
          $if(miss != vt_invalid_page) { misses.atomic(0u).fetch_add(1u); };
          const auto interp = cycles_svm_image_sampling_family(handles[b].interpolation);
          const auto ext = cycles_svm_image_sampling_extension(handles[b].extension);
          const auto dir_val = sample_cycles_texture_2d(direct_heap, b, uv, interp, ext);
          vt_res.write(b * count + i, vt_val);
          dir_res.write(b * count + i, dir_val);
        }
      });

  std::uint32_t zero_u32 = 0u;
  std::uint32_t host_misses = 0u;
  bool clean = false;
  for (int iter = 0; iter < 8; ++iter) {
    stream << miss_count.copy_from(&zero_u32)
           << test_shader(uv_buf, vt_out, direct_out, miss_count, n).dispatch(n)
           << miss_count.copy_to(&host_misses) << synchronize();
    if (host_misses == 0u) {
      clean = true;
      break;
    }
    vt->service_bitset_misses(stream);
  }
  require(clean, "Test 1 did not reach 0 misses after servicing");

  std::vector<luisa::float4> host_vt(n * 6u);
  std::vector<luisa::float4> host_dir(n * 6u);
  stream << vt_out.copy_to(luisa::span{host_vt})
         << direct_out.copy_to(luisa::span{host_dir}) << synchronize();

  const std::size_t texel_count = static_cast<std::size_t>(w) * h;
  for (std::uint32_t b = 0u; b < 6u; ++b) {
    const bool is_b4 = (b < 3u);
    const bool is_closest = (b == 0u || b == 3u);
    double max_err = 0.0;
    const std::size_t begin_i = is_closest ? 0u : texel_count;
    const std::size_t end_i = is_closest ? texel_count : n;
    for (std::size_t i = begin_i; i < end_i; ++i) {
      const auto a = host_vt[static_cast<std::size_t>(b) * n + i];
      const auto d = host_dir[static_cast<std::size_t>(b) * n + i];
      for (int c = 0; c < 4; ++c) {
        const float av = (&a.x)[c];
        const float dv = (&d.x)[c];
        require(std::isfinite(av), "non-finite VT value in Test 1");
        const double diff = std::abs(static_cast<double>(av) - static_cast<double>(dv));
        max_err = std::max(max_err, diff);
        if (is_closest) {
          require(av == dv,
                  "Test 1 Closest mismatch (> 0 ULP): binding=" +
                      std::to_string(b));
        } else if (is_b4) {
          // GPU hardware BYTE4 samplers quantize sub-texel lerp weights to ~8 bits.
          require(diff <= 0.006,
                  "Test 1 BYTE4 filtered mismatch: diff=" + std::to_string(diff));
        } else {
          const double tol = 2e-4 + 1e-4 * std::max(std::abs(av), std::abs(dv));
          require(diff <= tol,
                  "Test 1 FLOAT4 filtered mismatch: diff=" + std::to_string(diff));
        }
      }
    }
    std::cout << "  binding " << b << " max_abs_diff=" << max_err << '\n';
  }
}

void test_npot_wrap_filter_matrix(Device &device, Stream &stream) {
  std::cout << "[VT Test 2] NPOT 127x73 4x3 wrap x filter const & dynamic parity...\n";
  constexpr std::uint32_t w = 127u;
  constexpr std::uint32_t h = 73u;

  std::vector<std::uint8_t> rgba8(static_cast<std::size_t>(w) * h * 4u);
  for (std::uint32_t y = 0u; y < h; ++y) {
    for (std::uint32_t x = 0u; x < w; ++x) {
      const std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4u;
      rgba8[idx + 0u] = static_cast<std::uint8_t>((x * 29u + y * 7u + 13u) & 0xffu);
      rgba8[idx + 1u] = static_cast<std::uint8_t>((x * 11u + y * 53u + 41u) & 0xffu);
      rgba8[idx + 2u] = static_cast<std::uint8_t>((x * 43u + y * 19u + 79u) & 0xffu);
      rgba8[idx + 3u] = static_cast<std::uint8_t>((x * 5u + y * 37u + 128u) & 0xffu);
    }
  }
  const auto png_bytes = encode_png_rgba8(w, h, rgba8);
  const auto probed = probe_scene_image_info(png_bytes, "npot.png");
  require(probed.has_value() && probed->width == w && probed->height == h,
          "Test 2 PNG probe failed");
  const auto decoded = decode_scene_image(png_bytes, "npot.png");
  require(decoded.has_value(), "Test 2 PNG decode failed");

  auto direct_tex = device.create_image<float>(PixelStorage::BYTE4, w, h);
  BindlessArray direct_heap = device.create_bindless_array(16u);

  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  constexpr std::array exts{ImageExtension::repeat, ImageExtension::clip,
                            ImageExtension::extend, ImageExtension::mirror};
  constexpr std::array interps{ImageInterpolation::closest,
                               ImageInterpolation::linear,
                               ImageInterpolation::cubic};

  std::vector<ImageBinding> handles;
  std::vector<CyclesSvmImageBindingGpu> gpu_bindings;
  for (std::uint32_t ei = 0u; ei < exts.size(); ++ei) {
    for (std::uint32_t ii = 0u; ii < interps.size(); ++ii) {
      const auto b_idx = static_cast<std::uint32_t>(handles.size());
      handles.push_back(ImageBinding{.resource_id = 10u,
                                     .interpolation = interps[ii],
                                     .extension = exts[ei]});
      gpu_bindings.push_back(
          make_cycles_svm_image_binding(b_idx, interps[ii], exts[ei]));
      direct_heap.emplace_on_update(
          b_idx, direct_tex, cycles_svm_texture_sampler(interps[ii], exts[ei]));
    }
  }
  stream << direct_tex.copy_from(decoded->unorm8_pixels.data())
         << direct_heap.update() << synchronize();

  const std::vector<VtProbedSceneImage> probed_images{
      VtProbedSceneImage{.id = ImageId{10u},
                         .name = "npot.png",
                         .encoded_data = png_bytes,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *probed}};
  LuisaVirtualTextureCacheOptions vt_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u};
  auto vt = VirtualTextureCache::create(device, stream, vt_opts, probed_images,
                                        handles, gpu_bindings);

  // Build test points: corners, tile seam texels (59, 60, 61), interior, and out-of-range UVs.
  const std::array<float, 14> coord_x_texels{
      -1.75f * w, -0.01f * w, 0.0f, 0.5f, 1.5f, 59.5f, 60.2f,
      60.5f,      61.5f,      120.5f, 126.5f, 1.0f * w, 1.01f * w, 2.37f * w};
  const std::array<float, 12> coord_y_texels{
      -1.75f * h, -0.01f * h, 0.0f, 0.5f, 59.5f, 60.3f,
      61.5f,      72.5f,      1.0f * h, 1.01f * h, 2.37f * h, 36.25f};

  std::vector<luisa::float2> center_uvs;
  std::vector<luisa::float2> general_uvs;
  for (std::uint32_t y : {0u, 1u, 59u, 60u, 61u, 72u}) {
    for (std::uint32_t x : {0u, 1u, 59u, 60u, 61u, 119u, 120u, 126u}) {
      center_uvs.push_back(luisa::make_float2((static_cast<float>(x) + 0.5f) / w,
                                              (static_cast<float>(y) + 0.5f) / h));
    }
  }
  for (const float tx : coord_x_texels) {
    for (const float ty : coord_y_texels) {
      general_uvs.push_back(luisa::make_float2(tx / static_cast<float>(w),
                                               ty / static_cast<float>(h)));
    }
  }
  // Exact boundary normalized UVs requested in spec.
  for (const float u : {-1.75f, -0.01f, 0.0f, 0.5f, 1.0f, 1.01f, 2.37f}) {
    for (const float v : {-1.75f, -0.01f, 0.0f, 0.5f, 1.0f, 1.01f, 2.37f}) {
      general_uvs.push_back(luisa::make_float2(u, v));
    }
  }

  std::vector<luisa::float2> all_uvs = center_uvs;
  all_uvs.insert(all_uvs.end(), general_uvs.begin(), general_uvs.end());
  const std::size_t center_count = center_uvs.size();
  const std::uint32_t u_count = static_cast<std::uint32_t>(all_uvs.size());
  const std::uint32_t b_count = static_cast<std::uint32_t>(handles.size());
  const std::uint32_t total = u_count * b_count;

  auto uv_buf = device.create_buffer<luisa::float2>(u_count);
  auto const_out = device.create_buffer<luisa::float4>(total);
  auto dyn_out = device.create_buffer<luisa::float4>(total);
  auto ref_out = device.create_buffer<luisa::float4>(total);
  auto miss_count = device.create_buffer<luisa::uint>(1u);
  stream << uv_buf.copy_from(luisa::span{all_uvs}) << synchronize();

  auto eval_kernel = device.compile<1>(
      [&](BufferVar<luisa::float2> uvs, BufferVar<luisa::float4> out_c,
          BufferVar<luisa::float4> out_d, BufferVar<luisa::float4> out_ref,
          BufferVar<luisa::uint> misses, UInt n_uv) noexcept {
        const auto u_idx = dispatch_x();
        $if(u_idx >= n_uv) { $return(); };
        const auto uv = uvs.read(u_idx);
        for (std::uint32_t b = 0u; b < 12u; ++b) {
          UInt miss_c = vt_invalid_page;
          UInt miss_d = vt_invalid_page;
          const auto vc =
              vt->sample_2d_const(static_cast<std::int32_t>(b), uv, &miss_c);
          const auto vd = vt->sample_2d_dynamic(
              Int(static_cast<std::int32_t>(b)), uv, direct_heap, &miss_d);
          $if((miss_c != vt_invalid_page) | (miss_d != vt_invalid_page)) {
            misses.atomic(0u).fetch_add(1u);
          };
          const auto interp = cycles_svm_image_sampling_family(handles[b].interpolation);
          const auto ext = cycles_svm_image_sampling_extension(handles[b].extension);
          const auto vr = sample_cycles_texture_2d(direct_heap, b, uv, interp, ext);
          const auto out_idx = b * n_uv + u_idx;
          out_c.write(out_idx, vc);
          out_d.write(out_idx, vd);
          out_ref.write(out_idx, vr);
        }
      });

  std::uint32_t zero_u32 = 0u;
  std::uint32_t host_misses = 0u;
  bool clean = false;
  for (int iter = 0; iter < 8; ++iter) {
    stream << miss_count.copy_from(&zero_u32)
           << eval_kernel(uv_buf, const_out, dyn_out, ref_out, miss_count, u_count)
                  .dispatch(u_count)
           << miss_count.copy_to(&host_misses) << synchronize();
    if (host_misses == 0u) {
      clean = true;
      break;
    }
    vt->service_bitset_misses(stream);
  }
  require(clean, "Test 2 did not converge to 0 misses");

  std::vector<luisa::float4> h_c(total), h_d(total), h_r(total);
  stream << const_out.copy_to(luisa::span{h_c})
         << dyn_out.copy_to(luisa::span{h_d})
         << ref_out.copy_to(luisa::span{h_r}) << synchronize();

  for (std::uint32_t b = 0u; b < b_count; ++b) {
    const bool is_closest =
        (handles[b].interpolation == ImageInterpolation::closest);
    for (std::uint32_t u = 0u; u < u_count; ++u) {
      const std::size_t idx = static_cast<std::size_t>(b) * u_count + u;
      for (int c = 0; c < 4; ++c) {
        const float vc = (&h_c[idx].x)[c];
        const float vd = (&h_d[idx].x)[c];
        const float vr = (&h_r[idx].x)[c];
        require(std::isfinite(vc) && std::isfinite(vd), "non-finite Test 2 sample");
        // Const and dynamic VT sampling paths must agree (allowing tiny float rounding).
        require(std::abs(vc - vd) <= 1e-4f,
                "Test 2 const vs dynamic mismatch at binding=" +
                    std::to_string(b) + " u=" + std::to_string(u) +
                    " vc=" + std::to_string(vc) + " vd=" + std::to_string(vd) +
                    " vr=" + std::to_string(vr));
        const double diff = std::abs(static_cast<double>(vc) - static_cast<double>(vr));
        if (is_closest) {
          // Near integer texel edge u=1.0/0.0, floor vs hardware point rounding on
          // out-of-range/boundary coordinates may pick adjacent texel on float ULP;
          // texel-center UVs must match with 0-ULP exactness.
          if (u < center_count) {
            require(vc == vr, "Test 2 Closest seam/center mismatch (> 0 ULP)");
          } else {
            // Exact boundary u == 0.0 / 1.0 under REPEAT/EXTEND/MIRROR can differ by
            // at most 1 texel if floor(u*extent) straddles an integer boundary; CLIP matches.
            if (all_uvs[u].x != 0.0f && all_uvs[u].x != 1.0f &&
                all_uvs[u].y != 0.0f && all_uvs[u].y != 1.0f) {
              require(vc == vr, "Test 2 Closest non-integer-edge mismatch: got=" +
                                    std::to_string(vc) + " ref=" + std::to_string(vr));
            }
          }
        } else {
          // Bilinear / cubic hardware 8-bit subtexel weight quantization on BYTE4.
          require(diff <= 0.008, "Test 2 filtered mismatch at binding=" +
                                     std::to_string(b) + " u=" + std::to_string(u) +
                                     " diff=" + std::to_string(diff));
        }
      }
    }
  }
  std::cout << "  all 12 wrap x filter combos passed const/dynamic vs direct!\n";
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

[[nodiscard]] io::MemoryOutputSink render_test_scene(
    Device &device, const SceneSnapshot &scene, LuisaPathScheduler scheduler,
    const LuisaVirtualTextureCacheOptions &vt_options,
    std::uint32_t width, std::uint32_t height, std::uint32_t spp,
    VirtualTextureCacheStats *out_stats = nullptr,
    bool *out_eager_fallback = nullptr,
    io::MemoryOutputSink *out_warm_sink = nullptr,
    VirtualTextureCacheStats *out_warm_stats = nullptr) {
  LuisaPathTracerBackend renderer{
      Device{device.impl_shared()},
      {.next_event_estimation = true,
       .enable_fast_math = true,
       .scheduler = scheduler,
       .wavefront_frame_capacity = 4096u,
       .max_samples_per_dispatch = spp,
       .virtual_texture_cache = vt_options}};
  const auto compilation = renderer.compile_scene(scene);
  for (const auto &d : compilation.diagnostics) {
    std::cerr << "compile diagnostic: " << d.message << '\n';
  }
  require(compilation.ok(), "scene compilation failed");

  RenderSettings settings;
  settings.full_extent = {width, height};
  settings.window = {.x = 0u, .y = 0u, .width = width, .height = height};
  settings.seed = 12345u;
  settings.integrator.max_bounces = 4u;
  settings.integrator.diffuse_bounces = 2u;
  settings.integrator.glossy_bounces = 2u;
  settings.integrator.transparent_max_bounces = 8u;
  settings.passes = {{.kind = PassKind::combined, .name = "Combined", .channels = 4u}};

  auto session = renderer.create_session(*compilation.scene, settings);
  require(bool(session), "failed to create render session");
  io::MemoryOutputSink sink;
  require(session->render_samples(
              {.first = 0u, .count = spp, .offset = 0u, .total = spp}, sink),
          "render_samples failed");

  const auto *native =
      dynamic_cast<const LuisaCompiledScene *>(compilation.scene.get());
  require(native != nullptr, "missing LuisaCompiledScene");
  if (native->data()->virtual_texture_cache) {
    if (out_stats != nullptr) {
      *out_stats = native->data()->virtual_texture_cache->stats();
    }
    if (out_eager_fallback != nullptr) {
      *out_eager_fallback =
          native->data()->virtual_texture_cache->eager_fallback_active();
    }
  }
  if (out_warm_sink != nullptr) {
    auto session2 = renderer.create_session(*compilation.scene, settings);
    require(bool(session2), "failed to create warm render session");
    io::MemoryOutputSink warm_sink;
    require(session2->render_samples(
                {.first = 0u, .count = spp, .offset = 0u, .total = spp}, warm_sink),
            "warm render_samples failed");
    *out_warm_sink = std::move(warm_sink);
    if (out_warm_stats != nullptr && native->data()->virtual_texture_cache) {
      *out_warm_stats = native->data()->virtual_texture_cache->stats();
    }
  }
  return sink;
}

void test_lazy_decode_proof(Device &device) {
  std::cout << "[VT Test 3] Lazy decode proof (2 PNGs, 1 visible)...\n";
  auto make_png = [](std::uint8_t base_r, std::uint8_t base_g) {
    constexpr std::uint32_t s = 64u;
    std::vector<std::uint8_t> px(static_cast<std::size_t>(s) * s * 4u);
    for (std::uint32_t y = 0u; y < s; ++y) {
      for (std::uint32_t x = 0u; x < s; ++x) {
        const std::size_t i = (static_cast<std::size_t>(y) * s + x) * 4u;
        px[i + 0u] = static_cast<std::uint8_t>((base_r + x * 2u) & 0xffu);
        px[i + 1u] = static_cast<std::uint8_t>((base_g + y * 3u) & 0xffu);
        px[i + 2u] = 96u;
        px[i + 3u] = 255u;
      }
    }
    return encode_png_rgba8(s, s, px);
  };

  SceneSnapshot scene;
  scene.images.emplace(ImageId{1u}, ImageDesc{.name = "visible.png",
                                              .color_space = ImageColorSpace::data,
                                              .alpha_type = ImageAlphaType::straight,
                                              .width = 64u,
                                              .height = 64u,
                                              .encoded_data = make_png(20u, 40u)});
  scene.images.emplace(ImageId{2u}, ImageDesc{.name = "occluded.png",
                                              .color_space = ImageColorSpace::data,
                                              .alpha_type = ImageAlphaType::straight,
                                              .width = 64u,
                                              .height = 64u,
                                              .encoded_data = make_png(180u, 10u)});
  scene.materials.emplace(
      MaterialId{1u}, MaterialDesc{.name = "VisibleMat",
                                   .shader = make_textured_emission_shader(ImageId{1u})});
  scene.materials.emplace(
      MaterialId{2u}, MaterialDesc{.name = "OccludedMat",
                                   .shader = make_textured_emission_shader(ImageId{2u})});
  scene.geometries.emplace(
      GeometryId{1u}, make_quad_mesh(-5.0f, 5.0f, -5.0f, 5.0f, 0.0f, MaterialId{1u}));
  scene.geometries.emplace(
      GeometryId{2u}, make_quad_mesh(-2.0f, 2.0f, -2.0f, 2.0f, -4.0f, MaterialId{2u}));
  scene.instances.emplace(InstanceId{1u}, InstanceDesc{.name = "Front", .geometry = GeometryId{1u}});
  scene.instances.emplace(InstanceId{2u}, InstanceDesc{.name = "Back", .geometry = GeometryId{2u}});
  scene.cameras.emplace(CameraId{1u},
                        CameraDesc{.name = "Cam",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 2.0f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;

  const auto baseline = render_test_scene(
      device, scene, LuisaPathScheduler::wavefront_staged, {.enabled = false},
      16u, 16u, 2u);

  for (const auto sched :
       {LuisaPathScheduler::wavefront, LuisaPathScheduler::wavefront_staged}) {
    VirtualTextureCacheStats stats{};
    const auto vt_sink = render_test_scene(
        device, scene, sched,
        {.enabled = true, .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull},
        16u, 16u, 2u, &stats);
    const auto diff = compare_combined_pass(vt_sink, baseline);
    require(stats.virtual_images == 2u, "expected 2 virtual images");
    require(stats.images_decoded == 1u,
            "lazy decode failed: expected 1 decoded image, got " +
                std::to_string(stats.images_decoded));
    require(diff.invalid_pixels == 0u && diff.max_abs <= 0.01,
            "Test 3 pixel parity failed: max_abs=" + std::to_string(diff.max_abs));
    std::cout << "  scheduler=" << static_cast<int>(sched)
              << " decoded=" << stats.images_decoded << "/2 max_abs=" << diff.max_abs
              << '\n';
  }
}

void test_tiny_budget_eviction_stress(Device &device) {
  std::cout << "[VT Test 4] 256 KiB tiny-budget eviction stress...\n";
  constexpr std::uint32_t s = 512u;
  std::vector<std::uint8_t> px(static_cast<std::size_t>(s) * s * 4u);
  for (std::uint32_t y = 0u; y < s; ++y) {
    for (std::uint32_t x = 0u; x < s; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * s + x) * 4u;
      const auto tx = x / 60u;
      const auto ty = y / 60u;
      px[i + 0u] = static_cast<std::uint8_t>((tx * 29u + x) & 0xffu);
      px[i + 1u] = static_cast<std::uint8_t>((ty * 31u + y) & 0xffu);
      px[i + 2u] = static_cast<std::uint8_t>(((tx + ty) * 19u + 40u) & 0xffu);
      px[i + 3u] = 255u;
    }
  }
  const auto png_bytes = encode_png_rgba8(s, s, px);

  SceneSnapshot scene;
  scene.images.emplace(ImageId{1u}, ImageDesc{.name = "big.png",
                                              .color_space = ImageColorSpace::data,
                                              .alpha_type = ImageAlphaType::straight,
                                              .width = s,
                                              .height = s,
                                              .encoded_data = png_bytes});
  scene.materials.emplace(
      MaterialId{1u}, MaterialDesc{.name = "BigMat",
                                   .shader = make_textured_emission_shader(ImageId{1u})});
  scene.geometries.emplace(
      GeometryId{1u}, make_quad_mesh(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, MaterialId{1u}));
  scene.instances.emplace(InstanceId{1u}, InstanceDesc{.name = "Quad", .geometry = GeometryId{1u}});
  scene.cameras.emplace(CameraId{1u},
                        CameraDesc{.name = "Cam",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 2.0f});
  scene.active_camera = CameraId{1u};
  scene.world_sampling = WorldSampling::none;

  const auto baseline = render_test_scene(
      device, scene, LuisaPathScheduler::wavefront_staged, {.enabled = false},
      32u, 32u, 8u);

  for (const auto sched :
       {LuisaPathScheduler::wavefront, LuisaPathScheduler::wavefront_staged}) {
    VirtualTextureCacheStats stats{};
    const auto vt_sink = render_test_scene(
        device, scene, sched,
        {.enabled = true,
         .gpu_cache_budget_bytes = 256ull * 1024ull,
         .physical_tile_size = 64u},
        32u, 32u, 8u, &stats);
    const auto diff = compare_combined_pass(vt_sink, baseline);
    std::cout << "  scheduler=" << static_cast<int>(sched)
              << " pages=" << stats.byte4_virtual_pages
              << " cap=" << stats.byte4_cache_capacity
              << " peak=" << stats.byte4_peak_resident
              << " batches=" << stats.fault_batches
              << " loaded=" << stats.tiles_loaded
              << " evicted=" << stats.tiles_evicted
              << " max_abs=" << diff.max_abs << '\n';
    require(stats.byte4_virtual_pages == 81u, "expected 81 virtual pages for 512x512 at T=60");
    require(stats.byte4_cache_capacity == 16u, "expected 16-tile capacity for 256 KiB budget");
    require(stats.byte4_peak_resident <= stats.byte4_cache_capacity,
            "resident tiles exceeded bounded budget");
    require(stats.tiles_evicted > 0u, "expected evictions under 256 KiB budget");
    require(stats.fault_batches >= 2u, "expected >= 2 fault batches under eviction");
    require(diff.invalid_pixels == 0u && diff.max_abs <= 0.02,
            "Test 4 pixel parity failed: max_abs=" + std::to_string(diff.max_abs));
  }
}

void test_multistage_parity(Device &device) {
  std::cout << "[VT Test 5] Multi-stage parity (surface, shadow, area light, world env)...\n";
  auto make_checker_png = [](std::uint32_t w, std::uint32_t h,
                             std::array<std::uint8_t, 4> c0,
                             std::array<std::uint8_t, 4> c1) {
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 4u);
    for (std::uint32_t y = 0u; y < h; ++y) {
      for (std::uint32_t x = 0u; x < w; ++x) {
        const bool alt = ((x / 16u + y / 16u) & 1u) != 0u;
        const auto &c = alt ? c1 : c0;
        const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 4u;
        px[i + 0u] = c[0u];
        px[i + 1u] = c[1u];
        px[i + 2u] = c[2u];
        px[i + 3u] = c[3u];
      }
    }
    return encode_png_rgba8(w, h, px);
  };

  constexpr std::uint32_t env_w = 128u;
  constexpr std::uint32_t env_h = 64u;
  std::vector<float> env_px(static_cast<std::size_t>(env_w) * env_h * 4u);
  for (std::uint32_t y = 0u; y < env_h; ++y) {
    for (std::uint32_t x = 0u; x < env_w; ++x) {
      const float u = static_cast<float>(x) / static_cast<float>(env_w);
      const float v = static_cast<float>(y) / static_cast<float>(env_h);
      const std::size_t i = (static_cast<std::size_t>(y) * env_w + x) * 4u;
      env_px[i + 0u] = 0.25f + 0.45f * u;
      env_px[i + 1u] = 0.30f + 0.35f * (1.0f - v);
      env_px[i + 2u] = 0.55f + 0.25f * std::sin(u * 6.28318f);
      env_px[i + 3u] = 1.0f;
    }
  }

  SceneSnapshot scene;
  // Image 1 & 2: two surface textures.
  scene.images.emplace(
      ImageId{1u},
      ImageDesc{.name = "surf_a.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 128u,
                .height = 128u,
                .encoded_data = make_checker_png(128u, 128u, {200u, 80u, 60u, 255u},
                                                 {120u, 190u, 90u, 255u})});
  scene.images.emplace(
      ImageId{2u},
      ImageDesc{.name = "surf_b.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 128u,
                .height = 128u,
                .encoded_data = make_checker_png(128u, 128u, {230u, 210u, 90u, 255u},
                                                 {90u, 140u, 220u, 255u})});
  // Image 3: transparent shadow caster color modulation.
  scene.images.emplace(
      ImageId{3u},
      ImageDesc{.name = "shadow.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 128u,
                .height = 128u,
                .encoded_data = make_checker_png(128u, 128u, {220u, 120u, 120u, 255u},
                                                 {120u, 220u, 220u, 255u})});
  // Image 4: textured emissive area light.
  scene.images.emplace(
      ImageId{4u},
      ImageDesc{.name = "light.png",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = 128u,
                .height = 128u,
                .encoded_data = make_checker_png(128u, 128u, {255u, 230u, 160u, 255u},
                                                 {170u, 240u, 255u, 255u})});
  // Image 5: virtual HDR world environment texture.
  scene.images.emplace(
      ImageId{5u},
      ImageDesc{.name = "world.exr",
                .color_space = ImageColorSpace::data,
                .alpha_type = ImageAlphaType::straight,
                .width = env_w,
                .height = env_h,
                .encoded_data = encode_exr_rgba32f(env_w, env_h, env_px)});

  // Surface shader: blend Image 1 & Image 2 into diffuse_bsdf.
  {
    ShaderGraph g;
    const auto t1 = add_uv_image_node(g, ImageId{1u});
    const auto t2 = add_uv_image_node(g, ImageId{2u});
    const auto mix = g.add_node(node_type::mix_color, "MixSurf");
    const auto diff = g.add_node(node_type::diffuse_bsdf, "Diffuse");
    require(g.set_input(mix, "Factor", SocketValue::floating(0.5f)) &&
                g.connect({t1, "Color"}, mix, "A") &&
                g.connect({t2, "Color"}, mix, "B") &&
                g.connect({mix, "Color"}, diff, "Color") &&
                g.set_input(diff, "Roughness", SocketValue::floating(0.0f)),
            "failed to build Test 5 surface shader");
    g.set_root(ShaderDomain::surface, OutputRef{diff, "Closure"});
    scene.materials.emplace(MaterialId{1u},
                            MaterialDesc{.name = "ReceiverMat", .shader = std::move(g)});
  }

  // Transparent shadow caster shader (invisible to camera rays so shadow stage runs).
  {
    ShaderGraph g;
    const auto t3 = add_uv_image_node(g, ImageId{3u});
    const auto transp = g.add_node(node_type::transparent_bsdf, "Transp");
    require(g.connect({t3, "Color"}, transp, "Color"),
            "failed to build Test 5 transparent shader");
    g.set_root(ShaderDomain::surface, OutputRef{transp, "Closure"});
    scene.materials.emplace(MaterialId{2u},
                            MaterialDesc{.name = "CasterMat", .shader = std::move(g)});
  }

  // Area light emission shader.
  {
    ShaderGraph g;
    const auto geo = g.add_node(node_type::geometry, "Geo");
    const auto p2v = g.add_node(node_type::point_to_vector, "ParamToVec");
    const auto tex = g.add_node(node_type::image_texture, "LightTex");
    const auto em = g.add_node(node_type::emission, "LightEm");
    require(g.set_property(tex, "Image", SocketValue::unsigned_integer(4u)) &&
                g.set_property(tex, "Interpolation", SocketValue::string("Linear")) &&
                g.set_property(tex, "Extension", SocketValue::string("REPEAT")) &&
                g.set_property(tex, "Projection", SocketValue::string("FLAT")) &&
                g.set_property(tex, "ProjectionBlend", SocketValue::floating(0.0f)) &&
                g.set_property(tex, "ColorSpace", SocketValue::string("Non-Color")) &&
                g.set_property(tex, "UnassociateAlpha", SocketValue::boolean(false)) &&
                g.connect({geo, "Parametric"}, p2v, "Point") &&
                g.connect({p2v, "Vector"}, tex, "Vector") &&
                g.connect({tex, "Color"}, em, "Color") &&
                g.set_input(em, "Strength", SocketValue::floating(1.0f)),
            "failed to build Test 5 light shader");
    g.set_root(ShaderDomain::surface, OutputRef{em, "Closure"});
    scene.materials.emplace(MaterialId{3u},
                            MaterialDesc{.name = "LightMat", .shader = std::move(g)});
  }

  // World environment shader using environment_texture.
  {
    ShaderGraph g;
    const auto tc = g.add_node(node_type::texture_coordinate, "WorldTC");
    const auto p2v = g.add_node(node_type::point_to_vector, "GenToVec");
    const auto env = g.add_node(node_type::environment_texture, "EnvTex");
    const auto bg = g.add_node(node_type::background, "Bg");
    require(g.set_property(env, "Image", SocketValue::unsigned_integer(5u)) &&
                g.set_property(env, "Interpolation", SocketValue::string("Linear")) &&
                g.set_property(env, "Projection",
                               SocketValue::string("EQUIRECTANGULAR")) &&
                g.set_property(env, "ColorSpace", SocketValue::string("Non-Color")) &&
                g.connect({tc, "Generated"}, p2v, "Point") &&
                g.connect({p2v, "Vector"}, env, "Vector") &&
                g.connect({env, "Color"}, bg, "Color") &&
                g.set_input(bg, "Strength", SocketValue::floating(0.8f)),
            "failed to build Test 5 world shader");
    g.set_root(ShaderDomain::surface, OutputRef{bg, "Closure"});
    scene.materials.emplace(MaterialId{4u},
                            MaterialDesc{.name = "WorldMat", .shader = std::move(g)});
  }

  // Receiver quad on left/center covering part of screen so outer pixels hit World background.
  scene.geometries.emplace(
      GeometryId{1u}, make_quad_mesh(-1.2f, 0.35f, -0.9f, 0.9f, 0.0f, MaterialId{1u}));
  // Transparent shadow caster between receiver and light (invisible to camera rays so
  // camera rays hit receiver/light/background while shadow rays must shade CasterMat).
  scene.geometries.emplace(
      GeometryId{2u}, make_quad_mesh(-0.8f, 0.8f, -0.8f, 0.8f, 0.0f, MaterialId{2u}));
  scene.instances.emplace(InstanceId{1u},
                          InstanceDesc{.name = "Receiver",
                                       .geometry = GeometryId{1u},
                                       .cycles_object_index = 0u});
  scene.instances.emplace(InstanceId{2u},
                          InstanceDesc{.name = "TranspCaster",
                                       .geometry = GeometryId{2u},
                                       .transform = rotated_y_translated(
                                           1.5707963f, 0.55f, 0.1f, 0.35f),
                                       .visibility_mask =
                                           all_ray_visibility &
                                           ~visibility_bit(RayVisibility::camera),
                                       .cycles_object_index = 1u});

  // Area light tilted 120 deg around Y at (1.1, 0.15, 0.25) so emit_dir = (-0.866, 0, +0.5):
  // visible to primary camera rays (forward_light) AND illuminates receiver (NEE).
  scene.lights.emplace(
      LightId{1u},
      LightDesc{.name = "TexturedAreaLight",
                .type = LightType::area,
                .transform = rotated_y_translated(2.0943951f, 1.1f, 0.15f, 0.25f),
                .color = {1.0f, 1.0f, 1.0f},
                .power = 25.0f,
                .size = 0.9f,
                .size_y = 0.9f,
                .spread = 3.14159265f,
                .normalize = true,
                .is_sphere = false,
                .shader = MaterialId{3u},
                .use_mis = true,
                .cast_shadow = true,
                .visibility_mask = all_ray_visibility,
                .cycles_object_index = 2u});

  scene.cameras.emplace(CameraId{1u},
                        CameraDesc{.name = "Cam",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 3.6f});
  scene.active_camera = CameraId{1u};
  scene.world_shader = MaterialId{4u};
  scene.world_sampling = WorldSampling::automatic;
  scene.world_sample_map_resolution = 64u;

  const auto baseline = render_test_scene(
      device, scene, LuisaPathScheduler::wavefront_staged, {.enabled = false},
      48u, 32u, 4u);

  const auto baseline_wf = render_test_scene(
      device, scene, LuisaPathScheduler::wavefront, {.enabled = false},
      48u, 32u, 4u);
  {
    const auto wf_diff = compare_combined_pass(baseline_wf, baseline);
    std::cout << "  scheduler=2(VT OFF) rmse=" << wf_diff.rmse
              << " max_abs=" << wf_diff.max_abs << '\n';
  }

  const LuisaVirtualTextureCacheOptions vt_opts{
      .enabled = true,
      .gpu_cache_budget_bytes = 64ull * 1024ull * 1024ull,
      .physical_tile_size = 64u};

  bool all_ok = true;
  std::string fail_msg;
  auto print_top_diffs = [&](const io::MemoryOutputSink &sink_to_check,
                             std::size_t max_rows) {
    const auto *a = sink_to_check.find(PassKind::combined);
    const auto *b = baseline.find(PassKind::combined);
    struct PixDiff { std::uint32_t x, y; float d; float a[4]; float b[4]; };
    std::vector<PixDiff> top;
    for (std::uint32_t y = 0u; y < 32u; ++y) {
      for (std::uint32_t x = 0u; x < 48u; ++x) {
        const std::size_t off = (static_cast<std::size_t>(y) * 48u + x) * 4u;
        float md = 0.0f;
        for (int c = 0; c < 4; ++c) {
          md = std::max(md, std::abs(a->pixels[off + c] - b->pixels[off + c]));
        }
        if (md > 1e-3f) {
          PixDiff pd{x, y, md, {}, {}};
          for (int c = 0; c < 4; ++c) { pd.a[c] = a->pixels[off + c]; pd.b[c] = b->pixels[off + c]; }
          top.push_back(pd);
        }
      }
    }
    std::sort(top.begin(), top.end(), [](const PixDiff &p, const PixDiff &q) { return p.d > q.d; });
    std::cout << "    diff pixels (>1e-3): " << top.size() << '\n';
    for (std::size_t i = 0u; i < std::min<std::size_t>(top.size(), max_rows); ++i) {
      const auto &p = top[i];
      std::cout << "    px(" << p.x << "," << p.y << ") d=" << p.d
                << " got=(" << p.a[0] << "," << p.a[1] << "," << p.a[2] << "," << p.a[3] << ")"
                << " ref=(" << p.b[0] << "," << p.b[1] << "," << p.b[2] << "," << p.b[3] << ")\n";
    }
  };
  for (const auto sched : {LuisaPathScheduler::wavefront,
                           LuisaPathScheduler::wavefront_staged,
                           LuisaPathScheduler::megakernel}) {
    VirtualTextureCacheStats stats{};
    VirtualTextureCacheStats warm_stats{};
    bool eager = false;
    const bool do_warm =
        (sched == LuisaPathScheduler::wavefront ||
         sched == LuisaPathScheduler::wavefront_staged);
    io::MemoryOutputSink warm_sink;
    const auto sink = render_test_scene(
        device, scene, sched, vt_opts, 48u, 32u, 4u, &stats, &eager,
        do_warm ? &warm_sink : nullptr, do_warm ? &warm_stats : nullptr);
    if (sched == LuisaPathScheduler::megakernel) {
      require(eager, "expected megakernel to activate eager VT fallback");
    } else {
      require(!eager, "expected wavefront scheduler to keep VT active");
      require(stats.virtual_images == 5u, "expected 5 virtual images in Test 5");
    }
    const auto diff = compare_combined_pass(sink, baseline);
    std::cout << "  scheduler=" << static_cast<int>(sched)
              << " eager=" << eager << " decoded=" << stats.images_decoded
              << "/5 rmse=" << diff.rmse << " max_abs=" << diff.max_abs
              << " inv=" << diff.invalid_pixels << '\n';
    if (diff.invalid_pixels != 0u || diff.rmse > 2e-3 || diff.max_abs > 0.02) {
      all_ok = false;
      fail_msg = "Test 5 parity failed: rmse=" + std::to_string(diff.rmse) +
                 " max_abs=" + std::to_string(diff.max_abs);
      print_top_diffs(sink, 5u);
    }
    if (do_warm) {
      const auto warm_diff = compare_combined_pass(warm_sink, baseline);
      const auto batches_delta =
          warm_stats.fault_batches >= stats.fault_batches
              ? (warm_stats.fault_batches - stats.fault_batches)
              : 0ull;
      std::cout << "  scheduler=" << static_cast<int>(sched)
                << " WARM(batches_delta=" << batches_delta << ")"
                << " rmse=" << warm_diff.rmse << " max_abs=" << warm_diff.max_abs
                << " inv=" << warm_diff.invalid_pixels << '\n';
      if (batches_delta != 0u) {
        all_ok = false;
        if (fail_msg.empty()) {
          fail_msg = "Test 5 WARM parity failed: expected batches_delta == 0, got " +
                     std::to_string(batches_delta);
        }
      }
      if (warm_diff.invalid_pixels != 0u || warm_diff.rmse > 2e-3 ||
          warm_diff.max_abs > 0.02) {
        all_ok = false;
        if (fail_msg.empty()) {
          fail_msg = "Test 5 WARM parity failed: rmse=" +
                     std::to_string(warm_diff.rmse) +
                     " max_abs=" + std::to_string(warm_diff.max_abs);
        }
        print_top_diffs(warm_sink, 3u);
      }
    }
  }
  require(all_ok, fail_msg.empty() ? "Test 5 parity failed" : fail_msg);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    const bool test5_only =
        argc > 2 && std::string_view{argv[2]} == "test5";
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running Psycles VT tests on backend: " << backend << '\n';

    if (!test5_only) {
      test_byte4_and_float4_roundtrip(device, stream);
      test_npot_wrap_filter_matrix(device, stream);
      test_lazy_decode_proof(device);
      test_tiny_budget_eviction_stress(device);
    }
    test_multistage_parity(device);

    std::cout << (test5_only ? "VT TEST 5 PASSED on " : "ALL 5 VT TESTS PASSED on ")
              << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
