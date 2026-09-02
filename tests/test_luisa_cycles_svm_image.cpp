#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_image_sampling.h"
#include "cycles_svm_internal.h"
#include "cycles_texture_sampling.h"
#include "luisa_cycles_svm_test_kernel_globals.h"
#include "luisa_shader_shape_test_support.h"
#include "surface_image_box.h"
#include "surface_image_sampling.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;
namespace surface_detail = psycles::luisa_backend::detail;
using SceneImageBinding = surface_detail::CyclesSvmImageBindingGpu;

constexpr auto coordinate_offset = std::uint32_t{0u};
constexpr auto color_offset = std::uint32_t{16u};
constexpr auto alpha_offset = std::uint32_t{19u};
constexpr auto node_word_count = std::uint32_t{3u};

struct TextureSamplingShape {
  std::size_t native_samples{};
  std::size_t texel_reads{};
};

template<typename... Args>
[[nodiscard]] TextureSamplingShape texture_sampling_shape(
    const Kernel1D<Args...> &kernel) {
  auto module = xir::ast_to_xir_translate(
      kernel.function()->function(), {});
  TextureSamplingShape result;
  for (const auto *function : module->function_list()) {
    if (const auto *definition = function->definition()) {
      definition->traverse_instructions(
          [&](const xir::Instruction *instruction) noexcept {
            if (instruction->isa<xir::ResourceQueryInst>()) {
              const auto *query =
                  static_cast<const xir::ResourceQueryInst *>(instruction);
              result.native_samples +=
                  query->op() ==
                  xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER;
            }
            if (instruction->isa<xir::ResourceReadInst>()) {
              const auto *read =
                  static_cast<const xir::ResourceReadInst *>(instruction);
              result.texel_reads +=
                  read->op() ==
                  xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ;
            }
          });
    }
  }
  return result;
}

constexpr std::array source_interpolations{
    ImageInterpolation::closest,
    ImageInterpolation::linear,
    ImageInterpolation::cubic,
    ImageInterpolation::smart,
};
constexpr std::array source_extensions{
    ImageExtension::repeat,
    ImageExtension::extend,
    ImageExtension::clip,
    ImageExtension::mirror,
};
constexpr std::array canonical_interpolations{0u, 1u, 2u, 2u};
constexpr std::array canonical_extensions{0u, 2u, 1u, 3u};
constexpr auto sampling_case_count =
    static_cast<std::uint32_t>(source_interpolations.size() *
                               source_extensions.size());

[[nodiscard]] constexpr auto make_sampling_bindings() noexcept {
  std::array<ImageBinding, sampling_case_count> bindings{};
  auto index = std::size_t{};
  for (const auto interpolation : source_interpolations) {
    for (const auto extension : source_extensions) {
      bindings[index++] = ImageBinding{
          .resource_id = 0u,
          .interpolation = interpolation,
          .extension = extension};
    }
  }
  return bindings;
}

constexpr auto sampling_bindings = make_sampling_bindings();
constexpr std::array projection_binding{
    ImageBinding{.resource_id = 0u,
                 .interpolation = ImageInterpolation::linear,
                 .extension = ImageExtension::repeat}};

template<std::size_t Count>
[[nodiscard]] constexpr auto make_scene_image_bindings(
    const std::array<ImageBinding, Count> &bindings) noexcept {
  std::array<SceneImageBinding, Count> result{};
  for (auto index = std::size_t{}; index < Count; ++index) {
    result[index] = surface_detail::make_cycles_svm_image_binding(
        static_cast<std::uint32_t>(bindings[index].resource_id),
        bindings[index].interpolation, bindings[index].extension);
  }
  return result;
}

constexpr auto sampling_scene_bindings =
    make_scene_image_bindings(sampling_bindings);
constexpr auto projection_scene_binding =
    make_scene_image_bindings(projection_binding);

class ImageKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Expr<BindlessArray> _textures;
  Expr<Buffer<SceneImageBinding>> _bindings;

public:
  ImageKernelGlobals(Expr<BindlessArray> textures,
                     Expr<Buffer<SceneImageBinding>> bindings) noexcept
      : _textures{std::move(textures)}, _bindings{std::move(bindings)} {}

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &, Expr<std::int32_t> image_texture_id,
      const device_svm::Dual2 &uv) const noexcept override {
    return svm_detail::sample_scene_image_2d(
        _textures, _bindings, image_texture_id, uv);
  }
};

class DerivativeProbeKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &, Expr<std::int32_t>,
      const device_svm::Dual2 &uv) const noexcept override {
    return make_float4(uv.dx, uv.dy);
  }
};

class DirectBoxSampler final
    : public surface_detail::SurfaceImageBoxTextureSampler {
private:
  Expr<BindlessArray> _textures;

public:
  explicit DirectBoxSampler(Expr<BindlessArray> textures) noexcept
      : _textures{std::move(textures)} {}

  [[nodiscard]] Float4 sample(Expr<std::uint32_t> texture_handle,
                              Float2 uv) const noexcept override {
    return surface_detail::sample_cycles_texture_2d(
        _textures, texture_handle, uv, 1u, 0u);
  }
};

[[nodiscard]] device_svm::ShaderData make_shader_data(Float3 normal) noexcept {
  const auto identity = make_float4x4(1.0f);
  return device_svm::ShaderData{
      make_float3(0.0f), normal, normal, make_float3(0.0f, 0.0f, -1.0f),
      device_svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.2f, 0.3f, 0u,
      0.25f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      make_float3(1.0f, 0.0f, 0.0f),
      make_float3(0.0f, 1.0f, 0.0f), identity, identity};
}

[[nodiscard]] constexpr std::uint32_t pack_image_offsets(
    std::uint8_t flags) noexcept {
  return static_cast<std::uint32_t>(flags) |
         (coordinate_offset << 8u) |
         (color_offset << 16u) |
         (alpha_offset << 24u);
}

[[nodiscard]] Kernel1D<BindlessArray, Buffer<luisa::float4>>
make_shape_kernel(ImageInterpolation interpolation) {
  return [interpolation](BindlessVar textures,
                         BufferFloat4 output) noexcept {
    const device_svm::Dual2 uv{.val = make_float2(0.37f, 0.61f),
                               .dx = make_float2(0.0f),
                               .dy = make_float2(0.0f)};
    output.write(0u, svm_detail::sample_image_2d(
                         textures, 0u, uv, interpolation,
                         ImageExtension::repeat));
  };
}

[[nodiscard]] Kernel1D<BindlessArray, Buffer<SceneImageBinding>,
                       Buffer<luisa::float4>>
make_scene_table_shape_kernel() {
  return [](BindlessVar textures,
            BufferVar<SceneImageBinding> image_bindings,
            BufferFloat4 output) noexcept {
    const device_svm::Dual2 uv{.val = make_float2(0.37f, 0.61f),
                               .dx = make_float2(0.0f),
                               .dy = make_float2(0.0f)};
    output.write(0u, svm_detail::sample_scene_image_2d(
                         textures, image_bindings, 0, uv));
  };
}

[[nodiscard]] Kernel1D<BindlessArray, Buffer<SceneImageBinding>,
                       Buffer<std::uint32_t>,
                       Buffer<luisa::float3>, Buffer<luisa::float4>,
                       Buffer<std::uint32_t>>
make_sampling_kernel() {
  return [](BindlessVar textures, BufferVar<SceneImageBinding> image_bindings,
            BufferUInt words,
            BufferFloat3 coordinates, BufferFloat4 output,
            BufferUInt cursors) noexcept {
    const UInt index = dispatch_x();
    $if(index < sampling_case_count) {
      const Float3 coordinate = coordinates.read(index);
      svm_detail::Stack stack;
      svm_detail::stack_store_float3(stack, coordinate_offset, coordinate);
      svm_detail::stack_store_float3(
          stack, coordinate_offset + 3u,
          make_float3(0.013f, -0.021f, 0.0f));
      svm_detail::stack_store_float3(
          stack, coordinate_offset + 6u,
          make_float3(-0.017f, 0.009f, 0.0f));

      auto shader_data = make_shader_data(make_float3(0.0f, 0.0f, 1.0f));
      const ImageKernelGlobals kernel_globals{textures, image_bindings};
      UInt cursor_offset = index * node_word_count;
      svm_detail::Cursor cursor{words, cursor_offset};
      svm_detail::node_tex_image(
          cursor, stack, kernel_globals, shader_data, true);

      const auto color = svm_detail::stack_load_float3(stack, color_offset);
      const auto alpha = svm_detail::stack_load_float(stack, alpha_offset);
      output.write(index * 2u, make_float4(color, alpha));

      Float4 expected = make_float4(0.0f);
      for (auto interpolation = std::size_t{};
           interpolation < source_interpolations.size(); ++interpolation) {
        for (auto extension = std::size_t{};
             extension < source_extensions.size(); ++extension) {
          const auto case_index =
              interpolation * source_extensions.size() + extension;
          $if(index == static_cast<std::uint32_t>(case_index)) {
            expected = surface_detail::sample_cycles_texture_2d(
                textures, 0u,
                make_float2(coordinate.x, 1.0f - coordinate.y),
                canonical_interpolations[interpolation],
                canonical_extensions[extension]);
          };
        }
      }
      output.write(index * 2u + 1u, expected);
      cursors.write(index, cursor_offset - index * node_word_count);
    };
  };
}

enum class ProjectionHandler : std::uint32_t {
  image,
  box,
  environment,
};

struct ProjectionCase {
  ProjectionHandler handler;
  luisa::float3 coordinate;
  luisa::float3 normal;
  luisa::float2 expected_storage_uv;
  std::uint32_t projection;
  float blend;
  std::uint8_t flags;
};

constexpr auto inverse_sqrt_two = 0.70710678118654752440f;
constexpr std::array projection_cases{
    ProjectionCase{ProjectionHandler::image, {0.13f, 0.27f, 0.41f},
                   {0.0f, 0.0f, 1.0f}, {0.13f, 0.73f},
                   NODE_IMAGE_PROJ_FLAT, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::image, {0.5f, 0.5f, 1.0f},
                   {0.0f, 0.0f, 1.0f}, {0.5f, 0.0f},
                   NODE_IMAGE_PROJ_SPHERE, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::image, {0.5f, 0.5f, 0.5f},
                   {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f},
                   NODE_IMAGE_PROJ_SPHERE, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::image, {1.0f, 0.5f, 0.5f},
                   {0.0f, 0.0f, 1.0f}, {0.25f, 0.5f},
                   NODE_IMAGE_PROJ_TUBE, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::image, {0.5f, 0.5f, 0.5f},
                   {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f},
                   NODE_IMAGE_PROJ_TUBE, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::box, {0.13f, 0.27f, 0.41f},
                   {1.0f, 0.0f, 0.0f}, {}, 0u, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::box, {0.13f, 0.27f, 0.41f},
                   {-1.0f, 0.0f, 0.0f}, {}, 0u, 0.0f, 0u},
    ProjectionCase{ProjectionHandler::box, {0.83f, 0.67f, 0.51f},
                   {0.7f, 0.3f, 0.0f}, {}, 0u, 0.5f, 0u},
    ProjectionCase{ProjectionHandler::box, {0.43f, 0.77f, 0.21f},
                   {1.0f, 1.0f, 1.0f}, {}, 0u, 1.0f, 3u},
    ProjectionCase{ProjectionHandler::environment, {0.0f, 0.0f, 1.0f},
                   {0.0f, 0.0f, 1.0f}, {0.5f, 0.0f},
                   NODE_ENVIRONMENT_EQUIRECTANGULAR, 0.0f, 1u},
    ProjectionCase{ProjectionHandler::environment, {0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f},
                   NODE_ENVIRONMENT_EQUIRECTANGULAR, 0.0f, 1u},
    ProjectionCase{ProjectionHandler::environment, {1.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 1.0f},
                   {0.5f * (1.0f + inverse_sqrt_two), 0.5f},
                   NODE_ENVIRONMENT_MIRROR_BALL, 0.0f, 1u},
    ProjectionCase{ProjectionHandler::environment, {0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f},
                   NODE_ENVIRONMENT_MIRROR_BALL, 0.0f, 1u},
    ProjectionCase{ProjectionHandler::image, {0.45f, 0.25f, 0.0f},
                   {0.0f, 0.0f, 1.0f}, {0.45f, 0.75f},
                   NODE_IMAGE_PROJ_FLAT, 0.0f, 3u},
};
constexpr auto projection_case_count =
    static_cast<std::uint32_t>(projection_cases.size());

[[nodiscard]] Kernel1D<BindlessArray, Buffer<SceneImageBinding>,
                       Buffer<std::uint32_t>,
                       Buffer<std::uint32_t>, Buffer<luisa::float3>,
                       Buffer<luisa::float3>, Buffer<luisa::float2>,
                       Buffer<float>, Buffer<std::uint32_t>,
                       Buffer<luisa::float4>, Buffer<std::uint32_t>>
make_projection_kernel() {
  return [](BindlessVar textures, BufferVar<SceneImageBinding> image_bindings,
            BufferUInt words, BufferUInt handlers,
            BufferFloat3 coordinates, BufferFloat3 normals,
            BufferFloat2 expected_uvs, BufferFloat blends,
            BufferUInt flags, BufferFloat4 output,
            BufferUInt cursors) noexcept {
    const UInt index = dispatch_x();
    $if(index < projection_case_count) {
      const UInt handler = handlers.read(index);
      const UInt image_flags = flags.read(index);
      const Float3 coordinate = coordinates.read(index);
      const Float3 normal = normals.read(index);
      svm_detail::Stack stack;
      svm_detail::stack_store_float3(stack, coordinate_offset, coordinate);
      svm_detail::stack_store_float3(
          stack, coordinate_offset + 3u,
          make_float3(0.017f, -0.009f, 0.013f));
      svm_detail::stack_store_float3(
          stack, coordinate_offset + 6u,
          make_float3(-0.011f, 0.019f, -0.007f));

      auto shader_data = make_shader_data(normal);
      const auto identity = make_float4x4(1.0f);
      const device_svm::TransformState transforms{
          identity, identity, identity, identity};
      const ImageKernelGlobals kernel_globals{textures, image_bindings};
      UInt cursor_offset = index * node_word_count;
      svm_detail::Cursor cursor{words, cursor_offset};
      $if(handler == static_cast<std::uint32_t>(ProjectionHandler::box)) {
        svm_detail::node_tex_image_box(
            cursor, stack, kernel_globals, transforms, shader_data, true,
            false);
      }
      $elif(handler ==
            static_cast<std::uint32_t>(ProjectionHandler::environment)) {
        svm_detail::node_tex_environment(
            cursor, stack, kernel_globals, shader_data, true);
      }
      $else {
        svm_detail::node_tex_image(
            cursor, stack, kernel_globals, shader_data, true);
      };

      const auto color = svm_detail::stack_load_float3(stack, color_offset);
      const auto alpha = svm_detail::stack_load_float(stack, alpha_offset);
      output.write(index * 2u, make_float4(color, alpha));

      Float4 expected = make_float4(0.0f);
      $if(handler == static_cast<std::uint32_t>(ProjectionHandler::box)) {
        const DirectBoxSampler sampler{textures};
        const psycles::luisa_backend::SurfaceImageBoxInput input{
            .coordinate = coordinate,
            .signed_normal = normal,
            .blend = blends.read(index),
            .texture_handle = 0u,
            .unassociate_alpha =
                (image_flags & static_cast<std::uint32_t>(
                                   NODE_IMAGE_ALPHA_UNASSOCIATE)) != 0u,
            .encoded_as_srgb =
                (image_flags & static_cast<std::uint32_t>(
                                   NODE_IMAGE_COMPRESS_AS_SRGB)) != 0u};
        expected = surface_detail::evaluate_surface_image_box(input, sampler);
      }
      $else {
        expected = surface_detail::sample_cycles_texture_2d(
            textures, 0u, expected_uvs.read(index), 1u, 0u);
        expected = surface_detail::decode_surface_image_sample(
            expected,
            (image_flags & static_cast<std::uint32_t>(
                               NODE_IMAGE_ALPHA_UNASSOCIATE)) != 0u,
            (image_flags & static_cast<std::uint32_t>(
                               NODE_IMAGE_COMPRESS_AS_SRGB)) != 0u);
      };
      output.write(index * 2u + 1u, expected);
      cursors.write(index, cursor_offset - index * node_word_count);
    };
  };
}

[[nodiscard]] Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                       Buffer<std::uint32_t>>
make_dual_pole_kernel() {
  return [](BufferUInt words, BufferFloat4 output,
            BufferUInt cursors) noexcept {
    svm_detail::Stack stack;
    svm_detail::stack_store_float3(
        stack, coordinate_offset, make_float3(0.0f, 1.0f, 0.0f));
    svm_detail::stack_store_float3(
        stack, coordinate_offset + 3u, make_float3(1.0f, 0.0f, 0.0f));
    svm_detail::stack_store_float3(
        stack, coordinate_offset + 6u, make_float3(0.0f, 0.0f, 1.0f));
    auto shader_data = make_shader_data(make_float3(0.0f, 0.0f, 1.0f));
    const DerivativeProbeKernelGlobals kernel_globals;
    UInt cursor_offset = 0u;
    svm_detail::Cursor cursor{words, cursor_offset};
    svm_detail::node_tex_environment(
        cursor, stack, kernel_globals, shader_data, true);
    const auto color = svm_detail::stack_load_float3(stack, color_offset);
    const auto alpha = svm_detail::stack_load_float(stack, alpha_offset);
    output.write(0u, make_float4(color, alpha));
    cursors.write(0u, cursor_offset);
  };
}

[[nodiscard]] bool approximately_equal(luisa::float4 lhs,
                                       luisa::float4 rhs,
                                       float tolerance = 3.0e-5f) noexcept {
  return std::isfinite(lhs.x) && std::isfinite(lhs.y) &&
         std::isfinite(lhs.z) && std::isfinite(lhs.w) &&
         std::isfinite(rhs.x) && std::isfinite(rhs.y) &&
         std::isfinite(rhs.z) && std::isfinite(rhs.w) &&
         std::abs(lhs.x - rhs.x) <= tolerance &&
         std::abs(lhs.y - rhs.y) <= tolerance &&
         std::abs(lhs.z - rhs.z) <= tolerance &&
         std::abs(lhs.w - rhs.w) <= tolerance;
}

[[nodiscard]] bool compare_pairs(std::span<const luisa::float4> values,
                                 std::span<const std::uint32_t> cursors,
                                 std::string_view label,
                                 std::string_view backend) {
  for (auto index = std::size_t{}; index < cursors.size(); ++index) {
    if (cursors[index] != node_word_count) {
      std::cerr << label << " cursor " << index << " on " << backend
                << " consumed " << cursors[index] << " words, expected "
                << node_word_count << '\n';
      return false;
    }
    const auto actual = values[index * 2u];
    const auto expected = values[index * 2u + 1u];
    if (!approximately_equal(actual, expected)) {
      std::cerr << label << " case " << index << " mismatch on " << backend
                << ": actual={" << actual.x << ", " << actual.y << ", "
                << actual.z << ", " << actual.w << "}, expected={"
                << expected.x << ", " << expected.y << ", " << expected.z
                << ", " << expected.w << "}\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool verify_native_sampling_shape() {
  constexpr std::array expected_samples{1u, 1u, 4u, 4u};
  for (auto index = std::size_t{}; index < source_interpolations.size();
       ++index) {
    const auto shape = texture_sampling_shape(
        make_shape_kernel(source_interpolations[index]));
    if (shape.native_samples != expected_samples[index] ||
        shape.texel_reads != 0u) {
      std::cerr << "Cycles SVM image mode " << index
                << " lowered to " << shape.native_samples
                << " native samples and " << shape.texel_reads
                << " explicit texel reads; expected "
                << expected_samples[index] << " and 0\n";
      return false;
    }
  }
  // The runtime table emits the Cartesian product of three canonical
  // interpolation families and four extension modes exactly once. Its shape
  // is therefore 4 * (1 + 1 + 4) native samples for every scene size.
  constexpr auto expected_scene_table_samples = std::size_t{24u};
  const auto scene_table_shape =
      texture_sampling_shape(make_scene_table_shape_kernel());
  if (scene_table_shape.native_samples != expected_scene_table_samples ||
      scene_table_shape.texel_reads != 0u) {
    std::cerr << "Cycles SVM scene image table lowered to "
              << scene_table_shape.native_samples << " native samples and "
              << scene_table_shape.texel_reads
              << " explicit texel reads; expected "
              << expected_scene_table_samples << " and 0\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool test_sampling_modes(Device &device, Stream &stream,
                                       std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(sampling_case_count * node_word_count);
  std::array<luisa::float3, sampling_case_count> coordinates{};
  constexpr std::array extension_coordinates{
      luisa::float3{0.10f, 0.10f, 0.0f},
      luisa::float3{1.15f, -0.15f, 0.0f},
      luisa::float3{-0.10f, 0.35f, 0.0f},
      luisa::float3{1.10f, 0.65f, 0.0f},
  };
  for (auto index = std::uint32_t{}; index < sampling_case_count; ++index) {
    words.emplace_back(index);
    words.emplace_back(static_cast<std::uint32_t>(NODE_IMAGE_PROJ_FLAT));
    words.emplace_back(pack_image_offsets(0u));
    coordinates[index] =
        extension_coordinates[index % source_extensions.size()];
  }

  constexpr auto width = std::uint32_t{3u};
  constexpr auto height = std::uint32_t{2u};
  constexpr std::array pixels{
      luisa::float4{0.10f, 0.20f, 0.30f, 0.40f},
      luisa::float4{0.50f, 0.60f, 0.70f, 0.80f},
      luisa::float4{0.90f, 1.00f, 0.10f, 0.20f},
      luisa::float4{0.30f, 0.40f, 0.50f, 0.60f},
      luisa::float4{0.70f, 0.80f, 0.90f, 1.00f},
      luisa::float4{0.20f, 0.30f, 0.40f, 0.50f},
  };
  auto image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
  auto textures = device.create_bindless_array(1u);
  textures.emplace_on_update(0u, image, Sampler::linear_point_repeat());
  auto image_binding_buffer =
      device.create_buffer<SceneImageBinding>(sampling_scene_bindings.size());
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto coordinate_buffer =
      device.create_buffer<luisa::float3>(sampling_case_count);
  auto output_buffer =
      device.create_buffer<luisa::float4>(sampling_case_count * 2u);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(sampling_case_count);
  auto shader = device.compile(
      make_sampling_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, sampling_case_count * 2u> output{};
  std::array<std::uint32_t, sampling_case_count> cursors{};
  stream << image.copy_from(luisa::span{pixels})
         << textures.update()
         << image_binding_buffer.copy_from(
                luisa::span{sampling_scene_bindings})
         << word_buffer.copy_from(luisa::span{words})
         << coordinate_buffer.copy_from(luisa::span{coordinates})
         << shader(textures, image_binding_buffer, word_buffer,
                   coordinate_buffer, output_buffer, cursor_buffer)
                .dispatch(sampling_case_count)
         << output_buffer.copy_to(luisa::span{output})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  return compare_pairs(output, cursors, "Image sampler", backend);
}

[[nodiscard]] bool test_projections(Device &device, Stream &stream,
                                    std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(projection_case_count * node_word_count);
  std::array<std::uint32_t, projection_case_count> handlers{};
  std::array<luisa::float3, projection_case_count> coordinates{};
  std::array<luisa::float3, projection_case_count> normals{};
  std::array<luisa::float2, projection_case_count> expected_uvs{};
  std::array<float, projection_case_count> blends{};
  std::array<std::uint32_t, projection_case_count> flags{};
  for (auto index = std::size_t{}; index < projection_cases.size(); ++index) {
    const auto item = projection_cases[index];
    words.emplace_back(0u);
    words.emplace_back(
        item.handler == ProjectionHandler::box
            ? std::bit_cast<std::uint32_t>(item.blend)
            : item.projection);
    words.emplace_back(pack_image_offsets(item.flags));
    handlers[index] = static_cast<std::uint32_t>(item.handler);
    coordinates[index] = item.coordinate;
    normals[index] = item.normal;
    expected_uvs[index] = item.expected_storage_uv;
    blends[index] = item.blend;
    flags[index] = item.flags;
  }

  constexpr std::array pixels{
      luisa::float4{0.08f, 0.18f, 0.28f, 0.35f},
      luisa::float4{0.42f, 0.52f, 0.62f, 0.75f},
      luisa::float4{0.86f, 0.96f, 0.16f, 0.25f},
      luisa::float4{0.24f, 0.34f, 0.44f, 0.55f},
      luisa::float4{0.68f, 0.78f, 0.88f, 0.95f},
      luisa::float4{0.14f, 0.24f, 0.34f, 0.45f},
  };
  auto image = device.create_image<float>(PixelStorage::FLOAT4, 3u, 2u);
  auto textures = device.create_bindless_array(1u);
  textures.emplace_on_update(0u, image, Sampler::linear_point_repeat());
  auto image_binding_buffer = device.create_buffer<SceneImageBinding>(
      projection_scene_binding.size());
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto handler_buffer =
      device.create_buffer<std::uint32_t>(projection_case_count);
  auto coordinate_buffer =
      device.create_buffer<luisa::float3>(projection_case_count);
  auto normal_buffer =
      device.create_buffer<luisa::float3>(projection_case_count);
  auto uv_buffer =
      device.create_buffer<luisa::float2>(projection_case_count);
  auto blend_buffer = device.create_buffer<float>(projection_case_count);
  auto flag_buffer =
      device.create_buffer<std::uint32_t>(projection_case_count);
  auto output_buffer =
      device.create_buffer<luisa::float4>(projection_case_count * 2u);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(projection_case_count);
  auto shader = device.compile(
      make_projection_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, projection_case_count * 2u> output{};
  std::array<std::uint32_t, projection_case_count> cursors{};
  stream << image.copy_from(luisa::span{pixels})
         << textures.update()
         << image_binding_buffer.copy_from(
                luisa::span{projection_scene_binding})
         << word_buffer.copy_from(luisa::span{words})
         << handler_buffer.copy_from(luisa::span{handlers})
         << coordinate_buffer.copy_from(luisa::span{coordinates})
         << normal_buffer.copy_from(luisa::span{normals})
         << uv_buffer.copy_from(luisa::span{expected_uvs})
         << blend_buffer.copy_from(luisa::span{blends})
         << flag_buffer.copy_from(luisa::span{flags})
         << shader(textures, image_binding_buffer, word_buffer,
                   handler_buffer, coordinate_buffer, normal_buffer,
                   uv_buffer, blend_buffer, flag_buffer, output_buffer,
                   cursor_buffer)
                .dispatch(projection_case_count)
         << output_buffer.copy_to(luisa::span{output})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  return compare_pairs(output, cursors, "Image projection", backend);
}

[[nodiscard]] bool test_dual_mirrorball_pole(
    Device &device, Stream &stream, std::string_view backend) {
  const std::array words{
      std::uint32_t{0u},
      static_cast<std::uint32_t>(NODE_ENVIRONMENT_MIRROR_BALL),
      pack_image_offsets(0u),
  };
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      make_dual_pole_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, 1u> output{};
  std::array<std::uint32_t, 1u> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{output})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  const auto value = output.front();
  if (cursors.front() != node_word_count ||
      !std::isinf(value.x) ||
      !std::isnan(value.y) || !std::isnan(value.z) ||
      !std::isinf(value.w)) {
    std::cerr << "Cycles dual mirror-ball pole mismatch on " << backend
              << ": value={" << value.x << ", " << value.y << ", "
              << value.z << ", " << value.w << "}, cursor="
              << cursors.front() << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  if (!verify_native_sampling_shape()) {
    return EXIT_FAILURE;
  }
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_sampling_modes(device, stream, backend) &&
                 test_projections(device, stream, backend) &&
                 test_dual_mirrorball_pole(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
