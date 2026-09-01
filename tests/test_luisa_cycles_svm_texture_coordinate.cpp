#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

constexpr auto output_offset = std::uint8_t{8u};
constexpr auto node_word_count = 14u;
constexpr auto plain_case_count = 14u;
constexpr auto derivative_case_count = 15u;

[[nodiscard]] constexpr std::uint32_t pack_node(
    NodeTexCoord type,
    NodeBumpOffset bump_offset = NODE_BUMP_OFFSET_CENTER,
    bool store_derivatives = false) noexcept {
  return static_cast<std::uint32_t>(type) |
         (static_cast<std::uint32_t>(bump_offset) << 8u) |
         (static_cast<std::uint32_t>(store_derivatives) << 16u) |
         (static_cast<std::uint32_t>(output_offset) << 24u);
}

void append_node(std::vector<std::uint32_t> &words, NodeTexCoord type,
                 NodeBumpOffset bump_offset = NODE_BUMP_OFFSET_CENTER,
                 bool store_derivatives = false,
                 float filter_width = 0.0f) {
  words.emplace_back(pack_node(type, bump_offset, store_derivatives));
  words.emplace_back(std::bit_cast<std::uint32_t>(filter_width));

  // Cycles' PackedTransform payload is three float4 rows. Keeping a fixed
  // stride makes each probe independently addressable; only the explicit
  // Object case is allowed to consume these twelve words.
  static constexpr std::array transform{
      2.0f, 0.0f, 0.0f, 1.0f,
      0.0f, 3.0f, 0.0f, 2.0f,
      0.0f, 0.0f, 4.0f, 3.0f,
  };
  for (const auto value : transform) {
    words.emplace_back(std::bit_cast<std::uint32_t>(value));
  }
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 4.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

class ProbeKernelGlobals final : public device_svm::KernelGlobals {
public:
  [[nodiscard]] device_svm::TriangleVertices triangle_vertices(
      Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
    return {.v0 = make_float3(0.0f),
            .v1 = make_float3(1.0f, 0.0f, 0.0f),
            .v2 = make_float3(0.0f, 1.0f, 0.0f)};
  }

  [[nodiscard]] device_svm::TriangleVertices motion_triangle_vertices(
      Expr<std::uint32_t>, Expr<std::uint32_t>,
      Expr<float>) const noexcept override {
    return {.v0 = make_float3(0.0f),
            .v1 = make_float3(1.0f, 0.0f, 0.0f),
            .v2 = make_float3(0.0f, 1.0f, 0.0f)};
  }

  [[nodiscard]] Float3 film_rgb_to_y() const noexcept override {
    return make_float3(0.2126f, 0.7152f, 0.0722f);
  }

  [[nodiscard]] Float3 primitive_tangent(
      const device_svm::ShaderData &) const noexcept override {
    return make_float3(0.0f);
  }

  [[nodiscard]] device_svm::Dual3 primitive_tangent_derivative(
      const device_svm::ShaderData &) const noexcept override {
    return {.val = make_float3(0.0f),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
  }

  [[nodiscard]] UInt object_attribute_map_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 0u;
  }

  [[nodiscard]] Var<AttributeMap> attribute_map(
      Expr<std::uint32_t> offset) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(0u);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
    entry.pad = static_cast<std::uint8_t>(0u);
    $if(offset == 0u) {
      entry.id = static_cast<luisa::ulong>(ATTR_STD_GENERATED_TRANSFORM);
      entry.offset = 0;
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_OBJECT);
      entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT4);
    };
    return entry;
  }

  [[nodiscard]] Float attribute_float(
      Expr<std::int32_t>) const noexcept override {
    return 0.0f;
  }

  [[nodiscard]] Float2 attribute_float2(
      Expr<std::int32_t>) const noexcept override {
    return make_float2(0.0f);
  }

  [[nodiscard]] Var<packed_float3> attribute_float3(
      Expr<std::int32_t>) const noexcept override {
    Var<packed_float3> value;
    value.x = 0.0f;
    value.y = 0.0f;
    value.z = 0.0f;
    return value;
  }

  [[nodiscard]] Float4 attribute_float4(
      Expr<std::int32_t> offset) const noexcept override {
    Float4 value = make_float4(0.1f, 0.0f, 0.0f, 0.4f);
    $if(offset == 1) { value = make_float4(0.0f, 0.2f, 0.0f, 0.5f); }
    $elif(offset == 2) {
      value = make_float4(0.0f, 0.0f, 0.3f, 0.6f);
    };
    return value;
  }

  [[nodiscard]] Var<uchar4> attribute_uchar4(
      Expr<std::int32_t>) const noexcept override {
    Var<uchar4> value;
    value.x = static_cast<std::uint8_t>(0u);
    value.y = static_cast<std::uint8_t>(0u);
    value.z = static_cast<std::uint8_t>(0u);
    value.w = static_cast<std::uint8_t>(0u);
    return value;
  }

  [[nodiscard]] Var<packed_normal> attribute_normal(
      Expr<std::int32_t>) const noexcept override {
    Var<packed_normal> value;
    // Cycles' octahedral encoding of (0, 0, 1). All corners and motion steps
    // deliberately agree so a non-zero derivative would expose bad indexing.
    value.value = 0x80008000u;
    return value;
  }

  [[nodiscard]] UInt3 triangle_vertex_indices(
      Expr<std::uint32_t>) const noexcept override {
    return make_uint3(0u, 1u, 2u);
  }

  [[nodiscard]] Int object_normal_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 16;
  }

  [[nodiscard]] UInt object_num_geom_steps(
      Expr<std::uint32_t>) const noexcept override {
    return 3u;
  }

  [[nodiscard]] Int object_num_vertices(
      Expr<std::uint32_t>) const noexcept override {
    return 3;
  }

  [[nodiscard]] Int object_num_primitives(
      Expr<std::uint32_t>) const noexcept override {
    return 1;
  }

  [[nodiscard]] Float3 object_dupli_generated(
      Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.11f, 0.22f, 0.33f);
  }

  [[nodiscard]] Float3 object_dupli_uv(
      Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.44f, 0.55f, 0.0f);
  }

  [[nodiscard]] UInt camera_type() const noexcept override {
    return device_svm::camera_orthographic;
  }

  [[nodiscard]] Float camera_width() const noexcept override { return 640.0f; }

  [[nodiscard]] Float camera_height() const noexcept override {
    return 480.0f;
  }

  [[nodiscard]] Float3 camera_world_to_ndc(
      const device_svm::ShaderData &,
      Expr<luisa::float3> position) const noexcept override {
    return position * make_float3(0.1f, 0.2f, 0.3f) +
           make_float3(0.01f, 0.02f, 0.03f);
  }

  [[nodiscard]] Var<KernelCurve> curve(
      Expr<std::uint32_t>) const noexcept override {
    Var<KernelCurve> value;
    value.shader_id = 0;
    value.first_key = 0;
    value.num_keys = 0;
    value.type = 0;
    return value;
  }

  [[nodiscard]] Bool film_is_rec709() const noexcept override { return true; }

  [[nodiscard]] Float3 film_xyz_to_r() const noexcept override {
    return make_float3(3.2404542f, -1.5371385f, -0.4985314f);
  }

  [[nodiscard]] Float3 film_xyz_to_g() const noexcept override {
    return make_float3(-0.9692660f, 1.8760108f, 0.0415560f);
  }

  [[nodiscard]] Float3 film_xyz_to_b() const noexcept override {
    return make_float3(0.0556434f, -0.2040259f, 1.0572252f);
  }

  [[nodiscard]] Float3 film_rec709_to_r() const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_g() const noexcept override {
    return make_float3(0.0f, 1.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_b() const noexcept override {
    return make_float3(0.0f, 0.0f, 1.0f);
  }

  [[nodiscard]] Float3 object_inverse_position_transform_if_object(
      const device_svm::ShaderData &shader_data,
      Expr<luisa::float3> value) const noexcept override {
    Float3 result = value;
    $if(shader_data.object != device_svm::object_none) {
      result -= make_float3(1.0f, 2.0f, 3.0f);
    };
    return result;
  }

  [[nodiscard]] device_svm::Dual3
  object_inverse_position_transform_if_object_derivative(
      const device_svm::ShaderData &shader_data,
      const device_svm::Dual3 &value) const noexcept override {
    device_svm::Dual3 result{.val = value.val,
                             .dx = value.dx,
                             .dy = value.dy};
    $if(shader_data.object != device_svm::object_none) {
      result.val -= make_float3(1.0f, 2.0f, 3.0f);
    };
    return result;
  }

  [[nodiscard]] Float3 object_inverse_position_transform(
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value - make_float3(1.0f, 2.0f, 3.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &,
      Expr<std::int32_t>,
      const device_svm::Dual2 &) const noexcept override {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_3d(
      device_svm::ShaderData &, Expr<std::int32_t>, Expr<luisa::float3>,
      Expr<std::int32_t>, Expr<bool>) const noexcept override {
    return make_float4(0.0f);
  }
};

[[nodiscard]] std::vector<std::uint32_t> plain_nodes() {
  std::vector<std::uint32_t> words;
  words.reserve(plain_case_count * node_word_count);
  for (const auto type : {
           NODE_TEXCO_OBJECT,
           NODE_TEXCO_OBJECT_WITH_TRANSFORM,
           NODE_TEXCO_NORMAL,
           NODE_TEXCO_CAMERA,
           NODE_TEXCO_CAMERA,
           NODE_TEXCO_WINDOW,
           NODE_TEXCO_WINDOW,
           NODE_TEXCO_REFLECTION,
           NODE_TEXCO_REFLECTION,
           NODE_TEXCO_DUPLI_GENERATED,
           NODE_TEXCO_DUPLI_UV,
           NODE_TEXCO_DUPLI_GENERATED,
           NODE_TEXCO_VOLUME_GENERATED,
           static_cast<NodeTexCoord>(255u),
       }) {
    append_node(words, type);
  }
  return words;
}

[[nodiscard]] std::vector<std::uint32_t> derivative_nodes() {
  std::vector<std::uint32_t> words;
  words.reserve(derivative_case_count * node_word_count);
  append_node(words, NODE_TEXCO_OBJECT, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_OBJECT_WITH_TRANSFORM, NODE_BUMP_OFFSET_DX,
              true, 0.5f);
  append_node(words, NODE_TEXCO_NORMAL, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_NORMAL, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_NORMAL, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_CAMERA, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_CAMERA, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_WINDOW, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_WINDOW, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_REFLECTION, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_REFLECTION, NODE_BUMP_OFFSET_CENTER, true);
  append_node(words, NODE_TEXCO_DUPLI_GENERATED, NODE_BUMP_OFFSET_CENTER,
              true);
  append_node(words, NODE_TEXCO_VOLUME_GENERATED, NODE_BUMP_OFFSET_CENTER,
              true);
  append_node(words, static_cast<NodeTexCoord>(255u), NODE_BUMP_OFFSET_CENTER,
              true);
  append_node(words, NODE_TEXCO_OBJECT, NODE_BUMP_OFFSET_DY, true, 0.25f);
  return words;
}

[[nodiscard]] auto probe_kernel(std::uint32_t case_count, bool derivatives,
                                bool volume_enabled) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [case_count, derivatives, volume_enabled](BufferUInt words,
                                                BufferFloat4 output,
                                                BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < case_count) {
          UInt object = 0u;
          UInt primitive_type = device_svm::primitive_triangle;
          UInt shader = 0u;
          Float3 normal = make_float3(0.0f, 0.0f, 1.0f);

          if (derivatives) {
            $if(index == 2u) { normal = make_float3(3.0f, 4.0f, 0.0f); }
            $elif(index == 3u) { shader = device_svm::shader_smooth_normal; }
            $elif(index == 4u) {
              shader = device_svm::shader_smooth_normal;
              primitive_type = device_svm::primitive_triangle |
                               device_svm::primitive_motion;
            };
            $if((index == 6u) | (index == 8u) | (index == 10u)) {
              object = device_svm::object_none;
            };
          } else {
            $if((index == 4u) | (index == 6u) | (index == 8u) |
                (index == 11u)) {
              object = device_svm::object_none;
            };
          }

          const auto identity = make_float4x4(1.0f);
          const auto camera_to_world = make_float4x4(
              make_float4(1.0f, 0.0f, 0.0f, 0.0f),
              make_float4(0.0f, 1.0f, 0.0f, 0.0f),
              make_float4(0.0f, 0.0f, 1.0f, 0.0f),
              make_float4(10.0f, 20.0f, 30.0f, 1.0f));
          const auto world_to_camera = make_float4x4(
              make_float4(1.0f, 0.0f, 0.0f, 0.0f),
              make_float4(0.0f, 1.0f, 0.0f, 0.0f),
              make_float4(0.0f, 0.0f, 1.0f, 0.0f),
              make_float4(-10.0f, -20.0f, -30.0f, 1.0f));
          const device_svm::TransformState transforms{
              camera_to_world, world_to_camera, identity, identity};
          device_svm::ShaderData shader_data{
              make_float3(4.0f, 6.0f, 8.0f), normal, normal,
              make_float3(0.0f, 0.0f, -1.0f), primitive_type, shader, 0u,
              0u, 0u, 0.2f, 0.3f, object, 0.25f, 1.0f, 0.0f, 0.2f,
              0.1f, 0.2f, 0.3f, 0.4f, make_float3(1.0f, 0.0f, 0.0f),
              make_float3(0.0f, 1.0f, 0.0f), identity, identity};
          shader_data.ray_P = make_float3(9.0f, 10.0f, 11.0f);
          const device_svm::PathState path_state{
              device_svm::path_ray_visibility_camera, 0u};
          const ProbeKernelGlobals kernel_globals;
          device_svm::detail::Stack stack;
          UInt cursor_offset = index * node_word_count;
          device_svm::detail::Cursor cursor{words, cursor_offset};
          device_svm::detail::node_tex_coord(
              cursor, stack, kernel_globals, transforms, shader_data,
              path_state, derivatives, volume_enabled, true);

          const auto value = device_svm::detail::stack_load_float3(
              stack, static_cast<std::uint32_t>(output_offset));
          Float3 dx = make_float3(0.0f);
          Float3 dy = make_float3(0.0f);
          if (derivatives) {
            dx = device_svm::detail::stack_load_float3(
                stack, static_cast<std::uint32_t>(output_offset) + 3u);
            dy = device_svm::detail::stack_load_float3(
                stack, static_cast<std::uint32_t>(output_offset) + 6u);
          }
          output.write(index * 3u, make_float4(value, 0.0f));
          output.write(index * 3u + 1u, make_float4(dx, 0.0f));
          output.write(index * 3u + 2u, make_float4(dy, 0.0f));
          cursors.write(index, cursor_offset - index * node_word_count);
        };
      }};
}

template<std::size_t N>
[[nodiscard]] bool compare(const std::array<luisa::float4, N> &actual,
                           const std::array<luisa::float4, N> &expected,
                           std::string_view label,
                           std::string_view backend) noexcept {
  for (auto i = std::size_t{}; i < N; ++i) {
    if (!near(actual[i].x, expected[i].x) ||
        !near(actual[i].y, expected[i].y) ||
        !near(actual[i].z, expected[i].z)) {
      std::cerr << label << " lane " << i << " mismatch on " << backend
                << ": (" << actual[i].x << ", " << actual[i].y << ", "
                << actual[i].z << ") != (" << expected[i].x << ", "
                << expected[i].y << ", " << expected[i].z << ")\n";
      return false;
    }
  }
  return true;
}

template<std::size_t N>
[[nodiscard]] bool compare_cursors(
    const std::array<std::uint32_t, N> &actual,
    std::size_t explicit_transform_index,
    std::string_view backend) noexcept {
  for (auto i = std::size_t{}; i < N; ++i) {
    const auto expected = i == explicit_transform_index ? 14u : 2u;
    if (actual[i] != expected) {
      std::cerr << "Texture Coordinate cursor " << i << " mismatch on "
                << backend << ": " << actual[i] << " != " << expected
                << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_plain(Device &device, Stream &stream,
                              std::string_view backend) {
  const auto words = plain_nodes();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(plain_case_count * 3u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(plain_case_count);
  auto shader = device.compile(probe_kernel(plain_case_count, false, true),
                               ShaderOption{.enable_cache = false,
                                            .enable_fast_math = true});
  std::array<luisa::float4, plain_case_count * 3u> actual{};
  std::array<std::uint32_t, plain_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(plain_case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  static constexpr std::array<luisa::float4, plain_case_count * 3u> expected{
      luisa::float4{3.0f, 4.0f, 5.0f, 0.0f}, {}, {},
      luisa::float4{9.0f, 20.0f, 35.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, 1.0f, 0.0f}, {}, {},
      luisa::float4{-6.0f, -14.0f, -22.0f, 0.0f}, {}, {},
      luisa::float4{4.0f, 6.0f, 8.0f, 0.0f}, {}, {},
      luisa::float4{0.41f, 1.22f, 0.0f, 0.0f}, {}, {},
      luisa::float4{0.91f, 2.02f, 0.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, -1.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, -1.0f, 0.0f}, {}, {},
      luisa::float4{0.11f, 0.22f, 0.33f, 0.0f}, {}, {},
      luisa::float4{0.44f, 0.55f, 0.0f, 0.0f}, {}, {},
      luisa::float4{0.0f}, {}, {},
      luisa::float4{0.7f, 1.3f, 2.1f, 0.0f}, {}, {},
      luisa::float4{0.0f}, {}, {},
  };
  return compare(actual, expected, "Texture Coordinate plain", backend) &&
         compare_cursors(cursors, 1u, backend);
}

[[nodiscard]] bool test_derivative(Device &device, Stream &stream,
                                   std::string_view backend) {
  const auto words = derivative_nodes();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(derivative_case_count * 3u);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(derivative_case_count);
  auto shader = device.compile(probe_kernel(derivative_case_count, true, true),
                               ShaderOption{.enable_cache = false,
                                            .enable_fast_math = true});
  std::array<luisa::float4, derivative_case_count * 3u> actual{};
  std::array<std::uint32_t, derivative_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(derivative_case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  constexpr auto dx_window = 1.0f / 640.0f;
  constexpr auto dy_window = 1.0f / 480.0f;
  constexpr auto diagonal = 0.1414213562373095f;
  static constexpr std::array<luisa::float4, derivative_case_count * 3u>
      expected{
      luisa::float4{3.0f, 4.0f, 5.0f, 0.0f},
      luisa::float4{0.1f, 0.3f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
      luisa::float4{9.1f, 20.45f, 35.0f, 0.0f},
      luisa::float4{0.2f, 0.9f, 0.0f, 0.0f},
      luisa::float4{0.4f, 1.2f, 0.0f, 0.0f},
      luisa::float4{0.6f, 0.8f, 0.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, 1.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, 1.0f, 0.0f}, {}, {},
      luisa::float4{-6.0f, -14.0f, -22.0f, 0.0f},
      luisa::float4{0.1f, 0.3f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
      luisa::float4{4.0f, 6.0f, 8.0f, 0.0f},
      luisa::float4{0.1f, 0.3f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
      luisa::float4{0.41f, 1.22f, 0.0f, 0.0f},
      luisa::float4{dx_window, 0.0f, 0.0f, 0.0f},
      luisa::float4{0.0f, dy_window, 0.0f, 0.0f},
      luisa::float4{0.91f, 2.02f, 0.0f, 0.0f}, {}, {},
      luisa::float4{0.0f, 0.0f, -1.0f, 0.0f},
      luisa::float4{diagonal, -diagonal, 0.0f, 0.0f},
      luisa::float4{-diagonal, -diagonal, 0.0f, 0.0f},
      luisa::float4{0.0f, 0.0f, -1.0f, 0.0f},
      luisa::float4{-diagonal, diagonal, 0.0f, 0.0f},
      luisa::float4{diagonal, diagonal, 0.0f, 0.0f},
      luisa::float4{0.11f, 0.22f, 0.33f, 0.0f}, {}, {},
      luisa::float4{0.7f, 1.3f, 2.1f, 0.0f},
      luisa::float4{0.01f, 0.06f, 0.0f, 0.0f},
      luisa::float4{0.02f, 0.08f, 0.0f, 0.0f},
      luisa::float4{0.0f}, {}, {},
      luisa::float4{3.05f, 4.1f, 5.0f, 0.0f},
      luisa::float4{0.1f, 0.3f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
  };
  return compare(actual, expected, "Texture Coordinate derivative", backend) &&
         compare_cursors(cursors, 1u, backend);
}

[[nodiscard]] bool test_volume_disabled(Device &device, Stream &stream,
                                        std::string_view backend) {
  std::vector<std::uint32_t> words;
  append_node(words, NODE_TEXCO_VOLUME_GENERATED, NODE_BUMP_OFFSET_CENTER,
              true);
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(3u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(probe_kernel(1u, true, false),
                               ShaderOption{.enable_cache = false,
                                            .enable_fast_math = true});
  std::array<luisa::float4, 3u> actual{};
  std::array<std::uint32_t, 1u> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  static constexpr std::array<luisa::float4, 3u> expected{
      luisa::float4{4.0f, 6.0f, 8.0f, 0.0f},
      luisa::float4{0.1f, 0.3f, 0.0f, 0.0f},
      luisa::float4{0.2f, 0.4f, 0.0f, 0.0f},
  };
  return compare(actual, expected, "Texture Coordinate volume-off", backend) &&
         compare_cursors(cursors, 1u, backend);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_plain(device, stream, backend) &&
                 test_derivative(device, stream, backend) &&
                 test_volume_disabled(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
