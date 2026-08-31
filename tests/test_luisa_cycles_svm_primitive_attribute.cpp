#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

constexpr auto id_object = static_cast<std::uint64_t>(ATTR_STD_NUM) + 0u;
constexpr auto id_vertex = static_cast<std::uint64_t>(ATTR_STD_NUM) + 1u;
constexpr auto id_byte = static_cast<std::uint64_t>(ATTR_STD_NUM) + 2u;
constexpr auto id_normal = static_cast<std::uint64_t>(ATTR_STD_NUM) + 3u;
constexpr auto id_face = static_cast<std::uint64_t>(ATTR_STD_NUM) + 4u;
constexpr auto id_mesh = static_cast<std::uint64_t>(ATTR_STD_NUM) + 5u;
constexpr auto id_none = static_cast<std::uint64_t>(ATTR_STD_NUM) + 6u;
constexpr auto id_odd = static_cast<std::uint64_t>(ATTR_STD_NUM) + 7u;
constexpr auto id_missing = static_cast<std::uint64_t>(ATTR_STD_NUM) + 8u;
constexpr auto id_float3 = static_cast<std::uint64_t>(ATTR_STD_NUM) + 9u;
constexpr auto id_curve_key = static_cast<std::uint64_t>(ATTR_STD_NUM) + 10u;
constexpr auto id_curve = static_cast<std::uint64_t>(ATTR_STD_NUM) + 11u;
constexpr auto id_point = static_cast<std::uint64_t>(ATTR_STD_NUM) + 12u;

constexpr auto case_count = 17u;
constexpr auto vertex_color_case_count = 9u;

class BufferKernelGlobals final : public device_svm::KernelGlobals {
private:
  Expr<Buffer<AttributeMap>> _attribute_map;
  Expr<Buffer<float>> _attribute_float;
  Expr<Buffer<luisa::float2>> _attribute_float2;
  Expr<Buffer<packed_float3>> _attribute_float3;
  Expr<Buffer<luisa::float4>> _attribute_float4;
  Expr<Buffer<uchar4>> _attribute_uchar4;
  Expr<Buffer<packed_normal>> _attribute_normal;
  Expr<Buffer<luisa::uint3>> _triangle_indices;
  Expr<Buffer<KernelCurve>> _curves;
  Expr<Buffer<std::uint32_t>> _object_map_offsets;
  Bool _film_is_rec709;

public:
  BufferKernelGlobals(Expr<Buffer<AttributeMap>> attribute_map,
                      Expr<Buffer<float>> attribute_float,
                      Expr<Buffer<luisa::float2>> attribute_float2,
                      Expr<Buffer<packed_float3>> attribute_float3,
                      Expr<Buffer<luisa::float4>> attribute_float4,
                      Expr<Buffer<uchar4>> attribute_uchar4,
                      Expr<Buffer<packed_normal>> attribute_normal,
                      Expr<Buffer<luisa::uint3>> triangle_indices,
                      Expr<Buffer<KernelCurve>> curves,
                      Expr<Buffer<std::uint32_t>> object_map_offsets,
                      Expr<bool> film_is_rec709) noexcept
      : _attribute_map{attribute_map}, _attribute_float{attribute_float},
        _attribute_float2{attribute_float2},
        _attribute_float3{attribute_float3},
        _attribute_float4{attribute_float4},
        _attribute_uchar4{attribute_uchar4},
        _attribute_normal{attribute_normal},
        _triangle_indices{triangle_indices},
        _curves{curves},
        _object_map_offsets{object_map_offsets},
        _film_is_rec709{film_is_rec709} {}

  [[nodiscard]] device_svm::TriangleVertices
  triangle_vertices(Expr<std::uint32_t>,
                    Expr<std::uint32_t>) const noexcept override {
    return {.v0 = make_float3(0.0f),
            .v1 = make_float3(1.0f, 0.0f, 0.0f),
            .v2 = make_float3(0.0f, 1.0f, 0.0f)};
  }

  [[nodiscard]] device_svm::TriangleVertices
  motion_triangle_vertices(Expr<std::uint32_t> object, Expr<std::uint32_t> prim,
                           Expr<float>) const noexcept override {
    return triangle_vertices(object, prim);
  }

  [[nodiscard]] Float3 film_rgb_to_y() const noexcept override {
    return make_float3(0.2126f, 0.7152f, 0.0722f);
  }

  [[nodiscard]] Float3
  primitive_tangent(const device_svm::ShaderData &) const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] device_svm::Dual3 primitive_tangent_derivative(
      const device_svm::ShaderData &) const noexcept override {
    return {.val = make_float3(1.0f, 0.0f, 0.0f),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
  }

  [[nodiscard]] UInt object_attribute_map_offset(
      Expr<std::uint32_t> object) const noexcept override {
    return _object_map_offsets.read(object);
  }

  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    return _attribute_map.read(offset);
  }

  [[nodiscard]] Float
  attribute_float(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_float.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Float2
  attribute_float2(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_float2.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Var<packed_float3>
  attribute_float3(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_float3.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Float4
  attribute_float4(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_float4.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Var<uchar4>
  attribute_uchar4(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_uchar4.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Var<packed_normal>
  attribute_normal(Expr<std::int32_t> offset) const noexcept override {
    return _attribute_normal.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] UInt3
  triangle_vertex_indices(Expr<std::uint32_t> prim) const noexcept override {
    return _triangle_indices.read(prim);
  }

  [[nodiscard]] Int
  object_normal_offset(Expr<std::uint32_t>) const noexcept override {
    return 0;
  }
  [[nodiscard]] UInt
  object_num_geom_steps(Expr<std::uint32_t>) const noexcept override {
    return 2u;
  }
  [[nodiscard]] Int
  object_num_vertices(Expr<std::uint32_t>) const noexcept override {
    return 3;
  }
  [[nodiscard]] Int
  object_num_primitives(Expr<std::uint32_t>) const noexcept override {
    return 1;
  }
  [[nodiscard]] Float3
  object_dupli_generated(Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }
  [[nodiscard]] Float3
  object_dupli_uv(Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }
  [[nodiscard]] UInt camera_type() const noexcept override {
    return device_svm::camera_perspective;
  }
  [[nodiscard]] Float camera_width() const noexcept override { return 1.0f; }
  [[nodiscard]] Float camera_height() const noexcept override { return 1.0f; }
  [[nodiscard]] Float3 camera_world_to_ndc(
      const device_svm::ShaderData &,
      Expr<luisa::float3> position) const noexcept override {
    return position;
  }

  [[nodiscard]] Var<KernelCurve>
  curve(Expr<std::uint32_t> prim) const noexcept override {
    return _curves.read(prim);
  }

  [[nodiscard]] Bool film_is_rec709() const noexcept override {
    return _film_is_rec709;
  }

  [[nodiscard]] Float3 film_rec709_to_r() const noexcept override {
    return make_float3(0.0f, 1.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_g() const noexcept override {
    return make_float3(0.0f, 0.0f, 1.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_b() const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] Float3 object_inverse_position_transform_if_object(
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value;
  }

  [[nodiscard]] device_svm::Dual3
  object_inverse_position_transform_if_object_derivative(
      const device_svm::ShaderData &,
      const device_svm::Dual3 &value) const noexcept override {
    return value;
  }

  [[nodiscard]] Float3 object_inverse_position_transform(
      const device_svm::ShaderData &,
      Expr<luisa::float3> value) const noexcept override {
    return value;
  }

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &,
      Expr<std::int32_t>,
      const device_svm::Dual2 &) const noexcept override {
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  [[nodiscard]] Float4 kernel_image_interp_3d(
      device_svm::ShaderData &, Expr<std::int32_t>,
      Expr<luisa::float3>, Expr<std::int32_t>,
      Expr<bool>) const noexcept override {
    return make_float4(0.0f);
  }
};

[[nodiscard]] device_svm::ShaderData
make_shader_data(Expr<std::uint32_t> object,
                 Expr<std::uint32_t> prim,
                 Expr<std::uint32_t> primitive_type) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, -1.0f),
          primitive_type,
          0u,
          0u,
          0u,
          prim,
          0.2f,
          0.3f,
          object,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.1f,
          -0.2f,
          0.3f,
          0.4f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 4.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] AttributeMap map_entry(std::uint64_t id, std::int32_t offset,
                                     AttributeElement element,
                                     NodeAttributeType type) noexcept {
  return {.id = id,
          .offset = offset,
          .element = static_cast<std::uint16_t>(element),
          .type = static_cast<std::uint8_t>(type),
          .pad = 0u};
}

[[nodiscard]] AttributeMap terminator(bool chain,
                                      std::int32_t offset) noexcept {
  return {.id = static_cast<std::uint64_t>(ATTR_STD_NONE),
          .offset = chain ? offset : 0,
          .element = static_cast<std::uint16_t>(chain),
          .type = 0u,
          .pad = 0u};
}

[[nodiscard]] constexpr std::uint32_t pack_vertex_color_node(
    std::uint8_t layer_id, std::uint8_t color_offset,
    std::uint8_t alpha_offset, NodeBumpOffset bump_offset) noexcept {
  return static_cast<std::uint32_t>(layer_id) |
         (static_cast<std::uint32_t>(color_offset) << 8u) |
         (static_cast<std::uint32_t>(alpha_offset) << 16u) |
         (static_cast<std::uint32_t>(bump_offset) << 24u);
}

[[nodiscard]] std::array<bool, NODE_NUM> vertex_color_node_types() {
  std::array<bool, NODE_NUM> types{};
  for (const auto type : {NODE_END, NODE_SHADER_JUMP, NODE_VERTEX_COLOR,
                          NODE_EMISSION_WEIGHT, NODE_CLOSURE_EMISSION}) {
    types[type] = true;
  }
  return types;
}

} // namespace

int main(int argc, char **argv) {
  static_assert(sizeof(AttributeMap) == 16u);
  static_assert(offsetof(AttributeMap, offset) == 8u);
  static_assert(offsetof(AttributeMap, element) == 12u);
  static_assert(offsetof(AttributeMap, type) == 14u);
  static_assert(sizeof(uchar4) == 4u);
  static_assert(sizeof(packed_normal) == 4u);
  static_assert(sizeof(KernelCurve) == 16u);

  std::array<AttributeMap, 26u> maps{};
  maps[0] = map_entry(id_object, 0, ATTR_ELEMENT_OBJECT, NODE_ATTR_FLOAT);
  maps[2] = terminator(true, 4);
  maps[3] = terminator(true, 5);
  maps[4] = map_entry(id_vertex, 0, ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT2);
  // The geometry walk advances by ATTR_PRIM_TYPES. This deliberately valid
  // odd entry must therefore remain unreachable from a geometry map offset.
  maps[5] = map_entry(id_odd, 3, ATTR_ELEMENT_MESH, NODE_ATTR_FLOAT4);
  maps[6] = map_entry(id_byte, 0, ATTR_ELEMENT_CORNER_BYTE, NODE_ATTR_RGBA);
  maps[8] =
      map_entry(id_normal, 0, ATTR_ELEMENT_VERTEX_NORMAL, NODE_ATTR_FLOAT3);
  maps[10] = map_entry(id_face, 3, ATTR_ELEMENT_FACE, NODE_ATTR_FLOAT);
  maps[12] = map_entry(id_mesh, 3, ATTR_ELEMENT_MESH, NODE_ATTR_FLOAT4);
  maps[14] = map_entry(id_float3, 0, ATTR_ELEMENT_CORNER, NODE_ATTR_FLOAT3);
  maps[16] =
      map_entry(id_curve_key, 0, ATTR_ELEMENT_CURVE_KEY, NODE_ATTR_FLOAT2);
  maps[18] = map_entry(id_curve, 4, ATTR_ELEMENT_CURVE, NODE_ATTR_FLOAT4);
  maps[20] = map_entry(id_point, 3, ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT3);
  maps[22] = map_entry(id_none, 123, ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT);
  maps[24] = terminator(false, 0);
  maps[25] = terminator(false, 0);

  static constexpr std::array float_values{0.625f, 0.0f, 0.0f, 0.875f};
  static constexpr std::array float2_values{
      luisa::float2{0.1f, 0.2f},
      luisa::float2{0.4f, 0.8f},
      luisa::float2{0.9f, -0.2f},
      luisa::float2{0.2f, 0.6f},
      luisa::float2{0.8f, -0.4f},
  };
  static constexpr std::array float3_values{
      packed_float3{0.2f, 0.4f, 0.6f},
      packed_float3{-0.1f, 0.8f, 0.3f},
      packed_float3{0.9f, -0.2f, 0.5f},
      packed_float3{0.0f, 0.0f, 0.0f},
      packed_float3{0.7f, 0.8f, 0.9f},
  };
  static constexpr std::array float4_values{
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.11f, 0.22f, 0.33f, 0.44f},
      luisa::float4{0.55f, 0.65f, 0.75f, 0.85f},
  };
  static constexpr std::array byte_values{
      uchar4{0u, 64u, 128u, 255u},
      uchar4{255u, 128u, 64u, 128u},
      uchar4{32u, 200u, 10u, 0u},
  };
  static constexpr std::array normal_values{
      packed_normal{0x8000ffffu},
      packed_normal{0xffff8000u},
      packed_normal{0x80008000u},
  };
  static constexpr std::array triangle_indices{
      luisa::uint3{0u, 1u, 2u},
  };
  static constexpr std::array curves{
      KernelCurve{0, 2, 4, 0},
  };
  static constexpr std::array object_offsets{0u, 4u};

  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();

  auto map_buffer = device.create_buffer<AttributeMap>(maps.size());
  auto float_buffer = device.create_buffer<float>(float_values.size());
  auto float2_buffer =
      device.create_buffer<luisa::float2>(float2_values.size());
  auto float3_buffer =
      device.create_buffer<packed_float3>(float3_values.size());
  auto float4_buffer =
      device.create_buffer<luisa::float4>(float4_values.size());
  auto byte_buffer = device.create_buffer<uchar4>(byte_values.size());
  auto normal_buffer =
      device.create_buffer<packed_normal>(normal_values.size());
  auto triangle_buffer =
      device.create_buffer<luisa::uint3>(triangle_indices.size());
  auto curve_buffer = device.create_buffer<KernelCurve>(curves.size());
  auto object_offset_buffer =
      device.create_buffer<std::uint32_t>(object_offsets.size());
  auto output_buffer = device.create_buffer<luisa::float4>(case_count * 3u);
  auto descriptor_buffer = device.create_buffer<luisa::uint4>(case_count);

  const auto kernel = Kernel1D<
      Buffer<AttributeMap>, Buffer<float>, Buffer<luisa::float2>,
      Buffer<packed_float3>, Buffer<luisa::float4>, Buffer<uchar4>,
      Buffer<packed_normal>, Buffer<luisa::uint3>, Buffer<KernelCurve>,
      Buffer<std::uint32_t>, Buffer<luisa::float4>, Buffer<luisa::uint4>>{
      [](BufferVar<AttributeMap> attribute_map, BufferFloat attribute_float,
         BufferVar<luisa::float2> attribute_float2,
         BufferVar<packed_float3> attribute_float3,
         BufferFloat4 attribute_float4, BufferVar<uchar4> attribute_uchar4,
         BufferVar<packed_normal> attribute_normal,
         BufferVar<luisa::uint3> triangle_indices,
         BufferVar<KernelCurve> curves, BufferUInt object_map_offsets,
         BufferFloat4 output,
         BufferUInt4 descriptor_output) noexcept {
        const UInt index = dispatch_x();
        ULong id = static_cast<luisa::ulong>(id_object);
        UInt object = 0u;
        UInt prim = 0u;
        UInt primitive_type = device_svm::primitive_triangle;
        $if(index == 1u) { id = static_cast<luisa::ulong>(id_vertex); }
        $elif(index == 2u) { id = static_cast<luisa::ulong>(id_byte); }
        $elif(index == 3u) { id = static_cast<luisa::ulong>(id_normal); }
        $elif(index == 4u) { id = static_cast<luisa::ulong>(id_face); }
        $elif(index == 5u) {
          id = static_cast<luisa::ulong>(id_mesh);
          prim = device_svm::primitive_none;
        }
        $elif(index == 6u) { id = static_cast<luisa::ulong>(id_missing); }
        $elif(index == 7u) { id = static_cast<luisa::ulong>(id_none); }
        $elif(index == 8u) { id = static_cast<luisa::ulong>(id_odd); }
        $elif(index == 9u) { object = device_svm::object_none; }
        $elif(index == 10u) {
          id = static_cast<luisa::ulong>(id_vertex);
          object = 1u;
        }
        $elif(index == 11u) {
          id = static_cast<luisa::ulong>(id_vertex);
          prim = device_svm::primitive_none;
        }
        $elif(index == 12u) { id = static_cast<luisa::ulong>(id_float3); }
        $elif(index == 13u) { id = static_cast<luisa::ulong>(id_byte); }
        $elif(index == 14u) {
          id = static_cast<luisa::ulong>(id_curve_key);
          primitive_type =
              device_svm::primitive_curve_thick |
              (1u << device_svm::primitive_num_bits);
        }
        $elif(index == 15u) {
          id = static_cast<luisa::ulong>(id_curve);
          primitive_type = device_svm::primitive_curve_thick;
        }
        $elif(index == 16u) {
          id = static_cast<luisa::ulong>(id_point);
          primitive_type = device_svm::primitive_point;
          prim = 1u;
        };

        BufferKernelGlobals kernel_globals{attribute_map,      attribute_float,
                                           attribute_float2,   attribute_float3,
                                           attribute_float4,   attribute_uchar4,
                                           attribute_normal,   triangle_indices,
                                           curves,              object_map_offsets,
                                           index != 13u};
        auto shader_data =
            make_shader_data(object, prim, primitive_type);
        const auto descriptor =
            device_svm::find_attribute(kernel_globals, shader_data, id);
        Float4 value = make_float4(0.0f);
        Float4 dx = make_float4(0.0f);
        Float4 dy = make_float4(0.0f);
        $if(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
          const auto dual =
              device_svm::primitive_surface_attribute_float_derivative(
                  kernel_globals, shader_data, descriptor);
          value.x = dual.val;
          dx.x = dual.dx;
          dy.x = dual.dy;
        }
        $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
          const auto dual =
              device_svm::primitive_surface_attribute_float2_derivative(
                  kernel_globals, shader_data, descriptor);
          value = make_float4(dual.val.x, dual.val.y, 0.0f, 0.0f);
          dx = make_float4(dual.dx.x, dual.dx.y, 0.0f, 0.0f);
          dy = make_float4(dual.dy.x, dual.dy.y, 0.0f, 0.0f);
        }
        $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT3)) {
          const auto dual =
              device_svm::primitive_surface_attribute_float3_derivative(
                  kernel_globals, shader_data, descriptor);
          value = make_float4(dual.val, 0.0f);
          dx = make_float4(dual.dx, 0.0f);
          dy = make_float4(dual.dy, 0.0f);
        }
        $else {
          const auto dual =
              device_svm::primitive_surface_attribute_float4_derivative(
                  kernel_globals, shader_data, descriptor);
          value = dual.val;
          dx = dual.dx;
          dy = dual.dy;
        };
        output.write(index * 3u, value);
        output.write(index * 3u + 1u, dx);
        output.write(index * 3u + 2u, dy);
        descriptor_output.write(
            index,
            make_uint4(descriptor.element, descriptor.type,
                       descriptor.offset.bitcast<std::uint32_t>(),
                       select(0u, 1u,
                              descriptor.offset != static_cast<std::int32_t>(
                                                       ATTR_STD_NOT_FOUND))));
      }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});

  std::array<luisa::float4, case_count * 3u> actual{};
  std::array<luisa::uint4, case_count> descriptors{};
  stream << map_buffer.copy_from(luisa::span{maps})
         << float_buffer.copy_from(luisa::span{float_values})
         << float2_buffer.copy_from(luisa::span{float2_values})
         << float3_buffer.copy_from(luisa::span{float3_values})
         << float4_buffer.copy_from(luisa::span{float4_values})
         << byte_buffer.copy_from(luisa::span{byte_values})
         << normal_buffer.copy_from(luisa::span{normal_values})
         << triangle_buffer.copy_from(luisa::span{triangle_indices})
         << curve_buffer.copy_from(luisa::span{curves})
         << object_offset_buffer.copy_from(luisa::span{object_offsets})
         << shader(map_buffer, float_buffer, float2_buffer, float3_buffer,
                   float4_buffer, byte_buffer, normal_buffer, triangle_buffer,
                   curve_buffer, object_offset_buffer, output_buffer,
                   descriptor_buffer)
                .dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << descriptor_buffer.copy_to(luisa::span{descriptors})
         << synchronize();

  static constexpr auto not_found_bits = std::bit_cast<std::uint32_t>(
      static_cast<std::int32_t>(ATTR_STD_NOT_FOUND));
  static constexpr std::array<luisa::uint4, case_count> expected_descriptors{
      luisa::uint4{ATTR_ELEMENT_OBJECT, NODE_ATTR_FLOAT, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT2, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_CORNER_BYTE, NODE_ATTR_RGBA, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_VERTEX_NORMAL, NODE_ATTR_FLOAT3, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_FACE, NODE_ATTR_FLOAT, 3u, 1u},
      luisa::uint4{ATTR_ELEMENT_MESH, NODE_ATTR_FLOAT4, 3u, 1u},
      luisa::uint4{ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, not_found_bits, 0u},
      luisa::uint4{ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, not_found_bits, 0u},
      luisa::uint4{ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, not_found_bits, 0u},
      luisa::uint4{ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, not_found_bits, 0u},
      luisa::uint4{ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT2, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, not_found_bits, 0u},
      luisa::uint4{ATTR_ELEMENT_CORNER, NODE_ATTR_FLOAT3, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_CORNER_BYTE, NODE_ATTR_RGBA, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_CURVE_KEY, NODE_ATTR_FLOAT2, 0u, 1u},
      luisa::uint4{ATTR_ELEMENT_CURVE, NODE_ATTR_FLOAT4, 4u, 1u},
      luisa::uint4{ATTR_ELEMENT_VERTEX, NODE_ATTR_FLOAT3, 3u, 1u},
  };
  static constexpr std::array<luisa::float4, case_count * 3u> expected{
      luisa::float4{0.625f, 0.0f, 0.0f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.4f, 0.2f, 0.0f, 0.0f},
      luisa::float4{0.27f, -0.06f, 0.0f, 0.0f},
      luisa::float4{0.26f, -0.28f, 0.0f, 0.0f},
      luisa::float4{0.204333156f, 0.242081016f, 0.119094752f, 0.600392163f},
      luisa::float4{0.104333155f, 0.174352452f, -0.0803067014f, -0.349803925f},
      luisa::float4{-0.194222465f, 0.17760624f, -0.0522118993f, -0.300392151f},
      luisa::float4{0.50000459f, 0.200004578f, 0.299989343f, 0.0f},
      luisa::float4{-0.399995416f, 0.100004576f, 0.300004601f, 0.0f},
      luisa::float4{-0.199993894f, -0.199993894f, 0.400006086f, 0.0f},
      luisa::float4{0.875f, 0.0f, 0.0f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.11f, 0.22f, 0.33f, 0.44f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.4f, 0.2f, 0.0f, 0.0f},
      luisa::float4{0.27f, -0.06f, 0.0f, 0.0f},
      luisa::float4{0.26f, -0.28f, 0.0f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.35f, 0.3f, 0.51f, 0.0f},
      luisa::float4{0.18f, -0.14f, -0.06f, 0.0f},
      luisa::float4{0.34f, -0.32f, 0.02f, 0.0f},
      luisa::float4{0.242081016f, 0.119094752f, 0.204333156f, 0.600392163f},
      luisa::float4{0.174352452f, -0.0803067014f, 0.104333155f, -0.349803925f},
      luisa::float4{0.17760624f, -0.0522118993f, -0.194222465f, -0.300392151f},
      luisa::float4{0.32f, 0.4f, 0.0f, 0.0f},
      luisa::float4{0.06f, -0.1f, 0.0f, 0.0f},
      luisa::float4{-0.12f, 0.2f, 0.0f, 0.0f},
      luisa::float4{0.55f, 0.65f, 0.75f, 0.85f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.7f, 0.8f, 0.9f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
  };

  for (auto i = std::size_t{}; i < descriptors.size(); ++i) {
    if (descriptors[i].x != expected_descriptors[i].x ||
        descriptors[i].y != expected_descriptors[i].y ||
        descriptors[i].z != expected_descriptors[i].z ||
        descriptors[i].w != expected_descriptors[i].w) {
      std::cerr << "Cycles AttributeMap descriptor case " << i
                << " mismatch on " << backend << '\n';
      return EXIT_FAILURE;
    }
  }
  for (auto i = std::size_t{}; i < actual.size(); ++i) {
    if (!near(actual[i].x, expected[i].x) ||
        !near(actual[i].y, expected[i].y) ||
        !near(actual[i].z, expected[i].z) ||
        !near(actual[i].w, expected[i].w)) {
      std::cerr << "Cycles primitive attribute lane " << i << " mismatch on "
                << backend << ": (" << actual[i].x << ", " << actual[i].y
                << ", " << actual[i].z << ", " << actual[i].w << ") != ("
                << expected[i].x << ", " << expected[i].y << ", "
                << expected[i].z << ", " << expected[i].w << ")\n";
      return EXIT_FAILURE;
    }
  }

  static constexpr auto vertex_color_offset = std::uint8_t{4u};
  static constexpr auto vertex_alpha_offset = std::uint8_t{7u};
  static constexpr auto invalid_offset =
      static_cast<std::uint8_t>(SVM_STACK_INVALID);
  static constexpr std::array<std::uint32_t,
                              vertex_color_case_count * 2u>
      vertex_color_words{
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_mesh), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_byte), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_float3), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_missing), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_float3), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_DX),
          std::bit_cast<std::uint32_t>(0.5f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_float3), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_DY),
          std::bit_cast<std::uint32_t>(0.25f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_byte), vertex_color_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_DX),
          std::bit_cast<std::uint32_t>(0.5f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_mesh), invalid_offset,
              vertex_alpha_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
          pack_vertex_color_node(
              static_cast<std::uint8_t>(id_mesh), vertex_color_offset,
              invalid_offset, NODE_BUMP_OFFSET_CENTER),
          std::bit_cast<std::uint32_t>(0.0f),
      };
  auto vertex_color_word_buffer =
      device.create_buffer<std::uint32_t>(vertex_color_words.size());
  auto vertex_color_output_buffer =
      device.create_buffer<luisa::float4>(vertex_color_case_count);
  auto vertex_color_cursor_buffer =
      device.create_buffer<std::uint32_t>(vertex_color_case_count);
  const auto vertex_color_kernel =
      Kernel1D<Buffer<AttributeMap>, Buffer<float>, Buffer<luisa::float2>,
               Buffer<packed_float3>, Buffer<luisa::float4>, Buffer<uchar4>,
               Buffer<packed_normal>, Buffer<luisa::uint3>, Buffer<KernelCurve>,
               Buffer<std::uint32_t>, Buffer<std::uint32_t>,
               Buffer<luisa::float4>, Buffer<std::uint32_t>>{
          [](BufferVar<AttributeMap> attribute_map,
             BufferFloat attribute_float,
             BufferVar<luisa::float2> attribute_float2,
             BufferVar<packed_float3> attribute_float3,
             BufferFloat4 attribute_float4,
             BufferVar<uchar4> attribute_uchar4,
             BufferVar<packed_normal> attribute_normal,
             BufferVar<luisa::uint3> triangle_indices,
             BufferVar<KernelCurve> curves,
             BufferUInt object_map_offsets, BufferUInt words,
             BufferFloat4 output, BufferUInt cursor_output) noexcept {
            const UInt index = dispatch_x();
            BufferKernelGlobals kernel_globals{
                attribute_map, attribute_float, attribute_float2,
                attribute_float3, attribute_float4, attribute_uchar4,
                attribute_normal, triangle_indices, curves,
                object_map_offsets, true};
            const auto shader_data = make_shader_data(
                0u, 0u, device_svm::primitive_triangle);
            device_svm::detail::Stack stack;
            for (auto lane = vertex_color_offset;
                 lane <= vertex_alpha_offset; ++lane) {
              stack[lane] = -9.0f;
            }
            UInt offset = index * 2u;
            device_svm::detail::Cursor cursor{words, offset};
            $if((index >= 4u) & (index <= 6u)) {
              device_svm::detail::node_vertex_color_derivative(
                  cursor, stack, kernel_globals, shader_data);
            }
            $else {
              device_svm::detail::node_vertex_color(
                  cursor, stack, kernel_globals, shader_data);
            };
            output.write(
                index,
                make_float4(
                    device_svm::detail::stack_load_float3(
                        stack, static_cast<std::uint32_t>(
                                   vertex_color_offset)),
                    device_svm::detail::stack_load_float(
                        stack, static_cast<std::uint32_t>(
                                   vertex_alpha_offset))));
            cursor_output.write(index, offset);
          }};
  auto vertex_color_shader = device.compile(
      vertex_color_kernel, ShaderOption{.enable_cache = false});
  std::array<luisa::float4, vertex_color_case_count>
      vertex_color_actual{};
  std::array<std::uint32_t, vertex_color_case_count>
      vertex_color_cursors{};
  stream << vertex_color_word_buffer.copy_from(
                luisa::span{vertex_color_words})
         << vertex_color_shader(
                map_buffer, float_buffer, float2_buffer, float3_buffer,
                float4_buffer, byte_buffer, normal_buffer, triangle_buffer,
                curve_buffer, object_offset_buffer, vertex_color_word_buffer,
                vertex_color_output_buffer, vertex_color_cursor_buffer)
                .dispatch(vertex_color_case_count)
         << vertex_color_output_buffer.copy_to(
                luisa::span{vertex_color_actual})
         << vertex_color_cursor_buffer.copy_to(
                luisa::span{vertex_color_cursors})
         << synchronize();

  static constexpr std::array<luisa::float4, vertex_color_case_count>
      vertex_color_expected{
          luisa::float4{0.11f, 0.22f, 0.33f, 0.44f},
          luisa::float4{0.204333156f, 0.242081016f, 0.119094752f,
                        0.600392163f},
          luisa::float4{0.35f, 0.3f, 0.51f, 1.0f},
          luisa::float4{0.0f},
          luisa::float4{0.44f, 0.23f, 0.48f, 1.0f},
          luisa::float4{0.435f, 0.22f, 0.515f, 1.0f},
          luisa::float4{0.256499738f, 0.32925725f, 0.0789414048f,
                        0.425490201f},
          luisa::float4{-9.0f, -9.0f, -9.0f, 0.44f},
          luisa::float4{0.11f, 0.22f, 0.33f, -9.0f},
      };
  for (auto i = std::size_t{}; i < vertex_color_actual.size(); ++i) {
    const auto &value = vertex_color_actual[i];
    const auto &expected_value = vertex_color_expected[i];
    if (!near(value.x, expected_value.x) ||
        !near(value.y, expected_value.y) ||
        !near(value.z, expected_value.z) ||
        !near(value.w, expected_value.w) ||
        vertex_color_cursors[i] != i * 2u + 2u) {
      std::cerr << "Cycles Vertex Color case " << i << " mismatch on "
                << backend << ": (" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "), pc "
                << vertex_color_cursors[i] << '\n';
      return EXIT_FAILURE;
    }
  }

  // Exact local stream of shader 5, `SVM Vertex Color Named Color`, from the
  // Cycles 5.2.1 svm_vertex_color diagnostic dump. Unlike the direct handler
  // matrix above, this exercises ShaderJump, opcode dispatch, cursor advance,
  // EmissionWeight, ClosureEmission, and NODE_END as one interpreter path.
  static constexpr std::array<std::uint32_t, 17u>
      vertex_color_stream_words{
          0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u,
          0x00000017u, 0x00ff0023u, 0x00000000u, 0x00000007u,
          0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
          0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
          0x00000000u};
  auto vertex_color_stream_buffer =
      device.create_buffer<std::uint32_t>(vertex_color_stream_words.size());
  auto vertex_color_stream_output = device.create_buffer<luisa::float4>(1u);
  auto vertex_color_stream_status = device.create_buffer<luisa::uint4>(1u);
  auto vertex_color_stream_maps = maps;
  vertex_color_stream_maps[0] =
      map_entry(id_object, 3, ATTR_ELEMENT_MESH, NODE_ATTR_FLOAT4);
  const auto node_types = vertex_color_node_types();
  const auto vertex_color_stream_kernel =
      Kernel1D<Buffer<AttributeMap>, Buffer<float>, Buffer<luisa::float2>,
               Buffer<packed_float3>, Buffer<luisa::float4>, Buffer<uchar4>,
               Buffer<packed_normal>, Buffer<luisa::uint3>, Buffer<KernelCurve>,
               Buffer<std::uint32_t>, Buffer<std::uint32_t>,
               Buffer<luisa::float4>, Buffer<luisa::uint4>>{
          [node_types](BufferVar<AttributeMap> attribute_map,
                       BufferFloat attribute_float,
                       BufferVar<luisa::float2> attribute_float2,
                       BufferVar<packed_float3> attribute_float3,
                       BufferFloat4 attribute_float4,
                       BufferVar<uchar4> attribute_uchar4,
                       BufferVar<packed_normal> attribute_normal,
                       BufferVar<luisa::uint3> triangle_indices,
                       BufferVar<KernelCurve> curves,
                       BufferUInt object_map_offsets, BufferUInt words,
                       BufferFloat4 output,
                       BufferUInt4 status_output) noexcept {
            BufferKernelGlobals kernel_globals{
                attribute_map, attribute_float, attribute_float2,
                attribute_float3, attribute_float4, attribute_uchar4,
                attribute_normal, triangle_indices, curves,
                object_map_offsets, true};
            auto shader_data = make_shader_data(
                0u, 0u, device_svm::primitive_triangle);
            const auto identity = make_float4x4(1.0f);
            const device_svm::TransformState transforms{
                identity, identity, identity, identity};
            const device_svm::PathState path_state{
                device_svm::path_ray_visibility_camera, 0u};
            device_svm::EvaluationResult result;
            device_svm::eval_nodes(
                kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                device_svm::kernel_feature_node_emission, node_types,
                transforms, shader_data, path_state, result);
            output.write(
                0u, make_float4(shader_data.closure_emission_background,
                                result.closure_weight.x));
            status_output.write(
                0u, make_uint4(result.status, result.final_offset,
                               shader_data.flag, 0u));
          }};
  auto vertex_color_stream_shader = device.compile(
      vertex_color_stream_kernel, ShaderOption{.enable_cache = false});
  luisa::float4 vertex_color_stream_actual{};
  luisa::uint4 vertex_color_stream_state{};
  stream << map_buffer.copy_from(luisa::span{vertex_color_stream_maps})
         << vertex_color_stream_buffer.copy_from(
                luisa::span{vertex_color_stream_words})
         << vertex_color_stream_shader(
                map_buffer, float_buffer, float2_buffer, float3_buffer,
                float4_buffer, byte_buffer, normal_buffer, triangle_buffer,
                curve_buffer, object_offset_buffer,
                vertex_color_stream_buffer, vertex_color_stream_output,
                vertex_color_stream_status)
                .dispatch(1u)
         << vertex_color_stream_output.copy_to(
                luisa::span{&vertex_color_stream_actual, 1u})
         << vertex_color_stream_status.copy_to(
                luisa::span{&vertex_color_stream_state, 1u})
         << synchronize();
  if (!near(vertex_color_stream_actual.x, 0.11f) ||
      !near(vertex_color_stream_actual.y, 0.22f) ||
      !near(vertex_color_stream_actual.z, 0.33f) ||
      !near(vertex_color_stream_actual.w, 0.11f) ||
      vertex_color_stream_state.x !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) ||
      vertex_color_stream_state.y != 15u ||
      (vertex_color_stream_state.z & device_svm::shader_data_emission) == 0u) {
    std::cerr << "Cycles Vertex Color full stream mismatch on " << backend
              << ": (" << vertex_color_stream_actual.x << ", "
              << vertex_color_stream_actual.y << ", "
              << vertex_color_stream_actual.z << ", "
              << vertex_color_stream_actual.w << "), state ("
              << vertex_color_stream_state.x << ", "
              << vertex_color_stream_state.y << ", "
              << vertex_color_stream_state.z << ")\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
