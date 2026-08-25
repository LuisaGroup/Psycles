#pragma once

#include <cstdlib>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <psycles/compiler/shader_program.h>
#include <psycles/contract/surface.h>

namespace psycles::contract {
enum class DisplacementMethod : std::uint8_t;
}

namespace psycles::compiler {

template <typename Tag> struct ProgramId {
  static constexpr auto invalid_value =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t value{invalid_value};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return value != invalid_value;
  }

  auto operator<=>(const ProgramId &) const noexcept = default;
};

struct ParameterTag;
struct ValueExpressionTag;
struct ClosureExpressionTag;
struct VolumeExpressionTag;

using ParameterId = ProgramId<ParameterTag>;
using ValueExpressionId = ProgramId<ValueExpressionTag>;
using ClosureExpressionId = ProgramId<ClosureExpressionTag>;
using VolumeExpressionId = ProgramId<VolumeExpressionTag>;

enum class ParameterSource : std::uint8_t {
  input,
  property
};

// Static Mapping-node metadata. The numeric values are part of the compact
// ValueInstruction encoding consumed by surface backends.
enum class MappingVectorType : std::uint8_t {
  point = 0u,
  texture = 1u,
  vector = 2u,
  normal = 3u
};

struct ParameterDesc {
  ParameterId id;
  contract::NodeId node;
  std::string socket;
  contract::SocketType type{};
  contract::SocketValue default_value;
  ParameterSource source{ParameterSource::input};
};

enum class ValueOperation : std::uint8_t {
  parameter,
  passthrough,
  scalar_to_color,
  scalar_to_boolean,
  color_to_scalar,
  vector_to_scalar,
  add,
  subtract,
  multiply,
  divide,
  minimum,
  maximum,
  power,
  math,
  absolute,
  clamp01,
  clamp_range,
  map_range_float,
  map_range_vector,
  vector_math_value,
  vector_math_vector,
  mix_float,
  mix_vector,
  mix,
  multiply_color,
  hue_saturation,
  invert,
  gamma,
  brightness_contrast,
  blackbody,
  wavelength,
  surface_position,
  shading_normal,
  geometric_normal,
  incoming,
  tangent,
  uv,
  generated,
  object_position,
  object_position_with_transform,
  object_location,
  object_random,
  particle_index,
  particle_random,
  back_facing,
  pointiness,
  random_per_island,
  curve_is_strand,
  curve_intercept,
  curve_length,
  curve_thickness,
  curve_tangent_normal,
  curve_random,
  path_is_camera,
  path_is_shadow,
  path_is_diffuse,
  path_is_glossy,
  path_is_singular,
  path_is_reflection,
  path_is_transmission,
  path_is_volume_scatter,
  path_ray_length,
  path_ray_depth,
  path_diffuse_depth,
  path_glossy_depth,
  path_transparent_depth,
  path_transmission_depth,
  fresnel,
  layer_weight_fresnel,
  layer_weight_facing,
  mapping,
  image_color,
  image_alpha,
  environment_color,
  environment_alpha,
  attribute_color,
  attribute_factor,
  attribute_alpha,
  normal_map,
  bump,
  noise_factor,
  noise_color,
  white_noise_value,
  white_noise_color,
  checker_color,
  checker_factor,
  brick_color,
  brick_factor,
  magic_color,
  magic_factor,
  wave_color,
  wave_factor,
  voronoi_distance,
  voronoi_color,
  voronoi_position,
  voronoi_w,
  voronoi_radius,
  gradient,
  color_ramp,
  rgb_curve,
  separate_r,
  separate_g,
  separate_b,
  combine_color,
  hosek_wilkie_sky,
  nishita_sky
};

// Operand layouts are shared by lowering and backend AST construction. Named
// constexpr indices make each operation's IR contract explicit without
// storing socket names per instruction.
namespace value_operand {

struct unary {
  static constexpr std::size_t input = 0u;
  static constexpr std::size_t count = 1u;
};

struct binary {
  static constexpr std::size_t a = 0u;
  static constexpr std::size_t b = 1u;
  static constexpr std::size_t count = 2u;
};

struct ternary {
  static constexpr std::size_t a = 0u;
  static constexpr std::size_t b = 1u;
  static constexpr std::size_t c = 2u;
  static constexpr std::size_t count = 3u;
};

struct mapping {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t location = 1u;
  static constexpr std::size_t rotation = 2u;
  static constexpr std::size_t scale = 3u;
  static constexpr std::size_t count = 4u;
};

struct clamp_range {
  static constexpr std::size_t value = 0u;
  static constexpr std::size_t minimum = 1u;
  static constexpr std::size_t maximum = 2u;
  static constexpr std::size_t count = 3u;
};

struct map_range {
  static constexpr std::size_t value = 0u;
  static constexpr std::size_t from_min = 1u;
  static constexpr std::size_t from_max = 2u;
  static constexpr std::size_t to_min = 3u;
  static constexpr std::size_t to_max = 4u;
  static constexpr std::size_t steps = 5u;
  static constexpr std::size_t count = 6u;
};

struct vector_math {
  static constexpr std::size_t a = 0u;
  static constexpr std::size_t b = 1u;
  static constexpr std::size_t c = 2u;
  static constexpr std::size_t scale = 3u;
  static constexpr std::size_t count = 4u;
};

struct mix {
  static constexpr std::size_t a = 0u;
  static constexpr std::size_t b = 1u;
  static constexpr std::size_t factor = 2u;
  static constexpr std::size_t count = 3u;
};

struct hue_saturation {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t hue = 1u;
  static constexpr std::size_t saturation = 2u;
  static constexpr std::size_t value = 3u;
  static constexpr std::size_t factor = 4u;
  static constexpr std::size_t count = 5u;
};

struct color_factor {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t factor = 1u;
  static constexpr std::size_t count = 2u;
};

struct gamma {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t exponent = 1u;
  static constexpr std::size_t count = 2u;
};

struct blackbody {
  static constexpr std::size_t temperature = 0u;
  static constexpr std::size_t count = 1u;
};

struct wavelength {
  static constexpr std::size_t nanometers = 0u;
  static constexpr std::size_t count = 1u;
};

struct brightness_contrast {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t brightness = 1u;
  static constexpr std::size_t contrast = 2u;
  static constexpr std::size_t count = 3u;
};

struct normal_map {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t strength = 1u;
  static constexpr std::size_t uv_map = 2u;
  static constexpr std::size_t count = 3u;
};

struct bump {
  static constexpr std::size_t height = 0u;
  static constexpr std::size_t strength = 1u;
  static constexpr std::size_t distance = 2u;
  static constexpr std::size_t filter_width = 3u;
  static constexpr std::size_t normal = 4u;
  static constexpr std::size_t count = 5u;
};

struct layer_weight {
  static constexpr std::size_t blend = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t count = 2u;
};

struct fresnel {
  static constexpr std::size_t ior = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t count = 2u;
};

struct uv {
  static constexpr std::size_t map = 0u;
  static constexpr std::size_t count = 1u;
};

struct image_texture {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t image = 1u;
  static constexpr std::size_t projection_blend = 2u;
  static constexpr std::size_t count = 3u;
};

struct environment_texture {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t image = 1u;
  static constexpr std::size_t count = 2u;
};

struct attribute {
  static constexpr std::size_t id = 0u;
  static constexpr std::size_t count = 1u;
};

struct noise {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t scale = 1u;
  static constexpr std::size_t detail = 2u;
  static constexpr std::size_t roughness = 3u;
  static constexpr std::size_t lacunarity = 4u;
  static constexpr std::size_t distortion = 5u;
  static constexpr std::size_t w = 6u;
  static constexpr std::size_t offset = 7u;
  static constexpr std::size_t gain = 8u;
  static constexpr std::size_t count = 9u;
};

struct white_noise {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t w = 1u;
  static constexpr std::size_t count = 2u;
};

struct checker {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t color1 = 1u;
  static constexpr std::size_t color2 = 2u;
  static constexpr std::size_t scale = 3u;
  static constexpr std::size_t count = 4u;
};

struct brick {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t color1 = 1u;
  static constexpr std::size_t color2 = 2u;
  static constexpr std::size_t mortar = 3u;
  static constexpr std::size_t scale = 4u;
  static constexpr std::size_t mortar_size = 5u;
  static constexpr std::size_t mortar_smooth = 6u;
  static constexpr std::size_t bias = 7u;
  static constexpr std::size_t brick_width = 8u;
  static constexpr std::size_t row_height = 9u;
  static constexpr std::size_t offset_amount = 10u;
  static constexpr std::size_t offset_frequency = 11u;
  static constexpr std::size_t squash_amount = 12u;
  static constexpr std::size_t squash_frequency = 13u;
  static constexpr std::size_t count = 14u;
};

struct magic {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t scale = 1u;
  static constexpr std::size_t distortion = 2u;
  static constexpr std::size_t depth = 3u;
  static constexpr std::size_t count = 4u;
};

struct wave {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t scale = 1u;
  static constexpr std::size_t distortion = 2u;
  static constexpr std::size_t detail = 3u;
  static constexpr std::size_t detail_scale = 4u;
  static constexpr std::size_t detail_roughness = 5u;
  static constexpr std::size_t phase = 6u;
  static constexpr std::size_t count = 7u;
};

struct voronoi {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t w = 1u;
  static constexpr std::size_t scale = 2u;
  static constexpr std::size_t detail = 3u;
  static constexpr std::size_t roughness = 4u;
  static constexpr std::size_t lacunarity = 5u;
  static constexpr std::size_t smoothness = 6u;
  static constexpr std::size_t exponent = 7u;
  static constexpr std::size_t randomness = 8u;
  static constexpr std::size_t count = 9u;
};

struct rgb_curve {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t factor = 1u;
  static constexpr std::size_t min_x = 2u;
  static constexpr std::size_t max_x = 3u;
  static constexpr std::size_t extrapolate = 4u;
  static constexpr std::size_t count = 5u;
};

struct color_ramp {
  static constexpr std::size_t factor = 0u;
  static constexpr std::size_t count = 1u;
};

struct gradient {
  static constexpr std::size_t vector = 0u;
  static constexpr std::size_t count = 1u;
};

struct separate_color {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t count = 1u;
};

struct combine_color {
  static constexpr std::size_t r = 0u;
  static constexpr std::size_t g = 1u;
  static constexpr std::size_t b = 2u;
  static constexpr std::size_t count = 3u;
};

struct sky {
  static constexpr std::size_t direction = 0u;
  static constexpr std::size_t count = 1u;
};

struct nishita_sky {
  static constexpr std::size_t elevation = 0u;
  static constexpr std::size_t rotation = 1u;
  static constexpr std::size_t size = 2u;
  static constexpr std::size_t intensity = 3u;
  static constexpr std::size_t altitude = 4u;
  static constexpr std::size_t air = 5u;
  static constexpr std::size_t dust = 6u;
  static constexpr std::size_t ozone = 7u;
  static constexpr std::size_t direction = 8u;
  static constexpr std::size_t count = 9u;
};

}// namespace value_operand

struct ValueOperandAssignment {
  std::size_t index;
  ValueExpressionId value;
};

// Lowering names every socket-to-operand assignment with the same constexpr
// layout consumed by backends. This prevents a positional initializer from
// silently changing meaning when an operation's operand contract evolves.
template <typename Layout>
[[nodiscard]] std::vector<ValueExpressionId> make_value_operands(
    std::initializer_list<ValueOperandAssignment> assignments) {
  static_assert(Layout::count <= 64u);
  if (assignments.size() != Layout::count) {
    std::abort();
  }
  std::vector<ValueExpressionId> result(Layout::count);
  std::uint64_t assigned{};
  for (const auto [index, value] : assignments) {
    if (index >= Layout::count) {
      std::abort();
    }
    const auto bit = std::uint64_t{1u} << index;
    if ((assigned & bit) != 0u) {
      std::abort();
    }
    assigned |= bit;
    result[index] = value;
  }
  const auto expected = Layout::count == 64u
                            ? ~std::uint64_t{}
                            : (std::uint64_t{1u} << Layout::count) - 1u;
  if (assigned != expected) {
    std::abort();
  }
  return result;
}

[[nodiscard]] constexpr std::size_t
value_operation_operand_count(ValueOperation operation) noexcept {
  switch (operation) {
    case ValueOperation::parameter:
    case ValueOperation::surface_position:
    case ValueOperation::shading_normal:
    case ValueOperation::geometric_normal:
    case ValueOperation::incoming:
    case ValueOperation::tangent:
    case ValueOperation::generated:
    case ValueOperation::object_position:
    case ValueOperation::object_position_with_transform:
    case ValueOperation::object_location:
    case ValueOperation::object_random:
    case ValueOperation::particle_index:
    case ValueOperation::particle_random:
    case ValueOperation::back_facing:
    case ValueOperation::pointiness:
    case ValueOperation::random_per_island:
    case ValueOperation::curve_is_strand:
    case ValueOperation::curve_intercept:
    case ValueOperation::curve_length:
    case ValueOperation::curve_thickness:
    case ValueOperation::curve_tangent_normal:
    case ValueOperation::curve_random:
    case ValueOperation::path_is_camera:
    case ValueOperation::path_is_shadow:
    case ValueOperation::path_is_diffuse:
    case ValueOperation::path_is_glossy:
    case ValueOperation::path_is_singular:
    case ValueOperation::path_is_reflection:
    case ValueOperation::path_is_transmission:
    case ValueOperation::path_is_volume_scatter:
    case ValueOperation::path_ray_length:
    case ValueOperation::path_ray_depth:
    case ValueOperation::path_diffuse_depth:
    case ValueOperation::path_glossy_depth:
    case ValueOperation::path_transparent_depth:
    case ValueOperation::path_transmission_depth:
      return 0u;

    case ValueOperation::passthrough:
    case ValueOperation::scalar_to_color:
    case ValueOperation::scalar_to_boolean:
    case ValueOperation::color_to_scalar:
    case ValueOperation::vector_to_scalar:
    case ValueOperation::absolute:
    case ValueOperation::clamp01:
      return value_operand::unary::count;

    case ValueOperation::add:
    case ValueOperation::subtract:
    case ValueOperation::multiply:
    case ValueOperation::divide:
    case ValueOperation::minimum:
    case ValueOperation::maximum:
    case ValueOperation::power:
      return value_operand::binary::count;

    case ValueOperation::math:
      return value_operand::ternary::count;
    case ValueOperation::clamp_range:
      return value_operand::clamp_range::count;
    case ValueOperation::map_range_float:
    case ValueOperation::map_range_vector:
      return value_operand::map_range::count;
    case ValueOperation::vector_math_value:
    case ValueOperation::vector_math_vector:
      return value_operand::vector_math::count;
    case ValueOperation::mix_float:
    case ValueOperation::mix_vector:
    case ValueOperation::mix:
    case ValueOperation::multiply_color:
      return value_operand::mix::count;
    case ValueOperation::hue_saturation:
      return value_operand::hue_saturation::count;
    case ValueOperation::invert:
      return value_operand::color_factor::count;
    case ValueOperation::gamma:
      return value_operand::gamma::count;
    case ValueOperation::brightness_contrast:
      return value_operand::brightness_contrast::count;
    case ValueOperation::blackbody:
      return value_operand::blackbody::count;
    case ValueOperation::wavelength:
      return value_operand::wavelength::count;
    case ValueOperation::uv:
      return value_operand::uv::count;
    case ValueOperation::fresnel:
      return value_operand::fresnel::count;
    case ValueOperation::layer_weight_fresnel:
    case ValueOperation::layer_weight_facing:
      return value_operand::layer_weight::count;
    case ValueOperation::mapping:
      return value_operand::mapping::count;
    case ValueOperation::image_color:
    case ValueOperation::image_alpha:
      return value_operand::image_texture::count;
    case ValueOperation::environment_color:
    case ValueOperation::environment_alpha:
      return value_operand::environment_texture::count;
    case ValueOperation::attribute_color:
    case ValueOperation::attribute_factor:
    case ValueOperation::attribute_alpha:
      return value_operand::attribute::count;
    case ValueOperation::normal_map:
      return value_operand::normal_map::count;
    case ValueOperation::bump:
      return value_operand::bump::count;
    case ValueOperation::noise_factor:
    case ValueOperation::noise_color:
      return value_operand::noise::count;
    case ValueOperation::white_noise_value:
    case ValueOperation::white_noise_color:
      return value_operand::white_noise::count;
    case ValueOperation::checker_color:
    case ValueOperation::checker_factor:
      return value_operand::checker::count;
    case ValueOperation::brick_color:
    case ValueOperation::brick_factor:
      return value_operand::brick::count;
    case ValueOperation::magic_color:
    case ValueOperation::magic_factor:
      return value_operand::magic::count;
    case ValueOperation::wave_color:
    case ValueOperation::wave_factor:
      return value_operand::wave::count;
    case ValueOperation::voronoi_distance:
    case ValueOperation::voronoi_color:
    case ValueOperation::voronoi_position:
    case ValueOperation::voronoi_w:
    case ValueOperation::voronoi_radius:
      return value_operand::voronoi::count;
    case ValueOperation::gradient:
      return value_operand::gradient::count;
    case ValueOperation::color_ramp:
      return value_operand::color_ramp::count;
    case ValueOperation::rgb_curve:
      return value_operand::rgb_curve::count;
    case ValueOperation::separate_r:
    case ValueOperation::separate_g:
    case ValueOperation::separate_b:
      return value_operand::separate_color::count;
    case ValueOperation::combine_color:
      return value_operand::combine_color::count;
    case ValueOperation::hosek_wilkie_sky:
      return value_operand::sky::count;
    case ValueOperation::nishita_sky:
      return value_operand::nishita_sky::count;
  }
  std::abort();
}

// Blender 4.5.10 ShaderNodeMath operations. The values are part of the
// topology/static-property program signature and are selected while the
// Luisa DSL is constructed, not dynamically per shading point.
enum class MathOperation : std::uint8_t {
  add,
  subtract,
  multiply,
  divide,
  multiply_add,
  power,
  logarithm,
  square_root,
  inverse_square_root,
  absolute,
  exponent,
  minimum,
  maximum,
  less_than,
  greater_than,
  sign,
  compare,
  smooth_minimum,
  smooth_maximum,
  round,
  floor,
  ceil,
  trunc,
  fraction,
  modulo,
  floored_modulo,
  wrap,
  snap,
  ping_pong,
  sine,
  cosine,
  tangent,
  arcsine,
  arccosine,
  arctangent,
  arctangent2,
  hyperbolic_sine,
  hyperbolic_cosine,
  hyperbolic_tangent,
  radians,
  degrees
};

inline constexpr std::uint32_t math_operation_count =
    static_cast<std::uint32_t>(MathOperation::degrees) + 1u;

enum class VectorMathOperation : std::uint8_t {
  add,
  subtract,
  multiply,
  divide,
  multiply_add,
  cross_product,
  project,
  reflect,
  refract,
  faceforward,
  dot_product,
  distance,
  length,
  scale,
  normalize,
  absolute,
  power,
  sign,
  minimum,
  maximum,
  floor,
  ceil,
  fraction,
  modulo,
  wrap,
  snap,
  sine,
  cosine,
  tangent
};

inline constexpr std::uint32_t vector_math_operation_count =
    static_cast<std::uint32_t>(VectorMathOperation::tangent) + 1u;

enum class BlendOperation : std::uint8_t {
  mix,
  darken,
  multiply,
  burn,
  lighten,
  screen,
  dodge,
  add,
  overlay,
  soft_light,
  linear_light,
  difference,
  exclusion,
  subtract,
  divide,
  hue,
  saturation,
  color,
  value
};

enum class NormalMapSpace : std::uint8_t {
  tangent,
  object,
  world,
  blender_object,
  blender_world
};

enum class NormalMapBase : std::uint8_t {
  original,
  displaced
};

enum class NormalMapConvention : std::uint8_t {
  open_gl,
  direct_x
};

// Normal Map is a static shader-stage configuration. Keep its packed IR
// contract in one place so graph lowering and Luisa AST construction cannot
// silently assign different meanings to the same bits.
inline constexpr std::uint64_t normal_map_space_mask = 0xffu;
inline constexpr std::uint64_t normal_map_named_tangent = 1u << 8u;
inline constexpr std::uint64_t normal_map_displaced_base = 1u << 9u;
inline constexpr std::uint64_t normal_map_direct_x = 1u << 10u;

[[nodiscard]] constexpr std::uint64_t encode_normal_map_configuration(
    NormalMapSpace space,
    bool named_tangent,
    NormalMapBase base,
    NormalMapConvention convention) noexcept {
  return static_cast<std::uint64_t>(space) |
         (named_tangent ? normal_map_named_tangent : 0u) |
         (base == NormalMapBase::displaced ? normal_map_displaced_base : 0u) |
         (convention == NormalMapConvention::direct_x ? normal_map_direct_x
                                                       : 0u);
}

[[nodiscard]] constexpr NormalMapSpace decode_normal_map_space(
    std::uint64_t configuration) noexcept {
  return static_cast<NormalMapSpace>(configuration & normal_map_space_mask);
}

[[nodiscard]] constexpr bool normal_map_has_named_tangent(
    std::uint64_t configuration) noexcept {
  return (configuration & normal_map_named_tangent) != 0u;
}

[[nodiscard]] constexpr NormalMapBase decode_normal_map_base(
    std::uint64_t configuration) noexcept {
  return (configuration & normal_map_displaced_base) != 0u
             ? NormalMapBase::displaced
             : NormalMapBase::original;
}

[[nodiscard]] constexpr NormalMapConvention decode_normal_map_convention(
    std::uint64_t configuration) noexcept {
  return (configuration & normal_map_direct_x) != 0u
             ? NormalMapConvention::direct_x
             : NormalMapConvention::open_gl;
}

enum class NoiseType : std::uint8_t {
  multifractal,
  fbm,
  hybrid_multifractal,
  ridged_multifractal,
  hetero_terrain
};

enum class VoronoiFeature : std::uint8_t {
  f1,
  f2,
  smooth_f1,
  distance_to_edge,
  n_sphere_radius
};

enum class VoronoiDistanceMetric : std::uint8_t {
  euclidean,
  manhattan,
  chebychev,
  minkowski
};

enum class VoronoiOutput : std::uint8_t {
  distance,
  color,
  position,
  w,
  radius
};

// A single topologically ordered value stream is intentional. Blender shader
// graphs freely cross scalar/vector domains (texture color -> luminance ->
// math -> roughness, or vector -> texture -> alpha -> closure mix). Splitting
// these domains into independent instruction arrays makes valid graphs
// impossible to schedule without an additional dependency graph.
struct ValueInstruction {
  ValueOperation operation{ValueOperation::parameter};
  contract::NodeId source_node;
  contract::SocketType result_type{contract::SocketType::floating};
  ParameterId parameter;
  // SurfaceProgram is a host-side graph IR. Store its true arity instead of
  // paying for fourteen fixed operand slots on every instruction. If scene
  // profiling later identifies allocator/locality pressure here, replace the
  // storage with a small-vector or shared operand arena without changing the
  // iterable IR contract.
  std::vector<ValueExpressionId> operands;
  std::uint64_t static_u0{};
  std::uint64_t static_u1{};
  float static_f0{};
  float static_f1{};
  // Node-specific immutable data. It is part of the topology/static-property
  // JIT signature; variable-length material tables do not belong here.
  std::vector<float> static_table;

  [[nodiscard]] ValueExpressionId operand(std::size_t index) const noexcept {
    if (index >= operands.size()) {
      std::abort();
    }
    return operands[index];
  }
};

// Exact semantic dependence on the mutable ShaderData normal established by
// Cycles' SetNormal boundary. This deliberately excludes every other
// SurfacePoint field: the automatic-normal stage changes only shading_normal,
// so an endpoint may skip that stage iff neither its active value program nor
// its closure reduction observes this relation. The implementation is an
// exhaustive ValueOperation classifier; adding an operation therefore cannot
// silently inherit an optimistic "does not observe" default.
[[nodiscard]] bool value_instruction_observes_shading_normal(
    const ValueInstruction &instruction) noexcept;

enum class ClosureOperation : std::uint8_t {
  null_closure,
  diffuse,
  translucent,
  principled,
  glossy,
  glass,
  emission,
  transparent,
  subsurface,
  add,
  mix,
  refraction
};

// Static Cycles BSSRDF family selected by a Blender node property. Keeping
// this out of the dynamically typed value stream makes the shader topology
// (and therefore the Luisa JIT specialization) describe the transport model
// exactly; only authored numeric sockets remain device values.
enum class BssrdfMethod : std::uint8_t {
  burley,
  random_walk,
  random_walk_legacy,
  random_walk_skin
};

// Scheduling contract for Cycles' next-event light-shader evaluation. This is
// a proof about the closure program's structure, not a host-side evaluation of
// its radiance: parameter values remain runtime data and are evaluated by the
// Luisa device program.
enum class EmissionEvaluationMode : std::uint8_t { none, constant, deferred };

struct ClosureInstruction {
  ClosureOperation operation{ClosureOperation::diffuse};
  contract::NodeId source_node;
  ValueExpressionId color;
  ValueExpressionId normal;
  // Cycles' BsdfBaseNode::has_bump() is a graph-topology predicate: a
  // linked Normal uses bump unless its immediate source is a Geometry node.
  // Unlike socket literals, this fact is stable under parameter rebinding.
  bool normal_uses_bump{};
  ValueExpressionId roughness;
  ValueExpressionId diffuse_roughness;
  ValueExpressionId subsurface_weight;
  ValueExpressionId subsurface_radius;
  ValueExpressionId subsurface_scale;
  ValueExpressionId subsurface_ior;
  ValueExpressionId subsurface_anisotropy;
  BssrdfMethod subsurface_method{BssrdfMethod::random_walk};
  ValueExpressionId transmission_weight;
  ValueExpressionId metallic;
  ValueExpressionId ior;
  ValueExpressionId specular_ior_level;
  ValueExpressionId specular_tint;
  ValueExpressionId alpha;
  // Keep Thin Wall in the parameter stream. Cycles treats only an unlinked
  // direct true value as statically thin; linked values remain conservative.
  ValueExpressionId thin_wall;
  ValueExpressionId sheen_weight;
  ValueExpressionId sheen_roughness;
  ValueExpressionId sheen_tint;
  ValueExpressionId coat_weight;
  ValueExpressionId coat_roughness;
  ValueExpressionId coat_ior;
  ValueExpressionId coat_tint;
  ValueExpressionId coat_normal;
  // Cycles distinguishes an unlinked Coat Normal socket from a linked
  // expression whose numerical value happens to be zero. Preserve that
  // graph-topology fact explicitly instead of inferring it from the
  // lowered value operation.
  bool coat_normal_linked{};
  ValueExpressionId emission_color;
  ValueExpressionId emission_strength;
  bool preserve_ggx_energy{};
  bool beckmann{};
  ValueExpressionId strength;
  ValueExpressionId factor;
  ClosureExpressionId a;
  ClosureExpressionId b;
};

enum class VolumePhase : std::uint8_t {
  henyey_greenstein,
  fournier_forand,
  draine,
  rayleigh,
  mie
};

enum class VolumeOperation : std::uint8_t {
  null_volume,
  absorption,
  scatter,
  coefficients,
  emission,
  principled,
  add,
  mix
};

// Volume closures stay as a tree for the same reason as surface closures:
// Add/Mix weights are shading-point expressions and phase closures must
// remain individually selectable. Coefficients are evaluated in Luisa DSL;
// this IR never stores host-evaluated extinction or scattering.
struct VolumeInstruction {
  VolumeOperation operation{VolumeOperation::null_volume};
  contract::NodeId source_node;
  ValueExpressionId color;
  ValueExpressionId density;
  ValueExpressionId anisotropy;
  ValueExpressionId ior;
  ValueExpressionId backscatter;
  ValueExpressionId alpha;
  ValueExpressionId diameter;
  ValueExpressionId scatter_coefficients;
  ValueExpressionId absorption_coefficients;
  ValueExpressionId absorption_color;
  ValueExpressionId emission_coefficients;
  ValueExpressionId emission_strength;
  ValueExpressionId emission_color;
  ValueExpressionId blackbody_intensity;
  ValueExpressionId blackbody_tint;
  ValueExpressionId temperature;
  ValueExpressionId factor;
  VolumeExpressionId a;
  VolumeExpressionId b;
  VolumePhase phase{VolumePhase::henyey_greenstein};
};

class SurfaceProgram {

private:
  std::uint64_t _structure_signature{};
  std::vector<ParameterDesc> _parameters;
  std::vector<ValueInstruction> _value_instructions;
  std::vector<ClosureInstruction> _closure_instructions;
  std::vector<VolumeInstruction> _volume_instructions;
  ClosureExpressionId _root;
  VolumeExpressionId _volume_root;
  ValueExpressionId _surface_normal_root;
  ValueExpressionId _displacement_root;
  EmissionEvaluationMode _emission_evaluation{EmissionEvaluationMode::none};

public:
  SurfaceProgram(std::uint64_t structure_signature,
                 std::vector<ParameterDesc> parameters,
                 std::vector<ValueInstruction> value_instructions,
                 std::vector<ClosureInstruction> closure_instructions,
                 ClosureExpressionId root,
                 std::vector<VolumeInstruction> volume_instructions = {},
                 VolumeExpressionId volume_root = {},
                 ValueExpressionId surface_normal_root = {},
                 ValueExpressionId displacement_root = {}) noexcept;

  [[nodiscard]] std::uint64_t structure_signature() const noexcept {
    return _structure_signature;
  }
  [[nodiscard]] const std::vector<ParameterDesc> &parameters() const noexcept {
    return _parameters;
  }
  [[nodiscard]] const std::vector<ValueInstruction> &
  value_instructions() const noexcept {
    return _value_instructions;
  }
  [[nodiscard]] const std::vector<ClosureInstruction> &
  closure_instructions() const noexcept {
    return _closure_instructions;
  }
  [[nodiscard]] const std::vector<VolumeInstruction> &
  volume_instructions() const noexcept {
    return _volume_instructions;
  }
  [[nodiscard]] ClosureExpressionId root() const noexcept { return _root; }
  [[nodiscard]] VolumeExpressionId volume_root() const noexcept {
    return _volume_root;
  }
  [[nodiscard]] ValueExpressionId surface_normal_root() const noexcept {
    return _surface_normal_root;
  }
  [[nodiscard]] ValueExpressionId displacement_root() const noexcept {
    return _displacement_root;
  }
  [[nodiscard]] EmissionEvaluationMode emission_evaluation() const noexcept {
    return _emission_evaluation;
  }
};

class SurfaceParameterBlock {

private:
  std::vector<contract::SocketValue> _values;

public:
  SurfaceParameterBlock() = default;
  explicit SurfaceParameterBlock(const SurfaceProgram &program);

  [[nodiscard]] std::size_t size() const noexcept { return _values.size(); }
  [[nodiscard]] const contract::SocketValue *
  find(ParameterId id) const noexcept;
  [[nodiscard]] bool set(const SurfaceProgram &program, ParameterId id,
                         contract::SocketValue value);
};

// Host/JIT-stage reachability of the physical lobes produced by one
// Principled closure instruction. These bits describe code which may be
// needed by at least one parameter block sharing a SurfaceProgram topology;
// they never encode a shading-point decision or replace a device expression.
enum class PrincipledClosureFeature : std::uint32_t {
  alpha = 1u << 0u,
  sheen = 1u << 1u,
  coat = 1u << 2u,
  metallic = 1u << 3u,
  thick_transmission = 1u << 4u,
  thin_transmission = 1u << 5u,
  dielectric = 1u << 6u,
  thick_subsurface = 1u << 7u,
  thin_subsurface = 1u << 8u,
  diffuse = 1u << 9u,
  emission = 1u << 10u,
};

using PrincipledClosureFeatureMask = std::uint32_t;

[[nodiscard]] constexpr PrincipledClosureFeatureMask
principled_closure_feature_bit(PrincipledClosureFeature feature) noexcept {
  return static_cast<PrincipledClosureFeatureMask>(feature);
}

struct SurfaceClosurePlanEntry {
  bool reachable{};
  PrincipledClosureFeatureMask principled_features{};
};

// One entry per ClosureInstruction. Plans for material instances with the
// same structure signature are merged before Luisa records their shared
// material branch. This is the formal binding-time boundary: a direct socket
// literal may prove a branch/lobe unreachable, while a linked expression is
// conservatively retained.
class SurfaceClosurePlan {

private:
  std::vector<SurfaceClosurePlanEntry> _entries;

public:
  SurfaceClosurePlan() = default;
  explicit SurfaceClosurePlan(
      std::vector<SurfaceClosurePlanEntry> entries) noexcept
      : _entries{std::move(entries)} {}

  [[nodiscard]] const std::vector<SurfaceClosurePlanEntry> &entries()
      const noexcept {
    return _entries;
  }
  [[nodiscard]] bool compatible(const SurfaceProgram &program) const noexcept;
  [[nodiscard]] const SurfaceClosurePlanEntry &entry(
      ClosureExpressionId id) const noexcept;
  void merge(const SurfaceClosurePlan &other) noexcept;
};

// Topology-closed value schedules for the distinct surface consumers.  A
// mask contains an expression and all of its transitive operands, so walking
// the topologically ordered SurfaceProgram once evaluates every selected
// value exactly once.  The domains remain separate because evaluating a
// closure's disabled Coat texture while asking only for emission is both
// unnecessary runtime work and duplicated shader AST.
//
// These masks are derived exclusively from SurfaceClosurePlan reachability
// and immutable closure topology. Linked sockets therefore remain
// conservative device expressions; no material value is sampled, baked, or
// moved across the host/device boundary.
struct SurfaceValueDependencyPlan {
  std::vector<bool> physical;
  std::vector<bool> emission;
  std::vector<bool> preparation;
  // Exact roots consumed after the topological value stream has finished.
  // These are intentionally distinct from the transitive masks above: a
  // value can feed both another value instruction and a closure, so value
  // out-degree alone is not a sound last-use criterion for slot allocation.
  std::vector<bool> physical_outputs;
  std::vector<bool> emission_outputs;
  std::vector<bool> preparation_outputs;
  // Closure-tree liveness is distinct from value liveness: a value may be
  // shared by physical and emission leaves even though each consumer must
  // visit only its own closure domain.
  std::vector<bool> physical_closures;
  std::vector<bool> emission_closures;
  // Formal decomposition of the emission endpoint's SetNormal dependence.
  // Value instructions and the closure reduction are tracked separately so a
  // backend can project either execution domain without conflating their
  // causes. The union is the necessary-and-sufficient condition for running
  // an authored automatic-normal program before emission evaluation.
  bool emission_values_observe_shading_normal{};
  bool emission_closures_observe_shading_normal{};

  [[nodiscard]] bool emission_observes_shading_normal() const noexcept {
    return emission_values_observe_shading_normal ||
           emission_closures_observe_shading_normal;
  }

  [[nodiscard]] bool compatible(const SurfaceProgram &program) const noexcept;
};

[[nodiscard]] SurfaceValueDependencyPlan analyze_surface_value_dependencies(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan) noexcept;

// Conservative topology-only plan used by standalone GraphSurface clients
// which do not provide the scene's complete set of parameter bindings.
[[nodiscard]] SurfaceClosurePlan conservative_surface_closure_plan(
    const SurfaceProgram &program) noexcept;

// Material-specific proof. Mix branches selected by direct 0/1 factors and
// Principled lobes disabled by direct literals are omitted. Linked inputs are
// never host-evaluated and therefore remain reachable.
[[nodiscard]] SurfaceClosurePlan analyze_surface_closure_plan(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) noexcept;

// Cycles' host-stage output_estimate_emission metadata query. This evaluates
// only direct parameter literals and the closure topology; linked value
// expressions remain conservatively unknown (unit estimate). The returned
// value controls emitter discovery only and is never used as rendered
// radiance.
[[nodiscard]] Vec3f
estimate_surface_emission(const SurfaceProgram &program,
                          const SurfaceParameterBlock &parameters);

// Cycles' host-side Shader::has_surface_bssrdf flag. This is deliberately a
// topology/parameter query rather than a sampled closure evaluation: linked
// Principled inputs remain potentially non-zero, while direct zero literals
// can prove that the node does not prevent static geometry transforms.
[[nodiscard]] bool cycles_surface_has_bssrdf(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) noexcept;

// Cycles' host-side Shader::has_bssrdf_bump flag. This is the exact union of
// a real BSSRDF closure whose Normal topology uses bump and automatic bump
// from a linked displacement output, normalized as the surface-normal root,
// under BUMP/BOTH material policy.
[[nodiscard]] bool cycles_surface_has_bssrdf_bump(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters,
    contract::DisplacementMethod displacement_method) noexcept;

enum class SurfaceProgramDiagnosticCode : std::uint8_t {
  structure_mismatch,
  missing_surface_root,
  unsupported_node,
  missing_input,
  missing_output,
  type_mismatch
};

struct SurfaceProgramDiagnostic {
  SurfaceProgramDiagnosticCode code{};
  std::string message;
  std::optional<contract::NodeId> node;
  std::string socket;
};

struct SurfaceProgramCompilation {
  std::shared_ptr<const SurfaceProgram> program;
  std::vector<SurfaceProgramDiagnostic> diagnostics;

  [[nodiscard]] bool ok() const noexcept { return program != nullptr; }
};

[[nodiscard]] SurfaceProgramCompilation
compile_surface_program(const ShaderProgram &shader);

struct SurfaceParameterBinding {
  std::optional<SurfaceParameterBlock> parameters;
  std::vector<SurfaceProgramDiagnostic> diagnostics;

  [[nodiscard]] bool ok() const noexcept { return parameters.has_value(); }
};

[[nodiscard]] SurfaceParameterBinding
bind_surface_parameters(const SurfaceProgram &program,
                        const ShaderProgram &shader);

} // namespace psycles::compiler
