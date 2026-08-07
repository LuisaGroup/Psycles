#pragma once

#include <compare>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
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

struct ParameterDesc {
  ParameterId id;
  contract::NodeId node;
  std::string socket;
  contract::SocketType type{};
  contract::SocketValue default_value;
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
  ValueExpressionId a;
  ValueExpressionId b;
  ValueExpressionId c;
  ValueExpressionId d;
  ValueExpressionId e;
  ValueExpressionId f;
  ValueExpressionId g;
  ValueExpressionId h;
  ValueExpressionId i;
  ValueExpressionId j;
  std::uint64_t static_u0{};
  std::uint64_t static_u1{};
  float static_f0{};
  float static_f1{};
  // Node-specific immutable data, such as a sampled ColorRamp/RGB Curves
  // table. It is part of the topology/static-property JIT signature.
  std::vector<float> static_table;
};

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
