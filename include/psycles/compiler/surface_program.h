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

namespace psycles::compiler {

template<typename Tag>
struct ProgramId {
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

using ParameterId = ProgramId<ParameterTag>;
using ValueExpressionId = ProgramId<ValueExpressionTag>;
using ClosureExpressionId = ProgramId<ClosureExpressionTag>;

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
    color_to_scalar,
    vector_to_scalar,
    add,
    subtract,
    multiply,
    divide,
    minimum,
    maximum,
    power,
    absolute,
    clamp01,
    clamp_range,
    mix,
    multiply_color,
    hue_saturation,
    invert,
    gamma,
    brightness_contrast,
    surface_position,
    shading_normal,
    geometric_normal,
    incoming,
    tangent,
    uv,
    generated,
    object_position,
    object_location,
    object_random,
    particle_index,
    particle_random,
    back_facing,
    random_per_island,
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
    layer_weight_fresnel,
    layer_weight_facing,
    mapping,
    image_color,
    image_alpha,
    attribute_color,
    attribute_alpha,
    normal_map,
    bump,
    noise_factor,
    noise_color,
    brick_color,
    brick_factor,
    gradient,
    color_ramp,
    rgb_curve,
    separate_r,
    separate_g,
    separate_b,
    combine_color,
    nishita_sky
};

enum class NormalMapSpace : std::uint8_t {
    tangent,
    object,
    world,
    blender_object,
    blender_world
};

enum class NoiseType : std::uint8_t {
    multifractal,
    fbm,
    hybrid_multifractal,
    ridged_multifractal,
    hetero_terrain
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
    diffuse,
    principled,
    glossy,
    emission,
    transparent,
    add,
    mix
};

struct ClosureInstruction {
    ClosureOperation operation{ClosureOperation::diffuse};
    contract::NodeId source_node;
    ValueExpressionId color;
    ValueExpressionId normal;
    ValueExpressionId roughness;
    ValueExpressionId diffuse_roughness;
    ValueExpressionId metallic;
    ValueExpressionId ior;
    ValueExpressionId specular_ior_level;
    ValueExpressionId specular_tint;
    bool preserve_ggx_energy{};
    ValueExpressionId strength;
    ValueExpressionId factor;
    ClosureExpressionId a;
    ClosureExpressionId b;
};

class SurfaceProgram {

private:
    std::uint64_t _structure_signature{};
    std::vector<ParameterDesc> _parameters;
    std::vector<ValueInstruction> _value_instructions;
    std::vector<ClosureInstruction> _closure_instructions;
    ClosureExpressionId _root;

public:
    SurfaceProgram(
        std::uint64_t structure_signature,
        std::vector<ParameterDesc> parameters,
        std::vector<ValueInstruction> value_instructions,
        std::vector<ClosureInstruction> closure_instructions,
        ClosureExpressionId root) noexcept;

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
    [[nodiscard]] ClosureExpressionId root() const noexcept {
        return _root;
    }
};

class SurfaceParameterBlock {

private:
    std::vector<contract::SocketValue> _values;

public:
    SurfaceParameterBlock() = default;
    explicit SurfaceParameterBlock(const SurfaceProgram &program);

    [[nodiscard]] std::size_t size() const noexcept {
        return _values.size();
    }
    [[nodiscard]] const contract::SocketValue *find(
        ParameterId id) const noexcept;
    [[nodiscard]] bool set(
        const SurfaceProgram &program,
        ParameterId id,
        contract::SocketValue value);
};

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

    [[nodiscard]] bool ok() const noexcept {
        return program != nullptr;
    }
};

[[nodiscard]] SurfaceProgramCompilation compile_surface_program(
    const ShaderProgram &shader);

struct SurfaceParameterBinding {
    std::optional<SurfaceParameterBlock> parameters;
    std::vector<SurfaceProgramDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept {
        return parameters.has_value();
    }
};

[[nodiscard]] SurfaceParameterBinding bind_surface_parameters(
    const SurfaceProgram &program,
    const ShaderProgram &shader);

}// namespace psycles::compiler
