#include "compact_surface_program_test_support.h"

#include <psycles/compiler/core_nodes.h>

#include "luisa_surface_test_support.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_internal.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_surface_value_program.h"
#include "path_tracer_texture_sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

[[nodiscard]] std::size_t count_named_custom_calls(
    const Function &function, std::string_view prefix) noexcept {
    auto count = std::size_t{0u};
    traverse_expressions<true>(
        function.body(),
        [&](const Expression *expression) noexcept {
            if (expression->tag() != Expression::Tag::CALL) {
                return;
            }
            const auto *call = static_cast<const CallExpr *>(expression);
            if (call->is_custom() &&
                call->custom().name().starts_with(prefix)) {
                count += 1u;
            }
        },
        [](const Statement *) noexcept {},
        [](const Statement *) noexcept {});
    return count;
}

[[nodiscard]] bool domain_matches(const SurfaceValueProgramDomainView &view,
                                  const std::vector<std::uint32_t> &values,
                                  std::uint32_t offset) noexcept {
    return std::equal(view.value_variants.begin(), view.value_variants.end(),
                      values.begin(), values.end()) &&
           view.program_offset == offset;
}

[[nodiscard]] std::size_t
count_bump_configuration(const SurfaceSvmSceneImage &image,
                         std::uint32_t program_begin, std::uint32_t program_end,
                         std::uint16_t configuration) noexcept {
    auto count = std::size_t{0u};
    for (auto program = program_begin; program < program_end; ++program) {
        const auto &range = image.programs[program];
        for (auto instruction = range.instruction_begin;
             instruction < range.instruction_begin + range.instruction_count;
             ++instruction) {
            const auto &record = image.instructions[instruction];
            if (surface_svm_bytecode_kind(record) !=
                SurfaceSvmBytecodeKind::value) {
                continue;
            }
            const auto value = surface_svm_value_instruction(record);
            count += surface_value_operation(value) ==
                         ValueOperation::bump_samples &&
                     surface_value_svm_immediate(value) == configuration;
        }
    }
    return count;
}

} // namespace

ShaderGraph make_nested_mix_replay_graph(bool restore_after) {
    ShaderGraph graph;
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Nested mix diffuse");
    const auto glass = graph.add_node(
        node_type::glass_bsdf, "Nested mix Beckmann glass");
    const auto glossy = graph.add_node(
        node_type::glossy_bsdf, "Nested mix glossy");
    const auto transparent = graph.add_node(
        node_type::transparent_bsdf, "Nested mix transparent");
    const auto inner_mix = graph.add_node(
        node_type::mix_closure, "Nested physical mix level one");
    const auto middle_mix = graph.add_node(
        node_type::mix_closure, "Nested physical mix level two");
    const auto outer_mix = graph.add_node(
        node_type::mix_closure, "Nested physical mix level three");
    auto configured =
        graph.set_input(
            diffuse, "Color", SocketValue::color({0.22f, 0.51f, 0.76f})) &&
        graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.43f)) &&
        graph.set_input(
            diffuse, "Normal", SocketValue::normal({0.18f, 0.0f, 0.984f})) &&
        graph.set_input(
            glass, "Color", SocketValue::color({0.81f, 0.91f, 0.98f})) &&
        graph.set_input(
            glass, "Roughness", SocketValue::floating(0.18f)) &&
        graph.set_input(glass, "IOR", SocketValue::floating(1.37f)) &&
        graph.set_property(
            glass, "Distribution", SocketValue::string("BECKMANN")) &&
        graph.set_input(
            inner_mix, "Factor", SocketValue::floating(0.37f)) &&
        graph.set_input(
            middle_mix, "Factor", SocketValue::floating(0.29f)) &&
        graph.set_input(
            outer_mix, "Factor", SocketValue::floating(0.61f)) &&
        graph.set_input(
            glossy, "Color", SocketValue::color({0.48f, 0.21f, 0.07f})) &&
        graph.set_input(
            glossy, "Roughness", SocketValue::floating(0.31f)) &&
        graph.set_input(
            transparent, "Color", SocketValue::color({0.92f, 0.84f, 0.73f})) &&
        graph.connect(
            {.node = diffuse, .socket = "Closure"}, inner_mix, "A") &&
        graph.connect(
            {.node = glass, .socket = "Closure"}, inner_mix, "B") &&
        graph.connect(
            {.node = inner_mix, .socket = "Closure"}, middle_mix, "A") &&
        graph.connect(
            {.node = glossy, .socket = "Closure"}, middle_mix, "B") &&
        // Put transparency before the remaining physical tree. The compact
        // population path must keep this source position without replaying
        // the diffuse/glass/glossy suffix.
        graph.connect(
            {.node = transparent, .socket = "Closure"}, outer_mix, "A") &&
        graph.connect(
            {.node = middle_mix, .socket = "Closure"}, outer_mix, "B");
    auto root = outer_mix;
    if (restore_after) {
        const auto emission = graph.add_node(
            node_type::emission, "Nested explicit emission");
        root = graph.add_node(
            node_type::add_closure, "Physical plus emission");
        configured = configured &&
            graph.set_input(
                emission,
                "Color",
                SocketValue::color({0.17f, 0.41f, 0.89f})) &&
            graph.set_input(
                emission, "Strength", SocketValue::floating(2.3f)) &&
            graph.connect(
                {.node = outer_mix, .socket = "Closure"}, root, "A") &&
            graph.connect(
                {.node = emission, .socket = "Closure"}, root, "B");
    }
    if (!configured) {
        throw std::runtime_error{
            "failed to configure nested Mix/replay graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = root, .socket = "Closure"});
    return graph;
}

SurfacePoint make_surface_value_transaction_test_point() noexcept {
    auto point = make_surface_point();
    // Keep the displaced projection at the established fixture values. This
    // isolates the phase transition: ordinary programs remain unchanged,
    // while an automatic-normal prefix sees different undisplaced geometry.
    point.undisplaced_position = make_float3(-0.61f, 0.37f, -0.29f);
    point.undisplaced_object_position =
        make_float3(0.43f, -0.17f, 0.59f);
    point.undisplaced_shading_normal =
        normalize(make_float3(-0.41f, 0.22f, 0.88f));
    point.undisplaced_object_shading_normal =
        normalize(make_float3(0.36f, 0.27f, 0.89f));
    point.undisplaced_dPdx = make_float3(-0.12f, 0.06f, 0.08f);
    point.undisplaced_dPdy = make_float3(0.03f, -0.15f, 0.11f);
    point.undisplaced_object_dPdx =
        make_float3(-0.07f, 0.04f, -0.13f);
    point.undisplaced_object_dPdy =
        make_float3(0.15f, -0.02f, 0.05f);
    point.transmission_depth = 13u;
    return point;
}

namespace {

template<std::size_t StackCapacity>
[[nodiscard]] std::string validate_compact_surface_value_program_abi_for_stack(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SurfaceValueInstructionDispatcher &dispatcher) {
    using Stack = SurfaceValueStackBankFor<StackCapacity>;
    using ProbeCallable =
        Callable<void(Buffer<float>, Buffer<luisa::float3>, Buffer<float>,
                      BindlessArray, BindlessArray, SurfacePointCall &,
                      luisa::float3, bool, luisa::uint4, Stack &)>;
    ProbeCallable probe{
        [dispatcher](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, Var<SurfacePointCall> &point,
            Float3 transaction_shading_normal, Bool use_undisplaced_geometry,
            Var<luisa::uint4> instruction, Var<Stack> &stack) noexcept {
          const SurfaceValueLocalsView locals{stack.expression()};
          dispatcher(scalar_parameters, vector_parameters, cycles_bsdf_tables,
                     textures, geometry_heap, point, transaction_shading_normal,
                     use_undisplaced_geometry, instruction, locals, nullptr);
        }};
    const auto &function = probe.function();
    auto surface_point_arguments = std::size_t{0u};
    auto surface_point_is_reference = false;
    for (const auto &argument : function.arguments()) {
        if (argument.type() == Type::of<SurfacePointCall>()) {
            surface_point_arguments += 1u;
            surface_point_is_reference = argument.is_reference();
        }
    }
    // The unified PC loop owns transaction state. Its per-record dispatcher
    // observes the point by reference and returns no aggregate state.
    if (function.return_type() != nullptr ||
        surface_point_arguments != 1u || !surface_point_is_reference) {
        return "unified value dispatcher lost its narrow point ABI";
    }
    const auto handler_call_count =
        count_named_custom_calls(function, "surface_value_family_");
    std::vector<std::uint32_t> families;
    families.reserve(
        scene->surface_values->preparation_value_static_variants.size());
    for (const auto variant :
         scene->surface_values->preparation_value_static_variants) {
        if (variant >= scene->surface_values->value_variants.size()) {
            return "preparation domain contains an invalid value variant";
        }
        const auto &static_variant =
            scene->surface_values->value_variants[variant];
        if (static_variant.svm_immediates.empty()) {
            return "preparation variant has no SVM immediate domain";
        }
        auto bank = SurfaceValueBank::scalar;
        if (!classify_surface_value_type(
                static_variant.instruction.result_type, bank)) {
            return "preparation variant has no result bank";
        }
        const auto family = make_surface_value_handler_key(
            static_variant.instruction.operation, bank,
            static_variant.svm_immediates.front());
        for (const auto immediate : static_variant.svm_immediates) {
            if (make_surface_value_handler_key(
                    static_variant.instruction.operation, bank, immediate) !=
                family) {
                return "one provenance variant crossed SVM families";
            }
        }
        families.emplace_back(family);
    }
    std::sort(families.begin(), families.end());
    families.erase(std::unique(families.begin(), families.end()),
                   families.end());
    const auto expected = families.size();
    if (handler_call_count != expected) {
        return "unified value dispatcher lost its per-family callable "
               "boundary (calls=" +
               std::to_string(handler_call_count) + ", variants=" +
               std::to_string(
                   scene->surface_values
                       ->preparation_value_static_variants.size()) +
               ", families=" + std::to_string(expected) + ")";
    }
    return {};
}

} // namespace

std::string validate_compact_surface_value_program_abi(
    const std::shared_ptr<LuisaSceneData> &scene) {
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(0u, 1u);
    const auto dispatcher = make_surface_value_instruction_dispatcher(
        scene, texture_sampling, attribute_lookup,
        SurfaceValueProgramDomain::preparation);
    if (dispatcher.requires_ambient_occlusion()) {
        return "AO-free unified value dispatcher changed callable ABI";
    }
    switch (dispatcher.stack_capacity()) {
        case 32u:
            return validate_compact_surface_value_program_abi_for_stack<32u>(
                scene, dispatcher);
        case 64u:
            return validate_compact_surface_value_program_abi_for_stack<64u>(
                scene, dispatcher);
        case 128u:
            return validate_compact_surface_value_program_abi_for_stack<128u>(
                scene, dispatcher);
        case SurfaceValueRuntime::stack_capacity:
            return validate_compact_surface_value_program_abi_for_stack<
                SurfaceValueRuntime::stack_capacity>(scene, dispatcher);
        default:
            return "unified value dispatcher selected an invalid stack "
                   "capacity";
    }
}

std::string validate_surface_value_fresh_lifetime_seed() {
    constexpr std::array bucket_boundaries{
        std::pair{0u, 32u}, std::pair{1u, 32u},
        std::pair{32u, 32u}, std::pair{33u, 64u},
        std::pair{64u, 64u}, std::pair{65u, 128u},
        std::pair{128u, 128u},
        std::pair{129u, SurfaceValueRuntime::stack_capacity},
        std::pair{SurfaceValueRuntime::stack_capacity,
                  SurfaceValueRuntime::stack_capacity},
        std::pair{SurfaceValueRuntime::stack_capacity + 1u, 0u}};
    for (const auto [required, expected] : bucket_boundaries) {
        if (surface_value_stack_storage_lanes(required) != expected) {
            return "compact surface stack bucket selection is not the "
                   "smallest covering capacity";
        }
    }
    for (const auto capacity : surface_value_stack_lane_buckets) {
        Callable<void()> seed_callable{[capacity]() noexcept {
            SurfaceValueLocals locals{capacity};
        }};
        auto stack_seeds = std::size_t{0u};
        auto malformed_seeds = std::size_t{0u};
        const auto expected_type = Type::array(Type::of<float>(), capacity);
        traverse_expressions<true>(
            seed_callable.function().body(),
            [&](const Expression *expression) noexcept {
                if (expression->tag() != Expression::Tag::CALL) { return; }
                const auto *call = static_cast<const CallExpr *>(expression);
                if (call->op() != CallOp::UNDEFINED) { return; }
                malformed_seeds += !call->arguments().empty();
                stack_seeds += call->type() == expected_type;
            },
            [](const Statement *) noexcept {},
            [](const Statement *) noexcept {});
        if (stack_seeds != 1u || malformed_seeds != 0u) {
            return "compact surface stack bucket is not exactly one "
                   "argument-free fresh-lifetime seed";
        }
    }
    return {};
}

void print_compact_surface_sample_mismatch(
    const SurfaceSampleTraceCall &actual,
    const SurfaceSampleTraceCall &expected,
    std::string_view backend, std::size_t topology,
    std::size_t scenario) {
    const auto print_vector = [](const luisa::float3 &value) {
        std::cerr << '(' << value.x << ", " << value.y << ", " << value.z
                  << ')';
    };
    std::cerr << "compact population sample mismatch on " << backend
              << ", topology " << topology << ", scenario " << scenario
              << "\n  compact closure=" << actual.closure_index << '/'
              << actual.closure_type << ", expanded closure="
              << expected.closure_index << '/' << expected.closure_type
              << "\n  compact pdf=" << actual.pdf
              << ", expanded pdf=" << expected.pdf
              << "\n  compact wi=";
    print_vector(actual.wi);
    std::cerr << ", expanded wi=";
    print_vector(expected.wi);
    std::cerr << '\n';
}

ShaderGraph make_minimal_principled_graph() {
    ShaderGraph graph;
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Minimal Principled");
    const auto configured =
        graph.set_input(
            principled, "BaseColor",
            SocketValue::color({0.18f, 0.42f, 0.73f})) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.31f)) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "TransmissionWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SubsurfaceWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled, "EmissionStrength", SocketValue::floating(0.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure minimal Principled graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

ShaderGraph make_direct_math_convert_graph() {
    ShaderGraph graph;
    const auto geometry =
        graph.add_node(node_type::geometry, "Direct SVM Geometry");
    const auto normal_to_vector = graph.add_node(node_type::normal_to_vector,
                                                 "Direct SVM Normal to Vector");
    const auto vector_to_scalar = graph.add_node(node_type::vector_to_scalar,
                                                 "Direct SVM Vector to Scalar");
    const auto math =
        graph.add_node(node_type::math, "Direct SVM Multiply Add");
    const auto scalar_to_color = graph.add_node(node_type::scalar_to_color,
                                                "Direct SVM Scalar to Color");
    const auto color_to_scalar = graph.add_node(node_type::color_to_scalar,
                                                "Direct SVM Color to Scalar");
    const auto scalar_to_boolean = graph.add_node(
        node_type::scalar_to_boolean, "Direct SVM Scalar to Boolean");
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Direct SVM Principled");
    const auto configured =
        graph.connect({.node = geometry, .socket = "Normal"}, normal_to_vector,
                      "Normal") &&
        graph.connect({.node = normal_to_vector, .socket = "Vector"},
                      vector_to_scalar, "Vector") &&
        graph.connect({.node = vector_to_scalar, .socket = "Value"}, math,
                      "A") &&
        graph.set_input(math, "B", SocketValue::floating(2.5f)) &&
        graph.set_input(math, "C", SocketValue::floating(0.1f)) &&
        graph.set_property(math, "Operation",
                           SocketValue::string("MULTIPLY_ADD")) &&
        graph.connect({.node = math, .socket = "Value"}, scalar_to_color,
                      "Value") &&
        graph.connect({.node = scalar_to_color, .socket = "Color"},
                      color_to_scalar, "Color") &&
        graph.connect({.node = math, .socket = "Value"}, scalar_to_boolean,
                      "Value") &&
        graph.connect({.node = scalar_to_color, .socket = "Color"}, principled,
                      "BaseColor") &&
        graph.connect({.node = color_to_scalar, .socket = "Value"}, principled,
                      "Roughness") &&
        graph.connect({.node = scalar_to_boolean, .socket = "Boolean"},
                      principled, "ThinWall") &&
        graph.set_input(principled, "TransmissionWeight",
                        SocketValue::floating(0.65f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure direct Math/Convert SVM graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

ShaderGraph make_direct_texture_trunk_graph(bool box_projection) {
    ShaderGraph graph;
    const auto coordinates = graph.add_node(node_type::texture_coordinate,
                                            "Direct SVM texture coordinates");
    const auto mapping =
        graph.add_node(node_type::mapping, "Direct SVM Mapping");
    const auto image =
        graph.add_node(node_type::image_texture, "Direct SVM Image");
    const auto ramp =
        graph.add_node(node_type::color_ramp, "Direct SVM RGB Ramp");
    const auto mix =
        graph.add_node(node_type::mix_color, "Direct SVM Mix Color");
    const auto principled = graph.add_node(node_type::principled_bsdf,
                                           "Direct texture SVM Principled");
    const auto configured =
        graph.connect({.node = coordinates, .socket = "Generated"}, mapping,
                      "Vector") &&
        graph.set_input(mapping, "Location",
                        SocketValue::vector({0.13f, -0.07f, 0.21f})) &&
        graph.set_input(mapping, "Rotation",
                        SocketValue::vector({0.17f, -0.23f, 0.31f})) &&
        graph.set_input(mapping, "Scale",
                        SocketValue::vector({1.2f, 0.8f, 1.1f})) &&
        graph.set_property(mapping, "VectorType",
                           SocketValue::string("TEXTURE")) &&
        graph.set_property(mapping, "XMapping", SocketValue::string("Z")) &&
        graph.set_property(mapping, "YMapping", SocketValue::string("X")) &&
        graph.set_property(mapping, "ZMapping", SocketValue::string("Y")) &&
        graph.connect({.node = mapping, .socket = "Vector"}, image, "Vector") &&
        graph.set_property(image, "Image", SocketValue::unsigned_integer(0u)) &&
        graph.set_property(image, "ProjectionBlend",
                           SocketValue::floating(0.37f)) &&
        graph.set_property(
            image, "Projection",
            SocketValue::string(box_projection ? "BOX" : "FLAT")) &&
        graph.set_property(image, "Extension", SocketValue::string("REPEAT")) &&
        graph.set_property(image, "Interpolation",
                           SocketValue::string("Linear")) &&
        graph.set_property(image, "ColorSpace",
                           SocketValue::string("Non-Color")) &&
        graph.set_property(ramp, "Sampled", SocketValue::boolean(true)) &&
        graph.set_property(
            ramp, "Table",
            SocketValue::string(
                "0,0.1,0.3,0.8,0.2;0.5,0.8,0.2,0.1,0.6;1,0.2,0.9,0.4,0.9")) &&
        graph.connect({.node = image, .socket = "Alpha"}, ramp, "Factor") &&
        graph.connect({.node = ramp, .socket = "Color"}, mix, "A") &&
        graph.connect({.node = image, .socket = "Color"}, mix, "B") &&
        graph.connect({.node = ramp, .socket = "Alpha"}, mix, "Factor") &&
        graph.set_property(mix, "BlendMode", SocketValue::string("OVERLAY")) &&
        graph.set_property(mix, "ClampFactor", SocketValue::boolean(true)) &&
        graph.set_property(mix, "ClampResult", SocketValue::boolean(true)) &&
        graph.connect({.node = mix, .socket = "Color"}, principled,
                      "BaseColor") &&
        graph.connect({.node = image, .socket = "Alpha"}, principled,
                      "Roughness");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure direct texture-family SVM graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

ShaderGraph make_sampled_color_ramp_graph(
    std::string table, bool shifted_parameter) {
    ShaderGraph graph;
    NodeId factor_node{};
    if (shifted_parameter) {
        factor_node = graph.add_node(
            node_type::add_float,
            "Color Ramp parameter shift");
    }
    const auto ramp = graph.add_node(
        node_type::color_ramp,
        "Late-bound sampled Color Ramp");
    const auto emission = graph.add_node(
        node_type::emission,
        "Color Ramp emission");
    auto configured =
        graph.set_property(
            ramp, "Sampled", SocketValue::boolean(true)) &&
        graph.set_property(
            ramp, "Table", SocketValue::string(std::move(table))) &&
        graph.set_input(
            emission, "Strength", SocketValue::floating(1.0f)) &&
        graph.connect(
            {.node = ramp, .socket = "Color"}, emission, "Color");
    if (shifted_parameter) {
        configured = configured &&
                     graph.set_input(
                         factor_node, "A", SocketValue::floating(0.17f)) &&
                     graph.set_input(
                         factor_node, "B", SocketValue::floating(0.29f)) &&
                     graph.connect(
                         {.node = factor_node, .socket = "Value"},
                         ramp,
                         "Factor");
    } else {
        configured = configured &&
                     graph.set_input(
                         ramp, "Factor", SocketValue::floating(0.46f));
    }
    if (!configured) {
        throw std::runtime_error{
            "failed to configure sampled Color Ramp graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

ShaderGraph make_typed_clamp_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        "Clamp source geometry");
    const auto minmax = graph.add_node(
        node_type::clamp_range,
        "MINMAX clamp");
    const auto range = graph.add_node(
        node_type::clamp_range,
        "RANGE clamp");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Typed-clamp Principled");
    const auto configured =
        graph.set_property(
            minmax, "Mode", SocketValue::string("MINMAX")) &&
        graph.set_input(
            minmax, "Min", SocketValue::floating(0.12f)) &&
        graph.set_input(
            minmax, "Max", SocketValue::floating(0.68f)) &&
        graph.set_property(
            range, "Mode", SocketValue::string("RANGE")) &&
        graph.set_input(
            range, "Min", SocketValue::floating(0.61f)) &&
        graph.set_input(
            range, "Max", SocketValue::floating(0.17f)) &&
        graph.set_input(
            principled, "BaseColor",
            SocketValue::color({0.36f, 0.19f, 0.73f})) &&
        graph.set_input(
            principled, "TransmissionWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SubsurfaceWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled, "EmissionStrength", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = geometry, .socket = "Backfacing"}, minmax, "Value") &&
        graph.connect(
            {.node = geometry, .socket = "Backfacing"}, range, "Value") &&
        graph.connect(
            {.node = minmax, .socket = "Result"},
            principled, "Roughness") &&
        graph.connect(
            {.node = range, .socket = "Result"},
            principled, "Metallic");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure typed Clamp graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

namespace {

[[nodiscard]] ShaderGraph make_typed_map_range_graph(
    bool vector_mode,
    std::string interpolation,
    bool clamp_result) {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        vector_mode ? "Vector Map Range geometry"
                    : "Scalar Map Range geometry");
    const auto map_range = graph.add_node(
        node_type::map_range,
        vector_mode ? "Typed Vector Map Range"
                    : "Typed Scalar Map Range");
    const auto convert = graph.add_node(
        vector_mode ? node_type::vector_to_color
                    : node_type::scalar_to_color,
        "Map Range result color");
    const auto emission = graph.add_node(
        node_type::emission,
        "Map Range emission");
    auto configured =
        graph.set_property(
            map_range,
            "DataType",
            SocketValue::string(vector_mode ? "FLOAT_VECTOR" : "FLOAT")) &&
        graph.set_property(
            map_range,
            "Interpolation",
            SocketValue::string(std::move(interpolation))) &&
        graph.set_property(
            map_range,
            "Clamp",
            SocketValue::boolean(clamp_result));
    if (vector_mode) {
        configured = configured &&
                     graph.set_input(
                         map_range,
                         "FromMinVector",
                         SocketValue::vector({-0.2f, 0.1f, 0.4f})) &&
                     graph.set_input(
                         map_range,
                         "FromMaxVector",
                         SocketValue::vector({0.6f, 0.9f, -0.4f})) &&
                     graph.set_input(
                         map_range,
                         "ToMinVector",
                         SocketValue::vector({0.15f, 0.7f, 0.2f})) &&
                     graph.set_input(
                         map_range,
                         "ToMaxVector",
                         SocketValue::vector({0.75f, 0.1f, 0.9f})) &&
                     graph.set_input(
                         map_range,
                         "StepsVector",
                         SocketValue::vector({2.0f, 3.0f, 4.0f})) &&
                     graph.connect(
                         {.node = geometry, .socket = "Incoming"},
                         map_range,
                         "Vector") &&
                     graph.connect(
                         {.node = map_range, .socket = "Vector"},
                         convert,
                         "Vector");
    } else {
        configured = configured &&
                     graph.set_input(
                         map_range,
                         "FromMin",
                         SocketValue::floating(0.25f)) &&
                     graph.set_input(
                         map_range,
                         "FromMax",
                         SocketValue::floating(0.75f)) &&
                     graph.set_input(
                         map_range,
                         "ToMin",
                         SocketValue::floating(0.2f)) &&
                     graph.set_input(
                         map_range,
                         "ToMax",
                         SocketValue::floating(0.8f)) &&
                     graph.set_input(
                         map_range,
                         "Steps",
                         SocketValue::floating(3.0f)) &&
                     graph.connect(
                         {.node = geometry, .socket = "Backfacing"},
                         map_range,
                         "Value") &&
                     graph.connect(
                         {.node = map_range, .socket = "Result"},
                         convert,
                         "Value");
    }
    configured = configured &&
                 graph.connect(
                     {.node = convert, .socket = "Color"},
                     emission,
                     "Color");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure typed Map Range graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

} // namespace

std::vector<ShaderGraph> make_typed_map_range_graphs() {
    constexpr std::array interpolations{
        "LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"};
    std::vector<ShaderGraph> result;
    result.reserve(interpolations.size() * 4u);
    for (const auto *interpolation : interpolations) {
        for (const auto clamp_result : {false, true}) {
            result.emplace_back(make_typed_map_range_graph(
                false, interpolation, clamp_result));
            result.emplace_back(make_typed_map_range_graph(
                true, interpolation, clamp_result));
        }
    }
    return result;
}

bool has_typed_clamp_record_domain(
    const SurfaceValueRuntime &runtime) noexcept {
    const SurfaceValueStaticVariant *clamp_variant = nullptr;
    constexpr auto expected_key = make_surface_value_handler_key(
        ValueOperation::clamp_range,
        SurfaceValueBank::scalar,
        0u);
    for (const auto &variant : runtime.value_variants) {
        if (variant.instruction.operation != ValueOperation::clamp_range) {
            continue;
        }
        if (clamp_variant != nullptr) {
            return false;
        }
        auto bank = SurfaceValueBank::scalar;
        if (!classify_surface_value_type(
                variant.instruction.result_type, bank) ||
            variant.svm_immediates.empty() ||
            make_surface_value_handler_key(
                variant.instruction.operation,
                bank,
                variant.svm_immediates.front()) != expected_key) {
            return false;
        }
        clamp_variant = &variant;
    }
    return clamp_variant != nullptr &&
           clamp_variant->instruction.static_u0 == 0u &&
           clamp_variant->instruction.static_u1 == 0u &&
           clamp_variant->svm_immediates ==
               std::vector<std::uint16_t>{0u, 1u};
}

bool has_typed_map_range_record_domains(
    const SurfaceValueRuntime &runtime) noexcept {
    constexpr std::array expected_immediates{
        std::uint16_t{0u}, std::uint16_t{1u},
        std::uint16_t{2u}, std::uint16_t{3u},
        std::uint16_t{4u}, std::uint16_t{5u},
        std::uint16_t{6u}, std::uint16_t{7u}};
    auto scalar_count = std::size_t{0u};
    auto vector_count = std::size_t{0u};
    for (const auto &variant : runtime.value_variants) {
        if (variant.instruction.operation != ValueOperation::map_range_float &&
            variant.instruction.operation != ValueOperation::map_range_vector) {
            continue;
        }
        if (variant.instruction.static_u0 != 0u ||
            variant.instruction.static_u1 != 0u ||
            !std::equal(
                variant.svm_immediates.begin(),
                variant.svm_immediates.end(),
                expected_immediates.begin(),
                expected_immediates.end())) {
            return false;
        }
        if (variant.instruction.operation == ValueOperation::map_range_float) {
            ++scalar_count;
        } else {
            ++vector_count;
        }
    }
    return scalar_count == 1u && vector_count == 1u;
}

bool has_color_ramp_record_product(
    const SurfaceValueRuntime &runtime) noexcept {
    // The family subtype is the direct product of semantic operation and
    // physical result bank.  A graph that consumes both Color and Alpha must
    // therefore add a scalar provenance class without invalidating the
    // independently shared vector class proved by this regression.  Project
    // both sides of the relation to the vector subtype before checking that
    // table ParameterId remains late-bound rather than entering handler
    // identity.
    const auto variant_count = std::count_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [](const auto &variant) noexcept {
            auto bank = SurfaceValueBank::scalar;
            return variant.instruction.operation ==
                       ValueOperation::color_ramp &&
                   classify_surface_value_type(variant.instruction.result_type,
                                               bank) &&
                   bank == SurfaceValueBank::vector;
        });
    std::vector<std::uint32_t> parameters;
    for (const auto &instruction : runtime.svm_scene.instructions) {
        if (surface_svm_bytecode_kind(instruction) !=
            SurfaceSvmBytecodeKind::value) {
            continue;
        }
        const auto value = surface_svm_value_instruction(instruction);
        if (surface_value_operation(value) != ValueOperation::color_ramp ||
            surface_value_result_bank(value) != SurfaceValueBank::vector) {
            continue;
        }
        if (value.metadata_index >=
            runtime.svm_scene.value_metadata.size()) {
            return false;
        }
        const auto parameter =
            runtime.svm_scene.value_metadata[value.metadata_index].parameter;
        if (parameter == ~std::uint32_t{0u}) {
            return false;
        }
        if (std::find(parameters.begin(), parameters.end(), parameter) ==
            parameters.end()) {
            parameters.emplace_back(parameter);
        }
    }
    return variant_count == 1u && parameters.size() >= 2u;
}

CompactSurfaceProgramEvidence
inspect_compact_surface_program(const SurfaceValueRuntime &runtime) noexcept {
    CompactSurfaceProgramEvidence result;
    const auto preparation = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::preparation);
    const auto emission = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::emission);
    const auto bssrdf = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::bssrdf);
    result.domains_match =
        domain_matches(preparation, runtime.preparation_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset) &&
        domain_matches(emission, runtime.emission_value_static_variants,
                       SurfaceValueRuntime::emission_program_offset) &&
        domain_matches(bssrdf, runtime.bssrdf_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset);

    const auto &svm = runtime.svm_scene;
    const auto expected_program_count =
        runtime.topologies.size() * SurfaceValueRuntime::programs_per_topology;
    auto unified_scene_exact =
        svm.valid && svm.programs.size() == expected_program_count &&
        svm.side_ranges.size() == expected_program_count &&
        validate_surface_svm_scene_image(svm).empty() &&
        svm.maximum_stack_lanes <= SurfaceValueRuntime::stack_capacity;
    auto unified_variant_bijection =
        unified_scene_exact &&
        runtime.svm_instruction_variants.size() == svm.instructions.size();
    auto normal_transactions_exact = unified_variant_bijection;
    auto observed_leaf = false;
    auto observed_guard = false;
    std::vector<std::uint32_t> preparation_value_domain;
    std::vector<std::uint32_t> emission_value_domain;
    std::vector<SurfaceSvmClosureVariant> preparation_closure_domain;
    std::vector<SurfaceSvmClosureVariant> emission_closure_domain;
    const auto canonicalize = [](auto &domain) {
        std::sort(domain.begin(), domain.end());
        domain.erase(std::unique(domain.begin(), domain.end()), domain.end());
    };

    for (auto program = std::uint32_t{};
         unified_scene_exact && program < svm.programs.size(); ++program) {
        const auto &range = svm.programs[program];
        if (range.instruction_begin > svm.instructions.size() ||
            range.instruction_count >
                svm.instructions.size() - range.instruction_begin) {
            unified_scene_exact = false;
            break;
        }
        const auto emission_program =
            program % SurfaceValueRuntime::programs_per_topology ==
            SurfaceValueRuntime::emission_program_offset;
        const auto expected_endpoints =
            emission_program
                ? surface_closure_endpoint_bit(
                      SurfaceClosureEndpoint::emission)
                : all_surface_closure_endpoints;
        unified_scene_exact &= range.endpoints == expected_endpoints;

        auto normal_count = std::uint32_t{};
        auto end_count = std::uint32_t{};
        auto &value_domain = emission_program ? emission_value_domain
                                              : preparation_value_domain;
        auto &closure_domain = emission_program ? emission_closure_domain
                                                : preparation_closure_domain;
        for (auto offset = std::uint32_t{};
             offset < range.instruction_count; ++offset) {
            const auto instruction_index = range.instruction_begin + offset;
            const auto &instruction = svm.instructions[instruction_index];
            const auto kind = surface_svm_bytecode_kind(instruction);
            normal_count += kind == SurfaceSvmBytecodeKind::set_normal;
            end_count += kind == SurfaceSvmBytecodeKind::end;
            observed_leaf |= kind == SurfaceSvmBytecodeKind::closure_leaf;
            observed_guard |= kind == SurfaceSvmBytecodeKind::jump_if_one ||
                              kind == SurfaceSvmBytecodeKind::jump_if_zero;
            if (kind == SurfaceSvmBytecodeKind::closure_leaf) {
                closure_domain.emplace_back(SurfaceSvmClosureVariant{
                    .static_variant =
                        surface_svm_closure_control(instruction) &
                        surface_closure_static_variant_mask,
                    .principled_features = instruction.payload2});
            }
            if (kind == SurfaceSvmBytecodeKind::invalid) {
                unified_scene_exact = false;
            }
            const auto variant =
                runtime.svm_instruction_variants[instruction_index];
            if (kind != SurfaceSvmBytecodeKind::value) {
                unified_variant_bijection &=
                    variant == SurfaceValueAddress::invalid_value;
                continue;
            }
            if (variant >= runtime.value_variants.size()) {
                unified_variant_bijection = false;
                continue;
            }
            const auto value = surface_svm_value_instruction(instruction);
            const auto &static_variant = runtime.value_variants[variant];
            unified_variant_bijection &=
                surface_value_operation(value) ==
                    static_variant.instruction.operation &&
                surface_value_operand_count(value) ==
                    static_variant.operand_types.size() &&
                std::find(static_variant.svm_immediates.begin(),
                          static_variant.svm_immediates.end(),
                          surface_value_svm_immediate(value)) !=
                    static_variant.svm_immediates.end();
            value_domain.emplace_back(variant);
        }
        const auto final_kind =
            range.instruction_count == 0u
                ? SurfaceSvmBytecodeKind::invalid
                : surface_svm_bytecode_kind(
                      svm.instructions[range.instruction_begin +
                                       range.instruction_count - 1u]);
        const auto known_flags =
            (range.flags & ~surface_value_program_flag_mask) == 0u;
        const auto flags_require_transition =
            range.flags == 0u || normal_count == 1u;
        unified_scene_exact &= normal_count <= 1u && end_count == 1u &&
                               final_kind == SurfaceSvmBytecodeKind::end;
        normal_transactions_exact &=
            normal_count <= 1u && known_flags && flags_require_transition;
        if (!emission_program) {
            const auto topology =
                program / SurfaceValueRuntime::programs_per_topology;
            normal_transactions_exact &=
                topology < runtime.topologies.size() &&
                (normal_count == 1u) ==
                    runtime.topologies[topology]
                        .program->surface_normal_root().valid();
        } else {
            const auto topology =
                program / SurfaceValueRuntime::programs_per_topology;
            normal_transactions_exact &=
                normal_count == 0u ||
                (topology < runtime.topologies.size() &&
                 runtime.topologies[topology]
                     .program->surface_normal_root().valid());
        }
    }

    canonicalize(preparation_value_domain);
    canonicalize(emission_value_domain);
    canonicalize(preparation_closure_domain);
    canonicalize(emission_closure_domain);
    auto canonical_bssrdf_domain = runtime.bssrdf_svm_closure_variants;
    canonicalize(canonical_bssrdf_domain);
    unified_variant_bijection &=
        preparation_value_domain ==
            runtime.preparation_value_static_variants &&
        emission_value_domain == runtime.emission_value_static_variants;
    result.unified_scene_exact =
        unified_scene_exact && observed_leaf && observed_guard;
    result.unified_variant_bijection =
        unified_variant_bijection && result.unified_scene_exact;
    result.normal_transactions_exact =
        normal_transactions_exact && result.unified_scene_exact;
    result.unified_closure_domains_exact =
        result.unified_scene_exact &&
        preparation_closure_domain ==
            runtime.preparation_svm_closure_variants &&
        emission_closure_domain == runtime.emission_svm_closure_variants &&
        canonical_bssrdf_domain == runtime.bssrdf_svm_closure_variants &&
        std::includes(preparation_closure_domain.begin(),
                      preparation_closure_domain.end(),
                      canonical_bssrdf_domain.begin(),
                      canonical_bssrdf_domain.end());

    const auto bump = std::find_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [](const auto &variant) {
            return variant.instruction.operation ==
                   ValueOperation::bump_samples;
        });
    if (bump == runtime.value_variants.end()) {
        return result;
    }
    result.bump_variant = static_cast<std::uint32_t>(
        std::distance(runtime.value_variants.begin(), bump));
    const auto contains = [variant = result.bump_variant](
                              const auto &domain) noexcept {
        return std::find(domain.begin(), domain.end(), variant) != domain.end();
    };
    const auto has_no_recursive_bump = std::none_of(
        svm.instructions.begin(), svm.instructions.end(),
        [](const auto &instruction) noexcept {
            if (surface_svm_bytecode_kind(instruction) !=
                SurfaceSvmBytecodeKind::value) {
                return false;
            }
            return surface_value_operation(
                       surface_svm_value_instruction(instruction)) ==
                   ValueOperation::bump;
        });
    result.bump_stream_exact =
        result.unified_variant_bijection &&
        contains(runtime.preparation_value_static_variants) &&
        count_bump_configuration(
            svm, 0u, static_cast<std::uint32_t>(svm.programs.size()), 1u) !=
            0u &&
        count_bump_configuration(
            svm, 0u, static_cast<std::uint32_t>(svm.programs.size()), 0u) !=
            0u &&
        has_no_recursive_bump;
    return result;
}

namespace {

[[nodiscard]] bool finite(luisa::float2 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool finite(luisa::float3 value) noexcept {
    return std::isfinite(value.x) &&
           std::isfinite(value.y) &&
           std::isfinite(value.z);
}

void print(luisa::float3 value) {
    std::cerr << '{' << value.x << ", " << value.y << ", "
              << value.z << '}';
}

} // namespace

bool finite_compact_value(luisa::float3 value) noexcept {
    return finite(value);
}

void print_compact_value(luisa::float3 value) {
    print(value);
}

bool equal(luisa::float2 actual, luisa::float2 expected,
           float tolerance) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance);
}

bool equal(luisa::float3 actual, luisa::float3 expected,
           float tolerance) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

bool equal(const SurfacePreparationCall &actual,
           const SurfacePreparationCall &expected,
           float tolerance) noexcept {
    return actual.runtime_flags == expected.runtime_flags &&
           finite(actual.emission) && finite(actual.shading_normal) &&
           finite(actual.albedo) && finite(actual.glossy_albedo) &&
           finite(actual.transmission_albedo) && finite(actual.normal) &&
           finite(actual.transparency) && finite(actual.roughness) &&
           equal(actual.emission, expected.emission, tolerance) &&
           equal(actual.shading_normal, expected.shading_normal, tolerance) &&
           equal(actual.albedo, expected.albedo, tolerance) &&
           equal(actual.glossy_albedo, expected.glossy_albedo, tolerance) &&
           equal(actual.transmission_albedo,
                 expected.transmission_albedo,
                 tolerance) &&
           equal(actual.normal, expected.normal, tolerance) &&
           equal(actual.transparency, expected.transparency, tolerance) &&
           equal(actual.roughness, expected.roughness, tolerance);
}

bool equal(const SurfaceClosureTraceCall &actual,
           const SurfaceClosureTraceCall &expected,
           float tolerance) noexcept {
    return actual.count == expected.count &&
           actual.runtime_flags == expected.runtime_flags &&
           actual.index == expected.index && actual.type == expected.type &&
           actual.valid == expected.valid &&
           std::isfinite(actual.sample_weight) &&
           finite(actual.weight) && finite(actual.normal) &&
           approximately_equal(actual.sample_weight,
                               expected.sample_weight,
                               tolerance) &&
           equal(actual.weight, expected.weight, tolerance) &&
           equal(actual.normal, expected.normal, tolerance);
}

bool equal(const SurfaceEvaluationCall &actual,
           const SurfaceEvaluationCall &expected,
           float tolerance) noexcept {
    return actual.events == expected.events && finite(actual.f) &&
           finite(actual.diffuse_f) && finite(actual.glossy_f) &&
           std::isfinite(actual.pdf) && std::isfinite(actual.diffuse_pdf) &&
           std::isfinite(actual.average_roughness_squared) &&
           equal(actual.f, expected.f, tolerance) &&
           equal(actual.diffuse_f, expected.diffuse_f, tolerance) &&
           equal(actual.glossy_f, expected.glossy_f, tolerance) &&
           approximately_equal(actual.pdf, expected.pdf, tolerance) &&
           approximately_equal(actual.diffuse_pdf,
                               expected.diffuse_pdf,
                               tolerance) &&
           approximately_equal(actual.average_roughness_squared,
                               expected.average_roughness_squared,
                               tolerance);
}

bool equal(const SurfaceSampleTraceCall &actual,
           const SurfaceSampleTraceCall &expected,
           float tolerance) noexcept {
    return equal(
               SurfaceEvaluationCall{
                   .f = actual.f,
                   .pdf = actual.pdf,
                   .diffuse_f = actual.diffuse_f,
                   .glossy_f = actual.glossy_f,
                   .diffuse_pdf = actual.diffuse_pdf,
                   .average_roughness_squared =
                       actual.average_roughness_squared,
                   .events = actual.events},
               SurfaceEvaluationCall{
                   .f = expected.f,
                   .pdf = expected.pdf,
                   .diffuse_f = expected.diffuse_f,
                   .glossy_f = expected.glossy_f,
                   .diffuse_pdf = expected.diffuse_pdf,
                   .average_roughness_squared =
                       expected.average_roughness_squared,
                   .events = expected.events},
               tolerance) &&
           actual.runtime_flags == expected.runtime_flags &&
           actual.bssrdf_method == expected.bssrdf_method &&
           actual.valid == expected.valid &&
           actual.closure_index == expected.closure_index &&
           actual.closure_type == expected.closure_type &&
           actual.closure_valid == expected.closure_valid &&
           finite(actual.wi) && finite(actual.roughness) &&
           finite(actual.bssrdf_radius) && finite(actual.bssrdf_albedo) &&
           finite(actual.bssrdf_normal) && finite(actual.closure_weight) &&
           finite(actual.closure_normal) &&
           approximately_equal(actual.eta, expected.eta, tolerance) &&
           equal(actual.wi, expected.wi, tolerance) &&
           equal(actual.roughness, expected.roughness, tolerance) &&
           equal(actual.bssrdf_radius, expected.bssrdf_radius, tolerance) &&
           equal(actual.bssrdf_albedo, expected.bssrdf_albedo, tolerance) &&
           equal(actual.bssrdf_normal, expected.bssrdf_normal, tolerance) &&
           approximately_equal(actual.bssrdf_ior,
                               expected.bssrdf_ior,
                               tolerance) &&
           approximately_equal(actual.bssrdf_roughness,
                               expected.bssrdf_roughness,
                               tolerance) &&
           approximately_equal(actual.bssrdf_anisotropy,
                               expected.bssrdf_anisotropy,
                               tolerance) &&
           approximately_equal(actual.closure_sample_weight,
                               expected.closure_sample_weight,
                               tolerance) &&
           approximately_equal(actual.selection_rescaled,
                               expected.selection_rescaled,
                               tolerance) &&
           equal(actual.closure_weight,
                 expected.closure_weight,
                 tolerance) &&
           equal(actual.closure_normal,
                 expected.closure_normal,
                 tolerance);
}

void report_compact_surface_preparation_mismatch(
    std::string_view backend,
    std::size_t topology,
    std::size_t scenario,
    const SurfacePreparationCall &actual,
    const SurfacePreparationCall &expected) {
    std::cerr << "compact surface preparation mismatch on "
              << backend << ", topology " << topology
              << ", scenario " << scenario << '\n';
    std::cerr << "  compact emission=";
    print(actual.emission);
    std::cerr << ", expanded emission=";
    print(expected.emission);
    std::cerr << '\n' << "  compact shading normal=";
    print(actual.shading_normal);
    std::cerr << ", expanded shading normal=";
    print(expected.shading_normal);
    std::cerr << '\n' << "  compact albedo=";
    print(actual.albedo);
    std::cerr << ", expanded albedo=";
    print(expected.albedo);
    std::cerr << '\n' << "  compact glossy=";
    print(actual.glossy_albedo);
    std::cerr << ", expanded glossy=";
    print(expected.glossy_albedo);
    std::cerr << '\n' << "  compact transmission=";
    print(actual.transmission_albedo);
    std::cerr << ", expanded transmission=";
    print(expected.transmission_albedo);
    std::cerr << '\n' << "  compact normal=";
    print(actual.normal);
    std::cerr << ", expanded normal=";
    print(expected.normal);
    std::cerr << '\n' << "  compact transparency=";
    print(actual.transparency);
    std::cerr << ", expanded transparency=";
    print(expected.transparency);
    std::cerr << '\n' << "  compact roughness={"
              << actual.roughness.x << ", " << actual.roughness.y
              << "}, expanded roughness={" << expected.roughness.x
              << ", " << expected.roughness.y << "}\n"
              << "  compact flags=" << actual.runtime_flags
              << ", expanded flags=" << expected.runtime_flags << '\n';
}

} // namespace psycles::test_support
