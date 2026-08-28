#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

struct RecordConfiguration {
    std::uint64_t u0{};
    std::uint64_t u1{};
};

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] SocketValue default_value(SocketType type) {
    switch (type) {
        case SocketType::floating:
            return SocketValue::floating(0.5f);
        case SocketType::color:
            return SocketValue::color({0.25f, 0.5f, 0.75f});
        case SocketType::point:
            return SocketValue::point({0.25f, 0.5f, 0.75f});
        case SocketType::vector:
            return SocketValue::vector({0.25f, 0.5f, 0.75f});
        case SocketType::normal:
            return SocketValue::normal({0.0f, 0.0f, 1.0f});
        case SocketType::unsigned_integer:
            return SocketValue::unsigned_integer(7u);
        default:
            throw std::runtime_error{"unsupported test socket type"};
    }
}

[[nodiscard]] SurfaceProgram make_program(
    std::uint32_t tag,
    ValueOperation operation,
    SocketType result_type,
    std::span<const SocketType> operand_types,
    RecordConfiguration configuration) {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    std::vector<ValueExpressionId> operands;
    parameters.reserve(operand_types.size());
    values.reserve(operand_types.size() + 1u);
    operands.reserve(operand_types.size());
    for (auto index = std::size_t{0u}; index < operand_types.size(); ++index) {
        const auto parameter = static_cast<std::uint32_t>(index);
        parameters.emplace_back(ParameterDesc{
            .id = ParameterId{parameter},
            .node = NodeId{parameter + 1u},
            .socket = "Input" + std::to_string(index),
            .type = operand_types[index],
            .default_value = default_value(operand_types[index]),
            .source = ParameterSource::input});
        values.emplace_back(ValueInstruction{
            .operation = ValueOperation::parameter,
            .result_type = operand_types[index],
            .parameter = ParameterId{parameter}});
        operands.emplace_back(ValueExpressionId{parameter});
    }
    values.emplace_back(ValueInstruction{
        .operation = operation,
        .result_type = result_type,
        .operands = std::move(operands),
        .static_u0 = configuration.u0,
        .static_u1 = configuration.u1});
    return SurfaceProgram{
        tag, std::move(parameters), std::move(values), {}, {}};
}

[[nodiscard]] SurfaceValueStoragePlan make_plan(
    const SurfaceProgram &program) {
    const auto count = program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(
        program, std::vector<bool>(count, true), outputs);
}

void require_single_handler(
    ValueOperation operation,
    SocketType result_type,
    std::span<const SocketType> operand_types,
    std::span<const RecordConfiguration> configurations,
    const char *message) {
    std::vector<SurfaceProgram> programs;
    std::vector<SurfaceValueStoragePlan> plans;
    programs.reserve(configurations.size());
    plans.reserve(configurations.size());
    for (auto index = std::size_t{0u}; index < configurations.size(); ++index) {
        programs.emplace_back(make_program(
            static_cast<std::uint32_t>(100u + index),
            operation,
            result_type,
            operand_types,
            configurations[index]));
        plans.emplace_back(make_plan(programs.back()));
    }
    std::vector<SurfaceValueExecutionInput> inputs;
    inputs.reserve(programs.size());
    for (auto index = std::size_t{0u}; index < programs.size(); ++index) {
        inputs.emplace_back(SurfaceValueExecutionInput{
            .program = &programs[index], .storage = &plans[index]});
    }
    auto expected_immediates = std::vector<std::uint16_t>{};
    expected_immediates.reserve(configurations.size());
    for (const auto configuration : configurations) {
        expected_immediates.emplace_back(static_cast<std::uint16_t>(
            make_surface_value_svm_immediate(
                operation, configuration.u0, configuration.u1)));
    }
    std::sort(expected_immediates.begin(), expected_immediates.end());
    expected_immediates.erase(
        std::unique(expected_immediates.begin(), expected_immediates.end()),
        expected_immediates.end());

    const auto scene = build_surface_value_executable_scene(inputs);
    require(scene.valid && scene.variants.size() == 1u &&
                scene.instruction_variants ==
                    std::vector<std::uint32_t>(configurations.size(), 0u) &&
                scene.variants.front().instruction.static_u0 == 0u &&
                scene.variants.front().instruction.static_u1 == 0u &&
                scene.variants.front().svm_immediates == expected_immediates,
            message);
}

void require_rejected(
    ValueOperation operation,
    SocketType result_type,
    std::span<const SocketType> operand_types,
    RecordConfiguration configuration,
    const char *message) {
    const auto program = make_program(
        999u, operation, result_type, operand_types, configuration);
    require(!lower_surface_value_program(program, make_plan(program)).valid,
            message);
}

void test_typed_record_quotients() {
    constexpr SocketType uv_operands[]{SocketType::unsigned_integer};
    constexpr RecordConfiguration uv_configs[]{{0u, 0u}, {1u, 0u}};
    require_single_handler(
        ValueOperation::uv,
        SocketType::vector,
        uv_operands,
        uv_configs,
        "default and named UV records split their SVM handler");

    constexpr SocketType optional_normal_operands[]{
        SocketType::floating, SocketType::normal};
    constexpr RecordConfiguration optional_normal_configs[]{{0u, 0u},
                                                              {1u, 0u}};
    require_single_handler(
        ValueOperation::fresnel,
        SocketType::floating,
        optional_normal_operands,
        optional_normal_configs,
        "Fresnel linked-normal data split its SVM handler");
    require_single_handler(
        ValueOperation::layer_weight_facing,
        SocketType::floating,
        optional_normal_operands,
        optional_normal_configs,
        "Layer Weight linked-normal data split its SVM handler");
    require_single_handler(
        ValueOperation::layer_weight_fresnel,
        SocketType::floating,
        optional_normal_operands,
        optional_normal_configs,
        "Layer Weight Fresnel linked-normal data split its SVM handler");

    constexpr SocketType ambient_occlusion_operands[]{
        SocketType::floating,
        SocketType::normal,
        SocketType::unsigned_integer};
    constexpr RecordConfiguration ambient_occlusion_configs[]{
        {0u, 0u},
        {ambient_occlusion_only_local, 0u},
        {ambient_occlusion_inside | ambient_occlusion_global_radius |
             ambient_occlusion_normal_linked,
         0u}};
    require_single_handler(
        ValueOperation::ambient_occlusion,
        SocketType::floating,
        ambient_occlusion_operands,
        ambient_occlusion_configs,
        "Ambient Occlusion flags split its typed SVM handler");
    require_rejected(
        ValueOperation::ambient_occlusion,
        SocketType::floating,
        ambient_occlusion_operands,
        {1u << 8u, 0u},
        "Ambient Occlusion accepted an unknown configuration bit");
    require_rejected(
        ValueOperation::ambient_occlusion,
        SocketType::floating,
        ambient_occlusion_operands,
        {0u, 1u},
        "Ambient Occlusion accepted a non-canonical second immediate");

    constexpr SocketType clamp_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    constexpr RecordConfiguration clamp_configs[]{
        {static_cast<std::uint64_t>(ClampMode::minmax), 0u},
        {static_cast<std::uint64_t>(ClampMode::range), 0u}};
    require_single_handler(
        ValueOperation::clamp_range,
        SocketType::floating,
        clamp_operands,
        clamp_configs,
        "Clamp modes split their typed SVM handler");

    constexpr SocketType map_range_scalar_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    constexpr SocketType map_range_vector_operands[]{
        SocketType::vector,
        SocketType::vector,
        SocketType::vector,
        SocketType::vector,
        SocketType::vector,
        SocketType::vector};
    constexpr RecordConfiguration map_range_configs[]{
        {static_cast<std::uint64_t>(MapRangeInterpolation::linear), 0u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::linear), 1u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::stepped), 0u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::stepped), 1u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::smoothstep), 0u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::smoothstep), 1u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::smootherstep), 0u},
        {static_cast<std::uint64_t>(MapRangeInterpolation::smootherstep), 1u}};
    require_single_handler(
        ValueOperation::map_range_float,
        SocketType::floating,
        map_range_scalar_operands,
        map_range_configs,
        "scalar Map Range configurations split their typed SVM handler");
    require_single_handler(
        ValueOperation::map_range_vector,
        SocketType::vector,
        map_range_vector_operands,
        map_range_configs,
        "vector Map Range configurations split their typed SVM handler");

    constexpr SocketType gradient_operands[]{SocketType::vector};
    constexpr RecordConfiguration gradient_configs[]{{0u, 0u},
                                                       {4u, 0u},
                                                       {6u, 0u}};
    require_single_handler(
        ValueOperation::gradient,
        SocketType::floating,
        gradient_operands,
        gradient_configs,
        "Gradient modes split its SVM handler");

    constexpr SocketType noise_operands[]{
        SocketType::vector,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    constexpr RecordConfiguration noise_configs[]{{3u, 0x101u},
                                                    {3u, 0x100u},
                                                    {2u, 0x100u},
                                                    {3u, 0u}};
    require_single_handler(
        ValueOperation::noise_factor,
        SocketType::floating,
        noise_operands,
        noise_configs,
        "Noise record data split its SVM handler");

    constexpr SocketType wave_operands[]{
        SocketType::vector,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    constexpr RecordConfiguration wave_configs[]{{0x01030300u, 0u},
                                                   {0x00030300u, 0u},
                                                   {0x02020101u, 0u}};
    require_single_handler(
        ValueOperation::wave_factor,
        SocketType::floating,
        wave_operands,
        wave_configs,
        "Wave record data split its SVM handler");

    constexpr SocketType normal_map_operands[]{
        SocketType::color,
        SocketType::floating,
        SocketType::unsigned_integer};
    constexpr RecordConfiguration normal_map_configs[]{
        {encode_normal_map_configuration(
             NormalMapSpace::tangent,
             false,
             NormalMapBase::displaced,
             NormalMapConvention::open_gl),
         0u},
        {encode_normal_map_configuration(
             NormalMapSpace::object,
             false,
             NormalMapBase::displaced,
             NormalMapConvention::open_gl),
         0u},
        {encode_normal_map_configuration(
             NormalMapSpace::tangent,
             true,
             NormalMapBase::displaced,
             NormalMapConvention::direct_x),
         0u}};
    require_single_handler(
        ValueOperation::normal_map,
        SocketType::normal,
        normal_map_operands,
        normal_map_configs,
        "Normal Map record data split its SVM handler");

    constexpr SocketType bump_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::normal};
    constexpr RecordConfiguration bump_configs[]{{0u, 0u},
                                                   {1u, 0u},
                                                   {2u, 0u},
                                                   {7u, 0u}};
    require_single_handler(
        ValueOperation::bump,
        SocketType::normal,
        bump_operands,
        bump_configs,
        "Bump record data split its SVM handler");
}

void test_invalid_records_are_rejected() {
    constexpr SocketType uv_operands[]{SocketType::unsigned_integer};
    require_rejected(
        ValueOperation::uv,
        SocketType::vector,
        uv_operands,
        {2u, 0u},
        "UV accepted an invalid named-attribute flag");

    constexpr SocketType optional_normal_operands[]{
        SocketType::floating, SocketType::normal};
    require_rejected(
        ValueOperation::fresnel,
        SocketType::floating,
        optional_normal_operands,
        {2u, 0u},
        "Fresnel accepted an invalid linked-normal flag");

    constexpr SocketType clamp_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    require_rejected(
        ValueOperation::clamp_range,
        SocketType::floating,
        clamp_operands,
        {2u, 0u},
        "Clamp accepted an invalid mode");
    require_rejected(
        ValueOperation::clamp_range,
        SocketType::floating,
        clamp_operands,
        {0u, 1u},
        "Clamp accepted a foreign immutable field");

    constexpr SocketType map_range_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    require_rejected(
        ValueOperation::map_range_float,
        SocketType::floating,
        map_range_operands,
        {map_range_interpolation_count, 0u},
        "Map Range accepted an invalid interpolation");
    require_rejected(
        ValueOperation::map_range_float,
        SocketType::floating,
        map_range_operands,
        {0u, 2u},
        "Map Range accepted a non-Boolean clamp field");

    constexpr SocketType gradient_operands[]{SocketType::vector};
    require_rejected(
        ValueOperation::gradient,
        SocketType::floating,
        gradient_operands,
        {7u, 0u},
        "Gradient accepted an invalid mode");

    constexpr SocketType noise_operands[]{
        SocketType::vector,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    require_rejected(
        ValueOperation::noise_factor,
        SocketType::floating,
        noise_operands,
        {0u, 0x100u},
        "Noise accepted an invalid dimension");
    require_rejected(
        ValueOperation::noise_factor,
        SocketType::floating,
        noise_operands,
        {3u, 0x500u},
        "Noise accepted an invalid type");

    constexpr SocketType wave_operands[]{
        SocketType::vector,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    require_rejected(
        ValueOperation::wave_factor,
        SocketType::floating,
        wave_operands,
        {2u, 0u},
        "Wave accepted an invalid type");

    constexpr SocketType normal_map_operands[]{
        SocketType::color,
        SocketType::floating,
        SocketType::unsigned_integer};
    require_rejected(
        ValueOperation::normal_map,
        SocketType::normal,
        normal_map_operands,
        {5u, 0u},
        "Normal Map accepted an invalid space");

    constexpr SocketType bump_operands[]{
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::normal};
    require_rejected(
        ValueOperation::bump,
        SocketType::normal,
        bump_operands,
        {8u, 0u},
        "Bump accepted an invalid flag");
}

void test_serialized_record_coherence() {
    constexpr SocketType noise_operands[]{
        SocketType::vector,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating,
        SocketType::floating};
    const auto noise = make_program(
        1000u,
        ValueOperation::noise_factor,
        SocketType::floating,
        noise_operands,
        {3u, 0x100u});
    auto noise_image = lower_surface_value_program(noise, make_plan(noise));
    require(noise_image.valid && noise_image.instructions.size() == 1u,
            "valid Noise record did not lower to one instruction");
    noise_image.instructions.front().control |=
        surface_value_noise_normalize_immediate_bit
        << surface_value_svm_immediate_shift;
    const auto mismatched =
        build_surface_value_scene_image(std::vector{noise_image});
    require(!mismatched.valid &&
                mismatched.diagnostic.find(
                    "immediate disagrees with immutable metadata") !=
                    std::string::npos,
            "serialized Noise accepted an immediate/metadata mismatch");

    constexpr SocketType passthrough_operands[]{SocketType::floating};
    const auto passthrough = make_program(
        1001u,
        ValueOperation::passthrough,
        SocketType::floating,
        passthrough_operands,
        {});
    // Automatic planning contracts this pure identity to its parameter.
    // Keep the serialized-record validation independent by explicitly
    // constructing the legacy uncoalesced layout that old scene images may
    // still contain.
    const SurfaceValueStoragePlan legacy_passthrough_plan{
        .valid = true,
        .locations = {{.storage = SurfaceValueStorageClass::parameter,
                       .bank = SurfaceValueBank::scalar,
                       .index = 0u},
                      {.storage = SurfaceValueStorageClass::local_slot,
                       .bank = SurfaceValueBank::scalar,
                       .index = 0u}},
        .instructions = {ValueExpressionId{1u}},
        .scalar_slots = 1u,
        .active_values = 2u,
        .parameter_values = 1u};
    auto foreign =
        lower_surface_value_program(passthrough, legacy_passthrough_plan);
    require(foreign.valid && foreign.instructions.size() == 1u,
            "valid Passthrough record failed lowering");
    foreign.instructions.front().control |=
        1u << surface_value_svm_immediate_shift;
    const auto foreign_scene =
        build_surface_value_scene_image(std::vector{foreign});
    require(!foreign_scene.valid &&
                foreign_scene.diagnostic.find(
                    "without an immediate contract") != std::string::npos,
            "an opcode without a record immediate accepted control data");
}

void test_primary_handler_projection() {
    constexpr SocketType image_operands[]{
        SocketType::vector,
        SocketType::unsigned_integer,
        SocketType::floating};
    const auto flat = make_program(
        1100u,
        ValueOperation::image_color,
        SocketType::color,
        image_operands,
        {0u, 0u});
    const auto box = make_program(
        1101u,
        ValueOperation::image_color,
        SocketType::color,
        image_operands,
        {0u, 1u << surface_value_image_projection_shift});
    const auto sphere = make_program(
        1102u,
        ValueOperation::image_color,
        SocketType::color,
        image_operands,
        {0u, 2u << surface_value_image_projection_shift});
    const auto flat_image = lower_surface_value_program(flat, make_plan(flat));
    const auto box_image = lower_surface_value_program(box, make_plan(box));
    const auto sphere_image =
        lower_surface_value_program(sphere, make_plan(sphere));
    require(flat_image.valid && box_image.valid && sphere_image.valid &&
                flat_image.instructions.size() == 1u &&
                box_image.instructions.size() == 1u &&
                sphere_image.instructions.size() == 1u &&
                surface_value_handler_key(flat_image.instructions.front()) ==
                    surface_value_handler_key(sphere_image.instructions.front()) &&
                surface_value_handler_key(flat_image.instructions.front()) !=
                    surface_value_handler_key(box_image.instructions.front()),
            "primary handler projection did not isolate Image BOX exactly");

    constexpr SocketType mix_vector_uniform_operands[]{
        SocketType::vector, SocketType::vector, SocketType::floating};
    constexpr SocketType mix_vector_non_uniform_operands[]{
        SocketType::vector, SocketType::vector, SocketType::vector};
    const auto mix_vector_uniform =
        make_program(1103u, ValueOperation::mix_vector, SocketType::vector,
                     mix_vector_uniform_operands, {0u, 0u});
    const auto mix_vector_uniform_clamped =
        make_program(1104u, ValueOperation::mix_vector, SocketType::vector,
                     mix_vector_uniform_operands, {0u, 1u});
    const auto mix_vector_non_uniform =
        make_program(1105u, ValueOperation::mix_vector, SocketType::vector,
                     mix_vector_non_uniform_operands, {1u, 0u});
    const auto mix_vector_uniform_image = lower_surface_value_program(
        mix_vector_uniform, make_plan(mix_vector_uniform));
    const auto mix_vector_uniform_clamped_image = lower_surface_value_program(
        mix_vector_uniform_clamped, make_plan(mix_vector_uniform_clamped));
    const auto mix_vector_non_uniform_image = lower_surface_value_program(
        mix_vector_non_uniform, make_plan(mix_vector_non_uniform));
    require(
        mix_vector_uniform_image.valid &&
            mix_vector_uniform_clamped_image.valid &&
            mix_vector_non_uniform_image.valid &&
            surface_value_handler_key(
                mix_vector_uniform_image.instructions.front()) ==
                surface_value_handler_key(
                    mix_vector_uniform_clamped_image.instructions.front()) &&
            surface_value_handler_key(
                mix_vector_uniform_image.instructions.front()) !=
                surface_value_handler_key(
                    mix_vector_non_uniform_image.instructions.front()),
        "typed handler projection did not isolate scalar/vector Mix factors");

    const auto uniform_plan = make_plan(mix_vector_uniform);
    const auto uniform_clamped_plan = make_plan(mix_vector_uniform_clamped);
    const auto non_uniform_plan = make_plan(mix_vector_non_uniform);
    const std::array mix_vector_inputs{
        SurfaceValueExecutionInput{.program = &mix_vector_uniform,
                                   .storage = &uniform_plan},
        SurfaceValueExecutionInput{.program = &mix_vector_uniform_clamped,
                                   .storage = &uniform_clamped_plan},
        SurfaceValueExecutionInput{.program = &mix_vector_non_uniform,
                                   .storage = &non_uniform_plan}};
    const auto mix_vector_scene =
        build_surface_value_executable_scene(mix_vector_inputs);
    require(
        mix_vector_scene.valid && mix_vector_scene.variants.size() == 2u &&
            mix_vector_scene.instruction_variants[0u] ==
                mix_vector_scene.instruction_variants[1u] &&
            mix_vector_scene.instruction_variants[0u] !=
                mix_vector_scene.instruction_variants[2u],
        "Mix Vector data modes multiplied a handler or merged distinct typed "
        "factor ABIs");

    constexpr SocketType nishita_operands[]{
        SocketType::floating, SocketType::floating, SocketType::floating,
        SocketType::floating, SocketType::floating, SocketType::floating,
        SocketType::floating, SocketType::floating, SocketType::vector};
    const auto nishita_zero =
        make_program(1106u, ValueOperation::nishita_sky, SocketType::color,
                     nishita_operands, {0u, 0u});
    const auto nishita_large = make_program(
        1107u, ValueOperation::nishita_sky, SocketType::color, nishita_operands,
        {std::numeric_limits<std::uint32_t>::max(), 0u});
    const auto nishita_zero_plan = make_plan(nishita_zero);
    const auto nishita_large_plan = make_plan(nishita_large);
    const std::array nishita_inputs{
        SurfaceValueExecutionInput{.program = &nishita_zero,
                                   .storage = &nishita_zero_plan},
        SurfaceValueExecutionInput{.program = &nishita_large,
                                   .storage = &nishita_large_plan}};
    const auto nishita_scene =
        build_surface_value_executable_scene(nishita_inputs);
    require(
        nishita_scene.valid && nishita_scene.variants.size() == 1u &&
            nishita_scene.instruction_variants ==
                std::vector<std::uint32_t>{0u, 0u} &&
            nishita_scene.variants.front().instruction.static_u0 == 0u &&
            nishita_scene.variants.front().svm_immediates ==
                std::vector<std::uint16_t>{0u} &&
            nishita_scene.values.metadata.size() == 2u &&
            nishita_scene.values.metadata[0u].static_u0 == 0u &&
            nishita_scene.values.metadata[1u].static_u0 ==
                std::numeric_limits<std::uint32_t>::max(),
        "Nishita indices multiplied handlers or were truncated into the hot "
        "immediate");

    constexpr SocketType clamp_operands[]{
        SocketType::floating, SocketType::floating, SocketType::floating};
    const auto clamp_minmax = make_program(
        1110u,
        ValueOperation::clamp_range,
        SocketType::floating,
        clamp_operands,
        {0u, 0u});
    const auto clamp_range = make_program(
        1111u,
        ValueOperation::clamp_range,
        SocketType::floating,
        clamp_operands,
        {1u, 0u});
    const auto clamp_minmax_plan = make_plan(clamp_minmax);
    const auto clamp_range_plan = make_plan(clamp_range);
    const std::array clamp_inputs{
        SurfaceValueExecutionInput{.program = &clamp_minmax,
                                   .storage = &clamp_minmax_plan},
        SurfaceValueExecutionInput{.program = &clamp_range,
                                   .storage = &clamp_range_plan}};
    const auto clamp_scene = build_surface_value_executable_scene(clamp_inputs);
    require(clamp_scene.valid && clamp_scene.variants.size() == 1u &&
                clamp_scene.instruction_variants ==
                    std::vector<std::uint32_t>{0u, 0u} &&
                clamp_scene.variants.front().instruction.static_u0 == 0u &&
                clamp_scene.variants.front().instruction.static_u1 == 0u &&
                clamp_scene.variants.front().svm_immediates ==
                    std::vector<std::uint16_t>{0u, 1u} &&
                surface_value_handler_key(
                    clamp_scene.values.instructions[0u]) ==
                    surface_value_handler_key(
                        clamp_scene.values.instructions[1u]),
            "typed Clamp records did not form one exact handler quotient");

    const auto scalar_key = make_surface_value_handler_key(
        ValueOperation::passthrough, SurfaceValueBank::scalar, 0u);
    const auto vector_key = make_surface_value_handler_key(
        ValueOperation::passthrough, SurfaceValueBank::vector, 0u);
    require(scalar_key != vector_key,
            "primary handler projection omitted the typed result bank");
}

} // namespace

int main() {
    try {
        test_typed_record_quotients();
        test_invalid_records_are_rejected();
        test_serialized_record_coherence();
        test_primary_handler_projection();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Surface SVM record test failure: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
