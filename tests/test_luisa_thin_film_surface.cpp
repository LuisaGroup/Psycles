#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_surface_test_support.h"
#include "path_tracer_bsdf_tables.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::adapter;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::Vec3f;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr std::uint32_t result_stride = 6u;

namespace result_row {
constexpr std::uint32_t identity = 0u;
constexpr std::uint32_t optical = 1u;
constexpr std::uint32_t evaluation = 2u;
constexpr std::uint32_t sample = 3u;
constexpr std::uint32_t sample_direction = 4u;
constexpr std::uint32_t second_closure = 5u;
} // namespace result_row

namespace principled_case {
constexpr std::uint32_t regular_metallic = 0u;
constexpr std::uint32_t regular_dielectric = 1u;
constexpr std::uint32_t delta_metallic = 2u;
constexpr std::uint32_t delta_dielectric = 3u;
constexpr std::uint32_t thin_wall = 4u;
constexpr std::uint32_t count = 5u;
} // namespace principled_case

namespace glass_case {
constexpr std::uint32_t regular_front = 0u;
constexpr std::uint32_t delta_front = 1u;
constexpr std::uint32_t delta_back = 2u;
constexpr std::uint32_t count = 3u;
} // namespace glass_case

class TableParameterShaderServices final : public ParameterShaderServices {

private:
  const BufferFloat &_table;

public:
  TableParameterShaderServices(const BufferFloat4 &parameters,
                               const BufferFloat &table) noexcept
      : ParameterShaderServices{parameters}, _table{table} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }
};

struct PrincipledParameters {
  Vec3f base_color;
  float metallic;
  float roughness;
  float transmission_weight;
  Vec3f specular_tint;
  bool thin_wall;
};

[[nodiscard]] ShaderGraph
make_principled_graph(const PrincipledParameters &parameters) {
  CyclesNormalizedShaderGraph source;
  using Node = typename decltype(source.nodes)::value_type;
  source.nodes.emplace_back(Node{
      .id = 11u,
      .type = "principled_bsdf",
      .variant = {},
      .label = "Cycles 5.2 Thin Film Principled",
      .inputs = {{.socket = "Base Color",
                  .source = std::nullopt,
                  .value = SocketValue::color(parameters.base_color)},
                 {.socket = "Metallic",
                  .source = std::nullopt,
                  .value = SocketValue::floating(parameters.metallic)},
                 {.socket = "Roughness",
                  .source = std::nullopt,
                  .value = SocketValue::floating(parameters.roughness)},
                 {.socket = "Transmission Weight",
                  .source = std::nullopt,
                  .value =
                      SocketValue::floating(parameters.transmission_weight)},
                 {.socket = "IOR",
                  .source = std::nullopt,
                  .value = SocketValue::floating(1.5f)},
                 {.socket = "Specular IOR Level",
                  .source = std::nullopt,
                  .value = SocketValue::floating(0.5f)},
                 {.socket = "Specular Tint",
                  .source = std::nullopt,
                  .value = SocketValue::color(parameters.specular_tint)},
                 {.socket = "Thin Wall",
                  .source = std::nullopt,
                  .value = SocketValue::boolean(parameters.thin_wall)},
                 {.socket = "Thin Film Thickness",
                  .source = std::nullopt,
                  .value = SocketValue::floating(250.0f)},
                 {.socket = "Thin Film IOR",
                  .source = std::nullopt,
                  .value = SocketValue::floating(1.33f)}},
      .properties = {{"distribution", SocketValue::string("GGX")}}});
  source.set_root(ShaderDomain::surface,
                  CyclesOutputRef{.node = 11u, .socket = "BSDF"});
  auto adapted =
      adapt_cycles_shader_graph(source, make_core_cycles_node_mappings());
  if (!adapted.ok()) {
    throw std::runtime_error{
        "failed to adapt Cycles 5.2 Thin Film Principled graph"};
  }
  return std::move(*adapted.graph);
}

[[nodiscard]] ShaderGraph make_glass_graph(float roughness) {
  CyclesNormalizedShaderGraph source;
  using Node = typename decltype(source.nodes)::value_type;
  source.nodes.emplace_back(
      Node{.id = 17u,
           .type = "glass_bsdf",
           .variant = {},
           .label = "Cycles 5.2 Thin Film Glass",
           .inputs = {{.socket = "Color",
                       .source = std::nullopt,
                       .value = SocketValue::color({1.0f, 1.0f, 1.0f})},
                      {.socket = "Roughness",
                       .source = std::nullopt,
                       .value = SocketValue::floating(roughness)},
                      {.socket = "IOR",
                       .source = std::nullopt,
                       .value = SocketValue::floating(1.5f)},
                      {.socket = "Thin Film Thickness",
                       .source = std::nullopt,
                       .value = SocketValue::floating(250.0f)},
                      {.socket = "Thin Film IOR",
                       .source = std::nullopt,
                       .value = SocketValue::floating(1.33f)}},
           .properties = {{"distribution", SocketValue::string("GGX")}}});
  source.set_root(ShaderDomain::surface,
                  CyclesOutputRef{.node = 17u, .socket = "BSDF"});
  auto adapted =
      adapt_cycles_shader_graph(source, make_core_cycles_node_mappings());
  if (!adapted.ok()) {
    throw std::runtime_error{
        "failed to adapt Cycles 5.2 Thin Film Glass graph"};
  }
  return std::move(*adapted.graph);
}

struct CompiledSurface {
  std::shared_ptr<const SurfaceProgram> program;
  std::vector<luisa::float4> parameters;
};

[[nodiscard]] CompiledSurface compile_graph(ShaderCompiler &compiler,
                                            ShaderGraph graph) {
  const auto shader = compiler.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"failed to compile Thin Film shader graph"};
  }
  const auto surface = compile_surface_program(*shader.program);
  if (!surface.ok()) {
    throw std::runtime_error{"failed to lower Thin Film surface program"};
  }
  return {.program = surface.program,
          .parameters = parameter_data(*surface.program)};
}

void require_same_topology(const SurfaceProgram &reference,
                           const SurfaceProgram &candidate) {
  const auto &lhs = reference.parameters();
  const auto &rhs = candidate.parameters();
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error{"Thin Film topology changed its parameter extent"};
  }
  for (auto index = std::size_t{0u}; index < lhs.size(); ++index) {
    if (lhs[index].id != rhs[index].id || lhs[index].type != rhs[index].type) {
      throw std::runtime_error{
          "Thin Film topology changed its typed parameter ABI"};
    }
  }
}

[[nodiscard]] SurfaceQuery query() noexcept {
  return {.lobe_mask = ~std::uint32_t{0u},
          .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
          .glossy_filter_roughness = 0.0f,
          .reflective_caustics = true,
          .refractive_caustics = true};
}

[[nodiscard]] constexpr SurfaceClosureKindMask
kind_mask(SurfaceClosureKind kind) noexcept {
  return surface_closure_kind_bit(kind);
}

[[nodiscard]] constexpr SurfaceClosureLobeMask
lobe_mask(SurfaceClosureLobe lobe) noexcept {
  return surface_closure_lobe_bit(lobe);
}

constexpr auto principled_reachability = SurfaceClosureReachability{
    .kinds = kind_mask(SurfaceClosureKind::diffuse) |
             kind_mask(SurfaceClosureKind::principled) |
             kind_mask(SurfaceClosureKind::glossy) |
             kind_mask(SurfaceClosureKind::thin_glass_transmission) |
             kind_mask(SurfaceClosureKind::transparent),
    .principled_lobes = lobe_mask(SurfaceClosureLobe::metallic) |
                        lobe_mask(SurfaceClosureLobe::dielectric),
    .thin_film_principled_lobes = lobe_mask(SurfaceClosureLobe::metallic) |
                                  lobe_mask(SurfaceClosureLobe::dielectric)};

constexpr auto glass_reachability = SurfaceClosureReachability{
    .kinds = kind_mask(SurfaceClosureKind::glass),
    .thin_film_kinds = kind_mask(SurfaceClosureKind::glass)};

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z) && std::isfinite(value.w);
}

void print_row(std::string_view label, luisa::float4 value) {
  std::cerr << label << "={" << value.x << ", " << value.y << ", " << value.z
            << ", " << value.w << "}";
}

[[nodiscard]] bool expect_row(std::string_view backend, std::string_view label,
                              luisa::float4 actual, luisa::float4 expected,
                              float tolerance = 3.0e-5f) {
  if (finite(actual) && approximately_equal(actual, expected, tolerance)) {
    return true;
  }
  std::cerr << "Thin Film surface mismatch on " << backend << ", ";
  print_row(label, actual);
  std::cerr << ", ";
  print_row("expected", expected);
  std::cerr << '\n';
  return false;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  ShaderCompiler compiler{make_core_node_registry()};

  constexpr std::array<PrincipledParameters, principled_case::count>
      principled_inputs{
          PrincipledParameters{{0.72f, 0.31f, 0.08f},
                               1.0f,
                               0.4f,
                               0.0f,
                               {0.3f, 0.7f, 0.9f},
                               false},
          PrincipledParameters{
              {0.0f, 0.0f, 0.0f}, 0.0f, 0.4f, 0.0f, {2.0f, 1.0f, 0.5f}, false},
          PrincipledParameters{{0.72f, 0.31f, 0.08f},
                               1.0f,
                               0.0f,
                               0.0f,
                               {0.3f, 0.7f, 0.9f},
                               false},
          PrincipledParameters{
              {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 0.0f, {2.0f, 1.0f, 0.5f}, false},
          PrincipledParameters{{0.64f, 0.49f, 0.36f},
                               0.0f,
                               0.0f,
                               1.0f,
                               {1.0f, 1.0f, 1.0f},
                               true}};
  std::array<CompiledSurface, principled_inputs.size()> principled_programs;
  for (auto index = std::size_t{0u}; index < principled_inputs.size();
       ++index) {
    principled_programs[index] = compile_graph(
        compiler, make_principled_graph(principled_inputs[index]));
    if (index != 0u) {
      require_same_topology(*principled_programs.front().program,
                            *principled_programs[index].program);
    }
  }
  const auto principled_stride =
      static_cast<std::uint32_t>(principled_programs.front().parameters.size());
  std::vector<luisa::float4> principled_parameters;
  principled_parameters.reserve(principled_stride * principled_programs.size());
  for (const auto &surface : principled_programs) {
    principled_parameters.insert(principled_parameters.end(),
                                 surface.parameters.begin(),
                                 surface.parameters.end());
  }
  const auto principled_plan = merged_surface_closure_plan(
      *principled_programs.front().program,
      std::span<const luisa::float4>{principled_parameters});
  SurfaceDispatch principled_surfaces;
  const auto principled_tag = principled_surfaces.create<GraphSurface>(
      principled_programs.front().program, principled_plan);

  std::array<CompiledSurface, 2u> glass_programs{
      compile_graph(compiler, make_glass_graph(0.4f)),
      compile_graph(compiler, make_glass_graph(0.0f))};
  require_same_topology(*glass_programs[0u].program,
                        *glass_programs[1u].program);
  const auto glass_stride =
      static_cast<std::uint32_t>(glass_programs.front().parameters.size());
  std::vector<luisa::float4> glass_parameters;
  glass_parameters.reserve(glass_stride * glass_programs.size());
  for (const auto &surface : glass_programs) {
    glass_parameters.insert(glass_parameters.end(), surface.parameters.begin(),
                            surface.parameters.end());
  }
  const auto glass_plan = merged_surface_closure_plan(
      *glass_programs.front().program,
      std::span<const luisa::float4>{glass_parameters});
  SurfaceDispatch glass_surfaces;
  const auto glass_tag = glass_surfaces.create<GraphSurface>(
      glass_programs.front().program, glass_plan);

  Kernel1D test_principled = [&](BufferFloat table, BufferFloat4 parameters,
                                 BufferFloat4 output) noexcept {
    const auto case_index = dispatch_x();
    auto point = make_surface_point();
    point.parameter_block = case_index * principled_stride;
    point.incoming = select(normalize(make_float3(0.6f, 0.0f, 0.8f)),
                            make_float3(0.0f, 0.0f, 1.0f),
                            case_index == principled_case::thin_wall);
    TableParameterShaderServices services{parameters, table};
    SurfaceClosureSet closures{3u, SurfaceClosureStorageProfile::physical};
    const auto collection = principled_surfaces.collect_closures(
        UInt{principled_tag}, services, point, true, true, closures);
    const auto first = closures.entry(0u);
    const auto second = closures.entry(1u);
    const SurfaceClosureEvaluator evaluator{
        point, closures, collection.shading_normal, principled_reachability};
    const auto outgoing = normalize(make_float3(-0.2f, 0.3f, 0.9327379f));
    const auto evaluation = evaluator.evaluate(services, outgoing, query());
    const auto sample =
        evaluator.sample(services, 0.0f, make_float2(0.37f, 0.61f), query());
    const auto base = case_index * result_stride;
    output.write(base + result_row::identity,
                 make_float4(cast<float>(closures.count()),
                             cast<float>(first.kind), cast<float>(first.lobe),
                             select(0.0f, 1.0f, first.setup_valid)));
    output.write(base + result_row::optical,
                 make_float4(first.thin_film_thickness, first.thin_film_ior,
                             first.ior, first.weight.x));
    output.write(base + result_row::evaluation,
                 make_float4(evaluation.f, evaluation.pdf));
    output.write(base + result_row::sample,
                 make_float4(sample.evaluation.f * 1.0e-6f,
                             sample.evaluation.pdf * 1.0e-6f));
    output.write(base + result_row::sample_direction,
                 make_float4(sample.wi, select(0.0f, 1.0f, sample.valid)));
    output.write(base + result_row::second_closure,
                 make_float4(second.weight, cast<float>(second.kind)));
  };

  Kernel1D test_glass = [&](BufferFloat table, BufferFloat4 parameters,
                            BufferFloat4 output) noexcept {
    const auto case_index = dispatch_x();
    auto point = make_surface_point();
    point.parameter_block =
        min(case_index, glass_case::delta_front) * glass_stride;
    point.incoming = select(normalize(make_float3(0.6f, 0.0f, 0.8f)),
                            make_float3(0.0f, 0.0f, 1.0f),
                            case_index != glass_case::regular_front);
    point.back_facing = case_index == glass_case::delta_back;
    TableParameterShaderServices services{parameters, table};
    SurfaceClosureSet closures{1u, SurfaceClosureStorageProfile::physical};
    const auto collection = glass_surfaces.collect_closures(
        UInt{glass_tag}, services, point, true, true, closures);
    const auto first = closures.entry(0u);
    const SurfaceClosureEvaluator evaluator{
        point, closures, collection.shading_normal, glass_reachability};
    const auto outgoing = normalize(make_float3(-0.2f, 0.3f, 0.9327379f));
    const auto evaluation = evaluator.evaluate(services, outgoing, query());
    const auto sample =
        evaluator.sample(services, 0.0f, make_float2(0.37f, 0.61f), query());
    const auto base = case_index * result_stride;
    output.write(base + result_row::identity,
                 make_float4(cast<float>(closures.count()),
                             cast<float>(first.kind), cast<float>(first.lobe),
                             select(0.0f, 1.0f, first.setup_valid)));
    output.write(base + result_row::optical,
                 make_float4(first.thin_film_thickness, first.thin_film_ior,
                             first.ior, first.weight.x));
    output.write(base + result_row::evaluation,
                 make_float4(evaluation.f, evaluation.pdf));
    output.write(base + result_row::sample,
                 make_float4(sample.evaluation.f * 1.0e-6f,
                             sample.evaluation.pdf * 1.0e-6f));
    output.write(base + result_row::sample_direction,
                 make_float4(sample.wi, select(0.0f, 1.0f, sample.valid)));
    output.write(base + result_row::second_closure, make_float4(0.0f));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values = make_cycles_bsdf_table_values(ShaderColorSpace{});
  auto table_buffer = device.create_buffer<float>(table_values.size());
  auto principled_parameter_buffer =
      device.create_buffer<luisa::float4>(principled_parameters.size());
  auto glass_parameter_buffer =
      device.create_buffer<luisa::float4>(glass_parameters.size());
  std::array<luisa::float4, principled_inputs.size() * result_stride>
      principled_results{};
  std::array<luisa::float4, glass_case::count * result_stride> glass_results{};
  auto principled_output =
      device.create_buffer<luisa::float4>(principled_results.size());
  auto glass_output = device.create_buffer<luisa::float4>(glass_results.size());
  auto principled_shader = compile_named_kernel(
      device, "thin_film_surface_principled", test_principled);
  auto glass_shader =
      compile_named_kernel(device, "thin_film_surface_glass", test_glass);
  stream << table_buffer.copy_from(luisa::span{table_values})
         << principled_parameter_buffer.copy_from(
                luisa::span{principled_parameters})
         << glass_parameter_buffer.copy_from(luisa::span{glass_parameters})
         << principled_shader(table_buffer, principled_parameter_buffer,
                              principled_output)
                .dispatch(principled_inputs.size())
         << glass_shader(table_buffer, glass_parameter_buffer, glass_output)
                .dispatch(glass_case::count)
         << principled_output.copy_to(luisa::span{principled_results})
         << glass_output.copy_to(luisa::span{glass_results}) << synchronize();

  const auto row = [](auto &values, std::size_t case_index,
                      std::uint32_t offset) noexcept {
    return values[case_index * result_stride + offset];
  };
  auto ok = true;
  constexpr auto principled_kind =
      static_cast<float>(SurfaceClosureKind::principled);
  constexpr auto metallic_lobe =
      static_cast<float>(SurfaceClosureLobe::metallic);
  constexpr auto dielectric_lobe =
      static_cast<float>(SurfaceClosureLobe::dielectric);
  for (auto case_index = std::size_t{principled_case::regular_metallic};
       case_index <= principled_case::delta_dielectric; ++case_index) {
    const auto lobe = (case_index % 2u) == 0u ? metallic_lobe : dielectric_lobe;
    ok &= expect_row(backend, "Principled identity",
                     row(principled_results, case_index, result_row::identity),
                     {1.0f, principled_kind, lobe, 1.0f}, 0.0f);
    ok &= expect_row(
        backend, "Principled optical payload",
        row(principled_results, case_index, result_row::optical),
        {250.0f, 1.33f, (case_index % 2u) == 0u ? 1.0f : 1.5f, 1.0f});
  }

  // Emitted by Cycles 5.2.1 source revision
  // 9e2066aef7ef7e20c142ad7bd3303138a4304c93 using the real GGX evaluator.
  ok &= expect_row(backend, "Principled metallic regular evaluation",
                   row(principled_results, principled_case::regular_metallic,
                       result_row::evaluation),
                   {0.182558402f, 0.0572737530f, 0.00503868936f, 0.251224756f});
  ok &=
      expect_row(backend, "Principled dielectric regular evaluation",
                 row(principled_results, principled_case::regular_dielectric,
                     result_row::evaluation),
                 {0.0214305520f, 0.00838483870f, 0.00111866498f, 0.251224756f});
  ok &= expect_row(backend, "Principled metallic delta sample",
                   row(principled_results, principled_case::delta_metallic,
                       result_row::sample),
                   {0.721437275f, 0.273250937f, 0.00750176283f, 1.0f});
  ok &= expect_row(backend, "Principled dielectric delta sample",
                   row(principled_results, principled_case::delta_dielectric,
                       result_row::sample),
                   {0.0879736319f, 0.0415383764f, 0.00853416510f, 1.0f});
  for (const auto case_index :
       {principled_case::delta_metallic, principled_case::delta_dielectric}) {
    ok &= expect_row(
        backend, "Principled delta sample direction",
        row(principled_results, case_index, result_row::sample_direction),
        {-0.6f, 0.0f, 0.8f, 1.0f});
  }

  constexpr auto glossy_kind = static_cast<float>(SurfaceClosureKind::glossy);
  constexpr auto thin_transmission_kind =
      static_cast<float>(SurfaceClosureKind::thin_glass_transmission);
  constexpr auto transmission_lobe =
      static_cast<float>(SurfaceClosureLobe::transmission);
  ok &= expect_row(
      backend, "Thin Wall identity",
      row(principled_results, principled_case::thin_wall, result_row::identity),
      {2.0f, glossy_kind, transmission_lobe, 1.0f}, 0.0f);
  ok &= expect_row(
      backend, "Thin Wall reflected closure",
      row(principled_results, principled_case::thin_wall, result_row::optical),
      {0.0f, 0.0f, 1.0f, 0.0483776778f});
  ok &= expect_row(
      backend, "Thin Wall transmitted closure",
      row(principled_results, principled_case::thin_wall,
          result_row::second_closure),
      {0.602592528f, 0.462815374f, 0.343396068f, thin_transmission_kind});
  ok &= expect_row(backend, "Thin Wall delta sample direction",
                   row(principled_results, principled_case::thin_wall,
                       result_row::sample_direction),
                   {0.0f, 0.0f, 1.0f, 1.0f});

  constexpr auto glass_kind = static_cast<float>(SurfaceClosureKind::glass);
  for (auto case_index = std::size_t{glass_case::regular_front};
       case_index < glass_case::count; ++case_index) {
    ok &= expect_row(
        backend, "Glass identity",
        row(glass_results, case_index, result_row::identity),
        {1.0f, glass_kind, static_cast<float>(SurfaceClosureLobe::none), 1.0f},
        0.0f);
  }
  ok &= expect_row(
      backend, "Glass regular optical payload",
      row(glass_results, glass_case::regular_front, result_row::optical),
      {250.0f, 1.33f, 1.5f, 1.0f});
  ok &= expect_row(
      backend, "Glass regular evaluation",
      row(glass_results, glass_case::regular_front, result_row::evaluation),
      {0.0107306689f, 0.00838483963f, 0.00223733019f, 0.00712438952f});
  ok &= expect_row(
      backend, "Glass front delta sample",
      row(glass_results, glass_case::delta_front, result_row::sample),
      {0.0415148064f, 0.0259462222f, 0.00593409967f, 0.0244650429f});
  ok &= expect_row(
      backend, "Glass back optical payload",
      row(glass_results, glass_case::delta_back, result_row::optical),
      {250.0f, 1.33f / 1.5f, 1.0f / 1.5f, 1.0f});
  ok &=
      expect_row(backend, "Glass back delta sample",
                 row(glass_results, glass_case::delta_back, result_row::sample),
                 {0.0177951697f, 0.0305364169f, 0.0406331941f, 0.0296549276f});
  for (const auto case_index :
       {glass_case::delta_front, glass_case::delta_back}) {
    ok &=
        expect_row(backend, "Glass delta sample direction",
                   row(glass_results, case_index, result_row::sample_direction),
                   {0.0f, 0.0f, 1.0f, 1.0f});
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
