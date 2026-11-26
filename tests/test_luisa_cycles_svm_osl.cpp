#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/contract/shader_graph.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace compiler = psycles::compiler;
namespace compiler_svm = psycles::compiler::cycles_svm;
namespace contract = psycles::contract;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

[[nodiscard]] bool near_f(float a, float b, float eps = 1.0e-4f) noexcept {
  return std::abs(a - b) <= eps;
}

[[nodiscard]] bool near_f3(luisa::float3 a, luisa::float3 b,
                           float eps = 1.0e-4f) noexcept {
  return near_f(a.x, b.x, eps) && near_f(a.y, b.y, eps) &&
         near_f(a.z, b.z, eps);
}

[[nodiscard]] compiler_svm::CompiledShaderTable compile_graph(
    const contract::ShaderGraph &graph, bool enable_derivatives = false) {
  const compiler::ShaderCompiler frontend{
      compiler::make_core_node_registry()};
  const auto compiled = frontend.compile(graph);
  if (!compiled.ok()) {
    std::string msg = "ShaderGraph validation failed:";
    for (const auto &d : compiled.diagnostics) {
      msg += " ";
      msg += d.message;
    }
    throw std::runtime_error{std::move(msg)};
  }
  const std::array units{
      compiler_svm::ShaderTableCompileUnit{
          .shader_index = 0u,
          .shader = compiled.program.get(),
          .context =
              {.background = false,
               .displacement_method = contract::DisplacementMethod::bump,
               .enable_texture_derivatives = enable_derivatives},
          .kernel = {.name = "osl_unit_shader"},
      },
  };
  auto result = compiler_svm::compile_shader_table(units);
  require(result.table.valid, result.table.diagnostic);
  return result;
}

contract::NodeId add_osl_node(contract::ShaderGraph &graph,
                              std::string_view name, std::string_view oso,
                              std::string_view input_types,
                              std::string_view output_types,
                              bool has_closure_output = false) {
  const auto node =
      graph.add_node(compiler::node_type::osl_script, std::string{name});
  require(graph.set_property(node, "ScriptName",
                             contract::SocketValue::string(std::string{name})) &&
              graph.set_property(
                  node, "Bytecode",
                  contract::SocketValue::string(std::string{oso})) &&
              graph.set_property(
                  node, "InputTypes",
                  contract::SocketValue::string(std::string{input_types})) &&
              graph.set_property(
                  node, "OutputTypes",
                  contract::SocketValue::string(std::string{output_types})) &&
              graph.set_property(
                  node, "HasClosureOutput",
                  contract::SocketValue::boolean(has_closure_output)),
          "failed to configure osl.script node");
  return node;
}

[[nodiscard]] device_svm::ShaderData make_test_shader_data(
    Expr<luisa::float3> P, Expr<luisa::float3> N, Expr<luisa::float3> wi,
    Expr<std::uint32_t> shader_flag, Expr<std::uint32_t> object_id,
    Expr<float> u, Expr<float> v, Expr<float> du_dx, Expr<float> du_dy,
    Expr<float> dv_dx, Expr<float> dv_dy, Expr<luisa::float3> dPdu,
    Expr<luisa::float3> dPdv, device_svm::ClosurePool *pool) noexcept {
  const auto id = make_float4x4(1.0f);
  return {P,
          N,
          N,
          wi,
          device_svm::primitive_triangle,
          0u,
          shader_flag,
          0u,
          0u,
          u,
          v,
          object_id,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          du_dx,
          du_dy,
          dv_dx,
          dv_dy,
          dPdu,
          dPdv,
          id,
          id,
          0u,
          pool};
}

class BsdfTableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Expr<Buffer<float>> _table;

public:
  explicit BsdfTableKernelGlobals(Expr<Buffer<float>> table) noexcept
      : _table{table} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table.read(index);
  }
};

class AttrKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Expr<Buffer<compiler_svm::AttributeMap>> _maps;
  Expr<Buffer<luisa::float2>> _float2s;
  Expr<Buffer<compiler_svm::packed_float3>> _float3s;
  Expr<Buffer<luisa::uint3>> _tris;

public:
  AttrKernelGlobals(Expr<Buffer<compiler_svm::AttributeMap>> maps,
                    Expr<Buffer<luisa::float2>> float2s,
                    Expr<Buffer<compiler_svm::packed_float3>> float3s,
                    Expr<Buffer<luisa::uint3>> tris) noexcept
      : _maps{maps}, _float2s{float2s}, _float3s{float3s}, _tris{tris} {}

  [[nodiscard]] device_svm::SurfaceAttributeSummary
  surface_attribute_summary() const noexcept override {
    return {.element_mask =
                static_cast<std::uint32_t>(compiler_svm::ATTR_ELEMENT_CORNER |
                                           compiler_svm::ATTR_ELEMENT_VERTEX),
            .type_mask = (1u << compiler_svm::NODE_ATTR_FLOAT2) |
                         (1u << compiler_svm::NODE_ATTR_FLOAT3),
            .has_curves = false,
            .has_points = false,
            .has_undisplaced_normal = false,
            .has_undisplaced_position = false,
            .has_float4 = false};
  }

  [[nodiscard]] UInt
  object_attribute_map_offset(Expr<std::uint32_t>) const noexcept override {
    return 0u;
  }

  [[nodiscard]] Var<compiler_svm::AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    return _maps.read(offset);
  }

  [[nodiscard]] Float2
  attribute_float2(Expr<std::int32_t> offset) const noexcept override {
    return _float2s.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] Var<compiler_svm::packed_float3>
  attribute_float3(Expr<std::int32_t> offset) const noexcept override {
    return _float3s.read(offset.cast<std::uint32_t>());
  }

  [[nodiscard]] UInt3
  triangle_vertex_indices(Expr<std::uint32_t> prim) const noexcept override {
    return _tris.read(prim);
  }
};

void test_dynamic_inputs_init_and_isconnected(
    Device &device, Stream &stream, device_svm::SvmBackendStrategy strategy) {
  static constexpr std::string_view oso = R"oso(
OpenShadingLanguage 1.00
shader test_dyn_init
param	float	in_f	0.0
param	color	in_c	0 0 0
param	float	unlinked_f	1.0
param	point	Pos	0 0 0	%initexpr
oparam	color	out_c	0 0 0
global	point	P
local	int	c_in
local	int	c_un
local	int	c_pos
local	float	r_v
local	float	g_v
local	float	b_v
local	float	tmp0
local	float	tmp1
local	float	pos0
local	float	pos1
const	float	$c10	10.0
const	float	$c100	100.0
const	float	$c1000	1000.0
const	int	$i0	0
const	int	$i1	1
const	int	$i2	2
code Pos
	assign		Pos P
code ___main___
	isconnected	c_in in_f
	isconnected	c_un unlinked_f
	isconnected	c_pos Pos
	mul		r_v in_f unlinked_f
	mul		tmp0 c_in $c10
	add		r_v r_v tmp0
	mul		tmp0 c_un $c100
	add		r_v r_v tmp0
	mul		tmp0 c_pos $c1000
	add		r_v r_v tmp0
	compref		tmp1 in_c $i1
	compref		pos0 Pos $i0
	compref		pos1 Pos $i1
	add		g_v tmp1 pos0
	add		g_v g_v pos1
	compref		b_v Pos $i2
	color		out_c r_v g_v b_v
end
)oso";

  contract::ShaderGraph graph;
  const auto c_f =
      graph.add_node(compiler::node_type::constant_float, "InF");
  const auto c_c =
      graph.add_node(compiler::node_type::constant_color, "InC");
  const auto osl = add_osl_node(graph, "dyn_init", oso,
                                "in_f=float;in_c=color;unlinked_f=float;Pos=point",
                                "out_c=color", false);
  const auto emit = graph.add_node(compiler::node_type::emission, "Emit");
  require(
      graph.set_input(c_f, "Value", contract::SocketValue::floating(3.0f)) &&
          graph.set_input(c_c, "Color",
                          contract::SocketValue::color({0.2f, 0.7f, 0.4f})) &&
          graph.set_input(osl, "unlinked_f",
                          contract::SocketValue::floating(2.5f)) &&
          graph.connect({c_f, "Value"}, osl, "in_f") &&
          graph.connect({c_c, "Color"}, osl, "in_c") &&
          graph.connect({osl, "out_c"}, emit, "Color") &&
          graph.set_input(emit, "Strength",
                          contract::SocketValue::floating(1.0f)),
      "failed to wire dyn_init graph");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{emit, "Closure"});

  const auto compiled = compile_graph(graph);
  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto out_buf = device.create_buffer<luisa::float4>(1u);
  stream << words_buf.copy_from(compiled.table.words.data());

  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, id, id};
        device_svm::ClosurePool pool{4u};
        auto sd = make_test_shader_data(
            make_float3(1.25f, -0.5f, 2.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0u, device_svm::object_none, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), &pool);
        const device_svm::PathState path{device_svm::path_ray_visibility_camera,
                                         0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);
        out.write(0u, make_float4(sd.closure_emission_background,
                                  pool.count().cast<float>()));
      });

  luisa::float4 host_out{};
  stream << kernel(words_buf, out_buf).dispatch(1u)
         << out_buf.copy_to(&host_out) << synchronize();
  require(near_f3(host_out.xyz(), luisa::make_float3(17.5f, 1.45f, 2.0f)),
          "dynamic inputs / initexpr / isconnected mismatch");
}

void test_closures_and_mix_weight(Device &device, Stream &stream,
                                  device_svm::SvmBackendStrategy strategy,
                                  bool mixed) {
  static constexpr std::string_view oso = R"oso(
OpenShadingLanguage 1.00
surface test_closures
param	float	alpha	0.1
oparam	closure color	BSDF
global	normal	N
local	closure color	c_diff
local	closure color	c_spec
local	closure color	c_emit
local	closure color	c_trans
const	string	$s_diff	"diffuse"
const	string	$s_micro	"microfacet"
const	string	$s_ggx	"ggx"
const	string	$s_emit	"emission"
const	string	$s_trans	"transparent"
const	color	$w_diff	0.2 0.3 0.4
const	color	$w_spec	0.5 0.5 0.5
const	color	$w_emit	0.8 0.4 0.2
const	color	$w_trans	0.1 0.2 0.3
const	float	$f_ior	1.5
const	int	$i0	0
code ___main___
	closure		c_diff $s_diff N
	mul		c_diff c_diff $w_diff
	closure		c_spec $s_micro $s_ggx N alpha $f_ior $i0
	mul		c_spec c_spec $w_spec
	closure		c_emit $s_emit
	mul		c_emit c_emit $w_emit
	closure		c_trans $s_trans
	mul		c_trans c_trans $w_trans
	add		BSDF c_diff c_spec
	add		BSDF BSDF c_emit
	add		BSDF BSDF c_trans
end
)oso";

  contract::ShaderGraph graph;
  const auto osl =
      add_osl_node(graph, "closure_node", oso, "alpha=float", "BSDF=closure",
                   true);
  require(graph.set_input(osl, "alpha", contract::SocketValue::floating(0.25f)),
          "failed to set OSL alpha");
  if (!mixed) {
    graph.set_root(contract::ShaderDomain::surface,
                   contract::OutputRef{osl, "BSDF"});
  } else {
    const auto emit2 =
        graph.add_node(compiler::node_type::emission, "SecondEmit");
    const auto mix = graph.add_node(compiler::node_type::mix_closure, "Mix");
    require(graph.set_input(emit2, "Color",
                            contract::SocketValue::color({4.0f, 8.0f, 12.0f})) &&
                graph.set_input(emit2, "Strength",
                                contract::SocketValue::floating(1.0f)) &&
                graph.set_input(mix, "Factor",
                                contract::SocketValue::floating(0.75f)) &&
                graph.connect({osl, "BSDF"}, mix, "A") &&
                graph.connect({emit2, "Closure"}, mix, "B"),
            "failed to wire Mix Shader closure graph");
    graph.set_root(contract::ShaderDomain::surface,
                   contract::OutputRef{mix, "Closure"});
  }

  const auto compiled = compile_graph(graph);
  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  const auto bsdf_host =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          contract::ShaderColorSpace{});
  auto table_buf = device.create_buffer<float>(bsdf_host.size());
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto f4_out = device.create_buffer<luisa::float4>(5u);
  auto u4_out = device.create_buffer<luisa::uint4>(2u);
  stream << table_buf.copy_from(bsdf_host.data())
         << words_buf.copy_from(compiled.table.words.data());

  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat table, BufferFloat4 out_f4,
          BufferUInt4 out_u4) noexcept {
        const BsdfTableKernelGlobals kg{table};
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, id, id};
        device_svm::ClosurePool pool{8u};
        auto sd = make_test_shader_data(
            make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0u, device_svm::object_none, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), &pool);
        const device_svm::PathState path{device_svm::path_ray_visibility_camera,
                                         0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);

        const auto c0 = pool.common(0u);
        const auto c1 = pool.common(1u);
        const auto c2 = pool.common(2u);
        const auto m1 = pool.microfacet(1u);
        out_f4.write(0u, make_float4(c0.weight, c0.sample_weight));
        out_f4.write(1u, make_float4(c1.weight, m1.param.alpha_x));
        out_f4.write(2u, make_float4(m1.param.ior, c1.sample_weight, 0.0f, 0.0f));
        out_f4.write(3u, make_float4(c2.weight, c2.sample_weight));
        out_f4.write(4u, make_float4(sd.closure_emission_background,
                                     sd.flag.cast<float>()));
        out_u4.write(0u, make_uint4(pool.count(), c0.type, c1.type, c2.type));
        out_u4.write(1u, make_uint4(m1.param.fresnel_type, sd.flag, 0u, 0u));
      });

  std::array<luisa::float4, 5u> hf4{};
  std::array<luisa::uint4, 2u> hu4{};
  stream << kernel(words_buf, table_buf, f4_out, u4_out).dispatch(1u)
         << f4_out.copy_to(hf4.data()) << u4_out.copy_to(hu4.data())
         << synchronize();

  const float scale = mixed ? 0.25f : 1.0f;
  require(hu4[0].x == 3u, "expected 3 closure entries in ClosurePool");
  require(hu4[0].y == closure::type_diffuse, "slot 0 must be diffuse");
  require(hu4[0].z == closure::type_microfacet_ggx,
          "slot 1 must be microfacet ggx");
  require(hu4[0].w == closure::type_transparent,
          "slot 2 must be transparent");
  require(hu4[1].x ==
              static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::dielectric),
          "microfacet must retain dielectric fresnel");

  require(near_f3(hf4[0].xyz(),
                  luisa::make_float3(0.2f, 0.3f, 0.4f) * scale),
          "diffuse closure weight mismatch");
  require(near_f3(hf4[1].xyz(),
                  luisa::make_float3(0.5f, 0.5f, 0.5f) * scale),
          "microfacet closure weight mismatch");
  require(near_f(hf4[1].w, 0.25f), "microfacet alpha mismatch");
  require(near_f(hf4[2].x, 1.5f) && hf4[2].y > 0.0f,
          "microfacet ior or sample_weight invalid");
  require(near_f3(hf4[3].xyz(),
                  luisa::make_float3(0.1f, 0.2f, 0.3f) * scale),
          "transparent closure weight mismatch");
  const luisa::float3 expected_emit =
      mixed ? luisa::make_float3(0.8f, 0.4f, 0.2f) * 0.25f +
                  luisa::make_float3(4.0f, 8.0f, 12.0f) * 0.75f
            : luisa::make_float3(0.8f, 0.4f, 0.2f);
  require(near_f3(hf4[4].xyz(), expected_emit),
          "OSL emission closure background mismatch");
}

void test_transform_backfacing_raytype(
    Device &device, Stream &stream, device_svm::SvmBackendStrategy strategy) {
  static constexpr std::string_view oso = R"oso(
OpenShadingLanguage 1.00
shader test_space_flags
oparam	color	out_c	0 0 0
global	point	P
local	point	tp
local	int	ibf
local	int	ircam
local	int	irshad
local	float	r_v
local	float	g_v
local	float	b_v
local	float	t
const	string	$s_world	"world"
const	string	$s_obj	"object"
const	string	$s_cam	"camera"
const	string	$s_shad	"shadow"
const	float	$c10	10.0
const	float	$c20	20.0
const	float	$c100	100.0
const	int	$i0	0
const	int	$i1	1
const	int	$i2	2
code ___main___
	transform	tp $s_world $s_obj P
	backfacing	ibf
	raytype		ircam $s_cam
	raytype		irshad $s_shad
	compref		r_v tp $i0
	mul		t ibf $c100
	add		r_v r_v t
	compref		g_v tp $i1
	mul		t ircam $c10
	add		g_v g_v t
	compref		b_v tp $i2
	mul		t irshad $c20
	add		b_v b_v t
	color		out_c r_v g_v b_v
end
)oso";

  contract::ShaderGraph graph;
  const auto osl =
      add_osl_node(graph, "space_flags", oso, "", "out_c=color", false);
  const auto emit = graph.add_node(compiler::node_type::emission, "Emit");
  require(graph.connect({osl, "out_c"}, emit, "Color") &&
              graph.set_input(emit, "Strength",
                              contract::SocketValue::floating(1.0f)),
          "failed to wire space_flags graph");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{emit, "Closure"});

  const auto compiled = compile_graph(graph);
  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto out_buf = device.create_buffer<luisa::float4>(2u);
  stream << words_buf.copy_from(compiled.table.words.data());

  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        const UInt lane = dispatch_x();
        const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
        const auto o2w = make_float4x4(2.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 1.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 1.0f, 0.0f,
                                       10.0f, 20.0f, 30.0f, 1.0f);
        const auto w2o = make_float4x4(0.5f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 1.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 1.0f, 0.0f,
                                       -5.0f, -20.0f, -30.0f, 1.0f);
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, o2w, w2o};
        const Bool is_lane1 = lane == 1u;
        const UInt sd_flag =
            select(0u, device_svm::shader_data_backfacing, is_lane1);
        const UInt vis =
            select(device_svm::path_ray_visibility_camera,
                   device_svm::path_ray_visibility_shadow, is_lane1);
        device_svm::ClosurePool pool{4u};
        auto sd = make_test_shader_data(
            make_float3(14.0f, 23.0f, 35.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), sd_flag, 0u, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), &pool);
        const device_svm::PathState path{vis, 0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);
        out.write(lane, make_float4(sd.closure_emission_background, 1.0f));
      });

  std::array<luisa::float4, 2u> host_out{};
  stream << kernel(words_buf, out_buf).dispatch(2u)
         << out_buf.copy_to(host_out.data()) << synchronize();
  require(near_f3(host_out[0].xyz(), luisa::make_float3(2.0f, 13.0f, 5.0f)),
          "lane 0 transform/backfacing/raytype mismatch");
  require(near_f3(host_out[1].xyz(), luisa::make_float3(102.0f, 3.0f, 25.0f)),
          "lane 1 transform/backfacing/raytype mismatch");
}

void test_getattribute_uv_and_custom(Device &device, Stream &stream,
                                     device_svm::SvmBackendStrategy strategy) {
  static constexpr std::string_view oso = R"oso(
OpenShadingLanguage 1.00
shader test_attrs
oparam	color	out_c	0 0 0
oparam	float	out_f	0.0
local	point	uv_p
local	color	my_p
local	float	miss_f
local	int	ok_uv
local	int	ok_my
local	int	ok_miss
local	float	u_v
local	float	v_v
local	float	m0
local	float	r_v
local	float	g_v
local	float	b_v
local	float	t
const	string	$s_uv	"geom:uv"
const	string	$s_my	"my_attr"
const	string	$s_miss	"does_not_exist"
const	float	$c10	10.0
const	float	$c100	100.0
const	float	$c77	77.0
const	int	$i0	0
const	int	$i1	1
code ___main___
	assign		miss_f $c77
	getattribute	ok_uv $s_uv uv_p
	getattribute	ok_my $s_my my_p
	getattribute	ok_miss $s_miss miss_f
	compref		u_v uv_p $i0
	compref		v_v uv_p $i1
	compref		m0 my_p $i0
	mul		t ok_uv $c10
	add		r_v u_v t
	mul		t ok_my $c100
	add		g_v m0 t
	assign		b_v v_v
	color		out_c r_v g_v b_v
	mul		t ok_miss $c100
	add		out_f miss_f t
end
)oso";

  contract::ShaderGraph graph;
  const auto osl = add_osl_node(graph, "attrs_node", oso, "",
                                "out_c=color;out_f=float", false);
  const auto emit = graph.add_node(compiler::node_type::emission, "Emit");
  require(graph.connect({osl, "out_c"}, emit, "Color") &&
              graph.connect({osl, "out_f"}, emit, "Strength"),
          "failed to wire attr test graph");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{emit, "Closure"});

  const auto compiled = compile_graph(graph);
  std::uint64_t my_attr_id = 0u;
  bool found_id = false;
  for (const auto &[name, id] : compiled.named_attributes) {
    if (name == "my_attr") {
      my_attr_id = id;
      found_id = true;
      break;
    }
  }
  require(found_id, "compile_shader_table did not register custom 'my_attr'");

  const std::array maps{
      compiler_svm::AttributeMap{
          .id = static_cast<std::uint64_t>(compiler_svm::ATTR_STD_UV),
          .offset = 0,
          .element = static_cast<std::uint16_t>(compiler_svm::ATTR_ELEMENT_CORNER),
          .type = static_cast<std::uint8_t>(compiler_svm::NODE_ATTR_FLOAT2),
          .pad = 0u},
      compiler_svm::AttributeMap{},
      compiler_svm::AttributeMap{
          .id = my_attr_id,
          .offset = 0,
          .element = static_cast<std::uint16_t>(compiler_svm::ATTR_ELEMENT_VERTEX),
          .type = static_cast<std::uint8_t>(compiler_svm::NODE_ATTR_FLOAT3),
          .pad = 0u},
      compiler_svm::AttributeMap{},
      compiler_svm::AttributeMap{
          .id = static_cast<std::uint64_t>(compiler_svm::ATTR_STD_NONE),
          .offset = 0,
          .element = 0u,
          .type = 0u,
          .pad = 0u},
      compiler_svm::AttributeMap{
          .id = static_cast<std::uint64_t>(compiler_svm::ATTR_STD_NONE),
          .offset = 0,
          .element = 0u,
          .type = 0u,
          .pad = 0u},
  };
  const std::array uv_corners{
      luisa::make_float2(0.1f, 0.2f),
      luisa::make_float2(0.5f, 0.2f),
      luisa::make_float2(0.1f, 0.8f),
  };
  const std::array custom_verts{
      compiler_svm::packed_float3{1.0f, 2.0f, 3.0f},
      compiler_svm::packed_float3{3.0f, 2.0f, 5.0f},
      compiler_svm::packed_float3{5.0f, 6.0f, 3.0f},
  };
  const std::array tris{luisa::make_uint3(0u, 1u, 2u)};

  auto map_buf = device.create_buffer<compiler_svm::AttributeMap>(maps.size());
  auto f2_buf = device.create_buffer<luisa::float2>(uv_corners.size());
  auto f3_buf = device.create_buffer<compiler_svm::packed_float3>(custom_verts.size());
  auto tri_buf = device.create_buffer<luisa::uint3>(tris.size());
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto out_buf = device.create_buffer<luisa::float4>(1u);
  stream << map_buf.copy_from(maps.data()) << f2_buf.copy_from(uv_corners.data())
         << f3_buf.copy_from(custom_verts.data())
         << tri_buf.copy_from(tris.data())
         << words_buf.copy_from(compiled.table.words.data());

  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferVar<compiler_svm::AttributeMap> b_maps,
          BufferFloat2 b_f2, BufferVar<compiler_svm::packed_float3> b_f3,
          BufferVar<luisa::uint3> b_tris, BufferFloat4 out) noexcept {
        const AttrKernelGlobals kg{b_maps, b_f2, b_f3, b_tris};
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, id, id};
        device_svm::ClosurePool pool{4u};
        auto sd = make_test_shader_data(
            make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0u, 0u, 0.25f, 0.5f, 0.0f, 0.0f,
            0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), &pool);
        const device_svm::PathState path{device_svm::path_ray_visibility_camera,
                                         0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);
        out.write(0u, make_float4(sd.closure_emission_background, 0.0f));
      });

  luisa::float4 host_out{};
  stream << kernel(words_buf, map_buf, f2_buf, f3_buf, tri_buf, out_buf)
                .dispatch(1u)
         << out_buf.copy_to(&host_out) << synchronize();
  // uv = (0.2, 0.5), my_attr.x = 3.5, out_f = 77.0 (miss unchanged, ok_miss = 0)
  // out_c * 77.0 = (10.2 * 77, 103.5 * 77, 0.5 * 77) = (785.4, 7969.5, 38.5)
  require(near_f3(host_out.xyz(), luisa::make_float3(785.4f, 7969.5f, 38.5f),
                  1.0e-2f),
          "getattribute uv/custom mismatch");
}

void test_derivatives(Device &device, Stream &stream,
                      device_svm::SvmBackendStrategy strategy) {
  static constexpr std::string_view oso = R"oso(
OpenShadingLanguage 1.00
shader test_derivs
param	point	in_p	0 0 0
oparam	color	out_c	0 0 0
global	point	P
local	vector	dp
local	vector	dpy
local	vector	din
local	float	rx
local	float	gy
local	float	bz
const	int	$i0	0
const	int	$i1	1
const	int	$i2	2
code ___main___
	Dx		dp P
	Dy		dpy P
	Dx		din in_p
	compref		rx dp $i0
	compref		gy dpy $i1
	compref		bz din $i2
	color		out_c rx gy bz
end
)oso";

  contract::ShaderGraph graph;
  const auto geom = graph.add_node(compiler::node_type::geometry, "Geom");
  const auto osl =
      add_osl_node(graph, "deriv_node", oso, "in_p=point", "out_c=color", false);
  const auto emit = graph.add_node(compiler::node_type::emission, "Emit");
  require(graph.connect({geom, "Position"}, osl, "in_p") &&
              graph.connect({osl, "out_c"}, emit, "Color") &&
              graph.set_input(emit, "Strength",
                              contract::SocketValue::floating(1.0f)),
          "failed to wire derivative graph");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{emit, "Closure"});

  const auto compiled = compile_graph(graph, true);
  require(!compiled.table.osl_scripts.empty() &&
              !compiled.table.osl_scripts[0].params.empty() &&
              compiled.table.osl_scripts[0].params[0].has_derivatives,
          "OSL Dx propagation failed to request derivatives on upstream input");

  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto out_buf = device.create_buffer<luisa::float4>(1u);
  stream << words_buf.copy_from(compiled.table.words.data());

  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, id, id};
        device_svm::ClosurePool pool{4u};
        // dPdx = 0.5*(2,0,1) + 0.25*(0,3,-1) = (1.0, 0.75, 0.25)
        // dPdy = 0.25*(2,0,1) + 0.5*(0,3,-1) = (0.5, 1.5, -0.25)
        auto sd = make_test_shader_data(
            make_float3(1.0f, 2.0f, 3.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0u, device_svm::object_none, 0.0f,
            0.0f, 0.5f, 0.25f, 0.25f, 0.5f, make_float3(2.0f, 0.0f, 1.0f),
            make_float3(0.0f, 3.0f, -1.0f), &pool);
        const device_svm::PathState path{device_svm::path_ray_visibility_camera,
                                         0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);
        out.write(0u, make_float4(sd.closure_emission_background, 0.0f));
      });

  luisa::float4 host_out{};
  stream << kernel(words_buf, out_buf).dispatch(1u)
         << out_buf.copy_to(&host_out) << synchronize();
  require(near_f3(host_out.xyz(), luisa::make_float3(1.0f, 1.5f, 0.25f)),
          "OSL screen-space derivative mismatch");
}

void test_cycles_i_sign_and_perlin_noise(
    Device &device, Stream &stream, device_svm::SvmBackendStrategy strategy) {
  static constexpr std::string_view valid_oso = R"oso(
OpenShadingLanguage 1.00
shader test_noise_and_i
oparam	color	out_c	0 0 0
oparam	color	out_c2	0 0 0
global	normal	N
global	vector	I
local	float	ndoti
local	float	n1
local	float	n2
local	float	n4
local	color	nc3
local	float	nu3
local	float	c0
local	float	c1
local	float	c2
local	float	g_out
local	float	b_out
const	string	$s_perlin	"perlin"
const	string	$s_uperlin	"uperlin"
const	float	$f1	1.23
const	float	$f2	2.34
const	float	$f4	4.56
const	point	$p3	1.23 2.34 3.45
const	int	$i0	0
const	int	$i1	1
const	int	$i2	2
code ___main___
	dot		ndoti N I
	noise		n1 $s_perlin $f1
	noise		n2 $s_perlin $f1 $f2
	noise		n4 $s_perlin $p3 $f4
	noise		nc3 $s_perlin $p3
	noise		nu3 $s_uperlin $p3
	color		out_c ndoti n1 n2
	compref		c0 nc3 $i0
	compref		c1 nc3 $i1
	compref		c2 nc3 $i2
	add		g_out c1 n4
	add		b_out c2 nu3
	color		out_c2 c0 g_out b_out
end
)oso";

  contract::ShaderGraph graph;
  const auto osl = add_osl_node(graph, "test_noise.osl", valid_oso, "",
                                "out_c=color;out_c2=color", false);
  const auto emit_a = graph.add_node(compiler::node_type::emission, "EmitA");
  const auto emit_b = graph.add_node(compiler::node_type::emission, "EmitB");
  const auto add_c = graph.add_node(compiler::node_type::add_closure, "Add");
  require(graph.connect({osl, "out_c"}, emit_a, "Color") &&
              graph.set_input(emit_a, "Strength",
                              contract::SocketValue::floating(1.0f)) &&
              graph.connect({osl, "out_c2"}, emit_b, "Color") &&
              graph.set_input(emit_b, "Strength",
                              contract::SocketValue::floating(1000.0f)) &&
              graph.connect({emit_a, "Closure"}, add_c, "A") &&
              graph.connect({emit_b, "Closure"}, add_c, "B"),
          "failed to wire dual emission closure");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{add_c, "Closure"});

  const auto compiled = compile_graph(graph);
  const auto usage = compiled.table.usage_for(compiler_svm::SHADER_TYPE_SURFACE);
  auto words_buf = device.create_buffer<std::uint32_t>(compiled.table.words.size());
  auto out_buf = device.create_buffer<luisa::float4>(1u);
  stream << words_buf.copy_from(compiled.table.words.data());

  const auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
        const auto id = make_float4x4(1.0f);
        const device_svm::TransformState transforms{id, id, id, id};
        device_svm::ClosurePool pool{4u};
        auto sd = make_test_shader_data(
            make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), 0u, device_svm::object_none, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), &pool);
        const device_svm::PathState path{device_svm::path_ray_visibility_camera,
                                         0u};
        const device_svm::SvmTranslationRequest req{
            .kernel_globals = kg,
            .table = compiled.table,
            .words = words,
            .shader_type = compiler_svm::SHADER_TYPE_SURFACE,
            .kernel_features = compiled.kernel_features,
            .node_feature_mask = device_svm::kernel_feature_node_mask_surface,
            .node_types_used = usage.node_types_used,
            .transform_state = transforms,
            .shader_data = sd,
            .path_state = path,
            .stack_size = std::max(1u, usage.peak_stack_usage),
            .noise_usage = usage.noise_usage};
        device_svm::translate_svm_nodes_assume_valid(req, strategy);
        out.write(0u, make_float4(sd.closure_emission_background, 0.0f));
      });

  luisa::float4 host_out{};
  stream << kernel(words_buf, out_buf).dispatch(1u)
         << out_buf.copy_to(&host_out) << synchronize();
  const luisa::float3 expected_c1{1.0f, 0.1813715f, -0.1113906f};
  const luisa::float3 expected_c2{
      0.2050007f, -0.0061124f + (-0.1340143f), -0.0285183f + 0.6025003f};
  require(near_f3(host_out.xyz(), expected_c1 + expected_c2 * 1000.0f, 2.0e-2f),
          "OSL I sign or Cycles Perlin noise mismatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    for (const auto strategy :
         {device_svm::SvmBackendStrategy::direct_jit,
          device_svm::SvmBackendStrategy::bytecode_vm}) {
      const char *sname = strategy == device_svm::SvmBackendStrategy::direct_jit
                              ? "direct_jit"
                              : "bytecode_vm";
      test_dynamic_inputs_init_and_isconnected(device, stream, strategy);
      std::cout << "[" << backend << "/" << sname
                << "] dynamic_inputs_init_and_isconnected passed\n";
      test_closures_and_mix_weight(device, stream, strategy, false);
      test_closures_and_mix_weight(device, stream, strategy, true);
      std::cout << "[" << backend << "/" << sname
                << "] closures_and_mix_weight passed\n";
      test_transform_backfacing_raytype(device, stream, strategy);
      std::cout << "[" << backend << "/" << sname
                << "] transform_backfacing_raytype passed\n";
      test_getattribute_uv_and_custom(device, stream, strategy);
      std::cout << "[" << backend << "/" << sname
                << "] getattribute_uv_and_custom passed\n";
      test_derivatives(device, stream, strategy);
      std::cout << "[" << backend << "/" << sname
                << "] derivatives passed\n";
      test_cycles_i_sign_and_perlin_noise(device, stream, strategy);
      std::cout << "[" << backend << "/" << sname
                << "] cycles_i_sign_and_perlin_noise passed\n";
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &err) {
    std::cerr << "FAIL: " << err.what() << '\n';
    return EXIT_FAILURE;
  }
}
