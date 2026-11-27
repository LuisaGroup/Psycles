#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] constexpr std::uint32_t pack_bytes(std::uint32_t b0,
                                                 std::uint32_t b1 = 0u,
                                                 std::uint32_t b2 = 0u,
                                                 std::uint32_t b3 = 0u) noexcept {
  return (b0 & 0xffu) | ((b1 & 0xffu) << 8u) | ((b2 & 0xffu) << 16u) |
         ((b3 & 0xffu) << 24u);
}

[[nodiscard]] bool approx_eq(float actual, float expected, float eps = 1e-5f) noexcept {
  return std::fabs(actual - expected) <= eps;
}

bool run_nested_stack_test(Device &device, Stream &stream) {
  auto out_buf = device.create_buffer<luisa::float4>(8u);
  auto kernel = device.compile<1u>([&](BufferFloat4 out) noexcept {
    const UInt i = dispatch_x();
    auto stack = svm_detail::Stack::direct_jit(8u);

    // Root-scope literal, runtime SSA, and unwritten (implicit 0.0f) slots.
    svm_detail::stack_store_float(stack, 0u, 10.0f);
    svm_detail::stack_store_float(stack, 1u, i.cast<float>() + 100.0f);
    // Slot 2 intentionally left unwritten before nested branch.

    const Bool cond_a = (i & 1u) != 0u;
    const Bool cond_b = (i & 2u) != 0u;
    const Bool cond_c = (i & 4u) != 0u;

    $if(cond_a) {
      svm_detail::stack_store_float(stack, 0u, 20.0f);
      svm_detail::stack_store_float(stack, 1u, 200.0f);
      svm_detail::stack_store_float(stack, 2u, 30.0f);
      $if(cond_b) {
        svm_detail::stack_store_float(stack, 0u, 25.0f);
      };
    };

    const Float s0_after_a = svm_detail::stack_load_float(stack, 0u);
    const Float s1_after_a = svm_detail::stack_load_float(stack, 1u);
    const Float s2_after_a = svm_detail::stack_load_float(stack, 2u);

    // Overwrite slot 0 at root scope (var_expr != nullptr, use_var -> false),
    // then conditionally overwrite inside a second branch. Non-taken path must
    // preserve 77.0f rather than stale 20.0f / 25.0f from the first branch.
    svm_detail::stack_store_float(stack, 0u, 77.0f);
    $if(cond_c) {
      svm_detail::stack_store_float(stack, 0u, 88.0f);
    };
    const Float s0_after_c = svm_detail::stack_load_float(stack, 0u);

    out.write(i, make_float4(s0_after_a, s1_after_a, s2_after_a, s0_after_c));
  });

  std::vector<luisa::float4> host(8u);
  stream << kernel(out_buf).dispatch(8u) << out_buf.copy_to(host.data())
         << synchronize();

  for (std::uint32_t i = 0u; i < 8u; ++i) {
    const bool a = (i & 1u) != 0u;
    const bool b = (i & 2u) != 0u;
    const bool c = (i & 4u) != 0u;
    const float exp_s0_a = a ? (b ? 25.0f : 20.0f) : 10.0f;
    const float exp_s1_a = a ? 200.0f : (static_cast<float>(i) + 100.0f);
    const float exp_s2_a = a ? 30.0f : 0.0f;
    const float exp_s0_c = c ? 88.0f : 77.0f;
    if (!approx_eq(host[i].x, exp_s0_a) || !approx_eq(host[i].y, exp_s1_a) ||
        !approx_eq(host[i].z, exp_s2_a) || !approx_eq(host[i].w, exp_s0_c)) {
      std::cerr << "nested_stack mismatch lane " << i << ": got ("
                << host[i].x << ", " << host[i].y << ", " << host[i].z << ", "
                << host[i].w << ") expected (" << exp_s0_a << ", " << exp_s1_a
                << ", " << exp_s2_a << ", " << exp_s0_c << ")\n";
      return false;
    }
  }
  return true;
}

bool run_static_offset_nodes_test(Device &device, Stream &stream) {
  auto dummy_words = device.create_buffer<std::uint32_t>(4u);
  auto out_buf = device.create_buffer<luisa::float4>(2u);
  psycles::test_support::DefaultCyclesSvmKernelGlobals kg{};

  auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        auto stack = svm_detail::Stack::direct_jit(16u);
        // Sentinels on adjacent slots to detect clobbering by scalar convert.
        svm_detail::stack_store_float(stack, 5u, 123.0f);
        svm_detail::stack_store_float(stack, 6u, 456.0f);

        // Combine vector into slots 0, 1, 2 (NODE_COMBINE_VECTOR uses
        // out_off + vector_index; must stay static in DirectJit mode).
        const std::array<std::uint32_t, 2u> comb_x{
            float_bits(2.0f), pack_bytes(0u, 0u)};
        const std::array<std::uint32_t, 2u> comb_y{
            float_bits(4.0f), pack_bytes(1u, 0u)};
        const std::array<std::uint32_t, 2u> comb_z{
            float_bits(6.0f), pack_bytes(2u, 0u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_x}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_y}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_z}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }

        // Convert vector (slot 0..2) -> scalar float at slot 4 (NODE_CONVERT_VF).
        // Must not clobber slots 5 or 6.
        const std::array<std::uint32_t, 2u> conv_vf{
            static_cast<std::uint32_t>(NODE_CONVERT_VF), pack_bytes(0u, 4u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{conv_vf}, 0u};
          svm_detail::node_convert(c, stack, kg, false);
        }

        // Separate vector Y component from slot 0 into slot 7.
        const std::array<std::uint32_t, 4u> sep_y{
            SVM_INPUT_STACK_OFFSET_MASK | 0u, 0u, 0u, pack_bytes(1u, 7u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{sep_y}, 0u};
          svm_detail::node_separate_vector(c, stack, false);
        }

        out.write(0u, make_float4(svm_detail::stack_load_float(stack, 0u),
                                  svm_detail::stack_load_float(stack, 1u),
                                  svm_detail::stack_load_float(stack, 2u),
                                  svm_detail::stack_load_float(stack, 4u)));
        out.write(1u, make_float4(svm_detail::stack_load_float(stack, 5u),
                                  svm_detail::stack_load_float(stack, 6u),
                                  svm_detail::stack_load_float(stack, 7u),
                                  0.0f));
      });

  std::vector<luisa::float4> host(2u);
  stream << kernel(dummy_words, out_buf).dispatch(1u)
         << out_buf.copy_to(host.data()) << synchronize();

  if (!approx_eq(host[0].x, 2.0f) || !approx_eq(host[0].y, 4.0f) ||
      !approx_eq(host[0].z, 6.0f) || !approx_eq(host[0].w, 4.0f) ||
      !approx_eq(host[1].x, 123.0f) || !approx_eq(host[1].y, 456.0f) ||
      !approx_eq(host[1].z, 4.0f)) {
    std::cerr << "static_offset_nodes mismatch: out0=(" << host[0].x << ", "
              << host[0].y << ", " << host[0].z << ", " << host[0].w
              << ") out1=(" << host[1].x << ", " << host[1].y << ", "
              << host[1].z << ")\n";
    return false;
  }
  return true;
}

namespace closure = psycles::luisa_backend::cycles_closure;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace compiler = psycles::compiler;
namespace contract = psycles::contract;

class StructuralDedupKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  [[nodiscard]] static Float4 deterministic_sample(
      Expr<std::uint32_t> id, std::uint32_t channel_mode,
      Expr<luisa::float2> uv) noexcept {
    const auto f_id = id.cast<float>();
    const auto r = clamp(0.15f + 0.12f * f_id + 0.25f * uv.x, 0.05f, 0.95f);
    const auto g = clamp(0.20f + 0.10f * f_id + 0.20f * uv.y, 0.05f, 0.95f);
    const auto b = clamp(0.30f + 0.08f * f_id + 0.15f * uv.x, 0.05f, 0.95f);
    const auto a = clamp(0.45f + 0.09f * f_id, 0.10f, 0.95f);
    if (channel_mode == 1u) {
      return make_float4(r, r, r, 1.0f);
    }
    return make_float4(r, g, b, a);
  }

public:
  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t>) const noexcept override {
    return 0.5f;
  }

  [[nodiscard]] std::optional<device_svm::DirectImageSlotInfo>
  direct_image_slot_info(std::int32_t image_texture_id) const noexcept override {
    if (image_texture_id < 0 || image_texture_id > 4) {
      return std::nullopt;
    }
    return device_svm::DirectImageSlotInfo{
        .texture_slot = static_cast<std::uint32_t>(image_texture_id + 10),
        .channel_mode = image_texture_id == 4 ? 1u : 0u};
  }

  [[nodiscard]] Float4 kernel_image_interp_direct_slot(
      device_svm::ShaderData &, Expr<std::uint32_t> texture_slot,
      std::uint32_t channel_mode,
      const device_svm::Dual2 &uv) const noexcept override {
    return deterministic_sample(texture_slot - 10u, channel_mode, uv.val);
  }

  [[nodiscard]] Float4 kernel_image_interp_with_udim(
      device_svm::ShaderData &, Expr<std::int32_t> image_texture_id,
      const device_svm::Dual2 &uv) const noexcept override {
    const auto uid = image_texture_id.cast<std::uint32_t>();
    const auto base = deterministic_sample(uid, 0u, uv.val);
    return select(base, make_float4(base.x, base.x, base.x, 1.0f), uid == 4u);
  }
};

[[nodiscard]] device_svm::ShaderData
make_dedup_shader_data(device_svm::ClosurePool *closures,
                       Expr<std::uint32_t> shader_id) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          normalize(make_float3(0.2f, -0.1f, 0.9f)),
          device_svm::primitive_triangle,
          shader_id,
          device_svm::shader_data_use_bump_map_correction,
          0u,
          0u,
          0.25f,
          0.35f,
          0u,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity,
          0u,
          closures};
}

[[nodiscard]] bool approx_f4(const luisa::float4 &a, const luisa::float4 &b,
                             float eps = 1.0e-5f) noexcept {
  return approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) &&
         approx_eq(a.z, b.z, eps) && approx_eq(a.w, b.w, eps);
}

bool run_principled_structural_dedup_test(Device &device, Stream &stream) {
  static constexpr std::size_t shader_count = 5u;
  static constexpr std::array<float, shader_count> iors{1.45f, 1.55f, 1.80f,
                                                        1.00f, 1.65f};
  static constexpr std::array<float, shader_count> math_bs{0.35f, 0.55f, 0.75f,
                                                           0.45f, 0.65f};
  static constexpr std::array<float, shader_count> strengths{0.25f, 0.50f,
                                                             0.85f, 0.40f,
                                                             0.60f};

  AttributeIDMap attributes;
  ImageIDMap image_ids;
  const compiler::ShaderCompiler frontend{compiler::make_core_node_registry()};
  std::vector<ShaderImage> images(shader_count);
  std::uint32_t union_features = 0u;

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    contract::ShaderGraph graph;
    const auto geom =
        graph.add_node(compiler::node_type::geometry, "Geometry");
    const auto img =
        graph.add_node(compiler::node_type::image_texture, "Image");
    const auto math = graph.add_node(compiler::node_type::math, "Math");
    const auto nmap =
        graph.add_node(compiler::node_type::normal_map, "NormalMap");
    const auto bsdf =
        graph.add_node(compiler::node_type::principled_bsdf, "Principled");

    const bool wired =
        graph.connect({geom, "Incoming"}, img, "Vector") &&
        graph.set_property(
            img, "Image",
            contract::SocketValue::unsigned_integer(100u + idx)) &&
        graph.set_property(img, "Extension",
                           contract::SocketValue::string("REPEAT")) &&
        graph.set_property(img, "Interpolation",
                           contract::SocketValue::string("Linear")) &&
        graph.set_property(img, "Projection",
                           contract::SocketValue::string("FLAT")) &&
        graph.set_property(img, "ProjectionBlend",
                           contract::SocketValue::floating(0.0f)) &&
        graph.set_property(img, "ColorSpace",
                           contract::SocketValue::string("Non-Color")) &&
        graph.set_property(img, "UnassociateAlpha",
                           contract::SocketValue::boolean(false)) &&
        graph.set_property(math, "Operation",
                           contract::SocketValue::string("MULTIPLY")) &&
        graph.set_property(math, "Clamp",
                           contract::SocketValue::boolean(false)) &&
        graph.connect({img, "Alpha"}, math, "A") &&
        graph.set_input(math, "B",
                        contract::SocketValue::floating(math_bs[idx])) &&
        graph.set_property(nmap, "Space",
                           contract::SocketValue::string("OBJECT")) &&
        graph.connect({img, "Color"}, nmap, "Color") &&
        graph.set_input(nmap, "Strength",
                        contract::SocketValue::floating(strengths[idx])) &&
        graph.connect({img, "Color"}, bsdf, "BaseColor") &&
        graph.connect({math, "Value"}, bsdf, "Roughness") &&
        graph.connect({nmap, "Normal"}, bsdf, "Normal") &&
        graph.set_input(bsdf, "IOR",
                        contract::SocketValue::floating(iors[idx]));
    if (!wired) {
      std::cerr << "structural_dedup: failed to wire shader graph " << idx
                << '\n';
      return false;
    }
    graph.set_root(contract::ShaderDomain::surface,
                   contract::OutputRef{bsdf, "Closure"});

    const auto compiled_ir = frontend.compile(graph);
    if (!compiled_ir.ok()) {
      std::cerr << "structural_dedup: frontend compile failed for shader "
                << idx << '\n';
      for (const auto &d : compiled_ir.diagnostics) {
        std::cerr << "  diag: " << d.message << '\n';
      }
      return false;
    }
    images[idx] = compile_shader(*compiled_ir.program, attributes, image_ids,
                                 ShaderCompileContext{.background = false});
    if (!images[idx].valid ||
        !images[idx].node_types_used[NODE_TEX_IMAGE] ||
        !images[idx].node_types_used[NODE_MATH] ||
        !images[idx].node_types_used[NODE_NORMAL_MAP] ||
        !images[idx].node_types_used[NODE_CLOSURE_BSDF]) {
      std::cerr << "structural_dedup: compiled shader " << idx
                << " missing expected SVM node types\n";
      return false;
    }
    union_features |= images[idx].metadata.kernel_features;
  }

  const auto table = link_shader_table(images);
  if (!table.valid || !table.ir.valid || table.shader_count != shader_count) {
    std::cerr << "structural_dedup: link_shader_table failed: "
              << table.diagnostic << '\n';
    return false;
  }

  const StructuralDedupKernelGlobals kg_host{};
  const std::function<std::optional<device_svm::DirectImageSlotInfo>(std::int32_t)>
      slot_query = [&](std::int32_t image_id) noexcept {
        return kg_host.direct_image_slot_info(image_id);
      };
  const auto &r0 = table.ir.shaders[0u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r1 = table.ir.shaders[1u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r2 = table.ir.shaders[2u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r3 = table.ir.shaders[3u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r4 = table.ir.shaders[4u].routine_for(SHADER_TYPE_SURFACE);
  if (!device_svm::can_merge_svm_blocks_structurally(table.words, r0, r1, false,
                                                     slot_query) ||
      !device_svm::can_merge_svm_blocks_structurally(table.words, r0, r2, false,
                                                     slot_query)) {
    std::cerr << "structural_dedup: expected shaders 0,1,2 to merge\n";
    return false;
  }
  if (device_svm::can_merge_svm_blocks_structurally(table.words, r0, r3, false,
                                                    slot_query)) {
    std::cerr << "structural_dedup: IOR <= 1.0 shader 3 unexpectedly merged\n";
    return false;
  }
  if (device_svm::can_merge_svm_blocks_structurally(table.words, r0, r4, false,
                                                    slot_query)) {
    std::cerr
        << "structural_dedup: channel_mode mismatch shader 4 unexpectedly merged\n";
    return false;
  }

  const auto usage = table.usage_for(SHADER_TYPE_SURFACE);
  const auto stack_extent =
      std::max<std::size_t>(1u, usage.peak_stack_usage);
  auto words_buf = device.create_buffer<std::uint32_t>(table.words.size());
  stream << words_buf.copy_from(table.words.data());

  const auto run_strategy =
      [&](device_svm::SvmBackendStrategy strategy,
          std::vector<luisa::float4> &out_host,
          std::vector<luisa::uint4> &meta_host) {
        auto out_buf = device.create_buffer<luisa::float4>(shader_count * 4u);
        auto meta_buf = device.create_buffer<luisa::uint4>(shader_count);
        auto kernel = device.compile<1u>(
            [&](BufferUInt words, BufferFloat4 out,
                BufferUInt4 meta) noexcept {
              const UInt sid = dispatch_x();
              const StructuralDedupKernelGlobals kg{};
              const auto id = make_float4x4(1.0f);
              const device_svm::TransformState transforms{id, id, id, id};
              device_svm::ClosurePool pool{8u};
              auto sd = make_dedup_shader_data(&pool, sid);
              const device_svm::PathState path{
                  device_svm::path_ray_visibility_camera, 0u};
              const device_svm::SvmTranslationRequest req{
                  .kernel_globals = kg,
                  .table = table,
                  .words = words,
                  .shader_type = SHADER_TYPE_SURFACE,
                  .kernel_features = union_features,
                  .node_feature_mask =
                      device_svm::kernel_feature_node_mask_surface,
                  .node_types_used = usage.node_types_used,
                  .transform_state = transforms,
                  .shader_data = sd,
                  .path_state = path,
                  .stack_size = stack_extent,
                  .noise_usage = usage.noise_usage};
              device_svm::translate_svm_nodes_assume_valid(req, strategy);

              const auto c0 = pool.common(0u);
              const auto c1 = pool.common(1u);
              UInt fresnel_type = 0u;
              Float4 mf_out = make_float4(0.0f);
              $if(c0.type == closure::type_microfacet_ggx) {
                const auto mf = pool.microfacet(0u);
                fresnel_type = mf.param.fresnel_type;
                mf_out = make_float4(mf.param.alpha_x, mf.param.alpha_y,
                                     mf.param.ior, mf.param.energy_scale);
              };
              meta.write(sid, make_uint4(pool.count(),
                                         (pool.left() & 0xffffu) |
                                             ((fresnel_type & 0xffffu) << 16u),
                                         c0.type,
                                         select(0u, c1.type, pool.count() > 1u)));
              const auto base = sid * 4u;
              out.write(base + 0u, make_float4(c0.weight, c0.sample_weight));
              out.write(base + 1u, make_float4(c0.N, 0.0f));
              out.write(
                  base + 2u,
                  select(make_float4(0.0f),
                         make_float4(c1.weight, c1.sample_weight),
                         pool.count() > 1u));
              out.write(base + 3u, mf_out);
            });
        out_host.resize(shader_count * 4u);
        meta_host.resize(shader_count);
        stream << kernel(words_buf, out_buf, meta_buf).dispatch(shader_count)
               << out_buf.copy_to(out_host.data())
               << meta_buf.copy_to(meta_host.data()) << synchronize();
      };

  std::vector<luisa::float4> jit_out, vm_out;
  std::vector<luisa::uint4> jit_meta, vm_meta;
  run_strategy(device_svm::SvmBackendStrategy::direct_jit, jit_out, jit_meta);
  run_strategy(device_svm::SvmBackendStrategy::bytecode_vm, vm_out, vm_meta);

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    const auto jit_left = jit_meta[idx].y & 0xffffu;
    const auto jit_fresnel = jit_meta[idx].y >> 16u;
    const auto vm_left = vm_meta[idx].y & 0xffffu;
    const auto vm_fresnel = vm_meta[idx].y >> 16u;
    const auto expected_jit_left = iors[idx] > 1.0f ? 6u : 7u;
    const auto expected_vm_left = iors[idx] > 1.0f ? 5u : 7u;
    const auto expected_jit_fresnel =
        iors[idx] > 1.0f
            ? static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::dielectric)
            : 0u;
    const auto expected_vm_fresnel =
        iors[idx] > 1.0f
            ? static_cast<std::uint32_t>(
                  device_svm::MicrofacetFresnel::generalized_schlick)
            : 0u;
    if (jit_meta[idx].x != vm_meta[idx].x || jit_meta[idx].z != vm_meta[idx].z ||
        jit_meta[idx].w != vm_meta[idx].w || jit_left != expected_jit_left ||
        vm_left != expected_vm_left || jit_fresnel != expected_jit_fresnel ||
        vm_fresnel != expected_vm_fresnel) {
      std::cerr << "structural_dedup meta mismatch shader " << idx
                << ": jit=(cnt=" << jit_meta[idx].x << ",left=" << jit_left
                << ",fresnel=" << jit_fresnel << ",t0=" << jit_meta[idx].z
                << ",t1=" << jit_meta[idx].w << ") vm=(cnt=" << vm_meta[idx].x
                << ",left=" << vm_left << ",fresnel=" << vm_fresnel
                << ",t0=" << vm_meta[idx].z << ",t1=" << vm_meta[idx].w
                << ")\n";
      return false;
    }
    for (std::size_t k = 0u; k < 4u; ++k) {
      const auto &a = jit_out[idx * 4u + k];
      const auto &b = vm_out[idx * 4u + k];
      if (!approx_f4(a, b, 1.0e-5f)) {
        std::cerr << "structural_dedup value mismatch shader " << idx
                  << " slot " << k << ": jit=(" << a.x << "," << a.y << ","
                  << a.z << "," << a.w << ") vm=(" << b.x << "," << b.y << ","
                  << b.z << "," << b.w << ")\n";
        return false;
      }
    }
  }
  if (approx_f4(jit_out[0u * 4u + 1u], jit_out[1u * 4u + 1u], 1.0e-4f) ||
      approx_f4(jit_out[1u * 4u + 1u], jit_out[2u * 4u + 1u], 1.0e-4f) ||
      approx_f4(jit_out[0u * 4u + 3u], jit_out[1u * 4u + 3u], 1.0e-4f) ||
      approx_f4(jit_out[1u * 4u + 3u], jit_out[2u * 4u + 3u], 1.0e-4f)) {
    std::cerr << "structural_dedup: shaders 0,1,2 collapsed to identical outputs\n";
    return false;
  }
  return true;
}

bool run_value_f_structural_dedup_test(Device &device, Stream &stream) {
  static constexpr std::array<float, 2u> roughness_vals{0.05f, 0.65f};
  std::vector<ShaderImage> images(2u);
  for (std::size_t i = 0u; i < 2u; ++i) {
    auto &img = images[i];
    img.valid = true;
    img.words = {
        static_cast<std::uint32_t>(NODE_SHADER_JUMP),
        4u,
        19u,
        19u,
        static_cast<std::uint32_t>(NODE_VALUE_F),
        float_bits(roughness_vals[i]),
        pack_bytes(0u),
        static_cast<std::uint32_t>(NODE_CLOSURE_SET_WEIGHT),
        float_bits(0.6f + 0.15f * static_cast<float>(i)),
        float_bits(0.4f),
        float_bits(0.2f),
        static_cast<std::uint32_t>(NODE_CLOSURE_BSDF),
        static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID),
        pack_bytes(SVM_STACK_INVALID),
        float_bits(0.7f),
        float_bits(0.4f),
        float_bits(0.2f),
        SVM_INPUT_STACK_OFFSET_MASK | 0u,
        pack_bytes(SVM_STACK_INVALID),
        static_cast<std::uint32_t>(NODE_END),
    };
    img.node_types_used[NODE_SHADER_JUMP] = true;
    img.node_types_used[NODE_VALUE_F] = true;
    img.node_types_used[NODE_CLOSURE_SET_WEIGHT] = true;
    img.node_types_used[NODE_CLOSURE_BSDF] = true;
    img.node_types_used[NODE_END] = true;
    img.peak_stack_usage = 1u;
    SvmShaderIr shader_ir;
    shader_ir.routines[SHADER_TYPE_SURFACE] = {
        SvmOpIr{.instruction = {.opcode = NODE_VALUE_F,
                                .payload_offset = 5u,
                                .payload_word_count = 2u}},
        SvmOpIr{.instruction = {.opcode = NODE_CLOSURE_SET_WEIGHT,
                                .payload_offset = 8u,
                                .payload_word_count = 3u}},
        SvmOpIr{.instruction = {.opcode = NODE_CLOSURE_BSDF,
                                .payload_offset = 12u,
                                .payload_word_count = 7u}},
    };
    img.ir = std::move(shader_ir);
  }
  const auto table = link_shader_table(images);
  if (!table.valid || !table.ir.valid || table.shader_count != 2u) {
    std::cerr << "value_f_dedup: link_shader_table failed: "
              << table.diagnostic << '\n';
    return false;
  }
  if (!device_svm::can_merge_svm_blocks_structurally(
          table.words, table.ir.shaders[0u].routine_for(SHADER_TYPE_SURFACE),
          table.ir.shaders[1u].routine_for(SHADER_TYPE_SURFACE), false)) {
    std::cerr << "value_f_dedup: expected NODE_VALUE_F shaders to merge\n";
    return false;
  }
  const auto usage = table.usage_for(SHADER_TYPE_SURFACE);
  auto words_buf = device.create_buffer<std::uint32_t>(table.words.size());
  stream << words_buf.copy_from(table.words.data());

  const auto run_strategy =
      [&](device_svm::SvmBackendStrategy strategy,
          std::vector<luisa::float4> &out_host) {
        auto out_buf = device.create_buffer<luisa::float4>(4u);
        auto kernel = device.compile<1u>(
            [&](BufferUInt words, BufferFloat4 out) noexcept {
              const UInt sid = dispatch_x();
              const StructuralDedupKernelGlobals kg{};
              const auto id = make_float4x4(1.0f);
              const device_svm::TransformState transforms{id, id, id, id};
              device_svm::ClosurePool pool{4u};
              auto sd = make_dedup_shader_data(&pool, sid);
              const device_svm::PathState path{
                  device_svm::path_ray_visibility_camera, 0u};
              const device_svm::SvmTranslationRequest req{
                  .kernel_globals = kg,
                  .table = table,
                  .words = words,
                  .shader_type = SHADER_TYPE_SURFACE,
                  .kernel_features = device_svm::kernel_feature_node_bsdf,
                  .node_feature_mask =
                      device_svm::kernel_feature_node_mask_surface,
                  .node_types_used = usage.node_types_used,
                  .transform_state = transforms,
                  .shader_data = sd,
                  .path_state = path,
                  .stack_size = 1u,
                  .noise_usage = usage.noise_usage};
              device_svm::translate_svm_nodes_assume_valid(req, strategy);
              const auto on = pool.oren_nayar(0u);
              out.write(sid * 2u + 0u,
                        make_float4(on.common.weight, on.common.sample_weight));
              out.write(sid * 2u + 1u,
                        make_float4(on.param.roughness, on.param.a, on.param.b,
                                    on.common.type.cast<float>()));
            });
        out_host.resize(4u);
        stream << kernel(words_buf, out_buf).dispatch(2u)
               << out_buf.copy_to(out_host.data()) << synchronize();
      };

  std::vector<luisa::float4> jit_out, vm_out;
  run_strategy(device_svm::SvmBackendStrategy::direct_jit, jit_out);
  run_strategy(device_svm::SvmBackendStrategy::bytecode_vm, vm_out);
  for (std::size_t i = 0u; i < 4u; ++i) {
    if (!approx_f4(jit_out[i], vm_out[i], 1.0e-5f)) {
      std::cerr << "value_f_dedup parity mismatch slot " << i << ": jit=("
                << jit_out[i].x << "," << jit_out[i].y << "," << jit_out[i].z
                << "," << jit_out[i].w << ") vm=(" << vm_out[i].x << ","
                << vm_out[i].y << "," << vm_out[i].z << "," << vm_out[i].w
                << ")\n";
      return false;
    }
  }
  if (approx_eq(jit_out[1u].x, jit_out[3u].x, 1.0e-3f)) {
    std::cerr << "value_f_dedup: roughness did not parameterize across shaders\n";
    return false;
  }
  return true;
}

bool run_microfacet_node_data_view_dedup_test(Device &device, Stream &stream) {
  static constexpr std::size_t shader_count = 5u;
  static constexpr std::array<float, shader_count> roughnesses{
      0.10f, 0.35f, 0.70f, 0.05f, 0.40f};
  static constexpr std::array<float, shader_count> iors{
      1.00f, 1.00f, 1.00f, 1.33f, 1.80f};
  static constexpr std::array<psycles::Vec3f, shader_count> colors{{
      {0.85f, 0.30f, 0.20f},
      {0.30f, 0.80f, 0.25f},
      {0.25f, 0.35f, 0.90f},
      {0.92f, 0.85f, 0.75f},
      {0.60f, 0.78f, 0.95f},
  }};

  AttributeIDMap attributes;
  ImageIDMap image_ids;
  const compiler::ShaderCompiler frontend{compiler::make_core_node_registry()};
  std::vector<ShaderImage> images(shader_count);
  std::uint32_t union_features = 0u;

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    contract::ShaderGraph graph;
    const bool is_glass = idx >= 3u;
    const auto node = graph.add_node(
        is_glass ? compiler::node_type::glass_bsdf
                 : compiler::node_type::glossy_bsdf,
        is_glass ? "Glass" : "Glossy");
    const bool wired =
        graph.set_property(
            node, "Distribution",
            contract::SocketValue::string(is_glass ? "GGX" : "MULTI_GGX")) &&
        graph.set_input(node, "Color",
                        contract::SocketValue::color(colors[idx])) &&
        graph.set_input(node, "Roughness",
                        contract::SocketValue::floating(roughnesses[idx])) &&
        (!is_glass ||
         graph.set_input(node, "IOR",
                         contract::SocketValue::floating(iors[idx])));
    if (!wired) {
      std::cerr << "microfacet_dedup: failed to wire shader " << idx << '\n';
      return false;
    }
    graph.set_root(contract::ShaderDomain::surface,
                   contract::OutputRef{node, "Closure"});
    const auto compiled_ir = frontend.compile(graph);
    if (!compiled_ir.ok()) {
      std::cerr << "microfacet_dedup: frontend compile failed for shader "
                << idx << '\n';
      return false;
    }
    images[idx] = compile_shader(*compiled_ir.program, attributes, image_ids,
                                 ShaderCompileContext{.background = false});
    if (!images[idx].valid || !images[idx].node_types_used[NODE_CLOSURE_BSDF]) {
      std::cerr << "microfacet_dedup: compile_shader failed for " << idx
                << '\n';
      return false;
    }
    union_features |= images[idx].metadata.kernel_features;
  }

  const auto table = link_shader_table(images);
  if (!table.valid || !table.ir.valid || table.shader_count != shader_count) {
    std::cerr << "microfacet_dedup: link_shader_table failed: "
              << table.diagnostic << '\n';
    return false;
  }

  const auto &r0 = table.ir.shaders[0u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r1 = table.ir.shaders[1u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r2 = table.ir.shaders[2u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r3 = table.ir.shaders[3u].routine_for(SHADER_TYPE_SURFACE);
  const auto &r4 = table.ir.shaders[4u].routine_for(SHADER_TYPE_SURFACE);
  if (!device_svm::can_merge_svm_blocks_structurally(table.words, r0, r1, false) ||
      !device_svm::can_merge_svm_blocks_structurally(table.words, r0, r2, false) ||
      !device_svm::can_merge_svm_blocks_structurally(table.words, r3, r4, false)) {
    std::cerr << "microfacet_dedup: expected Glossy and Glass groups to merge\n";
    return false;
  }

  const auto usage = table.usage_for(SHADER_TYPE_SURFACE);
  const auto stack_extent =
      std::max<std::size_t>(1u, usage.peak_stack_usage);
  auto words_buf = device.create_buffer<std::uint32_t>(table.words.size());
  stream << words_buf.copy_from(table.words.data());

  const auto run_strategy = [&](device_svm::SvmBackendStrategy strategy,
                                std::vector<luisa::float4> &out_host,
                                std::vector<luisa::uint4> &meta_host) {
    auto out_buf = device.create_buffer<luisa::float4>(shader_count * 4u);
    auto meta_buf = device.create_buffer<luisa::uint4>(shader_count);
    auto kernel = device.compile<1u>(
        [&](BufferUInt words, BufferFloat4 out, BufferUInt4 meta) noexcept {
          const UInt sid = dispatch_x();
          const StructuralDedupKernelGlobals kg{};
          const auto id = make_float4x4(1.0f);
          const device_svm::TransformState transforms{id, id, id, id};
          device_svm::ClosurePool pool{8u};
          auto sd = make_dedup_shader_data(&pool, sid);
          const device_svm::PathState path{
              device_svm::path_ray_visibility_camera, 0u};
          const device_svm::SvmTranslationRequest req{
              .kernel_globals = kg,
              .table = table,
              .words = words,
              .shader_type = SHADER_TYPE_SURFACE,
              .kernel_features = union_features,
              .node_feature_mask =
                  device_svm::kernel_feature_node_mask_surface,
              .node_types_used = usage.node_types_used,
              .transform_state = transforms,
              .shader_data = sd,
              .path_state = path,
              .stack_size = stack_extent,
              .noise_usage = usage.noise_usage};
          device_svm::translate_svm_nodes_assume_valid(req, strategy);

          const auto c0 = pool.common(0u);
          const auto mf = pool.microfacet(0u);
          Float4 gs_refl = make_float4(0.0f);
          Float4 gs_f0 = make_float4(0.0f);
          $if(c0.type == closure::type_microfacet_ggx_glass) {
            const auto gs = mf.load_generalized_schlick();
            gs_refl = make_float4(gs.reflection_tint, 0.0f);
            gs_f0 = make_float4(gs.f0, 0.0f);
          };
          meta.write(sid, make_uint4(pool.count(), pool.left(), c0.type, 0u));
          const auto base = sid * 4u;
          out.write(base + 0u, make_float4(c0.weight, c0.sample_weight));
          out.write(base + 1u,
                    make_float4(mf.param.alpha_x, mf.param.alpha_y,
                                mf.param.ior, mf.param.energy_scale));
          out.write(base + 2u, gs_refl);
          out.write(base + 3u, gs_f0);
        });
    out_host.resize(shader_count * 4u);
    meta_host.resize(shader_count);
    stream << kernel(words_buf, out_buf, meta_buf).dispatch(shader_count)
           << out_buf.copy_to(out_host.data())
           << meta_buf.copy_to(meta_host.data()) << synchronize();
  };

  std::vector<luisa::float4> jit_out, vm_out;
  std::vector<luisa::uint4> jit_meta, vm_meta;
  run_strategy(device_svm::SvmBackendStrategy::direct_jit, jit_out, jit_meta);
  run_strategy(device_svm::SvmBackendStrategy::bytecode_vm, vm_out, vm_meta);

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    if (jit_meta[idx].x != vm_meta[idx].x ||
        jit_meta[idx].y != vm_meta[idx].y ||
        jit_meta[idx].z != vm_meta[idx].z || jit_meta[idx].x != 1u) {
      std::cerr << "microfacet_dedup meta mismatch shader " << idx << '\n';
      return false;
    }
    for (std::size_t k = 0u; k < 4u; ++k) {
      const auto &a = jit_out[idx * 4u + k];
      const auto &b = vm_out[idx * 4u + k];
      if (!approx_f4(a, b, 1.0e-5f)) {
        std::cerr << "microfacet_dedup mismatch shader " << idx << " slot " << k
                  << ": jit=(" << a.x << "," << a.y << "," << a.z << "," << a.w
                  << ") vm=(" << b.x << "," << b.y << "," << b.z << "," << b.w
                  << ")\n";
        return false;
      }
    }
  }
  if (approx_eq(jit_out[0u * 4u + 1u].x, jit_out[1u * 4u + 1u].x, 1.0e-2f) ||
      approx_eq(jit_out[1u * 4u + 1u].x, jit_out[2u * 4u + 1u].x, 1.0e-2f) ||
      approx_eq(jit_out[3u * 4u + 1u].x, jit_out[4u * 4u + 1u].x, 1.0e-2f) ||
      approx_eq(jit_out[3u * 4u + 1u].z, jit_out[4u * 4u + 1u].z, 1.0e-2f) ||
      approx_f4(jit_out[3u * 4u + 2u], jit_out[4u * 4u + 2u], 1.0e-2f)) {
    std::cerr << "microfacet_dedup: parameterized fields did not vary\n";
    return false;
  }
  return true;
}

class ScopedUsageKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  ClosureUsage _usage;
  bool _may_thin_film;

public:
  ScopedUsageKernelGlobals(ClosureUsage usage, bool may_thin_film) noexcept
      : _usage{usage}, _may_thin_film{may_thin_film} {}

  [[nodiscard]] ClosureUsage closure_usage() const noexcept override {
    return _usage;
  }

  [[nodiscard]] bool may_have_thin_film() const noexcept override {
    return _may_thin_film;
  }

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t>) const noexcept override {
    return 0.5f;
  }
};

bool run_principled_thin_wall_compact_pool_test(Device &device,
                                                Stream &stream) {
  static constexpr std::size_t shader_count = 2u;
  static constexpr std::array<float, shader_count> roughnesses{0.15f, 0.45f};
  static constexpr std::array<psycles::Vec3f, shader_count> base_colors{{
      {0.80f, 0.25f, 0.15f},
      {0.20f, 0.70f, 0.35f},
  }};

  AttributeIDMap attributes;
  ImageIDMap image_ids;
  const compiler::ShaderCompiler frontend{compiler::make_core_node_registry()};
  std::vector<ShaderImage> images(shader_count);
  std::uint32_t union_features = device_svm::kernel_feature_subsurface;
  ClosureUsage merged_usage = ClosureUsage::none();
  bool may_thin_film = false;

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    contract::ShaderGraph graph;
    const auto geom =
        graph.add_node(compiler::node_type::geometry, "Geometry");
    const auto math = graph.add_node(compiler::node_type::math, "Math");
    const auto bsdf =
        graph.add_node(compiler::node_type::principled_bsdf, "Principled");
    const bool wired =
        graph.set_property(math, "Operation",
                           contract::SocketValue::string("ADD")) &&
        graph.set_property(math, "Clamp",
                           contract::SocketValue::boolean(false)) &&
        graph.connect({geom, "Backfacing"}, math, "A") &&
        graph.set_input(math, "B", contract::SocketValue::floating(0.45f)) &&
        graph.set_input(bsdf, "ThinWall",
                        contract::SocketValue::boolean(true)) &&
        graph.connect({math, "Value"}, bsdf, "SubsurfaceWeight") &&
        graph.set_input(bsdf, "SubsurfaceScale",
                        contract::SocketValue::floating(0.05f)) &&
        graph.set_input(bsdf, "BaseColor",
                        contract::SocketValue::color(base_colors[idx])) &&
        graph.set_input(bsdf, "Roughness",
                        contract::SocketValue::floating(roughnesses[idx]));
    if (!wired) {
      std::cerr << "thin_wall_compact: failed to wire shader " << idx << '\n';
      return false;
    }
    graph.set_root(contract::ShaderDomain::surface,
                   contract::OutputRef{bsdf, "Closure"});
    const auto compiled_ir = frontend.compile(graph);
    if (!compiled_ir.ok()) {
      std::cerr << "thin_wall_compact: frontend compile failed for shader "
                << idx << '\n';
      return false;
    }
    images[idx] = compile_shader(*compiled_ir.program, attributes, image_ids,
                                 ShaderCompileContext{.background = false});
    if (!images[idx].valid) {
      std::cerr << "thin_wall_compact: compile_shader failed\n";
      return false;
    }
    union_features |= images[idx].metadata.kernel_features;
    merged_usage.merge(images[idx].metadata.surface_closure_usage);
    may_thin_film = may_thin_film || images[idx].metadata.may_have_thin_film;
  }

  constexpr std::uint64_t wide_closure_mask =
      ClosureUsage::bit(CLOSURE_BSDF_SHEEN_ID) |
      ClosureUsage::bit(CLOSURE_BSDF_HAIR_CHIANG_ID) |
      ClosureUsage::bit(CLOSURE_BSDF_HAIR_HUANG_ID) |
      ClosureUsage::bit(CLOSURE_BSSRDF_BURLEY_ID) |
      ClosureUsage::bit(CLOSURE_BSSRDF_RANDOM_WALK_ID) |
      ClosureUsage::bit(CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID) |
      ClosureUsage::bit(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID);
  if (!merged_usage.has_principled(ClosureUsage::principled_subsurface) ||
      !merged_usage.has_principled(ClosureUsage::principled_thin_wall) ||
      (merged_usage.surface_closure_mask & wide_closure_mask) != 0ull ||
      merged_usage.has_fresnel(ClosureUsage::fresnel_anisotropic)) {
    std::cerr << "thin_wall_compact: unexpected ClosureUsage flags\n";
    return false;
  }

  const auto table = link_shader_table(images);
  if (!table.valid || !table.ir.valid || table.shader_count != shader_count) {
    std::cerr << "thin_wall_compact: link_shader_table failed\n";
    return false;
  }
  if (!device_svm::can_merge_svm_blocks_structurally(
          table.words, table.ir.shaders[0u].routine_for(SHADER_TYPE_SURFACE),
          table.ir.shaders[1u].routine_for(SHADER_TYPE_SURFACE), false)) {
    std::cerr << "thin_wall_compact: expected ThinWall shaders to merge\n";
    return false;
  }

  const auto usage = table.usage_for(SHADER_TYPE_SURFACE);
  const auto stack_extent =
      std::max<std::size_t>(1u, usage.peak_stack_usage);
  auto words_buf = device.create_buffer<std::uint32_t>(table.words.size());
  stream << words_buf.copy_from(table.words.data());

  const auto run_strategy = [&](device_svm::SvmBackendStrategy strategy,
                                std::vector<luisa::float4> &out_host,
                                std::vector<luisa::uint4> &meta_host) {
    auto out_buf = device.create_buffer<luisa::float4>(shader_count * 4u);
    auto meta_buf = device.create_buffer<luisa::uint4>(shader_count);
    auto kernel = device.compile<1u>(
        [&](BufferUInt words, BufferFloat4 out, BufferUInt4 meta) noexcept {
          const UInt sid = dispatch_x();
          const ScopedUsageKernelGlobals kg{merged_usage, may_thin_film};
          const auto id = make_float4x4(1.0f);
          const device_svm::TransformState transforms{id, id, id, id};
          device_svm::ClosurePool pool{
              8u,
              {.may_have_thin_film = may_thin_film,
               .may_have_anisotropic_microfacet =
                   merged_usage.has_fresnel(ClosureUsage::fresnel_anisotropic),
               .may_have_custom_generalized_schlick = merged_usage.has_fresnel(
                   ClosureUsage::fresnel_custom_generalized_schlick),
               .may_have_positive_exponent = merged_usage.has_fresnel(
                   ClosureUsage::fresnel_positive_exponent),
               .may_have_extra_fresnel = merged_usage.has_fresnel(
                   ClosureUsage::fresnel_conductor |
                   ClosureUsage::fresnel_f82_tint |
                   ClosureUsage::fresnel_generalized_schlick),
               .compact_64b_slots = true}};
          auto sd = make_dedup_shader_data(&pool, sid);
          const device_svm::PathState path{
              device_svm::path_ray_visibility_camera, 0u};
          const device_svm::SvmTranslationRequest req{
              .kernel_globals = kg,
              .table = table,
              .words = words,
              .shader_type = SHADER_TYPE_SURFACE,
              .kernel_features = union_features,
              .node_feature_mask =
                  device_svm::kernel_feature_node_mask_surface,
              .node_types_used = usage.node_types_used,
              .transform_state = transforms,
              .shader_data = sd,
              .path_state = path,
              .stack_size = stack_extent,
              .noise_usage = usage.noise_usage};
          device_svm::translate_svm_nodes_assume_valid(req, strategy);

          const auto c0 = pool.common(0u);
          const auto c1 = pool.common(1u);
          const auto c2 = pool.common(2u);
          const auto c3 = pool.common(3u);
          meta.write(sid, make_uint4(pool.count(), pool.left(), c1.type, c2.type));
          const auto base = sid * 4u;
          out.write(base + 0u, make_float4(c0.weight, c0.sample_weight));
          out.write(base + 1u, make_float4(c1.weight, c1.sample_weight));
          out.write(base + 2u, make_float4(c2.weight, c2.sample_weight));
          out.write(base + 3u, make_float4(c3.weight, c3.sample_weight));
        });
    out_host.resize(shader_count * 4u);
    meta_host.resize(shader_count);
    stream << kernel(words_buf, out_buf, meta_buf).dispatch(shader_count)
           << out_buf.copy_to(out_host.data())
           << meta_buf.copy_to(meta_host.data()) << synchronize();
  };

  std::vector<luisa::float4> jit_out, vm_out;
  std::vector<luisa::uint4> jit_meta, vm_meta;
  run_strategy(device_svm::SvmBackendStrategy::direct_jit, jit_out, jit_meta);
  run_strategy(device_svm::SvmBackendStrategy::bytecode_vm, vm_out, vm_meta);

  for (std::size_t idx = 0u; idx < shader_count; ++idx) {
    if (jit_meta[idx].x != vm_meta[idx].x ||
        jit_meta[idx].z != vm_meta[idx].z ||
        jit_meta[idx].w != vm_meta[idx].w || jit_meta[idx].x < 3u) {
      std::cerr << "thin_wall_compact meta mismatch shader " << idx << '\n';
      return false;
    }
    for (std::size_t k = 0u; k < 4u; ++k) {
      if (!approx_f4(jit_out[idx * 4u + k], vm_out[idx * 4u + k], 1.0e-5f)) {
        std::cerr << "thin_wall_compact mismatch shader " << idx << " slot "
                  << k << '\n';
        return false;
      }
    }
  }
  if (approx_f4(jit_out[0u * 4u + 1u], jit_out[1u * 4u + 1u], 1.0e-2f)) {
    std::cerr << "thin_wall_compact: base color did not parameterize\n";
    return false;
  }
  return true;
}

bool run_parameterized_structural_dedup_test(Device &device, Stream &stream) {
  return run_principled_structural_dedup_test(device, stream) &&
         run_value_f_structural_dedup_test(device, stream) &&
         run_microfacet_node_data_view_dedup_test(device, stream) &&
         run_principled_thin_wall_compact_pool_test(device, stream);
}

} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    auto stream = device.create_stream();
    if (!run_nested_stack_test(device, stream)) {
      return EXIT_FAILURE;
    }
    if (!run_static_offset_nodes_test(device, stream)) {
      return EXIT_FAILURE;
    }
    if (!run_parameterized_structural_dedup_test(device, stream)) {
      return EXIT_FAILURE;
    }
    std::cout << "DirectJit stack & static offset tests PASSED\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
