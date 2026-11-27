#include "path_kernel_caustics_manifold.h"

#include "cycles_svm_internal.h"
#include "cycles_svm_microfacet_fresnel.h"
#include "cycles_svm_surface_shader.h"
#include "path_kernel_area_light.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_light.h"
#include "path_tracer_cycles_svm_shadow.h"
#include "surface_fresnel.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_light.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/cycles_svm_translator.h>
#include <psycles/luisa/cycles_transform.h>
#include <psycles/luisa/surface_ray.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
namespace svm = cycles_svm;
namespace svm_detail = svm::detail;
namespace abi = compiler::cycles_svm;

inline constexpr std::uint32_t kMaxChainLen = 4u;
inline constexpr std::uint32_t kMaxNewtonIters = 20u;
inline constexpr float kConvTol = 1.0e-4f;
inline constexpr std::uint32_t kSmsMaxNewtonIters = 24u;
inline constexpr float kSmsConvTol = 2.0e-5f;
inline constexpr float kMinStepBeta = 1.0e-3f;
inline constexpr float kRayEps = 1.0e-4f;

// Compact geometric state for a smooth specular caster vertex during MNEE
// seed discovery and Newton iteration. Keeping closure payload out of this
// struct avoids carrying Fresnel weights across Newton steps.
struct CasterVertexVar {
  Float3 p{make_float3(0.0f)};
  Float3 T1{make_float3(1.0f, 0.0f, 0.0f)};
  Float3 T2{make_float3(0.0f, 1.0f, 0.0f)};
  Float3 n{make_float3(0.0f, 0.0f, 1.0f)};
  Float3 dn_dx{make_float3(0.0f)};
  Float3 dn_dy{make_float3(0.0f)};
  Float3 ng{make_float3(0.0f, 0.0f, 1.0f)};
  Float2 bary{make_float2(0.0f)};
  Float eta{1.0f};
  UInt inst{surface_ray::invalid_primitive};
  UInt prim{surface_ray::invalid_primitive};
  UInt cycles_object{0u};
  Bool backfacing{false};
  // Phase 2B SMS fields (unused by MNEE).
  Bool is_reflection{false};
  Float2 h_offset{make_float2(0.0f)};
  Float seed_alpha2{0.0f};
  Bool seed_is_beckmann{false};
};

// Outlined 2x2 matrix inversion and manifold constraint callables to keep
// cold HIP/DX/VK JIT compact. Structs must stay outside anonymous namespace
// and be registered with LUISA_STRUCT before instantiating Var/Callable.
} // namespace

struct MnInv2x2 {
  luisa::float4 inv{};
  bool ok{};
};

struct ConstraintVertexEval {
  luisa::float2 C{};
  luisa::float4 A_prev{};
  luisa::float4 A_diag{};
  luisa::float4 A_next{};
  luisa::float4 B_light{};
  luisa::float3 h{};
  luisa::float3 wi{};
  luisa::float3 wo{};
  float c_err{};
  bool orientation_ok{};
};

struct SmsConstraintInput {
  luisa::float3 p_prev{};
  luisa::float3 p{};
  luisa::float3 p_next{};
  luisa::float3 n{};
  luisa::float3 T1{};
  luisa::float3 T2{};
  luisa::float3 dn_dx{};
  luisa::float3 dn_dy{};
  float eta{};
  luisa::float3 T1_prev{};
  luisa::float3 T2_prev{};
  luisa::float3 T1_next{};
  luisa::float3 T2_next{};
  bool has_prev{};
  bool has_next{};
  bool is_distant{};
  luisa::float3 light_e1{};
  luisa::float3 light_e2{};
  bool is_reflection{};
  luisa::float2 h_offset{};
  luisa::float3 ref_T{};
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::MnInv2x2, inv, ok){};
LUISA_STRUCT(psycles::luisa_backend::detail::ConstraintVertexEval,
             C, A_prev, A_diag, A_next, B_light, h, wi, wo, c_err,
             orientation_ok){};
LUISA_STRUCT(psycles::luisa_backend::detail::SmsConstraintInput,
             p_prev, p, p_next, n, T1, T2, dn_dx, dn_dy, eta, T1_prev, T2_prev,
             T1_next, T2_next, has_prev, has_next, is_distant, light_e1,
             light_e2, is_reflection, h_offset, ref_T){};

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] const Callable<MnInv2x2(luisa::float4)> &
invert_2x2_callable() noexcept {
  static auto callable = []() noexcept {
    // 2x2 matrix stored row-major as (m00, m01, m10, m11).
    Callable result{[](Float4 m) noexcept {
      Var<MnInv2x2> out{};
      const auto det = m.x * m.w - m.y * m.z;
      const auto ok = abs(det) > 1.0e-12f;
      const auto inv_det = select(0.0f, 1.0f / det, ok);
      out.ok = ok;
      out.inv = make_float4(m.w * inv_det, -m.y * inv_det,
                            -m.z * inv_det, m.x * inv_det);
      return out;
    }};
    result.set_name("mnee_invert_2x2");
    return result;
  }();
  return callable;
}

[[nodiscard]] const Callable<luisa::float4(luisa::float4, luisa::float4)> &
mul_2x2_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Float4 a, Float4 b) noexcept {
      return make_float4(
          a.x * b.x + a.y * b.z,
          a.x * b.y + a.y * b.w,
          a.z * b.x + a.w * b.z,
          a.z * b.y + a.w * b.w);
    }};
    result.set_name("mnee_mul_2x2");
    return result;
  }();
  return callable;
}

// Compute constraint C and Jacobian blocks (A_prev, A_diag, A_next, and
// optional B_light on the last vertex) at one manifold vertex i.
[[nodiscard]] const Callable<ConstraintVertexEval(
    luisa::float3, luisa::float3, luisa::float3, luisa::float3, luisa::float3,
    luisa::float3, luisa::float3, luisa::float3, float, luisa::float3,
    luisa::float3, luisa::float3, luisa::float3, bool, bool, bool,
    luisa::float3, luisa::float3)> &
eval_constraint_vertex_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Float3 p_prev, Float3 p, Float3 p_next, Float3 n,
                       Float3 T1, Float3 T2, Float3 dn_dx, Float3 dn_dy,
                       Float eta, Float3 T1_prev, Float3 T2_prev,
                       Float3 T1_next, Float3 T2_next, Bool has_prev,
                       Bool has_next, Bool is_distant, Float3 light_e1,
                       Float3 light_e2) noexcept {
      Var<ConstraintVertexEval> out{};
      const auto d_in = p_prev - p;
      const auto il = max(length(d_in), 1.0e-8f);
      const auto wi = d_in / il;

      Float3 wo = p_next; // distant unit direction when is_distant & !has_next
      Float ol = 1.0f;
      $if(has_next | !is_distant) {
        const auto d_out = p_next - p;
        ol = max(length(d_out), 1.0e-8f);
        wo = d_out / ol;
      };

      const auto s_eta = select(-1.0f, 1.0f, eta < 1.0f);
      const auto h_raw = s_eta * (wi + eta * wo);
      const auto h_len = max(length(h_raw), 1.0e-8f);
      const auto h = h_raw / h_len;

      // Orthonormal basis (s, t) around n using T1 (fallback T2 if parallel).
      const auto s_raw = T1 - n * dot(n, T1);
      const auto s_len2 = dot(s_raw, s_raw);
      const auto s_alt = T2 - n * dot(n, T2);
      const auto s_cand = select(s_alt, s_raw, s_len2 > 1.0e-6f);
      const auto s = normalize(s_cand);
      const auto t = cross(n, s);

      out.C = make_float2(dot(s, h), dot(t, h));
      out.c_err = length(out.C);
      out.h = h;
      out.wi = wi;
      out.wo = wo;
      out.orientation_ok =
          (dot(h, n) > 0.0f) & (dot(wi, n) > 0.0f) & (dot(wo, n) < 0.0f);

      const auto dwi_dq = [&](Float3 dp_prev_minus_dp) noexcept {
        return (dp_prev_minus_dp - wi * dot(wi, dp_prev_minus_dp)) / il;
      };
      const auto dwo_dq = [&](Float3 dp_next_minus_dp) noexcept {
        return select(make_float3(0.0f),
                      (dp_next_minus_dp - wo * dot(wo, dp_next_minus_dp)) / ol,
                      has_next | !is_distant);
      };
      const auto dh_from_dwi_dwo = [&](Float3 dwi, Float3 dwo) noexcept {
        const auto dh_raw = s_eta * (dwi + eta * dwo);
        return (dh_raw - h * dot(h, dh_raw)) / h_len;
      };
      const auto block_from_cols = [&](Float3 col_x, Float3 col_y) noexcept {
        return make_float4(dot(s, col_x), dot(s, col_y),
                           dot(t, col_x), dot(t, col_y));
      };

      // A_diag (q = x_i): dp_prev_minus_dp = -T1, -T2; dp_next_minus_dp = -T1, -T2.
      const auto dh_diag_x =
          dh_from_dwi_dwo(dwi_dq(-T1), dwo_dq(-T1)) - dot(h, n) * dn_dx;
      const auto dh_diag_y =
          dh_from_dwi_dwo(dwi_dq(-T2), dwo_dq(-T2)) - dot(h, n) * dn_dy;
      out.A_diag = block_from_cols(dh_diag_x, dh_diag_y);

      // A_prev (q = x_{i-1}).
      $if(has_prev) {
        const auto dh_px = dh_from_dwi_dwo(dwi_dq(T1_prev), make_float3(0.0f));
        const auto dh_py = dh_from_dwi_dwo(dwi_dq(T2_prev), make_float3(0.0f));
        out.A_prev = block_from_cols(dh_px, dh_py);
      };

      // A_next (q = x_{i+1}).
      $if(has_next) {
        const auto dh_nx = dh_from_dwi_dwo(make_float3(0.0f), dwo_dq(T1_next));
        const auto dh_ny = dh_from_dwi_dwo(make_float3(0.0f), dwo_dq(T2_next));
        out.A_next = block_from_cols(dh_nx, dh_ny);
      };

      // B_light (q = light coords (e1, e2), only at last vertex).
      $if(!has_next) {
        Float3 dwo_lx;
        Float3 dwo_ly;
        $if(is_distant) {
          dwo_lx = light_e1;
          dwo_ly = light_e2;
        }
        $else {
          dwo_lx = (light_e1 - wo * dot(wo, light_e1)) / ol;
          dwo_ly = (light_e2 - wo * dot(wo, light_e2)) / ol;
        };
        const auto dh_lx = dh_from_dwi_dwo(make_float3(0.0f), dwo_lx);
        const auto dh_ly = dh_from_dwi_dwo(make_float3(0.0f), dwo_ly);
        out.B_light = block_from_cols(dh_lx, dh_ly);
      };
      return out;
    }};
    result.set_name("mnee_eval_constraint_vertex");
    return result;
  }();
  return callable;
}

// Block-tridiagonal 2x2 Thomas solver for k <= 4 vertices:
//   A_prev[i]*x[i-1] + A_diag[i]*x[i] + A_next[i]*x[i+1] = rhs[i]
// We solve simultaneously for a 2D vector RHS (Newton step) or two columns of
// a 2x2 matrix (Y_0 = -A^{-1} B) via two RHS columns stored in float4 (c0, c1).
void solve_block_tridiagonal(
    const std::array<Float4, kMaxChainLen> &A_prev,
    const std::array<Float4, kMaxChainLen> &A_diag,
    const std::array<Float4, kMaxChainLen> &A_next,
    std::array<Float4, kMaxChainLen> &rhs_cols,
    Expr<std::uint32_t> k,
    Bool &ok,
    std::array<Float4, kMaxChainLen> &sol_cols) noexcept {
  const auto &inv2x2 = invert_2x2_callable();
  const auto &mul2x2 = mul_2x2_callable();
  std::array<Float4, kMaxChainLen> M_inv;
  std::array<Float4, kMaxChainLen> modified_diag;
  for (std::uint32_t i = 0u; i < kMaxChainLen; ++i) {
    M_inv[i] = make_float4(0.0f);
    modified_diag[i] = A_diag[i];
    sol_cols[i] = make_float4(0.0f);
  }
  ok = true;
  // Forward elimination across i = 0 .. k-1.
  for (std::uint32_t i = 0u; i < kMaxChainLen; ++i) {
    $if(ok & (UInt{i} < k)) {
      if (i > 0u) {
        const auto factor = mul2x2(A_prev[i], M_inv[i - 1u]);
        modified_diag[i] = A_diag[i] - mul2x2(factor, A_next[i - 1u]);
        rhs_cols[i] = rhs_cols[i] - mul2x2(factor, rhs_cols[i - 1u]);
      }
      const auto inv = inv2x2(modified_diag[i]);
      M_inv[i] = inv.inv;
      ok &= inv.ok;
    };
  }
  // Backward substitution.
  $if(ok) {
    // Handle exact index k - 1 down to 0 statically via conditional mux.
    for (int idx = static_cast<int>(kMaxChainLen) - 1; idx >= 0; --idx) {
      const auto ui = static_cast<std::uint32_t>(idx);
      $if(UInt{ui} < k) {
        Float4 r = rhs_cols[ui];
        if (ui + 1u < kMaxChainLen) {
          $if(UInt{ui + 1u} < k) {
            r -= mul2x2(A_next[ui], sol_cols[ui + 1u]);
          };
        }
        sol_cols[ui] = mul2x2(M_inv[ui], r);
      };
    }
  };
}

[[nodiscard]] svm::ClosurePool::Options closure_pool_options_for_kg(
    const PathCyclesSvmKernelGlobals &kg, bool spectral) noexcept {
  const auto usage = kg.closure_usage();
  return {.may_have_thin_film = kg.may_have_thin_film(),
          .may_have_anisotropic_microfacet =
              usage.has_fresnel(abi::ClosureUsage::fresnel_anisotropic),
          .may_have_custom_generalized_schlick =
              usage.has_fresnel(abi::ClosureUsage::fresnel_custom_generalized_schlick),
          .may_have_positive_exponent =
              usage.has_fresnel(abi::ClosureUsage::fresnel_positive_exponent),
          .may_have_extra_fresnel =
              usage.has_fresnel(abi::ClosureUsage::fresnel_conductor |
                                abi::ClosureUsage::fresnel_f82_tint |
                                abi::ClosureUsage::fresnel_generalized_schlick),
          .spectral = spectral};
}

// Evaluate masked SVM at a surface point (used for both caster and receiver).
void evaluate_masked_svm(
    const std::shared_ptr<LuisaSceneData> &scene,
    const PathCyclesSvmKernelGlobals &kg,
    const svm::TransformState &transforms,
    svm::ShaderData &sd,
    const svm::PathState &path_state,
    std::span<const std::uint8_t> shader_mask) noexcept {
  const auto usage = shader_mask.empty()
                         ? cycles_svm_domain_entry_usage(
                               *scene, PathCyclesSvmShaderDomain::surface_only)
                         : cycles_svm_masked_entry_usage(*scene, shader_mask);
  const auto backend_name = scene->device.backend_name();
  const bool is_simd = backend_name == "simd";
  const bool is_simd_wide = is_simd && scene->device.compute_warp_size() > 1u;
  const bool cpu_backend = backend_name == "fallback" || is_simd;
  if (sd.closure != nullptr) {
    svm_detail::surface_shader_initialize_closures(
        *sd.closure, path_state.visibility, path_state.flag);
  }
  svm::translate_svm_nodes_assume_valid(
      svm::SvmTranslationRequest{
          .kernel_globals = kg,
          .table = scene->cycles_svm->compilation.table,
          .words = *scene->cycles_svm->word_buffer,
          .shader_type = abi::SHADER_TYPE_SURFACE,
          .kernel_features = scene->cycles_svm->kernel_features,
          .node_feature_mask = svm::kernel_feature_node_mask_surface_shadow,
          .node_types_used = usage.node_types_used,
          .transform_state = transforms,
          .shader_data = sd,
          .path_state = path_state,
          .stack_size = std::max<std::size_t>(1u, usage.peak_stack_usage),
          .noise_usage = usage.noise_usage,
          .active_shader_mask = shader_mask,
          .max_groups_per_outlined_chunk =
              is_simd_wide ? std::size_t{2u}
                           : (backend_name == "fallback" ? std::size_t{16u}
                                                         : std::size_t{0u}),
          .max_chunk_jit_weight = is_simd_wide ? std::size_t{180u}
                                               : std::size_t{900u},
          .outline_chunk_dispatcher = !is_simd_wide},
      cpu_backend ? svm::SvmBackendStrategy::direct_jit
                  : svm::SvmBackendStrategy::auto_select);
}

// Compute world-space geometry, area-preserving tangent frame (T1, T2), and
// smooth normal with tangent-space derivatives from ray setup data.
void build_caster_geometry_frame(
    const svm::ShaderData &sd,
    const svm::TransformState &transforms,
    const svm::TriangleVertices &vertices,
    const svm::TriangleNormals &normals,
    Bool backfacing,
    CasterVertexVar &v) noexcept {
  const auto transform_applied =
      (sd.object_flag & static_cast<unsigned>(abi::SD_OBJECT_TRANSFORM_APPLIED)) != 0u;
  const auto o2w = transforms.fetch_object_to_world();
  const auto w2o = transforms.fetch_world_to_object();

  Float3 dp_du = vertices.v1 - vertices.v0;
  Float3 dp_dv = vertices.v2 - vertices.v0;
  $if(!transform_applied) {
    dp_du = cycles_transform::direction(o2w, dp_du);
    dp_dv = cycles_transform::direction(o2w, dp_dv);
  };

  const auto L_u = max(length(dp_du), 1.0e-8f);
  v.T1 = dp_du / L_u;
  const auto proj = dot(v.T1, dp_dv);
  const auto w = dp_dv - proj * v.T1;
  const auto L_w = max(length(w), 1.0e-8f);
  v.T2 = w / L_w;

  const auto negative_scale_applied =
      (sd.object_flag & static_cast<unsigned>(abi::SD_OBJECT_NEGATIVE_SCALE)) != 0u;
  const auto ng_raw = svm_detail::safe_normalize_cycles(
      select(cross(dp_du, dp_dv), cross(dp_dv, dp_du), negative_scale_applied));
  const auto side_sign = select(1.0f, -1.0f, backfacing);
  v.ng = side_sign * ng_raw;

  v.p = sd.P;
  v.bary = make_float2(sd.u, sd.v);
  v.backfacing = backfacing;

  const auto smooth = (sd.shader & svm::shader_smooth_normal) != 0u;
  $if(smooth) {
    auto wn0 = normals.n0;
    auto wn1 = normals.n1;
    auto wn2 = normals.n2;
    $if(!transform_applied) {
      wn0 = cycles_transform::direction_transposed(w2o, wn0);
      wn1 = cycles_transform::direction_transposed(w2o, wn1);
      wn2 = cycles_transform::direction_transposed(w2o, wn2);
    };
    const auto N_unnorm =
        (1.0f - sd.u - sd.v) * wn0 + sd.u * wn1 + sd.v * wn2;
    const auto n_mag = length(N_unnorm);
    $if(n_mag > 1.0e-8f) {
      const auto inv_len = 1.0f / n_mag;
      const auto n_raw = N_unnorm * inv_len;
      const auto outward_match =
          select(1.0f, -1.0f, dot(n_raw, ng_raw) < 0.0f);
      const auto flip_s = outward_match * side_sign;
      v.n = flip_s * n_raw;
      const auto dN_du = wn1 - wn0;
      const auto dN_dv = wn2 - wn0;
      const auto dn_du = (dN_du - n_raw * dot(n_raw, dN_du)) * inv_len;
      const auto dn_dv = (dN_dv - n_raw * dot(n_raw, dN_dv)) * inv_len;
      v.dn_dx = flip_s * (dn_du / L_u);
      v.dn_dy = flip_s * ((dn_dv - (proj / L_u) * dn_du) / L_w);
    }
    $else {
      v.n = v.ng;
      v.dn_dx = make_float3(0.0f);
      v.dn_dy = make_float3(0.0f);
    };
  }
  $else {
    v.n = v.ng;
    v.dn_dx = make_float3(0.0f);
    v.dn_dy = make_float3(0.0f);
  };
}

// Trace nearest triangle hit in `scene->accel` excluding (skip_inst, skip_prim),
// optionally restricting candidates to one Cycles object (when target_obj != invalid)
// or returning any closest surface hit (for full-scene blocking).
struct MnTraceHit {
  Bool hit{false};
  UInt inst{surface_ray::invalid_primitive};
  UInt prim{surface_ray::invalid_primitive};
  Float2 bary{make_float2(0.0f)};
  Float t{0.0f};
  UInt cycles_object{0u};
  UInt object_flags{0u};
};

[[nodiscard]] MnTraceHit trace_mnee_surface(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<luisa::compute::Ray> &ray,
    Expr<std::uint32_t> skip_inst0, Expr<std::uint32_t> skip_prim0,
    Expr<std::uint32_t> skip_inst1, Expr<std::uint32_t> skip_prim1,
    Expr<std::uint32_t> target_cycles_object,
    Expr<std::uint32_t> ignore_light_cycles_object) noexcept {
  MnTraceHit result{};
  auto committed =
      scene->accel->traverse(ray, {.visibility_mask = 0xffu})
          .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
          const auto h = candidate.hit();
          const auto skipped0 = (h->inst == skip_inst0) & (h->prim == skip_prim0);
          const auto skipped1 = (h->inst == skip_inst1) & (h->prim == skip_prim1);
          $if(!skipped0 & !skipped1) {
            const auto instance = scene->instance_buffer->read(h->inst);
            const auto cobj = instance.cycles_object_index;
            Bool accept = true;
            $if(target_cycles_object != surface_ray::invalid_primitive) {
              accept &= (cobj == target_cycles_object);
            };
            $if(ignore_light_cycles_object != surface_ray::invalid_primitive) {
              accept &= (cobj != ignore_light_cycles_object);
            };
            $if(accept) { candidate.commit(); };
          };
        })
          .trace();
  result.hit = committed->miss() == false;
  $if(result.hit) {
    result.inst = committed->inst;
    result.prim = committed->prim;
    result.bary = committed->bary;
    result.t = committed->committed_ray_t;
    const auto instance = scene->instance_buffer->read(result.inst);
    result.cycles_object = instance.cycles_object_index;
    result.object_flags =
        scene->cycles_svm->objects->object_flag_buffer->read(result.cycles_object);
  };
  return result;
}

[[nodiscard]] Float4 eval_caster_transmission(
    const PathCyclesSvmKernelGlobals &kg,
    const svm::ClosurePool &closures,
    Expr<std::uint32_t> closure_idx,
    Expr<std::uint32_t> shader_id,
    Expr<float> cos_i,
    Expr<float> adjusted_ior,
    const cycles_spectrum::SpectralContext *spectral,
    Bool &any_dispersive) noexcept {
  const auto microfacet = closures.microfacet(closure_idx);
  const auto common = microfacet.common;
  const auto is_fresnel_glass =
      microfacet.param.fresnel_type !=
      static_cast<std::uint32_t>(svm::MicrofacetFresnel::none);

  if (spectral != nullptr) {
    const Float4 rgb_up = cycles_spectrum::rgb_to_spectrum_reflectance(
        *spectral->tables, common.weight, spectral->wavelengths.lambdas);
    const Float4 base_weight = select(
        rgb_up, closures.spectral_weight(closure_idx),
        closures.has_spectral_weight(closure_idx));
    const Float abbe =
        kg.has_dispersive_materials() ? kg.shader_dispersion_abbe(shader_id)
                                      : Float{0.0f};
    const Bool is_disp =
        kg.has_dispersive_materials() ? (abbe > 0.0f) : Bool{false};
    any_dispersive |= is_disp;
    Float4 transmittance = make_float4(1.0f);
    $if(is_fresnel_glass) {
      const Float4 eta_4 =
          kg.has_dispersive_materials()
              ? cycles_spectrum::cauchy_ior(adjusted_ior, abbe,
                                            spectral->wavelengths.lambdas)
              : make_float4(adjusted_ior);
      const auto fresnel_res = svm_detail::microfacet_fresnel_spectral(
          kg, microfacet, cos_i, eta_4, *spectral, is_disp, true);
      transmittance = fresnel_res.transmittance;
    };
    return base_weight * transmittance;
  }

  Float3 transmittance = make_float3(1.0f);
  $if(is_fresnel_glass) {
    const auto fresnel_res =
        svm_detail::microfacet_fresnel(kg, microfacet, cos_i, true);
    transmittance = fresnel_res.transmittance;
  };
  return make_float4(common.weight * transmittance, 0.0f);
}

// Find the dominant refractive/glass closure in an evaluated ClosurePool.
struct SelectedRefractiveClosure {
  Bool found{false};
  UInt index{0u};
  Float eta_d{1.0f};
};

[[nodiscard]] SelectedRefractiveClosure
select_refractive_closure(const svm::ClosurePool &closures) noexcept {
  SelectedRefractiveClosure out{};
  Float best_weight = -1.0f;
  UInt i = 0u;
  $while(i < closures.count()) {
    const auto common = closures.common(i);
    const auto is_refr =
        cycles_closure::is_glass(common.type) |
        cycles_closure::is_refraction_microfacet(common.type);
    $if(is_refr) {
      const auto m = closures.microfacet(i);
      const auto ok = (abs(m.param.ior - 1.0f) > 1.0e-4f) &
                      (common.sample_weight > best_weight);
      $if(ok) {
        best_weight = common.sample_weight;
        out.found = true;
        out.index = i;
        out.eta_d = m.param.ior;
      };
    };
    i += 1u;
  };
  return out;
}

// =============================================================================
// Phase 2B: Specular Manifold Sampling (SMS — Zeltner et al. 2020)
// =============================================================================

struct SmsTangentFrame {
  Float3 s;
  Float3 t;
  Float3 ref_axis;
  Float inv_s;
};

[[nodiscard]] SmsTangentFrame smooth_sms_tangent_frame(
    Float3 n, Float3 ref_T, Float3 T1, Float3 T2) noexcept {
  const auto s0 = ref_T - n * dot(n, ref_T);
  const auto s1 = T1 - n * dot(n, T1);
  const auto s2 = T2 - n * dot(n, T2);
  const auto use_ref = dot(s0, s0) > 1.0e-6f;
  const auto use_t1 = dot(s1, s1) > 1.0e-6f;
  const auto ref_axis = select(select(T2, T1, use_t1), ref_T, use_ref);
  const auto s_raw = select(select(s2, s1, use_t1), s0, use_ref);
  const auto inv_s = rsqrt(max(dot(s_raw, s_raw), 1.0e-12f));
  const auto s = s_raw * inv_s;
  const auto t = cross(n, s);
  return {.s = s, .t = t, .ref_axis = ref_axis, .inv_s = inv_s};
}

[[nodiscard]] const Callable<ConstraintVertexEval(SmsConstraintInput)> &
eval_sms_constraint_vertex_callable() noexcept {
  static auto callable = []() noexcept {
    Callable result{[](Var<SmsConstraintInput> in) noexcept {
      Var<ConstraintVertexEval> out{};
      const auto d_in = in.p_prev - in.p;
      const auto il = max(length(d_in), 1.0e-8f);
      const auto wi = d_in / il;

      Float3 wo = in.p_next;
      Float ol = 1.0f;
      $if(in.has_next | !in.is_distant) {
        const auto d_out = in.p_next - in.p;
        ol = max(length(d_out), 1.0e-8f);
        wo = d_out / ol;
      };

      const auto eta_eff = select(in.eta, 1.0f, in.is_reflection);
      const auto s_eta =
          select(select(-1.0f, 1.0f, in.eta < 1.0f), 1.0f, in.is_reflection);
      const auto H = s_eta * (wi + eta_eff * wo);
      const auto h_len = max(length(H), 1.0e-8f);
      const auto h = H / h_len;

      const auto frame =
          smooth_sms_tangent_frame(in.n, in.ref_T, in.T1, in.T2);
      const auto s = frame.s;
      const auto t = frame.t;
      const auto hz = sqrt(max(1.0f - dot(in.h_offset, in.h_offset), 1.0e-6f));

      out.C = make_float2(dot(s, H) * hz - in.h_offset.x * dot(in.n, H),
                          dot(t, H) * hz - in.h_offset.y * dot(in.n, H));
      out.c_err = length(make_float2(dot(s, h) - in.h_offset.x,
                                     dot(t, h) - in.h_offset.y));
      out.h = h;
      out.wi = wi;
      out.wo = wo;

      const auto refl_ok = (dot(h, in.n) > 1.0e-4f) & (dot(wi, in.n) > 0.0f) &
                           (dot(wo, in.n) > 0.0f) & (dot(wi, h) > 0.0f) &
                           (dot(wo, h) > 0.0f);
      const auto refr_ok = (dot(h, in.n) > 1.0e-4f) & (dot(wi, in.n) > 0.0f) &
                           (dot(wo, in.n) < 0.0f) & (dot(wi, h) > 0.0f) &
                           (dot(wo, h) < 0.0f);
      out.orientation_ok = select(refr_ok, refl_ok, in.is_reflection);

      const auto dwi_dq = [&](Float3 dp_prev_minus_dp) noexcept {
        return (dp_prev_minus_dp - wi * dot(wi, dp_prev_minus_dp)) / il;
      };
      const auto dwo_dq = [&](Float3 dp_next_minus_dp) noexcept {
        return select(make_float3(0.0f),
                      (dp_next_minus_dp - wo * dot(wo, dp_next_minus_dp)) / ol,
                      in.has_next | !in.is_distant);
      };
      const auto dH_from_dwi_dwo = [&](Float3 dwi, Float3 dwo) noexcept {
        return s_eta * (dwi + eta_eff * dwo);
      };
      const auto block_from_dH_cols = [&](Float3 col_x, Float3 col_y) noexcept {
        return make_float4(
            dot(s, col_x) * hz - in.h_offset.x * dot(in.n, col_x),
            dot(s, col_y) * hz - in.h_offset.x * dot(in.n, col_y),
            dot(t, col_x) * hz - in.h_offset.y * dot(in.n, col_x),
            dot(t, col_y) * hz - in.h_offset.y * dot(in.n, col_y));
      };

      const auto compute_ds = [&](Float3 dn) noexcept {
        const auto dsr =
            -dn * dot(in.n, frame.ref_axis) - in.n * dot(dn, frame.ref_axis);
        return (dsr - s * dot(s, dsr)) * frame.inv_s;
      };
      const auto ds_dx = compute_ds(in.dn_dx);
      const auto ds_dy = compute_ds(in.dn_dy);
      const auto dt_dx = cross(in.dn_dx, s) + cross(in.n, ds_dx);
      const auto dt_dy = cross(in.dn_dy, s) + cross(in.n, ds_dy);

      const auto dH_dx = dH_from_dwi_dwo(dwi_dq(-in.T1), dwo_dq(-in.T1));
      const auto dH_dy = dH_from_dwi_dwo(dwi_dq(-in.T2), dwo_dq(-in.T2));
      out.A_diag = make_float4(
          (dot(s, dH_dx) + dot(ds_dx, H)) * hz -
              in.h_offset.x * (dot(in.n, dH_dx) + dot(in.dn_dx, H)),
          (dot(s, dH_dy) + dot(ds_dy, H)) * hz -
              in.h_offset.x * (dot(in.n, dH_dy) + dot(in.dn_dy, H)),
          (dot(t, dH_dx) + dot(dt_dx, H)) * hz -
              in.h_offset.y * (dot(in.n, dH_dx) + dot(in.dn_dx, H)),
          (dot(t, dH_dy) + dot(dt_dy, H)) * hz -
              in.h_offset.y * (dot(in.n, dH_dy) + dot(in.dn_dy, H)));

      $if(in.has_prev) {
        const auto dH_px =
            dH_from_dwi_dwo(dwi_dq(in.T1_prev), make_float3(0.0f));
        const auto dH_py =
            dH_from_dwi_dwo(dwi_dq(in.T2_prev), make_float3(0.0f));
        out.A_prev = block_from_dH_cols(dH_px, dH_py);
      };

      $if(in.has_next) {
        const auto dH_nx =
            dH_from_dwi_dwo(make_float3(0.0f), dwo_dq(in.T1_next));
        const auto dH_ny =
            dH_from_dwi_dwo(make_float3(0.0f), dwo_dq(in.T2_next));
        out.A_next = block_from_dH_cols(dH_nx, dH_ny);
      };

      $if(!in.has_next) {
        Float3 dwo_lx;
        Float3 dwo_ly;
        $if(in.is_distant) {
          dwo_lx = in.light_e1;
          dwo_ly = in.light_e2;
        }
        $else {
          dwo_lx = (in.light_e1 - wo * dot(wo, in.light_e1)) / ol;
          dwo_ly = (in.light_e2 - wo * dot(wo, in.light_e2)) / ol;
        };
        const auto dH_lx = dH_from_dwi_dwo(make_float3(0.0f), dwo_lx);
        const auto dH_ly = dH_from_dwi_dwo(make_float3(0.0f), dwo_ly);
        out.B_light = block_from_dH_cols(dH_lx, dH_ly);
      };
      return out;
    }};
    result.set_name("sms_eval_constraint_vertex");
    return result;
  }();
  return callable;
}

struct SmsClosureSelection {
  Bool has_refl{false};
  UInt refl_index{0u};
  Float refl_alpha2{0.0f};
  Bool refl_is_beckmann{false};
  Float refl_approx_f{0.0f};

  Bool has_refr{false};
  UInt refr_index{0u};
  Float refr_eta_d{1.0f};
  Float refr_alpha2{0.0f};
  Bool refr_is_beckmann{false};
};

[[nodiscard]] SmsClosureSelection
select_sms_closures(const PathCyclesSvmKernelGlobals &kg,
                    const svm::ClosurePool &closures,
                    Expr<float> cos_wi_macro) noexcept {
  SmsClosureSelection out{};
  Float best_refl_w = -1.0f;
  Float best_refr_w = -1.0f;
  UInt i = 0u;
  $while(i < closures.count()) {
    const auto common = closures.common(i);
    const auto is_glass_cl = cycles_closure::is_glass(common.type);
    const auto is_refl_cl =
        cycles_closure::is_reflection_microfacet(common.type) | is_glass_cl;
    const auto is_refr_cl =
        cycles_closure::is_refraction_microfacet(common.type) | is_glass_cl;
    $if(is_refl_cl | is_refr_cl) {
      const auto m = closures.microfacet(i);
      const auto alpha2 = m.param.alpha_x * m.param.alpha_y;
      const auto rough_ok = alpha2 <= kMaxSmsAlphaSquared;
      const auto is_beckmann = cycles_closure::is_beckmann_microfacet(common.type);
      $if(is_refl_cl & rough_ok & (common.sample_weight > best_refl_w)) {
        Float fresnel_approx = 1.0f;
        $if(is_glass_cl) {
          const auto fr =
              svm_detail::microfacet_fresnel(kg, m, max(cos_wi_macro, 1.0e-4f),
                                             true);
          const auto r_l = (fr.reflectance.x + fr.reflectance.y + fr.reflectance.z) *
                           (1.0f / 3.0f);
          const auto t_l = (fr.transmittance.x + fr.transmittance.y +
                            fr.transmittance.z) *
                           (1.0f / 3.0f);
          const auto denom = r_l + t_l;
          fresnel_approx = select(0.5f, r_l / denom, denom > 1.0e-6f);
        };
        $if(!is_glass_cl | (fresnel_approx > 1.0e-5f)) {
          best_refl_w = common.sample_weight;
          out.has_refl = true;
          out.refl_index = i;
          out.refl_alpha2 = alpha2;
          out.refl_is_beckmann = is_beckmann;
          out.refl_approx_f = fresnel_approx;
        };
      };
      $if(is_refr_cl & rough_ok & (abs(m.param.ior - 1.0f) > 1.0e-4f) &
          (common.sample_weight > best_refr_w)) {
        best_refr_w = common.sample_weight;
        out.has_refr = true;
        out.refr_index = i;
        out.refr_eta_d = m.param.ior;
        out.refr_alpha2 = alpha2;
        out.refr_is_beckmann = is_beckmann;
      };
    };
    i += 1u;
  };
  return out;
}

[[nodiscard]] Float2 sample_sms_h_offset(Expr<float> alpha2,
                                         Expr<bool> is_beckmann,
                                         Expr<luisa::float2> u_rand) noexcept {
  Float2 out = make_float2(0.0f);
  $if(alpha2 > cycles_closure::microfacet_singular_alpha_product) {
    const auto u1 = clamp(u_rand.x, 1.0e-6f, 1.0f - 1.0e-6f);
    const auto u2 = clamp(u_rand.y, 1.0e-6f, 1.0f - 1.0e-6f);
    Float cos2_theta = 1.0f;
    $if(is_beckmann) {
      const auto tan2 = -alpha2 * log(max(1.0f - u1, 1.0e-7f));
      cos2_theta = 1.0f / (1.0f + max(tan2, 0.0f));
    }
    $else {
      cos2_theta = (1.0f - u1) / max(1.0f + (alpha2 - 1.0f) * u1, 1.0e-12f);
    };
    cos2_theta = clamp(cos2_theta, 1.0e-6f, 1.0f);
    const auto sin_theta = sqrt(max(1.0f - cos2_theta, 0.0f));
    const auto phi = 6.28318530717958647692f * u2;
    out = make_float2(sin_theta * cos(phi), sin_theta * sin(phi));
  };
  return out;
}

[[nodiscard]] Float microfacet_lambda_sms(Expr<float> alpha2,
                                          Expr<float> cos_theta,
                                          Expr<bool> is_beckmann) noexcept {
  const auto c2 = max(cos_theta * cos_theta, 1.0e-8f);
  const auto sq_tan = alpha2 * max(1.0f / c2 - 1.0f, 0.0f);
  Float out = 0.0f;
  $if(is_beckmann) {
    $if(sq_tan >= 0.39f) {
      const auto a = rsqrt(sq_tan);
      out = ((0.396f * a - 1.259f) * a + 1.0f) / ((2.181f * a + 3.535f) * a);
    };
  }
  $else {
    out = 0.5f * (sqrt(1.0f + sq_tan) - 1.0f);
  };
  return out;
}

[[nodiscard]] Float microfacet_ndf_sms(Expr<float> alpha2,
                                       Expr<float> cos_h,
                                       Expr<bool> is_beckmann) noexcept {
  const auto c2 = min(max(cos_h * cos_h, 1.0e-8f), 1.0f);
  const auto s2 = 1.0f - c2;
  Float out = 0.0f;
  $if(is_beckmann) {
    out = 1.0f / (exp(s2 / max(c2 * alpha2, 1.0e-12f)) *
                  3.14159265358979323846f * alpha2 * c2 * c2);
  }
  $else {
    const auto denom = s2 + alpha2 * c2;
    out = alpha2 / (3.14159265358979323846f * denom * denom);
  };
  return out;
}

[[nodiscard]] Float4 eval_sms_caster_weight(
    const PathCyclesSvmKernelGlobals &kg,
    const svm::ClosurePool &closures,
    Expr<std::uint32_t> closure_idx,
    Expr<std::uint32_t> shader_id,
    Expr<bool> is_reflection,
    Expr<float> cos_ih,
    Expr<float> cos_in,
    Expr<float> cos_on,
    Expr<float> cos_hn,
    Expr<float> adjusted_ior,
    Expr<float> seed_alpha2,
    Expr<bool> seed_is_beckmann,
    const cycles_spectrum::SpectralContext *spectral,
    Bool &any_refr_dispersive) noexcept {
  const auto microfacet = closures.microfacet(closure_idx);
  const auto common = microfacet.common;
  const auto sol_alpha2 = microfacet.param.alpha_x * microfacet.param.alpha_y;
  const auto sol_is_beckmann =
      cycles_closure::is_beckmann_microfacet(common.type);
  const auto is_fresnel =
      microfacet.param.fresnel_type !=
      static_cast<std::uint32_t>(svm::MicrofacetFresnel::none);

  // Glossy microfacet weight matching Cycles microfacet_eval / sample:
  //   weight = base * energy_scale * F * G_2 * |wi.h| / (|wi.n| * |h.n|) * (D_sol / D_seed)
  Float glossy_factor = 1.0f;
  $if(sol_alpha2 > cycles_closure::microfacet_singular_alpha_product) {
    const auto l_i = microfacet_lambda_sms(sol_alpha2, cos_in, sol_is_beckmann);
    const auto l_o = microfacet_lambda_sms(sol_alpha2, cos_on, sol_is_beckmann);
    const auto G2 = 1.0f / (1.0f + l_i + l_o);
    const auto energy_scale =
        select(microfacet.param.energy_scale, 1.0f, sol_is_beckmann);
    Float ndf_ratio = 1.0f;
    $if((seed_alpha2 > cycles_closure::microfacet_singular_alpha_product) &
        ((abs(sol_alpha2 - seed_alpha2) > 1.0e-6f) |
         (sol_is_beckmann != seed_is_beckmann))) {
      const auto d_sol = microfacet_ndf_sms(sol_alpha2, cos_hn, sol_is_beckmann);
      const auto d_seed =
          microfacet_ndf_sms(seed_alpha2, cos_hn, seed_is_beckmann);
      ndf_ratio = select(0.0f, d_sol / d_seed, d_seed > 1.0e-12f);
    };
    glossy_factor = energy_scale * G2 *
                    max(cos_ih, 0.0f) / max(abs(cos_in) * abs(cos_hn), 1.0e-8f) *
                    ndf_ratio;
  };

  if (spectral != nullptr) {
    const Float4 rgb_up = cycles_spectrum::rgb_to_spectrum_reflectance(
        *spectral->tables, common.weight, spectral->wavelengths.lambdas);
    const Float4 base_weight = select(
        rgb_up, closures.spectral_weight(closure_idx),
        closures.has_spectral_weight(closure_idx));
    const Float abbe =
        kg.has_dispersive_materials() ? kg.shader_dispersion_abbe(shader_id)
                                      : Float{0.0f};
    const Bool is_disp =
        kg.has_dispersive_materials() ? (abbe > 0.0f) : Bool{false};
    any_refr_dispersive |= (is_disp & !is_reflection);
    Float4 fresnel_val = make_float4(1.0f);
    $if(is_fresnel) {
      const Float4 eta_4 =
          kg.has_dispersive_materials()
              ? cycles_spectrum::cauchy_ior(adjusted_ior, abbe,
                                            spectral->wavelengths.lambdas)
              : make_float4(adjusted_ior);
      const auto fresnel_res = svm_detail::microfacet_fresnel_spectral(
          kg, microfacet, max(cos_ih, 0.0f), eta_4, *spectral, is_disp, true);
      fresnel_val = select(fresnel_res.transmittance, fresnel_res.reflectance,
                           is_reflection);
    };
    return base_weight * fresnel_val * glossy_factor;
  }

  Float3 fresnel_val = make_float3(1.0f);
  $if(is_fresnel) {
    const auto fresnel_res = svm_detail::microfacet_fresnel(
        kg, microfacet, max(cos_ih, 0.0f), true);
    fresnel_val = select(fresnel_res.transmittance, fresnel_res.reflectance,
                         is_reflection);
  };
  return make_float4(common.weight * fresnel_val * glossy_factor, 0.0f);
}

[[nodiscard]] Bool manifold_jacobian_within_bounds(
    Expr<float> cos_v0, Expr<float> det_Y0, Expr<float> G_term) noexcept {
  constexpr float kMaxManifoldJacobian = 1.0e4f;
  return (cos_v0 > 1.0e-4f) & (det_Y0 <= kMaxManifoldJacobian) &
         (G_term <= kMaxManifoldJacobian) &
         !(luisa::compute::isnan(G_term) | luisa::compute::isinf(G_term));
}

[[nodiscard]] Float3 eval_xyz3_caustic_raw_rgb(
    const cycles_spectrum::SpectralTablesGpu &tables,
    const cycles_spectrum::SpectralContext &spec_ctx,
    Expr<luisa::float4> path_throughput_and_cf,
    Expr<luisa::float4> bsdf_spec,
    Expr<luisa::float4> chain_spec,
    Expr<luisa::float4> light_factor_spec,
    Expr<float> g_term,
    Expr<float> hero_lambda,
    Expr<bool> secondary_terminated,
    Expr<bool> any_dispersive) noexcept {
  const auto prod_eval4 = bsdf_spec * chain_spec * light_factor_spec * g_term;
  const auto prod_xyz3 =
      cycles_spectrum::Spectrum::from_eval4(&spec_ctx, prod_eval4).xyz3();
  const auto collapsed_prev_xyz3 = path_throughput_and_cf.xyz() * prod_eval4.x;
  const auto t_h = cycles_spectrum::xyz3_hero(path_throughput_and_cf.xyz(),
                                              *spec_ctx.hero_index);
  const auto cf = path_throughput_and_cf.w;
  const auto w_raw = cycles_spectrum::xyz3_collapse_weight(tables, hero_lambda);
  const auto collapsed_local_xyz3 = (t_h * cf * prod_eval4.x) * w_raw;
  const auto base_xyz3 = select(path_throughput_and_cf.xyz() * prod_xyz3,
                                collapsed_local_xyz3, any_dispersive);
  const auto raw_xyz3 =
      select(base_xyz3, collapsed_prev_xyz3, secondary_terminated);
  return cycles_spectrum::xyz3_to_rgb(tables, raw_xyz3);
}

void compute_xyz3_bsdf_component_weights(
    const cycles_spectrum::SpectralTablesGpu &tables,
    const cycles_spectrum::SpectralContext &spec_ctx,
    const LightComponentRatioCallable &light_component_ratio,
    Expr<luisa::float4> bsdf_spec,
    Expr<luisa::float4> diff_spec,
    Expr<luisa::float4> gloss_spec,
    Float3 &out_diff_w,
    Float3 &out_gloss_w) noexcept {
  const auto bsdf_rgb = cycles_spectrum::xyz3_reflectance_to_rgb(
      tables,
      cycles_spectrum::Spectrum::from_eval4(&spec_ctx, bsdf_spec).xyz3(),
      *spec_ctx.lambdas_xyz);
  const auto diff_rgb = cycles_spectrum::xyz3_reflectance_to_rgb(
      tables,
      cycles_spectrum::Spectrum::from_eval4(&spec_ctx, diff_spec).xyz3(),
      *spec_ctx.lambdas_xyz);
  const auto gloss_rgb = cycles_spectrum::xyz3_reflectance_to_rgb(
      tables,
      cycles_spectrum::Spectrum::from_eval4(&spec_ctx, gloss_spec).xyz3(),
      *spec_ctx.lambdas_xyz);
  out_diff_w = light_component_ratio(diff_rgb, bsdf_rgb);
  out_gloss_w = light_component_ratio(gloss_rgb, bsdf_rgb);
}

[[nodiscard]] Float3 evaluate_sms_caustics(
    const CausticsTaskEvaluator &eval,
    const Var<CausticTaskCall> &task,
    const Var<RenderKernelParameters> &parameters,
    Bool &out_valid,
    Float3 *out_diffuse_weight,
    Float3 *out_glossy_weight) noexcept {
  const auto &scene = eval.scene;
  const bool need_weights =
      out_diffuse_weight != nullptr && out_glossy_weight != nullptr;
  const bool is_xyz3 = eval.spectral_mode == contract::SpectralMode::xyz3;
  Float3 result_rgb = make_float3(0.0f);
  Bool eval_valid = false;
  Float3 eval_diff_w = make_float3(0.0f);
  Float3 eval_gloss_w = make_float3(0.0f);

  $outline_with_name(is_xyz3 ? "sms_evaluate_xyz3"
                             : (eval.spectral_rendering ? "sms_evaluate_spectral"
                                                        : "sms_evaluate_rgb")) {
    const auto load_f4 = [](const std::array<Float4, kMaxChainLen> &arr,
                            Expr<std::uint32_t> idx) noexcept -> Float4 {
      Float4 res = arr[0u];
      for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
        res = select(res, arr[k], idx == k);
      }
      return res;
    };
    const auto store_f4 = [](std::array<Float4, kMaxChainLen> &arr,
                             Expr<std::uint32_t> idx,
                             Expr<luisa::float4> val) noexcept {
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        $if(idx == k) { arr[k] = val; };
      }
    };
    const auto load_f2 = [](const std::array<Float2, kMaxChainLen> &arr,
                            Expr<std::uint32_t> idx) noexcept -> Float2 {
      Float2 res = arr[0u];
      for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
        res = select(res, arr[k], idx == k);
      }
      return res;
    };
    const auto store_f2 = [](std::array<Float2, kMaxChainLen> &arr,
                             Expr<std::uint32_t> idx,
                             Expr<luisa::float2> val) noexcept {
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        $if(idx == k) { arr[k] = val; };
      }
    };
    static_cast<void>(store_f2);
    const auto load_vertex =
        [](const std::array<CasterVertexVar, kMaxChainLen> &arr,
           Expr<std::uint32_t> idx) noexcept -> CasterVertexVar {
      CasterVertexVar res = arr[0u];
      for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
        $if(idx == k) { res = arr[k]; };
      }
      return res;
    };
    const auto store_vertex =
        [](std::array<CasterVertexVar, kMaxChainLen> &arr,
           Expr<std::uint32_t> idx,
           const CasterVertexVar &val) noexcept {
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        $if(idx == k) { arr[k] = val; };
      }
    };

    // Per-task 32-bit PCG PRNG.
    UInt rng_state = cycles_noise::hash_uint3(
        task.rng_hash ^ (task.pixel * 0x9e3779b1u + 0x517cc1b7u),
        task.rng_offset ^ (task.path_depth * 0x85ebca6bu),
        task.sample_index ^ 0x27d4eb2fu) | 1u;
    const auto next_float = [&]() noexcept -> Float {
      rng_state = rng_state * 747796405u + 2891336453u;
      const auto word =
          ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
      const auto bits = (word >> 22u) ^ word;
      return min(bits.cast<float>() * (1.0f / 4294967296.0f), 0.99999994f);
    };

    // Fixed world reference tangent direction and fixed canonical microfacet
    // random draws shared across ALL trials of this task.
    const auto z_ref = 1.0f - 2.0f * next_float();
    const auto r_ref = sqrt(max(1.0f - z_ref * z_ref, 0.0f));
    const auto phi_ref = 6.28318530717958647692f * next_float();
    const Float3 ref_T =
        make_float3(r_ref * cos(phi_ref), r_ref * sin(phi_ref), z_ref);
    std::array<Float2, kMaxChainLen> u_h;
    for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
      u_h[k] = make_float2(next_float(), next_float());
    }

    // Optional spectral context reconstructed from scalar u_lambda.
    std::optional<cycles_spectrum::SpectralContext> spec_ctx;
    Float hero_lambda = 550.0f;
    Bool secondary_terminated = false;
    if (eval.spectral_rendering && eval.spectral_tables) {
      secondary_terminated =
          (task.path_flags &
           cycles_path_state::flag_spectral_secondary_terminated) != 0u;
      spec_ctx.emplace(cycles_spectrum::make_spectral_context(
          eval.spectral_tables.get(), eval.spectral_mode, task.u_lambda,
          secondary_terminated));
      hero_lambda = is_xyz3
                        ? cycles_spectrum::xyz3_hero(*spec_ctx->lambdas_xyz,
                                                     *spec_ctx->hero_index)
                        : spec_ctx->wavelengths.lambdas.x;
    }

    const PathCyclesSvmKernelGlobals kg_caster{
        scene,
        parameters,
        eval.camera_projection,
        true,
        true,
        PathCyclesSvmShaderDomain::surface_only,
        scene->caustics_caster_shader_mask,
        false,
        nullptr,
        nullptr,
        eval.spectral_rendering,
        eval.spectral_mode};

    const svm_detail::ClosureTypeMask caster_closure_types =
        svm_detail::closure_types_for_kernel_features(
            scene->cycles_svm->kernel_features);
    const auto caster_pool_opts =
        closure_pool_options_for_kg(kg_caster, eval.spectral_rendering);
    const std::size_t closure_cap = std::clamp<std::size_t>(
        std::max<std::size_t>(
            scene->cycles_svm->compilation.max_surface_closures, 2u),
        2u, svm::maximum_closure_capacity);

    // Caster surface evaluation helper supporting both seeding/projection and
    // Step D closure evaluation.
    const auto eval_sms_caster =
        [&](const MnTraceHit &hit, const Var<luisa::compute::Ray> &ray,
            CasterVertexVar &v, UInt mode_select,
            Expr<bool> want_fixed_refl, Expr<bool> force_backfacing_from_v,
            Expr<bool> fixed_backfacing, Expr<bool> reject_wrong_side,
            SmsClosureSelection *out_sel, Float *out_alpha2,
            Bool *out_is_beckmann, Float4 *out_weight,
            Bool &any_disp_out, Expr<bool> eval_weight_at_conv,
            Expr<float> cos_ih_eval, Expr<float> cos_hn_eval,
            Expr<float3> wo_eval, Expr<float> seed_alpha2_eval,
            Expr<bool> seed_is_beck_eval) noexcept -> Bool {
      Bool found = false;
      svm::ClosurePool closures{closure_cap, caster_pool_opts};
      svm::TriangleVertices vertices{};
      svm::TriangleNormals normals{};
      Var<InstanceGpu> inst_gpu{};
      const auto lcg = cycles_noise::hash_uint3(
          task.rng_hash ^ 0x6c8e9cf5u, task.rng_offset, task.sample_index);
      auto setup = setup_cycles_svm_ray_shader_data(
          scene, kg_caster, ray, hit.inst, hit.prim, hit.bary, hit.t,
          task.ray_dP, task.ray_dD, task.ray_time, lcg, parameters,
          &inst_gpu, &vertices, &normals);
      auto &sd = setup.shader_data;
      sd.closure = &closures;
      if (spec_ctx) {
        sd.spectral = *spec_ctx;
      }
      const svm::PathState state{
          svm::path_ray_visibility_diffuse |
              svm::path_ray_visibility_transmit,
          0u,
          task.path_depth + 1u,
          task.transparent_depth,
          task.diffuse_depth,
          task.glossy_depth,
          task.transmission_depth,
          task.portal_depth};
      evaluate_masked_svm(scene, kg_caster, setup.transforms, sd, state,
                          scene->caustics_caster_shader_mask);
      const auto actual_backfacing =
          (sd.flag & static_cast<unsigned>(abi::SD_BACKFACING)) != 0u;
      const auto side_mismatch =
          reject_wrong_side & (actual_backfacing != fixed_backfacing);
      $if(!side_mismatch) {
        const auto bf =
            select(actual_backfacing, fixed_backfacing, force_backfacing_from_v);
        const auto cos_macro = abs(dot(-ray->direction(), sd.Ng));
        const auto sel = select_sms_closures(kg_caster, closures, cos_macro);
        if (out_sel != nullptr) {
          *out_sel = sel;
        }
        // mode_select: 0 = caller decides from out_sel, 1 = require want_fixed_refl.
        Bool has_mode = sel.has_refl | sel.has_refr;
        $if(mode_select == 1u) {
          has_mode = select(sel.has_refr, sel.has_refl, want_fixed_refl);
        };
        $if(has_mode) {
          found = true;
          build_caster_geometry_frame(sd, setup.transforms, vertices, normals,
                                      bf, v);
          v.inst = hit.inst;
          v.prim = hit.prim;
          v.cycles_object = hit.cycles_object;

          const auto is_refl = want_fixed_refl;
          Float eta_d = select(sel.refr_eta_d, 1.0f, is_refl);
          $if(!is_refl & force_backfacing_from_v &
              (actual_backfacing != fixed_backfacing)) {
            eta_d = 1.0f / max(eta_d, 1.0e-5f);
          };
          if (spec_ctx) {
            const Float abbe =
                kg_caster.has_dispersive_materials()
                    ? kg_caster.shader_dispersion_abbe(sd.shader)
                    : Float{0.0f};
            v.eta = cycles_spectrum::cauchy_ior(
                        eta_d, abbe, make_float4(hero_lambda))
                        .x;
          } else {
            v.eta = eta_d;
          }
          if (out_alpha2 != nullptr && out_is_beckmann != nullptr) {
            *out_alpha2 = select(sel.refr_alpha2, sel.refl_alpha2, is_refl);
            *out_is_beckmann =
                select(sel.refr_is_beckmann, sel.refl_is_beckmann, is_refl);
          }
          if (out_weight != nullptr) {
            $if(eval_weight_at_conv) {
              const auto c_idx =
                  select(sel.refr_index, sel.refl_index, is_refl);
              const auto cos_in_val = abs(dot(-ray->direction(), v.n));
              const auto cos_on_val = abs(dot(wo_eval, v.n));
              *out_weight = eval_sms_caster_weight(
                  kg_caster, closures, c_idx, sd.shader, is_refl, cos_ih_eval,
                  cos_in_val, cos_on_val, cos_hn_eval, eta_d,
                  seed_alpha2_eval, seed_is_beck_eval,
                  spec_ctx ? &*spec_ctx : nullptr, any_disp_out);
            };
          }
        };
      };
      return found;
    };

    const Var<LightGpu> light = scene->light_buffer->read(task.light_index);
    const auto is_distant =
        light.type == static_cast<std::uint32_t>(LightType::distant);
    const auto is_area =
        light.type == static_cast<std::uint32_t>(LightType::area);
    const auto &caster_tris = *scene->caustics_caster_triangles;
    const auto &caster_objs = *scene->caustics_caster_objects;
    const std::uint32_t obj_count = scene->caustics_caster_object_count;

    // Receiver front-facing geometric normal for above-horizon caster culling.
    const Float3 recv_n =
        select(-task.receiver_ng, task.receiver_ng,
               dot(task.receiver_ng, task.receiver_wi) >= 0.0f);
    const auto caster_solid_angle_weight =
        [&](const Var<CausticsCasterObjectGpu> &obj) noexcept -> Float {
          const Float3 obj_center = obj.center_radius.xyz();
          const Float obj_radius = obj.center_radius.w;
          const Float3 to_c = obj_center - task.receiver_p;
          const Float dist2 =
              max(length_squared(to_c), obj_radius * obj_radius);
          const Bool above_horizon =
              (dot(to_c, recv_n) + obj_radius) > -1.0e-3f;
          return select(0.0f, obj.area / max(dist2, 1.0e-6f), above_horizon);
        };

    Float sum_omega = 0.0f;
    UInt vis_count = 0u;
    if (obj_count == 1u) {
      const auto w0 = caster_solid_angle_weight(caster_objs->read(0u));
      $if(w0 > 0.0f) {
        sum_omega = w0;
        vis_count = 1u;
      };
    } else {
      UInt oi = 0u;
      $while(oi < obj_count) {
        const auto w = caster_solid_angle_weight(caster_objs->read(oi));
        $if(w > 0.0f) {
          sum_omega += w;
          vis_count += 1u;
        };
        oi += 1u;
      };
    }

    const auto sample_caster_object =
        [&](Expr<float> u_obj, UInt &out_offset, UInt &out_count,
            Float &out_pdf) noexcept {
          if (obj_count == 1u) {
            const auto obj0 = caster_objs->read(0u);
            out_offset = obj0.tri_offset;
            out_count = obj0.tri_count;
            out_pdf = 1.0f;
            return;
          }
          constexpr float kSolidAngleMix = 0.85f;
          constexpr float kUniformMix = 0.15f;
          const Float inv_vis = 1.0f / max(vis_count.cast<float>(), 1.0f);
          const Float inv_sum = 1.0f / max(sum_omega, 1.0e-20f);
          Float cdf = 0.0f;
          Bool picked = false;
          out_offset = 0u;
          out_count = 1u;
          out_pdf = inv_vis;
          UInt oi = 0u;
          $while(oi < obj_count) {
            const auto obj = caster_objs->read(oi);
            const auto w = caster_solid_angle_weight(obj);
            $if(w > 0.0f) {
              const Float pm =
                  kSolidAngleMix * (w * inv_sum) + kUniformMix * inv_vis;
              cdf += pm;
              $if(!picked) {
                out_offset = obj.tri_offset;
                out_count = obj.tri_count;
                out_pdf = pm;
                $if(u_obj <= cdf) { picked = true; };
              };
            };
            oi += 1u;
          };
        };

    // Sample a random point within one caster object weighted by world area.
    const auto sample_caster_seed_point =
        [&](Expr<std::uint32_t> tri_offset, Expr<std::uint32_t> tri_count_obj,
            Float3 &out_p, UInt &out_cobj) noexcept {
          const auto u_tri = next_float();
          const auto ub0 = next_float();
          const auto ub1 = next_float();
          UInt lo = tri_offset;
          UInt hi = tri_offset + tri_count_obj - 1u;
          $while(lo < hi) {
            const auto mid = (lo + hi) >> 1u;
            const auto c = caster_tris->read(mid).cdf;
            $if(c < u_tri) { lo = mid + 1u; }
            $else { hi = mid; };
          };
          const auto rec = caster_tris->read(lo);
          out_cobj = rec.cycles_object_index;
          const auto inst = scene->instance_buffer->read(rec.instance_id);
          const auto cprim = inst.cycles_primitive_offset + rec.primitive_id;
          const auto indices = kg_caster.triangle_vertex_indices(cprim);
          const auto obj =
              scene->cycles_svm->objects->object_buffer->read(out_cobj);
          const auto obj_flag =
              scene->cycles_svm->objects->object_flag_buffer->read(out_cobj);
          const auto pos_offset = obj.position_offset.cast<unsigned>();
          const auto &vbuf = scene->cycles_svm->geometry->triangle_vertex_buffer;
          const auto unpack_v = [](Var<abi::packed_float3> pv) noexcept {
            return make_float3(pv.x, pv.y, pv.z);
          };
          Float3 p0 = unpack_v(vbuf->read(pos_offset + indices.x));
          Float3 p1 = unpack_v(vbuf->read(pos_offset + indices.y));
          Float3 p2 = unpack_v(vbuf->read(pos_offset + indices.z));
          const auto tfm_applied =
              (obj_flag &
               static_cast<unsigned>(abi::SD_OBJECT_TRANSFORM_APPLIED)) != 0u;
          $if(!tfm_applied) {
            const auto &t = obj.tfm;
            const auto o2w = svm_detail::transform_from_rows(
                make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
                make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
                make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
            p0 = cycles_transform::point(o2w, p0);
            p1 = cycles_transform::point(o2w, p1);
            p2 = cycles_transform::point(o2w, p2);
          };
          const auto s_b = sqrt(ub0);
          const auto b0 = 1.0f - s_b;
          const auto b1 = ub1 * s_b;
          const auto b2 = 1.0f - b0 - b1;
          out_p = b0 * p0 + b1 * p1 + b2 * p2;
        };

    // Unbiased vs Biased state preserved across the single trial loop.
    const std::uint32_t max_bernoulli =
        eval.sms_unbiased
            ? std::max<std::uint32_t>(eval.sms_max_trials * 8u, 64u)
            : std::max<std::uint32_t>(eval.sms_max_trials, 1u);
    const std::uint32_t max_loop_iters =
        eval.sms_unbiased ? (1u + max_bernoulli) : max_bernoulli;

    const auto &eval_cv = eval_sms_constraint_vertex_callable();

    $if(vis_count > 0u) {
      UInt chosen_offset = 0u;
      UInt chosen_count = 1u;
      Float chosen_pdf = 1.0f;
      if (eval.sms_unbiased) {
        sample_caster_object(next_float(), chosen_offset, chosen_count,
                             chosen_pdf);
      }

      UInt phase = 0u; // Unbiased: 0 = discovery, 1 = bernoulli.
      UInt trial_idx = 0u;
      Bool loop_done = false;
      Float inv_prob = 1.0f;

      // Discovered star solution (unbiased) & unique solution dedup cache (biased).
      UInt star_len = 0u;
      UInt star_mode_mask = 0u;
      Float3 star_d0 = make_float3(0.0f);
      Float3 star_dlast = make_float3(0.0f);
      std::array<Float2, kMaxChainLen> star_h_offsets;
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        star_h_offsets[k] = make_float2(0.0f);
      }
      Float3 star_contrib = make_float3(0.0f);
      Float3 star_diff_w = make_float3(0.0f);
      Float3 star_gloss_w = make_float3(0.0f);

      constexpr std::uint32_t kMaxUnique = 4u;
      UInt unique_count = 0u;
      std::array<UInt, kMaxUnique> uniq_len;
      std::array<UInt, kMaxUnique> uniq_mode;
      std::array<Float3, kMaxUnique> uniq_d0;
      std::array<Float3, kMaxUnique> uniq_dlast;
      for (std::uint32_t k = 0u; k < kMaxUnique; ++k) {
        uniq_len[k] = 0u;
        uniq_mode[k] = 0u;
        uniq_d0[k] = make_float3(0.0f);
        uniq_dlast[k] = make_float3(0.0f);
      }
      Float3 biased_accum = make_float3(0.0f);
      Float3 biased_diff_w = make_float3(0.0f);
      Float3 biased_gloss_w = make_float3(0.0f);
      UInt biased_max_depth = task.path_depth;

      $while((trial_idx < max_loop_iters) & !loop_done) {
        const Bool in_bernoulli =
            eval.sms_unbiased ? (phase == 1u) : Bool{false};

        // =====================================================================
        // Step A: Solid-Angle Conditioned Area-Weighted Chain Seeding
        // =====================================================================
        std::array<CasterVertexVar, kMaxChainLen> v;
        UInt chain_len = 0u;
        UInt mode_mask = 0u;
        UInt num_refl = 0u;
        UInt num_refr = 0u;
        Bool chain_ok = true;
        Bool unused_disp = false;

        Float3 p_seed = make_float3(0.0f);
        UInt seed_cobj = 0u;
        if (eval.sms_unbiased) {
          sample_caster_seed_point(chosen_offset, chosen_count, p_seed,
                                   seed_cobj);
        } else {
          UInt trial_offset = 0u;
          UInt trial_count = 1u;
          Float trial_pdf = 1.0f;
          sample_caster_object(next_float(), trial_offset, trial_count,
                               trial_pdf);
          static_cast<void>(trial_pdf);
          sample_caster_seed_point(trial_offset, trial_count, p_seed,
                                   seed_cobj);
        }

      {
        const auto off0 = p_seed - task.receiver_p;
        const auto dist0 = length(off0);
        Bool done_seed = false;
        Float3 cur_p = task.receiver_p;
        Float3 cur_dir = make_float3(0.0f, 0.0f, 1.0f);
        UInt skip_inst = task.receiver_inst;
        UInt skip_prim = task.receiver_prim;
        UInt cur_target_cobj = seed_cobj;

        $if(dist0 <= 1.0e-5f) {
          chain_ok = false;
          done_seed = true;
        }
        $else {
          cur_dir = off0 / dist0;
        };

        UInt seed_i = 0u;
        $while((seed_i < kMaxChainLen) & !done_seed & chain_ok) {
          const auto coin_u = next_float();
          const auto stop_u = next_float();
          const auto seed_ray =
              make_ray(cur_p, cur_dir, kRayEps, cycles_light::distant_ray_length);
          const auto h = trace_mnee_surface(
              scene, seed_ray, skip_inst, skip_prim,
              surface_ray::invalid_primitive, surface_ray::invalid_primitive,
              cur_target_cobj, task.light_cycles_object);
          cur_target_cobj = surface_ray::invalid_primitive;

          $if(!h.hit) {
            $if(seed_i == 0u) { chain_ok = false; };
            done_seed = true;
          }
          $else {
            const auto is_caster =
                (h.object_flags &
                 static_cast<unsigned>(abi::SD_OBJECT_CAUSTICS_CASTER)) != 0u;
            $if(!is_caster) {
              $if(seed_i == 0u) { chain_ok = false; };
              done_seed = true;
            }
            $else {
              CasterVertexVar hit_v{};
              SmsClosureSelection sel{};
              const Bool found_any = eval_sms_caster(
                  h, seed_ray, hit_v, 0u, false, false, false, false, &sel,
                  nullptr, nullptr, nullptr, unused_disp, false, 0.0f, 1.0f,
                  make_float3(0.0f), 0.0f, false);
              $if(!found_any) {
                chain_ok = false;
                done_seed = true;
              }
              $else {
                // Choose R vs T mode at this caster vertex.
                Bool is_refl = false;
                $if(sel.has_refl & !sel.has_refr) { is_refl = true; }
                $elif(!sel.has_refl & sel.has_refr) { is_refl = false; }
                $else {
                  const auto p_refl = clamp(sel.refl_approx_f, 0.05f, 0.45f);
                  is_refl = coin_u < p_refl;
                };

                // Recompute geometry & eta matching the chosen mode (when
                // reflection was chosen on a dielectric, set eta = 1).
                const auto c_alpha2 =
                    select(sel.refr_alpha2, sel.refl_alpha2, is_refl);
                const auto c_is_beck =
                    select(sel.refr_is_beckmann, sel.refl_is_beckmann, is_refl);
                Float eta_v = 1.0f;
                $if(!is_refl) {
                  // When found_any evaluated with want_fixed_refl=false,
                  // hit_v.eta already holds the hero-wavelength refractive eta.
                  eta_v = hit_v.eta;
                };

                const auto u_pair = load_f2(u_h, seed_i);
                const auto sampled_off =
                    sample_sms_h_offset(c_alpha2, c_is_beck, u_pair);
                const auto fixed_off = load_f2(star_h_offsets, seed_i);
                const auto h_off = select(sampled_off, fixed_off, in_bernoulli);
                const auto frame = smooth_sms_tangent_frame(
                    hit_v.n, ref_T, hit_v.T1, hit_v.T2);
                const auto hz = sqrt(max(1.0f - dot(h_off, h_off), 1.0e-6f));
                const auto h_seed =
                    normalize(h_off.x * frame.s + h_off.y * frame.t + hz * hit_v.n);

                const auto wi_v = -cur_dir;
                const auto cos_ih = dot(wi_v, h_seed);
                $if(cos_ih <= 1.0e-4f) {
                  chain_ok = false;
                  done_seed = true;
                }
                $else {
                  Float3 wo_spec = make_float3(0.0f);
                  Bool step_valid = true;
                  $if(is_refl) {
                    wo_spec = normalize(2.0f * cos_ih * h_seed - wi_v);
                    step_valid = dot(wo_spec, hit_v.n) > 1.0e-5f;
                  }
                  $else {
                    const auto inv_eta = 1.0f / max(eta_v, 1.0e-5f);
                    const auto cos2_t =
                        1.0f - inv_eta * inv_eta * (1.0f - cos_ih * cos_ih);
                    $if(cos2_t <= 1.0e-6f) {
                      // Total internal reflection: switch to reflection if supported.
                      $if(sel.has_refl) {
                        is_refl = true;
                        eta_v = 1.0f;
                        wo_spec = normalize(2.0f * cos_ih * h_seed - wi_v);
                        step_valid = dot(wo_spec, hit_v.n) > 1.0e-5f;
                      }
                      $else { step_valid = false; };
                    }
                    $else {
                      wo_spec = normalize(
                          (inv_eta * cos_ih - sqrt(cos2_t)) * h_seed -
                          inv_eta * wi_v);
                      step_valid = dot(wo_spec, hit_v.n) < -1.0e-5f;
                    };
                  };

                  $if(!step_valid) {
                    chain_ok = false;
                    done_seed = true;
                  }
                  $else {
                    hit_v.eta = eta_v;
                    hit_v.is_reflection = is_refl;
                    hit_v.h_offset = h_off;
                    hit_v.seed_alpha2 = c_alpha2;
                    hit_v.seed_is_beckmann = c_is_beck;
                    store_vertex(v, seed_i, hit_v);
                    mode_mask |= select(0u, 1u << seed_i, is_refl);
                    num_refl += select(0u, 1u, is_refl);
                    num_refr += select(1u, 0u, is_refl);
                    chain_len += 1u;

                    // Decide whether the ray continues inside/between casters or exits.
                    const Bool stays_inside_or_enters =
                        sel.has_refr & ((!is_refl & !hit_v.backfacing) |
                                        (is_refl & hit_v.backfacing));
                    cur_p = hit_v.p;
                    cur_dir = wo_spec;
                    skip_inst = h.inst;
                    skip_prim = h.prim;

                    $if(seed_i + 1u == kMaxChainLen) {
                      $if(stays_inside_or_enters) { chain_ok = false; };
                      done_seed = true;
                    }
                    $elif(!stays_inside_or_enters) {
                      // External reflection or exiting refraction: optionally
                      // chain another caster bounce with small probability or on a
                      // different object, otherwise terminate the specular chain.
                      $if(stop_u < 0.75f) { done_seed = true; };
                    };
                  };
                };
              };
            };
          };
          seed_i += 1u;
        };
      }

      chain_ok &= (chain_len >= 1u) &
                  (task.path_depth + chain_len <= parameters.max_bounces) &
                  (task.transmission_depth + num_refr <=
                   parameters.max_transmission_bounces) &
                  (task.glossy_depth + num_refl <=
                   parameters.max_glossy_bounces);

      // =======================================================================
      // Step B: Damped Block-Tridiagonal Newton Walk (Unified Smooth + Glossy)
      // =======================================================================
      Bool converged = false;
      $if(chain_ok) {
        Float beta = 1.0f;
        Bool stop_walk = false;
        UInt iter = 0u;
        $while((iter < kSmsMaxNewtonIters) & !stop_walk) {
          std::array<Float4, kMaxChainLen> A_prev;
          std::array<Float4, kMaxChainLen> A_diag;
          std::array<Float4, kMaxChainLen> A_next;
          std::array<Float4, kMaxChainLen> rhs_cols;
          for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
            A_prev[k] = make_float4(0.0f);
            A_diag[k] = make_float4(0.0f);
            A_next[k] = make_float4(0.0f);
            rhs_cols[k] = make_float4(0.0f);
          }
          Float max_err = 0.0f;
          Bool all_orient_ok = true;

          UInt cv_i = 0u;
          $while(cv_i < chain_len) {
            const Bool has_prev = (cv_i > 0u);
            const Bool has_next = (cv_i + 1u < chain_len);
            const auto vi = load_vertex(v, cv_i);
            const auto v_prev = load_vertex(v, select(0u, cv_i - 1u, has_prev));
            const auto v_next =
                load_vertex(v, min(cv_i + 1u, kMaxChainLen - 1u));
            const Float3 p_light =
                select(task.light_p, task.light_dir, is_distant);

            Var<SmsConstraintInput> cin{};
            cin.p_prev = select(task.receiver_p, v_prev.p, has_prev);
            cin.p = vi.p;
            cin.p_next = select(p_light, v_next.p, has_next);
            cin.n = vi.n;
            cin.T1 = vi.T1;
            cin.T2 = vi.T2;
            cin.dn_dx = vi.dn_dx;
            cin.dn_dy = vi.dn_dy;
            cin.eta = vi.eta;
            cin.T1_prev = select(make_float3(0.0f), v_prev.T1, has_prev);
            cin.T2_prev = select(make_float3(0.0f), v_prev.T2, has_prev);
            cin.T1_next = v_next.T1;
            cin.T2_next = v_next.T2;
            cin.has_prev = has_prev;
            cin.has_next = has_next;
            cin.is_distant = is_distant;
            cin.light_e1 = make_float3(0.0f);
            cin.light_e2 = make_float3(0.0f);
            cin.is_reflection = vi.is_reflection;
            cin.h_offset = vi.h_offset;
            cin.ref_T = ref_T;

            const auto cv = eval_cv(cin);
            store_f4(A_prev, cv_i, cv.A_prev);
            store_f4(A_diag, cv_i, cv.A_diag);
            store_f4(A_next, cv_i, cv.A_next);
            store_f4(rhs_cols, cv_i, make_float4(cv.C.x, 0.0f, cv.C.y, 0.0f));
            max_err = max(max_err, cv.c_err);
            all_orient_ok &= cv.orientation_ok;
            cv_i += 1u;
          };

          $if((max_err < kSmsConvTol) & all_orient_ok) {
            converged = true;
            stop_walk = true;
          }
          $else {
            Bool solve_ok = false;
            std::array<Float4, kMaxChainLen> delta_cols;
            solve_block_tridiagonal(A_prev, A_diag, A_next, rhs_cols, chain_len,
                                    solve_ok, delta_cols);
            $if(!solve_ok) { stop_walk = true; }
            $else {
              Float max_step = 0.0f;
              for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
                $if(UInt{k} < chain_len) {
                  const auto d2 = make_float2(delta_cols[k].x, delta_cols[k].z);
                  max_step = max(max_step, length(d2));
                };
              }
              $if((max_step < 1.0e-6f) & (max_err < 1.0e-4f) & all_orient_ok) {
                converged = true;
                stop_walk = true;
              }
              $else {
                Float min_seg = length(v[0u].p - task.receiver_p);
                for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
                  $if(UInt{k} < chain_len) {
                    min_seg = min(min_seg, length(v[k].p - v[k - 1u].p));
                  };
                }
                const Float step_cap = clamp(0.45f * min_seg, 1.0e-3f, 100.0f);
                const Float step_scale =
                    select(1.0f, step_cap / max_step, max_step > step_cap);

                std::array<CasterVertexVar, kMaxChainLen> next_v;
                Bool step_accepted = true;
                Float3 prev_p = task.receiver_p;
                UInt skip_inst = task.receiver_inst;
                UInt skip_prim = task.receiver_prim;

                UInt pi = 0u;
                $while(step_accepted & (pi < chain_len)) {
                  const auto vi = load_vertex(v, pi);
                  const auto dc = load_f4(delta_cols, pi);
                  const auto d2 = make_float2(dc.x, dc.z) * (beta * step_scale);
                  const auto p_prop = vi.p - d2.x * vi.T1 - d2.y * vi.T2;
                  const auto dir_vec = p_prop - prev_p;
                  const auto dir_len = length(dir_vec);
                  $if(dir_len <= 1.0e-7f) { step_accepted = false; }
                  $else {
                    const auto proj_ray =
                        make_ray(prev_p, dir_vec / dir_len, kRayEps,
                                 cycles_light::distant_ray_length);
                    const auto h = trace_mnee_surface(
                        scene, proj_ray, skip_inst, skip_prim,
                        surface_ray::invalid_primitive,
                        surface_ray::invalid_primitive, vi.cycles_object,
                        surface_ray::invalid_primitive);
                    $if(!h.hit) { step_accepted = false; }
                    $else {
                      CasterVertexVar proj_v{};
                      const Bool found_proj = eval_sms_caster(
                          h, proj_ray, proj_v, 1u, vi.is_reflection, true,
                          vi.backfacing, true, nullptr, nullptr, nullptr,
                          nullptr, unused_disp, false, 0.0f, 1.0f,
                          make_float3(0.0f), 0.0f, false);
                      $if(!found_proj) { step_accepted = false; }
                      $else {
                        proj_v.is_reflection = vi.is_reflection;
                        proj_v.h_offset = vi.h_offset;
                        proj_v.seed_alpha2 = vi.seed_alpha2;
                        proj_v.seed_is_beckmann = vi.seed_is_beckmann;
                        store_vertex(next_v, pi, proj_v);
                        prev_p = proj_v.p;
                        skip_inst = h.inst;
                        skip_prim = h.prim;
                      };
                    };
                  };
                  pi += 1u;
                };

                $if(step_accepted) {
                  for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
                    $if(UInt{k} < chain_len) { v[k] = next_v[k]; };
                  }
                  beta = min(1.0f, 2.0f * beta);
                }
                $else {
                  beta *= 0.5f;
                  $if(beta < kMinStepBeta) { stop_walk = true; };
                };
              };
            };
          };
          iter += 1u;
        };
      };

      const auto conv_last_v =
          load_vertex(v, select(0u, chain_len - 1u, chain_len > 0u));
      const auto conv_p0 = v[0u].p;
      const auto conv_plast = conv_last_v.p;
      const Float3 conv_d0_raw = conv_p0 - task.receiver_p;
      const Float3 conv_d0 = conv_d0_raw / max(length(conv_d0_raw), 1.0e-8f);
      const Float3 conv_dlast_seg =
          select(task.light_p - conv_plast, conv_plast - conv_p0,
                 is_distant & (chain_len > 1u));
      const Float3 conv_dlast_raw =
          select(conv_dlast_seg, conv_d0, is_distant & (chain_len <= 1u));
      const Float3 conv_dlast =
          conv_dlast_raw / max(length(conv_dlast_raw), 1.0e-8f);

      // Branch between:
      //  - Unbiased Phase 1 (Bernoulli trial): only test match against star.
      //  - Discovery (Unbiased Phase 0) / Biased unique solution: run C & D.
      $if(in_bernoulli) {
        const auto matched =
            converged & (chain_len == star_len) &
            (mode_mask == star_mode_mask) &
            (dot(conv_d0, star_d0) > 0.99998f) &
            (dot(conv_dlast, star_dlast) > 0.99998f);
        $if(matched) { loop_done = true; }
        $else { inv_prob += 1.0f; };
      }
      $else {
        Bool should_eval_cd = converged;
        if (!eval.sms_unbiased) {
          Bool already_seen = false;
          for (std::uint32_t k = 0u; k < kMaxUnique; ++k) {
            $if(UInt{k} < unique_count) {
              const auto same =
                  (chain_len == uniq_len[k]) & (mode_mask == uniq_mode[k]) &
                  (dot(conv_d0, uniq_d0[k]) > 0.99998f) &
                  (dot(conv_dlast, uniq_dlast[k]) > 0.99998f);
              already_seen |= same;
            };
          }
          should_eval_cd &= !already_seen & (unique_count < kMaxUnique);
        }

        Bool cd_valid = false;
        Float3 cd_rgb = make_float3(0.0f);
        Float3 cd_diff_w = make_float3(0.0f);
        Float3 cd_gloss_w = make_float3(0.0f);

        // =====================================================================
        // Step C: Full-Scene Occlusion Check
        // =====================================================================
        $if(should_eval_cd) {
          Bool vis_ok = true;
          UInt vis_i = 0u;
          $while(vis_ok & (vis_i < chain_len)) {
            const auto has_prev = (vis_i > 0u);
            const auto vp = load_vertex(v, select(0u, vis_i - 1u, has_prev));
            const auto vi = load_vertex(v, vis_i);
            const Float3 prev_p = select(task.receiver_p, vp.p, has_prev);
            const UInt p_inst = select(task.receiver_inst, vp.inst, has_prev);
            const UInt p_prim = select(task.receiver_prim, vp.prim, has_prev);
            const auto seg = vi.p - prev_p;
            const auto seg_len = length(seg);
            $if(seg_len <= 1.0e-5f) { vis_ok = false; }
            $else {
              const auto seg_ray =
                  make_ray(prev_p, seg / seg_len, kRayEps,
                           max(seg_len - kRayEps, kRayEps));
              const auto blk = trace_mnee_surface(
                  scene, seg_ray, p_inst, p_prim, vi.inst, vi.prim,
                  surface_ray::invalid_primitive, task.light_cycles_object);
              $if(blk.hit) { vis_ok = false; };
            };
            vis_i += 1u;
          };
          $if(vis_ok) {
            Float3 to_light_dir = task.light_dir;
            Float tmax = cycles_light::distant_ray_length;
            $if(!is_distant) {
              const auto off = task.light_p - conv_plast;
              const auto dist = length(off);
              $if(dist <= 1.0e-5f) { vis_ok = false; }
              $else {
                to_light_dir = off / dist;
                tmax = max(dist - kRayEps, kRayEps);
              };
            };
            $if(vis_ok) {
              const auto l_ray =
                  make_ray(conv_plast, to_light_dir, kRayEps, tmax);
              const auto blk = trace_mnee_surface(
                  scene, l_ray, conv_last_v.inst, conv_last_v.prim,
                  surface_ray::invalid_primitive,
                  surface_ray::invalid_primitive,
                  surface_ray::invalid_primitive, task.light_cycles_object);
              $if(blk.hit) { vis_ok = false; };
            };
          };

          // ===================================================================
          // Step D: Manifold Jacobian, Cycles Glossy/Smooth Weight & Radiance
          // ===================================================================
          $if(vis_ok) {
            Float3 last_wo = task.light_dir;
            Float last_dist = cycles_light::distant_ray_length;
            $if(!is_distant) {
              const auto off = task.light_p - conv_plast;
              last_dist = max(length(off), 1.0e-6f);
              last_wo = off / last_dist;
            };

            Float3 light_e1 = make_float3(1.0f, 0.0f, 0.0f);
            Float3 light_e2 = make_float3(0.0f, 1.0f, 0.0f);
            $if(is_area) {
              light_e1 = light.axis_x;
              light_e2 = light.axis_y;
            }
            $elif(is_distant) {
              const auto basis =
                  cycles_sample_mapping::make_orthonormals(last_wo);
              light_e1 = basis.tangent;
              light_e2 = basis.bitangent;
            }
            $else {
              const auto basis_ref =
                  select(last_wo, task.light_normal, light.radius > 0.0f);
              const auto basis =
                  cycles_sample_mapping::make_orthonormals(basis_ref);
              light_e1 = basis.tangent;
              light_e2 = basis.bitangent;
            };

            std::array<Float4, kMaxChainLen> A_prev;
            std::array<Float4, kMaxChainLen> A_diag;
            std::array<Float4, kMaxChainLen> A_next;
            std::array<Float4, kMaxChainLen> B_cols;
            for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
              A_prev[k] = make_float4(0.0f);
              A_diag[k] = make_float4(0.0f);
              A_next[k] = make_float4(0.0f);
              B_cols[k] = make_float4(0.0f);
            }
            Float4 chain_weight = make_float4(1.0f);
            Bool any_refr_dispersive = false;
            Bool chain_valid = true;

            UInt d_i = 0u;
            $while(d_i < chain_len) {
              const Bool has_prev = (d_i > 0u);
              const Bool has_next = (d_i + 1u < chain_len);
              const auto vi = load_vertex(v, d_i);
              const auto v_prev =
                  load_vertex(v, select(0u, d_i - 1u, has_prev));
              const auto v_next =
                  load_vertex(v, min(d_i + 1u, kMaxChainLen - 1u));
              const Float3 p_prev = select(task.receiver_p, v_prev.p, has_prev);
              const Float3 p_light =
                  select(task.light_p, task.light_dir, is_distant);

              Var<SmsConstraintInput> cin{};
              cin.p_prev = p_prev;
              cin.p = vi.p;
              cin.p_next = select(p_light, v_next.p, has_next);
              cin.n = vi.n;
              cin.T1 = vi.T1;
              cin.T2 = vi.T2;
              cin.dn_dx = vi.dn_dx;
              cin.dn_dy = vi.dn_dy;
              cin.eta = vi.eta;
              cin.T1_prev = select(make_float3(0.0f), v_prev.T1, has_prev);
              cin.T2_prev = select(make_float3(0.0f), v_prev.T2, has_prev);
              cin.T1_next = v_next.T1;
              cin.T2_next = v_next.T2;
              cin.has_prev = has_prev;
              cin.has_next = has_next;
              cin.is_distant = is_distant;
              cin.light_e1 = light_e1;
              cin.light_e2 = light_e2;
              cin.is_reflection = vi.is_reflection;
              cin.h_offset = vi.h_offset;
              cin.ref_T = ref_T;

              const auto cv = eval_cv(cin);
              store_f4(A_prev, d_i, cv.A_prev);
              store_f4(A_diag, d_i, cv.A_diag);
              store_f4(A_next, d_i, cv.A_next);
              $if(!has_next) { store_f4(B_cols, d_i, -cv.B_light); };
              chain_valid &= cv.orientation_ok & (cv.c_err < 5.0e-4f) &
                             (dot(cv.h, vi.n) > 1.0e-4f);

              const auto seg_vec = vi.p - p_prev;
              const auto seg_d = max(length(seg_vec), 1.0e-6f);
              const auto cam_ray =
                  make_ray(p_prev, seg_vec / seg_d, kRayEps,
                           cycles_light::distant_ray_length);
              MnTraceHit dummy_h{.hit = true,
                                 .inst = vi.inst,
                                 .prim = vi.prim,
                                 .bary = vi.bary,
                                 .t = seg_d,
                                 .cycles_object = vi.cycles_object,
                                 .object_flags = 0u};
              CasterVertexVar unused_v{};
              Float4 vert_w = make_float4(0.0f);
              const auto cos_ih_val = dot(cv.wi, cv.h);
              const auto cos_hn_val = dot(cv.h, vi.n);
              const Bool found_w = eval_sms_caster(
                  dummy_h, cam_ray, unused_v, 1u, vi.is_reflection, true,
                  vi.backfacing, false, nullptr, nullptr, nullptr, &vert_w,
                  any_refr_dispersive, true, cos_ih_val, cos_hn_val, cv.wo,
                  vi.seed_alpha2, vi.seed_is_beckmann);
              chain_valid &= found_w;
              chain_weight *= vert_w;
              d_i += 1u;
            };

            Bool sens_ok = false;
            std::array<Float4, kMaxChainLen> Y_cols;
            solve_block_tridiagonal(A_prev, A_diag, A_next, B_cols, chain_len,
                                    sens_ok, Y_cols);
            chain_valid &= sens_ok;

            Float3 light_factor_rgb = make_float3(0.0f);
            Float4 light_factor_spec = make_float4(0.0f);
            Bool light_ok = false;

            $if(chain_valid) {
              Float scalar_eval_over_pdf = 0.0f;
              $if(is_area) {
                const auto area_hit = AreaLightSampling{}.from_intersection(
                    area_light_sample_input(light, conv_plast,
                                            make_float2(0.5f)),
                    last_wo, last_dist);
                light_ok = area_hit.valid;
                scalar_eval_over_pdf = area_hit.evaluation_factor /
                                       max(task.light_area_pdf, 1.0e-20f);
              }
              $elif(is_distant) {
                const auto half_angle = 0.5f * max(light.angle, 0.0f);
                const auto finite = half_angle > 0.0f;
                const auto sine_half = sin(half_angle);
                const auto disk_area = pi * sine_half * sine_half;
                const auto normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                const auto distant_eval =
                    select(1.0f, 1.0f / max(disk_area, 1.0e-20f),
                           normalize_power & finite);
                light_ok = true;
                scalar_eval_over_pdf =
                    distant_eval / max(task.light_area_pdf, 1.0e-20f);
              }
              $else {
                const auto spot =
                    light.type == static_cast<std::uint32_t>(LightType::spot);
                const auto normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                const auto pt_factor =
                    analytic_light_sampling::point_eval_factor(
                        light.radius, normalize_power);
                Float atten = 1.0f;
                Bool side_ok = (light.radius == 0.0f) |
                               (dot(-last_wo, task.light_normal) > 0.0f);
                $if(spot) {
                  const auto l_tfm =
                      analytic_light_sampling::light_linear_transform(
                          light.axis_x, light.axis_y, light.axis_z,
                          light.axis_scale);
                  const auto local_ray =
                      analytic_light_sampling::spot_light_to_local(
                          -last_wo, l_tfm);
                  atten = analytic_light_sampling::spot_light_attenuation(
                      local_ray, light.spot);
                };
                light_ok = side_ok & (atten > 0.0f);
                scalar_eval_over_pdf =
                    (pt_factor * atten) / max(task.light_area_pdf, 1.0e-20f);
              };

              $if(light_ok & (scalar_eval_over_pdf > 0.0f)) {
                Float4 shader_val =
                    (is_xyz3 && spec_ctx)
                        ? cycles_spectrum::xyz3_rotate_eval(
                              task.light_shader.xyz(), *spec_ctx->hero_index)
                        : task.light_shader;
                $if(task.light_constant_shader == 0u) {
                  Var<DirectLightTaskCall> dl_task{};
                  dl_task.ray_origin = conv_plast;
                  dl_task.ray_direction = last_wo;
                  dl_task.ray_minimum = kRayEps;
                  dl_task.ray_maximum = last_dist;
                  dl_task.ray_dP = task.ray_dP;
                  dl_task.ray_dD = task.ray_dD;
                  dl_task.ray_time = task.ray_time;
                  dl_task.sample_index = task.sample_index;
                  dl_task.rng_hash = task.rng_hash;
                  dl_task.rng_offset = task.rng_offset;
                  dl_task.light_object = task.light_cycles_object;
                  dl_task.light_primitive = task.light_index;
                  dl_task.shader_flags = task.light_shader_flags;
                  dl_task.path_depth = task.path_depth + chain_len;
                  dl_task.diffuse_depth = task.diffuse_depth;
                  dl_task.glossy_depth = task.glossy_depth + num_refl;
                  dl_task.transmission_depth =
                      task.transmission_depth + num_refr;
                  dl_task.transparent_depth =
                      task.transparent_depth.cast<std::uint16_t>();
                  dl_task.portal_depth =
                      task.portal_depth.cast<std::uint16_t>();
                  if (eval.spectral_rendering && spec_ctx) {
                    shader_val = eval.light_emission->evaluate_spectral(
                        dl_task, parameters, *spec_ctx);
                  } else {
                    shader_val = make_float4(
                        eval.light_emission->evaluate(dl_task, parameters),
                        0.0f);
                  }
                };
                if (eval.spectral_rendering && spec_ctx) {
                  const auto color_spec =
                      cycles_spectrum::rgb_to_spectrum_reflectance(
                          *spec_ctx->tables, light.color,
                          spec_ctx->wavelengths.lambdas);
                  light_factor_spec = color_spec * shader_val *
                                      (light.power * scalar_eval_over_pdf);
                } else {
                  light_factor_rgb = light.color * shader_val.xyz() *
                                     (light.power * scalar_eval_over_pdf);
                }
              };
            };

            $if(chain_valid & light_ok) {
              const auto Y0 = Y_cols[0u];
              const auto det_Y0 = abs(Y0.x * Y0.w - Y0.y * Y0.z);
              const auto to_v0 = conv_p0 - task.receiver_p;
              const auto dist0_sq = max(length_squared(to_v0), 1.0e-12f);
              const auto wo_receiver = to_v0 / sqrt(dist0_sq);
              const auto cos_v0 = abs(dot(wo_receiver, v[0u].ng));
              const auto G_term = (cos_v0 / dist0_sq) * det_Y0;

              $if(manifold_jacobian_within_bounds(cos_v0, det_Y0, G_term)) {
                const PathCyclesSvmKernelGlobals kg_recv{
                    scene,
                    parameters,
                    eval.camera_projection,
                    true,
                    true,
                    PathCyclesSvmShaderDomain::surface_only,
                    scene->caustics_receiver_shader_mask,
                    false,
                    nullptr,
                    nullptr,
                    eval.spectral_rendering,
                    eval.spectral_mode};
                const auto recv_pool_opts =
                    closure_pool_options_for_kg(kg_recv, eval.spectral_rendering);
                svm::ClosurePool recv_closures{closure_cap, recv_pool_opts};
                const auto recv_cam_ray =
                    make_ray(task.receiver_p + task.receiver_wi,
                             -task.receiver_wi, 0.0f, 2.0f);
                const auto recv_lcg = cycles_noise::hash_uint3(
                    task.rng_hash ^ 0x9e3779b9u, task.rng_offset,
                    task.sample_index);
                auto recv_setup = setup_cycles_svm_ray_shader_data(
                    scene, kg_recv, recv_cam_ray, task.receiver_inst,
                    task.receiver_prim, task.receiver_bary, 0.0f, task.ray_dP,
                    task.ray_dD, task.ray_time, recv_lcg, parameters);
                auto &sd_recv = recv_setup.shader_data;
                sd_recv.closure = &recv_closures;
                if (spec_ctx) {
                  sd_recv.spectral = *spec_ctx;
                }
                const svm::PathState recv_state{
                    task.path_visibility,
                    0u,
                    task.path_depth,
                    task.transparent_depth,
                    task.diffuse_depth,
                    task.glossy_depth,
                    task.transmission_depth,
                    task.portal_depth};
                evaluate_masked_svm(scene, kg_recv, recv_setup.transforms,
                                    sd_recv, recv_state,
                                    scene->caustics_receiver_shader_mask);
                const auto bsdf_eval = svm_detail::surface_shader_bsdf_eval(
                    kg_recv, sd_recv, wo_receiver, task.light_shader_flags,
                    caster_closure_types);

                if (is_xyz3 && spec_ctx) {
                  const auto bsdf_spec = bsdf_eval.spectral_sum.value_or(
                      make_float4(bsdf_eval.sum, 0.0f));
                  const auto raw_rgb = eval_xyz3_caustic_raw_rgb(
                      *eval.spectral_tables, *spec_ctx, task.path_throughput,
                      bsdf_spec, chain_weight, light_factor_spec, G_term,
                      hero_lambda, secondary_terminated, any_refr_dispersive);
                  const auto finite_nz =
                      !(any(luisa::compute::isnan(raw_rgb)) |
                        any(luisa::compute::isinf(raw_rgb))) &
                      any(raw_rgb != 0.0f);
                  $if(finite_nz) {
                    cd_rgb = raw_rgb;
                    cd_valid = true;
                    if (need_weights) {
                      compute_xyz3_bsdf_component_weights(
                          *eval.spectral_tables, *spec_ctx,
                          eval.light_component_ratio, bsdf_spec,
                          bsdf_eval.spectral_diffuse.value_or(
                              make_float4(bsdf_eval.diffuse, 0.0f)),
                          bsdf_eval.spectral_glossy.value_or(
                              make_float4(bsdf_eval.glossy, 0.0f)),
                          cd_diff_w, cd_gloss_w);
                    }
                  };
                } else if (eval.spectral_rendering && spec_ctx) {
                  const auto bsdf_spec = bsdf_eval.spectral_sum.value_or(
                      make_float4(bsdf_eval.sum, 0.0f));
                  const auto unclamped_spec = task.path_throughput * bsdf_spec *
                                              chain_weight * light_factor_spec *
                                              G_term;
                  const auto raw_rgb = cycles_spectrum::spectrum_to_rgb(
                      *eval.spectral_tables, unclamped_spec,
                      spec_ctx->wavelengths,
                      secondary_terminated | any_refr_dispersive);
                  const auto finite_pos =
                      !(any(luisa::compute::isnan(raw_rgb)) |
                        any(luisa::compute::isinf(raw_rgb))) &
                      any(raw_rgb > 0.0f);
                  $if(finite_pos) {
                    cd_rgb = max(raw_rgb, make_float3(0.0f));
                    cd_valid = true;
                    if (need_weights) {
                      const auto bsdf_rgb = cycles_spectrum::spectrum_to_rgb(
                          *eval.spectral_tables, bsdf_spec,
                          spec_ctx->wavelengths, secondary_terminated);
                      const auto diff_rgb = cycles_spectrum::spectrum_to_rgb(
                          *eval.spectral_tables,
                          bsdf_eval.spectral_diffuse.value_or(
                              make_float4(bsdf_eval.diffuse, 0.0f)),
                          spec_ctx->wavelengths, secondary_terminated);
                      const auto gloss_rgb = cycles_spectrum::spectrum_to_rgb(
                          *eval.spectral_tables,
                          bsdf_eval.spectral_glossy.value_or(
                              make_float4(bsdf_eval.glossy, 0.0f)),
                          spec_ctx->wavelengths, secondary_terminated);
                      cd_diff_w =
                          eval.light_component_ratio(diff_rgb, bsdf_rgb);
                      cd_gloss_w =
                          eval.light_component_ratio(gloss_rgb, bsdf_rgb);
                    }
                  };
                } else {
                  const auto raw_rgb = task.path_throughput.xyz() *
                                       bsdf_eval.sum * chain_weight.xyz() *
                                       light_factor_rgb * G_term;
                  const auto finite_pos =
                      !(any(luisa::compute::isnan(raw_rgb)) |
                        any(luisa::compute::isinf(raw_rgb))) &
                      any(raw_rgb > 0.0f);
                  $if(finite_pos) {
                    cd_rgb = max(raw_rgb, make_float3(0.0f));
                    cd_valid = true;
                    if (need_weights) {
                      cd_diff_w = eval.light_component_ratio(
                          bsdf_eval.diffuse, bsdf_eval.sum);
                      cd_gloss_w = eval.light_component_ratio(
                          bsdf_eval.glossy, bsdf_eval.sum);
                    }
                  };
                }
              };
            };
          };
        };

        if (eval.sms_unbiased) {
          $if(cd_valid) {
            star_len = chain_len;
            star_mode_mask = mode_mask;
            star_d0 = conv_d0;
            star_dlast = conv_dlast;
            for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
              star_h_offsets[k] = v[k].h_offset;
            }
            star_contrib = cd_rgb;
            star_diff_w = cd_diff_w;
            star_gloss_w = cd_gloss_w;
            phase = 1u;
            inv_prob = 1.0f;
          }
          $else {
            // Discovery trial found no solution: finish in 1 trial.
            loop_done = true;
          };
        } else {
          $if(cd_valid) {
            for (std::uint32_t k = 0u; k < kMaxUnique; ++k) {
              $if(unique_count == k) {
                uniq_len[k] = chain_len;
                uniq_mode[k] = mode_mask;
                uniq_d0[k] = conv_d0;
                uniq_dlast[k] = conv_dlast;
              };
            }
            unique_count += 1u;
            biased_accum += cd_rgb;
            $if(unique_count == 1u) {
              biased_diff_w = cd_diff_w;
              biased_gloss_w = cd_gloss_w;
            };
            biased_max_depth =
                max(biased_max_depth, task.path_depth + chain_len);
          };
        }
      };
        trial_idx += 1u;
      };

      if (eval.sms_unbiased) {
        $if(phase == 1u) {
          const auto unbiased_rgb =
              star_contrib * (inv_prob / max(chosen_pdf, 1.0e-6f));
          const auto clamped = eval.clamp_contribution(
              unbiased_rgb, task.path_depth + star_len,
              parameters.sample_clamp_direct, parameters.sample_clamp_indirect);
          if (is_xyz3) {
            const auto ok = !(any(luisa::compute::isnan(clamped)) |
                              any(luisa::compute::isinf(clamped))) &
                            any(clamped != 0.0f);
            result_rgb = select(make_float3(0.0f), clamped, ok);
            eval_valid = ok;
          } else {
            const auto ok = !(any(luisa::compute::isnan(clamped)) |
                              any(luisa::compute::isinf(clamped))) &
                            any(clamped > 0.0f);
            result_rgb =
                select(make_float3(0.0f), max(clamped, make_float3(0.0f)), ok);
            eval_valid = ok;
          }
          if (need_weights) {
            eval_diff_w = star_diff_w;
            eval_gloss_w = star_gloss_w;
          }
        };
      } else {
        $if(unique_count > 0u) {
          const auto clamped = eval.clamp_contribution(
              biased_accum, biased_max_depth,
              parameters.sample_clamp_direct, parameters.sample_clamp_indirect);
          if (is_xyz3) {
            const auto ok = !(any(luisa::compute::isnan(clamped)) |
                              any(luisa::compute::isinf(clamped))) &
                            any(clamped != 0.0f);
            result_rgb = select(make_float3(0.0f), clamped, ok);
            eval_valid = ok;
          } else {
            const auto ok = !(any(luisa::compute::isnan(clamped)) |
                              any(luisa::compute::isinf(clamped))) &
                            any(clamped > 0.0f);
            result_rgb =
                select(make_float3(0.0f), max(clamped, make_float3(0.0f)), ok);
            eval_valid = ok;
          }
          if (need_weights) {
            eval_diff_w = biased_diff_w;
            eval_gloss_w = biased_gloss_w;
          }
        };
      }
    };
  };

  out_valid = eval_valid;
  if (need_weights) {
    *out_diffuse_weight = eval_diff_w;
    *out_glossy_weight = eval_gloss_w;
  }
  return result_rgb;
}

} // namespace

Float3 CausticsTaskEvaluator::evaluate(
    const Var<CausticTaskCall> &task,
    const Var<RenderKernelParameters> &parameters,
    Bool &valid,
    Float3 *out_diffuse_weight,
    Float3 *out_glossy_weight) const noexcept {
  valid = false;
  if (out_diffuse_weight != nullptr) {
    *out_diffuse_weight = make_float3(0.0f);
  }
  if (out_glossy_weight != nullptr) {
    *out_glossy_weight = make_float3(0.0f);
  }
  if (!scene || !scene->cycles_svm || !scene->cycles_svm->geometry ||
      !scene->cycles_svm->objects) {
    return make_float3(0.0f);
  }
  if (caustics_method == contract::CausticsMethod::sms &&
      scene->caustics_caster_triangles &&
      scene->caustics_caster_triangle_count > 0u &&
      scene->caustics_caster_objects &&
      scene->caustics_caster_object_count > 0u) {
    return evaluate_sms_caustics(*this, task, parameters, valid,
                                 out_diffuse_weight, out_glossy_weight);
  }

  Float3 result_rgb = make_float3(0.0f);
  Bool eval_valid = false;
  Float3 eval_diff_w = make_float3(0.0f);
  Float3 eval_gloss_w = make_float3(0.0f);
  const bool need_weights =
      out_diffuse_weight != nullptr && out_glossy_weight != nullptr;
  const bool is_xyz3 = spectral_mode == contract::SpectralMode::xyz3;

  $outline_with_name(is_xyz3 ? "mnee_evaluate_xyz3"
                             : (spectral_rendering ? "mnee_evaluate_spectral"
                                                   : "mnee_evaluate_rgb")) {
    const auto load_f4 = [](const std::array<Float4, kMaxChainLen> &arr,
                            Expr<std::uint32_t> idx) noexcept -> Float4 {
      Float4 res = arr[0u];
      for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
        res = select(res, arr[k], idx == k);
      }
      return res;
    };
    const auto store_f4 = [](std::array<Float4, kMaxChainLen> &arr,
                             Expr<std::uint32_t> idx,
                             Expr<luisa::float4> val) noexcept {
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        $if(idx == k) { arr[k] = val; };
      }
    };
    const auto load_vertex =
        [](const std::array<CasterVertexVar, kMaxChainLen> &arr,
           Expr<std::uint32_t> idx) noexcept -> CasterVertexVar {
      CasterVertexVar res = arr[0u];
      for (std::uint32_t k = 1u; k < kMaxChainLen; ++k) {
        $if(idx == k) { res = arr[k]; };
      }
      return res;
    };
    const auto store_vertex =
        [](std::array<CasterVertexVar, kMaxChainLen> &arr,
           Expr<std::uint32_t> idx,
           const CasterVertexVar &val) noexcept {
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        $if(idx == k) { arr[k] = val; };
      }
    };

    // Optional spectral context reconstructed from scalar u_lambda.
    std::optional<cycles_spectrum::SpectralContext> spec_ctx;
    Float hero_lambda = 550.0f;
    Bool secondary_terminated = false;
    if (spectral_rendering && spectral_tables) {
      secondary_terminated =
          (task.path_flags &
           cycles_path_state::flag_spectral_secondary_terminated) != 0u;
      spec_ctx.emplace(cycles_spectrum::make_spectral_context(
          spectral_tables.get(), spectral_mode, task.u_lambda,
          secondary_terminated));
      hero_lambda = is_xyz3
                        ? cycles_spectrum::xyz3_hero(*spec_ctx->lambdas_xyz,
                                                     *spec_ctx->hero_index)
                        : spec_ctx->wavelengths.lambdas.x;
    }

    const PathCyclesSvmKernelGlobals kg_caster{
        scene,
        parameters,
        camera_projection,
        true,
        true,
        PathCyclesSvmShaderDomain::surface_only,
        scene->caustics_caster_shader_mask,
        false,
        nullptr,
        nullptr,
        spectral_rendering,
        spectral_mode};

    const svm_detail::ClosureTypeMask caster_closure_types =
        svm_detail::closure_types_for_kernel_features(
            scene->cycles_svm->kernel_features);
    const auto caster_pool_opts =
        closure_pool_options_for_kg(kg_caster, spectral_rendering);
    const std::size_t closure_cap = std::clamp<std::size_t>(
        std::max<std::size_t>(
            scene->cycles_svm->compilation.max_surface_closures, 2u),
        2u, svm::maximum_closure_capacity);

    const auto eval_caster_hit =
        [&](const MnTraceHit &hit, const Var<luisa::compute::Ray> &ray,
            CasterVertexVar &v, Bool &found_refractive,
            Bool &any_disp_out, Float4 *out_transmittance,
            Bool force_backfacing_from_v,
            Bool fixed_backfacing,
            Bool reject_wrong_side) noexcept {
          found_refractive = false;
          svm::ClosurePool closures{closure_cap, caster_pool_opts};
          svm::TriangleVertices vertices{};
          svm::TriangleNormals normals{};
          Var<InstanceGpu> inst_gpu{};
          const auto lcg = cycles_noise::hash_uint3(
              task.rng_hash ^ 0x6c8e9cf5u, task.rng_offset, task.sample_index);
          auto setup = setup_cycles_svm_ray_shader_data(
              scene, kg_caster, ray, hit.inst, hit.prim, hit.bary, hit.t,
              task.ray_dP, task.ray_dD, task.ray_time, lcg, parameters,
              &inst_gpu, &vertices, &normals);
          auto &sd = setup.shader_data;
          sd.closure = &closures;
          if (spec_ctx) {
            sd.spectral = *spec_ctx;
          }
          const svm::PathState state{
              svm::path_ray_visibility_diffuse |
                  svm::path_ray_visibility_transmit,
              0u,
              task.path_depth + 1u,
              task.transparent_depth,
              task.diffuse_depth,
              task.glossy_depth,
              task.transmission_depth,
              task.portal_depth};
          evaluate_masked_svm(scene, kg_caster, setup.transforms, sd, state,
                              scene->caustics_caster_shader_mask);
          const auto sel = select_refractive_closure(closures);
          $if(sel.found) {
            const auto actual_backfacing =
                (sd.flag & static_cast<unsigned>(abi::SD_BACKFACING)) != 0u;
            const auto side_mismatch =
                reject_wrong_side & (actual_backfacing != fixed_backfacing);
            $if(!side_mismatch) {
              found_refractive = true;
              const auto bf = select(actual_backfacing, fixed_backfacing,
                                     force_backfacing_from_v);
              build_caster_geometry_frame(sd, setup.transforms, vertices, normals,
                                          bf, v);
              v.inst = hit.inst;
              v.prim = hit.prim;
              v.cycles_object = hit.cycles_object;

              // If fixed_backfacing disagrees with setup's backfacing detection
              // during Step D re-eval, invert IOR back once so eta matches
              // the active side of the surface.
              Float eta_d = sel.eta_d;
              $if(force_backfacing_from_v &
                  (actual_backfacing != fixed_backfacing)) {
                eta_d = 1.0f / max(eta_d, 1.0e-5f);
              };
              if (spec_ctx) {
                const Float abbe =
                    kg_caster.has_dispersive_materials()
                        ? kg_caster.shader_dispersion_abbe(sd.shader)
                        : Float{0.0f};
                v.eta = cycles_spectrum::cauchy_ior(
                            eta_d, abbe, make_float4(hero_lambda))
                            .x;
              } else {
                v.eta = eta_d;
              }
              if (out_transmittance != nullptr) {
                const auto cos_i = abs(dot(-ray->direction(), v.n));
                *out_transmittance = eval_caster_transmission(
                    kg_caster, closures, sel.index, sd.shader, cos_i, eta_d,
                    spec_ctx ? &*spec_ctx : nullptr, any_disp_out);
              }
            };
          };
        };

    const Var<LightGpu> light = scene->light_buffer->read(task.light_index);
    const auto is_distant =
        light.type == static_cast<std::uint32_t>(LightType::distant);

    // =========================================================================
    // Step A: Seed chain discovery from receiver toward light
    // =========================================================================
    std::array<CasterVertexVar, kMaxChainLen> v;
    UInt chain_len = 0u;
    Bool chain_ok = true;
    Bool unused_disp = false;
    {
      Float3 cur_p = task.receiver_p;
      UInt skip_inst = task.receiver_inst;
      UInt skip_prim = task.receiver_prim;
      Bool done_seed = false;
      UInt seed_i = 0u;
      $while((seed_i < kMaxChainLen) & !done_seed & chain_ok) {
        Float3 seed_dir = task.light_dir;
        Float tmax = cycles_light::distant_ray_length;
        $if(!is_distant) {
          const auto offset = task.light_p - cur_p;
          const auto dist = length(offset);
          $if(dist <= 1.0e-5f) {
            chain_ok = false;
            done_seed = true;
          }
          $else {
            seed_dir = offset / dist;
            tmax = max(dist - kRayEps, 0.0f);
          };
        };
        $if(!done_seed & chain_ok) {
          const auto seed_ray = make_ray(cur_p, seed_dir, kRayEps, tmax);
          const auto h = trace_mnee_surface(
              scene, seed_ray, skip_inst, skip_prim,
              surface_ray::invalid_primitive, surface_ray::invalid_primitive,
              surface_ray::invalid_primitive, task.light_cycles_object);
          $if(!h.hit) { done_seed = true; }
          $else {
            const auto is_caster =
                (h.object_flags &
                 static_cast<unsigned>(abi::SD_OBJECT_CAUSTICS_CASTER)) != 0u;
            $if(!is_caster) {
              chain_ok = false;
              done_seed = true;
            }
            $else {
              Bool found_refr = false;
              CasterVertexVar hit_v{};
              eval_caster_hit(h, seed_ray, hit_v, found_refr, unused_disp,
                              nullptr, false, false, false);
              $if(!found_refr) {
                chain_ok = false;
                done_seed = true;
              }
              $else {
                store_vertex(v, seed_i, hit_v);
                cur_p = hit_v.p;
                skip_inst = h.inst;
                skip_prim = h.prim;
                chain_len += 1u;
              };
            };
          };
        };
        seed_i += 1u;
      };
      // If all 4 slots filled, ensure no extra caster/blocker hides the light.
      $if(chain_ok & (chain_len == kMaxChainLen)) {
        Float3 tail_dir = task.light_dir;
        Float tmax = cycles_light::distant_ray_length;
        $if(!is_distant) {
          const auto off = task.light_p - cur_p;
          const auto d = length(off);
          tail_dir = select(task.light_dir, off / max(d, 1.0e-6f), d > 1.0e-5f);
          tmax = max(d - kRayEps, 0.0f);
        };
        const auto tail_ray = make_ray(cur_p, tail_dir, kRayEps, tmax);
        const auto tail_h = trace_mnee_surface(
            scene, tail_ray, skip_inst, skip_prim,
            surface_ray::invalid_primitive, surface_ray::invalid_primitive,
            surface_ray::invalid_primitive, task.light_cycles_object);
        $if(tail_h.hit) { chain_ok = false; };
      };
    }

    chain_ok &= (chain_len >= 1u) &
                (task.path_depth + chain_len <= parameters.max_bounces) &
                (task.transmission_depth + chain_len <=
                 parameters.max_transmission_bounces);

    // =========================================================================
    // Step B: Damped Block-Tridiagonal Newton Manifold Walk
    // =========================================================================
    Bool converged = false;
    const auto &eval_cv = eval_constraint_vertex_callable();
    $if(chain_ok) {
      Float beta = 1.0f;
      Bool stop_walk = false;
      UInt iter = 0u;
      $while((iter < kMaxNewtonIters) & !stop_walk) {
        std::array<Float4, kMaxChainLen> A_prev;
        std::array<Float4, kMaxChainLen> A_diag;
        std::array<Float4, kMaxChainLen> A_next;
        std::array<Float4, kMaxChainLen> rhs_cols;
        for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
          A_prev[k] = make_float4(0.0f);
          A_diag[k] = make_float4(0.0f);
          A_next[k] = make_float4(0.0f);
          rhs_cols[k] = make_float4(0.0f);
        }
        Float max_err = 0.0f;
        Bool all_orient_ok = true;

        UInt cv_i = 0u;
        $while(cv_i < chain_len) {
          const Bool has_prev = (cv_i > 0u);
          const Bool has_next = (cv_i + 1u < chain_len);
          const auto vi = load_vertex(v, cv_i);
          const auto v_prev = load_vertex(v, select(0u, cv_i - 1u, has_prev));
          const auto v_next =
              load_vertex(v, min(cv_i + 1u, kMaxChainLen - 1u));
          const Float3 p_prev = select(task.receiver_p, v_prev.p, has_prev);
          const Float3 T1_prev =
              select(make_float3(0.0f), v_prev.T1, has_prev);
          const Float3 T2_prev =
              select(make_float3(0.0f), v_prev.T2, has_prev);
          const Float3 p_light =
              select(task.light_p, task.light_dir, is_distant);
          const Float3 p_next = select(p_light, v_next.p, has_next);

          const auto cv = eval_cv(
              p_prev, vi.p, p_next, vi.n, vi.T1, vi.T2, vi.dn_dx, vi.dn_dy,
              vi.eta, T1_prev, T2_prev, v_next.T1, v_next.T2, has_prev,
              has_next, is_distant, make_float3(0.0f), make_float3(0.0f));
          store_f4(A_prev, cv_i, cv.A_prev);
          store_f4(A_diag, cv_i, cv.A_diag);
          store_f4(A_next, cv_i, cv.A_next);
          store_f4(rhs_cols, cv_i, make_float4(cv.C.x, 0.0f, cv.C.y, 0.0f));
          max_err = max(max_err, length(cv.C));
          all_orient_ok &= cv.orientation_ok;
          cv_i += 1u;
        };

        $if((max_err < kConvTol) & all_orient_ok) {
          converged = true;
          stop_walk = true;
        }
        $else {
          Bool solve_ok = false;
          std::array<Float4, kMaxChainLen> delta_cols;
          solve_block_tridiagonal(A_prev, A_diag, A_next, rhs_cols, chain_len,
                                  solve_ok, delta_cols);
          $if(!solve_ok) { stop_walk = true; }
          $else {
            Float max_step = 0.0f;
            for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
              $if(UInt{k} < chain_len) {
                const auto d2 = make_float2(delta_cols[k].x, delta_cols[k].z);
                max_step = max(max_step, length(d2));
              };
            }
            $if((max_step < 1.0e-6f) & (max_err < 5.0e-4f) & all_orient_ok) {
              converged = true;
              stop_walk = true;
            }
            $else {
              const Float step_cap = 1.0f;
              const Float step_scale =
                  select(1.0f, step_cap / max_step, max_step > step_cap);

              std::array<CasterVertexVar, kMaxChainLen> next_v;
              Bool step_accepted = true;
              Float3 prev_p = task.receiver_p;
              UInt skip_inst = task.receiver_inst;
              UInt skip_prim = task.receiver_prim;

              UInt pi = 0u;
              $while(step_accepted & (pi < chain_len)) {
                const auto vi = load_vertex(v, pi);
                const auto dc = load_f4(delta_cols, pi);
                const auto d2 = make_float2(dc.x, dc.z) * (beta * step_scale);
                const auto p_prop = vi.p - d2.x * vi.T1 - d2.y * vi.T2;
                const auto dir_vec = p_prop - prev_p;
                const auto dir_len = length(dir_vec);
                $if(dir_len <= 1.0e-7f) { step_accepted = false; }
                $else {
                  const auto proj_ray =
                      make_ray(prev_p, dir_vec / dir_len, kRayEps,
                               cycles_light::distant_ray_length);
                  const auto h = trace_mnee_surface(
                      scene, proj_ray, skip_inst, skip_prim,
                      surface_ray::invalid_primitive,
                      surface_ray::invalid_primitive, vi.cycles_object,
                      surface_ray::invalid_primitive);
                  $if(!h.hit) { step_accepted = false; }
                  $else {
                    Bool found_refr = false;
                    CasterVertexVar proj_v{};
                    eval_caster_hit(h, proj_ray, proj_v, found_refr,
                                    unused_disp, nullptr, true, vi.backfacing,
                                    true);
                    $if(!found_refr) { step_accepted = false; }
                    $else {
                      store_vertex(next_v, pi, proj_v);
                      prev_p = proj_v.p;
                      skip_inst = h.inst;
                      skip_prim = h.prim;
                    };
                  };
                };
                pi += 1u;
              };

              $if(step_accepted) {
                for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
                  $if(UInt{k} < chain_len) { v[k] = next_v[k]; };
                }
                beta = min(1.0f, 2.0f * beta);
              }
              $else {
                beta *= 0.5f;
                $if(beta < kMinStepBeta) { stop_walk = true; };
              };
            };
          };
        };
        iter += 1u;
      };
    };

    // =========================================================================
    // Step C: Snell Orientation, Light Orientation & Full-Scene Occlusion Check
    // =========================================================================
    $if(converged) {
      Bool vis_ok = true;
      UInt vis_i = 0u;
      $while(vis_ok & (vis_i < chain_len)) {
        const auto has_prev = (vis_i > 0u);
        const auto vp = load_vertex(v, select(0u, vis_i - 1u, has_prev));
        const auto vi = load_vertex(v, vis_i);
        const Float3 prev_p = select(task.receiver_p, vp.p, has_prev);
        const UInt p_inst = select(task.receiver_inst, vp.inst, has_prev);
        const UInt p_prim = select(task.receiver_prim, vp.prim, has_prev);
        const auto seg = vi.p - prev_p;
        const auto seg_len = length(seg);
        $if(seg_len <= 1.0e-5f) { vis_ok = false; }
        $else {
          const auto seg_ray = make_ray(prev_p, seg / seg_len, kRayEps,
                                        max(seg_len - kRayEps, kRayEps));
          const auto blk = trace_mnee_surface(
              scene, seg_ray, p_inst, p_prim, vi.inst, vi.prim,
              surface_ray::invalid_primitive, task.light_cycles_object);
          $if(blk.hit) { vis_ok = false; };
        };
        vis_i += 1u;
      };
      $if(vis_ok) {
        const auto last_v = load_vertex(v, chain_len - 1u);
        Float3 to_light_dir = task.light_dir;
        Float tmax = cycles_light::distant_ray_length;
        $if(!is_distant) {
          const auto off = task.light_p - last_v.p;
          const auto dist = length(off);
          $if(dist <= 1.0e-5f) { vis_ok = false; }
          $else {
            to_light_dir = off / dist;
            tmax = max(dist - kRayEps, kRayEps);
          };
        };
        $if(vis_ok) {
          const auto l_ray = make_ray(last_v.p, to_light_dir, kRayEps, tmax);
          const auto blk = trace_mnee_surface(
              scene, l_ray, last_v.inst, last_v.prim,
              surface_ray::invalid_primitive, surface_ray::invalid_primitive,
              surface_ray::invalid_primitive, task.light_cycles_object);
          $if(blk.hit) { vis_ok = false; };
        };
      };
      eval_valid = vis_ok;
    };

    // =========================================================================
    // Step D: Generalized Geometric Term G_manifold & Radiance Contribution
    // =========================================================================
    $if(eval_valid) {
      const auto last_v = load_vertex(v, chain_len - 1u);
      Float3 last_wo = task.light_dir;
      Float last_dist = cycles_light::distant_ray_length;
      $if(!is_distant) {
        const auto off = task.light_p - last_v.p;
        last_dist = max(length(off), 1.0e-6f);
        last_wo = off / last_dist;
      };

      // Tangent basis (light_e1, light_e2) for B_{k-1}.
      Float3 light_e1 = make_float3(1.0f, 0.0f, 0.0f);
      Float3 light_e2 = make_float3(0.0f, 1.0f, 0.0f);
      const auto is_area =
          light.type == static_cast<std::uint32_t>(LightType::area);
      $if(is_area) {
        light_e1 = light.axis_x;
        light_e2 = light.axis_y;
      }
      $elif(is_distant) {
        const auto basis = cycles_sample_mapping::make_orthonormals(last_wo);
        light_e1 = basis.tangent;
        light_e2 = basis.bitangent;
      }
      $else {
        const auto basis_ref =
            select(last_wo, task.light_normal, light.radius > 0.0f);
        const auto basis = cycles_sample_mapping::make_orthonormals(basis_ref);
        light_e1 = basis.tangent;
        light_e2 = basis.bitangent;
      };

      std::array<Float4, kMaxChainLen> A_prev;
      std::array<Float4, kMaxChainLen> A_diag;
      std::array<Float4, kMaxChainLen> A_next;
      std::array<Float4, kMaxChainLen> B_cols;
      for (std::uint32_t k = 0u; k < kMaxChainLen; ++k) {
        A_prev[k] = make_float4(0.0f);
        A_diag[k] = make_float4(0.0f);
        A_next[k] = make_float4(0.0f);
        B_cols[k] = make_float4(0.0f);
      }
      Float eta_prod = 1.0f;
      Float4 chain_transmittance = make_float4(1.0f);
      Bool any_dispersive = false;
      Bool chain_valid = true;

      UInt d_i = 0u;
      $while(d_i < chain_len) {
        const Bool has_prev = (d_i > 0u);
        const Bool has_next = (d_i + 1u < chain_len);
        const auto vi = load_vertex(v, d_i);
        const auto v_prev = load_vertex(v, select(0u, d_i - 1u, has_prev));
        const auto v_next = load_vertex(v, min(d_i + 1u, kMaxChainLen - 1u));
        const Float3 p_prev = select(task.receiver_p, v_prev.p, has_prev);
        const Float3 T1_prev = select(make_float3(0.0f), v_prev.T1, has_prev);
        const Float3 T2_prev = select(make_float3(0.0f), v_prev.T2, has_prev);
        const Float3 p_light =
            select(task.light_p, task.light_dir, is_distant);
        const Float3 p_next = select(p_light, v_next.p, has_next);

        const auto cv = eval_cv(
            p_prev, vi.p, p_next, vi.n, vi.T1, vi.T2, vi.dn_dx, vi.dn_dy,
            vi.eta, T1_prev, T2_prev, v_next.T1, v_next.T2, has_prev, has_next,
            is_distant, light_e1, light_e2);
        store_f4(A_prev, d_i, cv.A_prev);
        store_f4(A_diag, d_i, cv.A_diag);
        store_f4(A_next, d_i, cv.A_next);
        $if(!has_next) { store_f4(B_cols, d_i, -cv.B_light); };
        chain_valid &= cv.orientation_ok & (dot(cv.h, vi.n) > 0.99f);
        eta_prod *= vi.eta;

        // Re-evaluate exact caster closure transmittance at converged hit.
        const auto seg_vec = vi.p - p_prev;
        const auto seg_d = max(length(seg_vec), 1.0e-6f);
        const auto cam_ray = make_ray(p_prev, seg_vec / seg_d, kRayEps,
                                      cycles_light::distant_ray_length);
        MnTraceHit dummy_h{.hit = true,
                           .inst = vi.inst,
                           .prim = vi.prim,
                           .bary = vi.bary,
                           .t = seg_d,
                           .cycles_object = vi.cycles_object,
                           .object_flags = 0u};
        CasterVertexVar unused_v{};
        Bool found_refr = false;
        Float4 trans = make_float4(0.0f);
        eval_caster_hit(dummy_h, cam_ray, unused_v, found_refr, any_dispersive,
                        &trans, true, vi.backfacing, false);
        chain_valid &= found_refr;
        chain_transmittance *= trans;
        d_i += 1u;
      };

      Bool sens_ok = false;
      std::array<Float4, kMaxChainLen> Y_cols;
      solve_block_tridiagonal(A_prev, A_diag, A_next, B_cols, chain_len,
                              sens_ok, Y_cols);
      chain_valid &= sens_ok;

      // Evaluate light factor at the converged (last_v.p, last_wo).
      Float3 light_factor_rgb = make_float3(0.0f);
      Float4 light_factor_spec = make_float4(0.0f);
      Bool light_ok = false;

      $if(chain_valid) {
        Float scalar_eval_over_pdf = 0.0f;
        $if(is_area) {
          const auto area_hit = AreaLightSampling{}.from_intersection(
              area_light_sample_input(light, last_v.p, make_float2(0.5f)),
              last_wo, last_dist);
          light_ok = area_hit.valid;
          scalar_eval_over_pdf =
              area_hit.evaluation_factor / max(task.light_area_pdf, 1.0e-20f);
        }
        $elif(is_distant) {
          const auto half_angle = 0.5f * max(light.angle, 0.0f);
          const auto finite = half_angle > 0.0f;
          const auto sine_half = sin(half_angle);
          const auto disk_area = pi * sine_half * sine_half;
          const auto normalize_power =
              (light.flags & light_flag_normalize) != 0u;
          const auto distant_eval =
              select(1.0f, 1.0f / max(disk_area, 1.0e-20f),
                     normalize_power & finite);
          light_ok = true;
          scalar_eval_over_pdf =
              distant_eval / max(task.light_area_pdf, 1.0e-20f);
        }
        $else {
          const auto spot =
              light.type == static_cast<std::uint32_t>(LightType::spot);
          const auto normalize_power =
              (light.flags & light_flag_normalize) != 0u;
          const auto pt_factor = analytic_light_sampling::point_eval_factor(
              light.radius, normalize_power);
          Float atten = 1.0f;
          Bool side_ok = (light.radius == 0.0f) |
                         (dot(-last_wo, task.light_normal) > 0.0f);
          $if(spot) {
            const auto l_tfm = analytic_light_sampling::light_linear_transform(
                light.axis_x, light.axis_y, light.axis_z, light.axis_scale);
            const auto local_ray =
                analytic_light_sampling::spot_light_to_local(-last_wo, l_tfm);
            atten = analytic_light_sampling::spot_light_attenuation(local_ray,
                                                                    light.spot);
          };
          light_ok = side_ok & (atten > 0.0f);
          scalar_eval_over_pdf =
              (pt_factor * atten) / max(task.light_area_pdf, 1.0e-20f);
        };

        $if(light_ok & (scalar_eval_over_pdf > 0.0f)) {
          Float4 shader_val =
              (is_xyz3 && spec_ctx)
                  ? cycles_spectrum::xyz3_rotate_eval(task.light_shader.xyz(),
                                                      *spec_ctx->hero_index)
                  : task.light_shader;
          $if(task.light_constant_shader == 0u) {
            Var<DirectLightTaskCall> dl_task{};
            dl_task.ray_origin = last_v.p;
            dl_task.ray_direction = last_wo;
            dl_task.ray_minimum = kRayEps;
            dl_task.ray_maximum = last_dist;
            dl_task.ray_dP = task.ray_dP;
            dl_task.ray_dD = task.ray_dD;
            dl_task.ray_time = task.ray_time;
            dl_task.sample_index = task.sample_index;
            dl_task.rng_hash = task.rng_hash;
            dl_task.rng_offset = task.rng_offset;
            dl_task.light_object = task.light_cycles_object;
            dl_task.light_primitive = task.light_index;
            dl_task.shader_flags = task.light_shader_flags;
            dl_task.path_depth = task.path_depth + chain_len;
            dl_task.diffuse_depth = task.diffuse_depth;
            dl_task.glossy_depth = task.glossy_depth;
            dl_task.transmission_depth = task.transmission_depth + chain_len;
            dl_task.transparent_depth =
                task.transparent_depth.cast<std::uint16_t>();
            dl_task.portal_depth = task.portal_depth.cast<std::uint16_t>();
            if (spectral_rendering && spec_ctx) {
              shader_val = light_emission->evaluate_spectral(dl_task,
                                                               parameters,
                                                               *spec_ctx);
            } else {
              shader_val = make_float4(
                  light_emission->evaluate(dl_task, parameters), 0.0f);
            }
          };
          if (spectral_rendering && spec_ctx) {
            const auto color_spec =
                cycles_spectrum::rgb_to_spectrum_reflectance(
                    *spec_ctx->tables, light.color,
                    spec_ctx->wavelengths.lambdas);
            light_factor_spec =
                color_spec * shader_val * (light.power * scalar_eval_over_pdf);
          } else {
            light_factor_rgb = light.color * shader_val.xyz() *
                               (light.power * scalar_eval_over_pdf);
          }
        };
      };

      $if(chain_valid & light_ok) {
        const auto Y0 = Y_cols[0u];
        const auto det_Y0 = abs(Y0.x * Y0.w - Y0.y * Y0.z);
        const auto to_v0 = v[0u].p - task.receiver_p;
        const auto dist0_sq = max(length_squared(to_v0), 1.0e-12f);
        const auto wo_receiver = to_v0 / sqrt(dist0_sq);
        const auto cos_v0 = abs(dot(wo_receiver, v[0u].ng));
        const auto eta_sq_factor = 1.0f / max(eta_prod * eta_prod, 1.0e-8f);
        const auto G_term = (cos_v0 / dist0_sq) * det_Y0 * eta_sq_factor;

        $if(manifold_jacobian_within_bounds(cos_v0, det_Y0, G_term)) {
          // Receiver BSDF evaluation (masked to caustics_receiver_shader_mask).
          const PathCyclesSvmKernelGlobals kg_recv{
              scene,
              parameters,
              camera_projection,
              true,
              true,
              PathCyclesSvmShaderDomain::surface_only,
              scene->caustics_receiver_shader_mask,
              false,
              nullptr,
              nullptr,
              spectral_rendering,
              spectral_mode};
          const auto recv_pool_opts =
              closure_pool_options_for_kg(kg_recv, spectral_rendering);
          svm::ClosurePool recv_closures{closure_cap, recv_pool_opts};
          const auto recv_cam_ray = make_ray(
              task.receiver_p + task.receiver_wi, -task.receiver_wi, 0.0f, 2.0f);
          const auto recv_lcg = cycles_noise::hash_uint3(
              task.rng_hash ^ 0x9e3779b9u, task.rng_offset, task.sample_index);
          auto recv_setup = setup_cycles_svm_ray_shader_data(
              scene, kg_recv, recv_cam_ray, task.receiver_inst,
              task.receiver_prim, task.receiver_bary, 0.0f, task.ray_dP,
              task.ray_dD, task.ray_time, recv_lcg, parameters);
          auto &sd_recv = recv_setup.shader_data;
          sd_recv.closure = &recv_closures;
          if (spec_ctx) {
            sd_recv.spectral = *spec_ctx;
          }
          const svm::PathState recv_state{
              task.path_visibility,
              0u,
              task.path_depth,
              task.transparent_depth,
              task.diffuse_depth,
              task.glossy_depth,
              task.transmission_depth,
              task.portal_depth};
          evaluate_masked_svm(scene, kg_recv, recv_setup.transforms, sd_recv,
                              recv_state, scene->caustics_receiver_shader_mask);
          const auto bsdf_eval = svm_detail::surface_shader_bsdf_eval(
              kg_recv, sd_recv, wo_receiver, task.light_shader_flags,
              caster_closure_types);

          if (is_xyz3 && spec_ctx) {
            const auto bsdf_spec = bsdf_eval.spectral_sum.value_or(
                make_float4(bsdf_eval.sum, 0.0f));
            const auto raw_rgb = eval_xyz3_caustic_raw_rgb(
                *spectral_tables, *spec_ctx, task.path_throughput, bsdf_spec,
                chain_transmittance, light_factor_spec, G_term, hero_lambda,
                secondary_terminated, any_dispersive);
            const auto clamped = clamp_contribution(
                raw_rgb, task.path_depth + chain_len,
                parameters.sample_clamp_direct, parameters.sample_clamp_indirect);
            const auto finite_nz =
                !(any(luisa::compute::isnan(clamped)) |
                  any(luisa::compute::isinf(clamped))) &
                any(clamped != 0.0f);
            result_rgb = select(make_float3(0.0f), clamped, finite_nz);
            eval_valid = finite_nz;
            if (need_weights) {
              compute_xyz3_bsdf_component_weights(
                  *spectral_tables, *spec_ctx, light_component_ratio, bsdf_spec,
                  bsdf_eval.spectral_diffuse.value_or(
                      make_float4(bsdf_eval.diffuse, 0.0f)),
                  bsdf_eval.spectral_glossy.value_or(
                      make_float4(bsdf_eval.glossy, 0.0f)),
                  eval_diff_w, eval_gloss_w);
            }
          } else if (spectral_rendering && spec_ctx) {
            const auto bsdf_spec = bsdf_eval.spectral_sum.value_or(
                make_float4(bsdf_eval.sum, 0.0f));
            const auto unclamped_spec =
                task.path_throughput * bsdf_spec * chain_transmittance *
                light_factor_spec * G_term;
            const auto raw_rgb = cycles_spectrum::spectrum_to_rgb(
                *spectral_tables, unclamped_spec, spec_ctx->wavelengths,
                secondary_terminated | any_dispersive);
            const auto clamped = clamp_contribution(
                raw_rgb, task.path_depth + chain_len,
                parameters.sample_clamp_direct, parameters.sample_clamp_indirect);
            const auto finite_pos =
                !(any(luisa::compute::isnan(clamped)) |
                  any(luisa::compute::isinf(clamped))) &
                any(clamped > 0.0f);
            result_rgb = select(
                make_float3(0.0f), max(clamped, make_float3(0.0f)), finite_pos);
            eval_valid = finite_pos;
            if (need_weights) {
              const auto bsdf_rgb = cycles_spectrum::spectrum_to_rgb(
                  *spectral_tables, bsdf_spec, spec_ctx->wavelengths,
                  secondary_terminated);
              const auto diff_rgb = cycles_spectrum::spectrum_to_rgb(
                  *spectral_tables,
                  bsdf_eval.spectral_diffuse.value_or(
                      make_float4(bsdf_eval.diffuse, 0.0f)),
                  spec_ctx->wavelengths, secondary_terminated);
              const auto gloss_rgb = cycles_spectrum::spectrum_to_rgb(
                  *spectral_tables,
                  bsdf_eval.spectral_glossy.value_or(
                      make_float4(bsdf_eval.glossy, 0.0f)),
                  spec_ctx->wavelengths, secondary_terminated);
              eval_diff_w = light_component_ratio(diff_rgb, bsdf_rgb);
              eval_gloss_w = light_component_ratio(gloss_rgb, bsdf_rgb);
            }
          } else {
            const auto unclamped_rgb =
                task.path_throughput.xyz() * bsdf_eval.sum *
                chain_transmittance.xyz() * light_factor_rgb * G_term;
            const auto clamped = clamp_contribution(
                unclamped_rgb, task.path_depth + chain_len,
                parameters.sample_clamp_direct, parameters.sample_clamp_indirect);
            const auto finite_pos =
                !(any(luisa::compute::isnan(clamped)) |
                  any(luisa::compute::isinf(clamped))) &
                any(clamped > 0.0f);
            result_rgb = select(
                make_float3(0.0f), max(clamped, make_float3(0.0f)), finite_pos);
            eval_valid = finite_pos;
            if (need_weights) {
              eval_diff_w =
                  light_component_ratio(bsdf_eval.diffuse, bsdf_eval.sum);
              eval_gloss_w =
                  light_component_ratio(bsdf_eval.glossy, bsdf_eval.sum);
            }
          }
        }
        $else {
          eval_valid = false;
        };
      }
      $else {
        eval_valid = false;
      };
    };
  };

  valid = eval_valid;
  if (need_weights) {
    *out_diffuse_weight = eval_diff_w;
    *out_glossy_weight = eval_gloss_w;
  }
  return result_rgb;
}

void CausticsTaskEvaluator::evaluate_and_accumulate_atomic(
    const Var<CausticTaskCall> &task, const DirectLightTaskFilm &film,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Bool valid = false;
  Float3 diff_w = make_float3(0.0f);
  Float3 gloss_w = make_float3(0.0f);
  const auto contrib = evaluate(task, parameters, valid, &diff_w, &gloss_w);
  $if(valid & any(contrib != 0.0f)) {
    atomic_accumulate_radiance(
        film.combined, film.volume_guiding_raw, task.pixel,
        task.pixel * volume_guiding::raw_pixel_stride, volume_guiding,
        task.path_flags, task.path_visibility, task.path_depth, contrib);
    const auto no_pass =
        (task.path_flags & cycles_path_state::flag_any_pass) == 0u;
    const auto pass_flags =
        select(task.path_flags,
               task.path_flags | cycles_path_state::flag_surface_pass, no_pass);
    const auto pass_diff = select(task.path_diffuse_weight, diff_w, no_pass);
    const auto pass_gloss = select(task.path_glossy_weight, gloss_w, no_pass);
    atomic_accumulate_scattered_light_passes(
        film.light_passes, task.pixel * light_pass_buffer_count, contrib,
        pass_flags, pass_diff, pass_gloss, false);
  };
}

void CausticsTaskEvaluator::evaluate_and_accumulate_sample(
    PathSampleContext &sample, const Var<CausticTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Bool valid = false;
  Float3 diff_w = make_float3(0.0f);
  Float3 gloss_w = make_float3(0.0f);
  const auto contrib = evaluate(task, parameters, valid, &diff_w, &gloss_w);
  $if(valid & any(contrib != 0.0f)) {
    sample.accumulate_radiance_at_state(
        contrib, task.path_flags, task.path_visibility, task.path_depth);
    UInt pass_flags = task.path_flags;
    Float3 d_w = sample.path_diffuse_weight;
    Float3 g_w = sample.path_glossy_weight;
    $if((pass_flags & cycles_path_state::flag_any_pass) == 0u) {
      pass_flags |= cycles_path_state::flag_surface_pass;
      d_w = diff_w;
      g_w = gloss_w;
    };
    sample.accumulate_scattered_light_at_state(
        contrib, pass_flags, d_w, g_w, false);
  };
}

CausticsTaskEvaluator
make_caustics_task_evaluator(const PathKernelConfig &config) noexcept {
  return {.scene = config.scene,
          .clamp_contribution = config.light_transport.clamp_light_contribution,
          .light_component_ratio = config.light_transport.light_component_ratio,
          .light_emission = make_cycles_svm_light_emission_component(
              config.scene, config.camera_projection,
              config.reflective_caustics, config.refractive_caustics),
          .spectral_tables =
              config.spectral_rendering && config.scene ? config.scene->spectral_tables
                                                        : nullptr,
          .camera_projection = config.camera_projection,
          .spectral_rendering = config.spectral_rendering,
          .spectral_mode = config.effective_spectral_mode(),
          .volume_guiding = config.volume_state != nullptr,
          .reflective_caustics = config.reflective_caustics,
          .refractive_caustics = config.refractive_caustics,
          .caustics_method = config.caustics_method,
          .sms_unbiased = config.sms_unbiased,
          .sms_max_trials = config.sms_max_trials};
}

} // namespace psycles::luisa_backend::detail
