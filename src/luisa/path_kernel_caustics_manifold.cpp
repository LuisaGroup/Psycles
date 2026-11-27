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
  bool orientation_ok{};
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::MnInv2x2, inv, ok){};
LUISA_STRUCT(psycles::luisa_backend::detail::ConstraintVertexEval,
             C, A_prev, A_diag, A_next, B_light, h, wi, wo, orientation_ok){};

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

  Float3 result_rgb = make_float3(0.0f);
  Bool eval_valid = false;
  Float3 eval_diff_w = make_float3(0.0f);
  Float3 eval_gloss_w = make_float3(0.0f);
  const bool need_weights =
      out_diffuse_weight != nullptr && out_glossy_weight != nullptr;

  $outline_with_name(spectral_rendering ? "mnee_evaluate_spectral"
                                        : "mnee_evaluate_rgb") {
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
      spec_ctx.emplace(cycles_spectrum::SpectralContext{
          .tables = spectral_tables.get(),
          .wavelengths =
              cycles_spectrum::sample_visible_wavelengths(task.u_lambda),
          .secondary_terminated = secondary_terminated});
      hero_lambda = spec_ctx->wavelengths.lambdas.x;
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
        spectral_rendering};

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
          Float4 shader_val = task.light_shader;
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
            spectral_rendering};
        const auto recv_pool_opts =
            closure_pool_options_for_kg(kg_recv, spectral_rendering);
        svm::ClosurePool recv_closures{closure_cap, recv_pool_opts};
        const auto recv_cam_ray = make_ray(
            task.receiver_p + task.receiver_wi, -task.receiver_wi, 0.0f, 2.0f);
        const auto recv_lcg = cycles_noise::hash_uint3(
            task.rng_hash ^ 0x9e3779b9u, task.rng_offset, task.sample_index);
        auto recv_setup = setup_cycles_svm_ray_shader_data(
            scene, kg_recv, recv_cam_ray, task.receiver_inst, task.receiver_prim,
            task.receiver_bary, 0.0f, task.ray_dP, task.ray_dD, task.ray_time,
            recv_lcg, parameters);
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

        if (spectral_rendering && spec_ctx) {
          const auto bsdf_spec =
              bsdf_eval.spectral_sum.value_or(make_float4(bsdf_eval.sum, 0.0f));
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
          result_rgb = select(make_float3(0.0f), max(clamped, make_float3(0.0f)),
                              finite_pos);
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
          result_rgb = select(make_float3(0.0f), max(clamped, make_float3(0.0f)),
                              finite_pos);
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
          .volume_guiding = config.volume_state != nullptr,
          .reflective_caustics = config.reflective_caustics,
          .refractive_caustics = config.refractive_caustics};
}

} // namespace psycles::luisa_backend::detail
