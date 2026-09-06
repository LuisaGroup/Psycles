#include "path_tracer_cycles_svm_shader_data.h"

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::detail {
using namespace luisa::compute;
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;
namespace sd_detail = svm::detail;

void cycles_svm_shader_setup_backfacing(svm::ShaderData &sd) noexcept {
  $if(dot(sd.Ng, sd.wi) < 0.0f) {
    sd.flag |= static_cast<unsigned>(abi::SD_BACKFACING);
    sd.N = -sd.N;
    sd.Ng = -sd.Ng;
    sd.dPdu = -sd.dPdu;
    sd.dPdv = -sd.dPdv;
  };
}

// Cycles differential_dudv_compact, including its stable-axis selection and
// exact zero-determinant branch. There is no application epsilon clamp.
void cycles_svm_shader_setup_dudv(svm::ShaderData &sd) noexcept {
  auto dP = sd_detail::differential_from_compact(sd.Ng, sd.dP);
  auto dPdu = def(sd.dPdu);
  auto dPdv = def(sd.dPdv);
  const auto n = abs(sd.Ng);
  $if((n.z < n.x) | (n.z < n.y)) {
    $if((n.y < n.x) | (n.y < n.z)) {
      dPdu.x = dPdu.y;
      dPdv.x = dPdv.y;
      dP.dx.x = dP.dx.y;
      dP.dy.x = dP.dy.y;
    };
    dPdu.y = dPdu.z;
    dPdv.y = dPdv.z;
    dP.dx.y = dP.dx.z;
    dP.dy.y = dP.dy.z;
  };
  Float determinant = dPdu.x * dPdv.y - dPdv.x * dPdu.y;
  $if(determinant != 0.0f) { determinant = 1.0f / determinant; };
  sd.du.dx = (dP.dx.x * dPdv.y - dP.dx.y * dPdv.x) * determinant;
  sd.dv.dx = (dP.dx.y * dPdu.x - dP.dx.x * dPdu.y) * determinant;
  sd.du.dy = (dP.dy.x * dPdv.y - dP.dy.y * dPdv.x) * determinant;
  sd.dv.dy = (dP.dy.y * dPdu.x - dP.dy.x * dPdu.y) * determinant;
}

void cycles_svm_triangle_shader_setup(const svm::KernelGlobals &kg,
                                      const svm::TransformState &transforms,
                                      const svm::TriangleVertices &vertices,
                                      svm::ShaderData &sd) noexcept {
  const auto transform_applied =
      (sd.object_flag & svm::shader_data_object_transform_applied) != 0u;
  sd.dPdu = vertices.v1 - vertices.v0;
  sd.dPdv = vertices.v2 - vertices.v0;
  sd.P = vertices.v0 + sd.u * sd.dPdu + sd.v * sd.dPdv;
  $if(!transform_applied) {
    sd.P = cycles_transform::point(transforms.object_to_world, sd.P);
  };
  const auto negative_scale_applied =
      transform_applied &
      ((sd.object_flag &
        static_cast<unsigned>(abi::SD_OBJECT_NEGATIVE_SCALE)) != 0u);
  sd.Ng = sd_detail::normalize_cycles(select(cross(sd.dPdu, sd.dPdv),
                                             cross(sd.dPdv, sd.dPdu),
                                             negative_scale_applied));
  sd.N = sd.Ng;
  $if((sd.shader & svm::shader_smooth_normal) != 0u) {
    const auto normals = sd_detail::triangle_normals(kg, sd);
    const auto smooth =
        sd_detail::safe_normalize_cycles((1.0f - sd.u - sd.v) * normals.n0 +
                                         sd.u * normals.n1 + sd.v * normals.n2);
    sd.N = select(smooth, sd.Ng, all(smooth == 0.0f));
  };
  $if(!transform_applied) {
    sd_detail::object_normal_transform(sd.N, transforms, sd, false);
    sd_detail::object_normal_transform(sd.Ng, transforms, sd, false);
    sd_detail::object_dir_transform(sd.dPdu, transforms, sd, false);
    sd_detail::object_dir_transform(sd.dPdv, transforms, sd, false);
  };
}

} // namespace psycles::luisa_backend::detail
