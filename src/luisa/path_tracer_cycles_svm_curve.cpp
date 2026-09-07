/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_tracer_cycles_svm_curve.h"

#include "curve_ribbon_component.h"
#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::detail {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

void cycles_svm_curve_shader_setup(
    const std::shared_ptr<LuisaSceneData> &scene,
    const svm::KernelGlobals &kg, const svm::TransformState &transforms,
    const Var<InstanceGpu> &instance, Expr<unsigned> segment_id,
    const Var<luisa::compute::Ray> &ray, svm::ShaderData &sd) noexcept {
  const auto geometry = scene->geometry_buffer->read(instance.geometry_index);
  const auto segment = scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base)
                           .read(segment_id);
  sd.prim = segment.cycles_curve_index;
  const auto curve = kg.curve(sd.prim);
  const auto segment_index = segment.key_begin - curve.first_key.cast<unsigned>();
  sd.type = curve.type.cast<unsigned>() | (segment_index << svm::primitive_num_bits);
  // The scene upload admits static ribbons only. Other curve types must get
  // their own Cycles intersection/setup, never be reinterpreted as ribbons.
  assume((sd.type & unsigned(abi::PRIMITIVE_ALL)) == svm::primitive_curve_ribbon);
  const auto first = curve.first_key;
  const auto k0 = first + segment_index.cast<int>();
  const auto k1 = k0 + 1;
  const auto position_offset = kg.object_position_offset(sd.object);
  const CurveControlPoints keys{
      .before = kg.curve_key(position_offset + max(k0 - 1, first)),
      .begin = kg.curve_key(position_offset + k0),
      .end = kg.curve_key(position_offset + k1),
      .after = kg.curve_key(position_offset + min(k1 + 1, first + curve.num_keys - 1))};
  const auto applied = (sd.object_flag & svm::shader_data_object_transform_applied) != 0u;
  Float3 P = ray->origin(), D = ray->direction();
  $if(!applied) {
    P = cycles_transform::point(transforms.world_to_object, P);
    D = cycles_transform::direction(transforms.world_to_object, D);
  };
  const auto ribbon = make_curve_ribbon_component();
  // A procedural CommittedHit has no u/v payload. Reconstruct the same
  // intersection coordinates without replacing its committed world ray t.
  const auto local_ray = make_ray(P, D, ray->t_min(), ray->t_max());
  const auto intersection = ribbon->intersect(local_ray, keys, geometry.curve_subdivision_level);
  sd.u = intersection.u;
  sd.v = intersection.v;

  // Cycles curve_shader_setup transforms D*t and uses safe_normalize_len,
  // whereas the ribbon intersection above permits a non-unit direction.
  Float t = sd.ray_length;
  $if(!applied) {
    D = cycles_transform::direction(transforms.world_to_object, ray->direction() * t);
    t = length(D);
    D = select(D, D / t, t != 0.0f);
  };
  P += D * t;
  sd.dPdu = ribbon->derivative(keys, sd.u).xyz();
  const auto tangent = normalize(sd.dPdu);
  const auto bitangent = normalize(cross(tangent, -D));
  const auto cosine = sqrt(max(1.0f - sd.v * sd.v, 0.0f));
  sd.N = normalize(sd.v * bitangent - cosine * normalize(cross(tangent, bitangent)));
  $if(!applied) {
    P = cycles_transform::point(transforms.object_to_world, P);
    svm::detail::object_normal_transform(sd.N, transforms, sd, false);
    svm::detail::object_dir_transform(sd.dPdu, transforms, sd, false);
  };
  sd.P = P;
  sd.Ng = sd.wi;
  sd.dPdv = cross(sd.dPdu, sd.Ng);
  sd.shader = curve.shader_id.cast<unsigned>();
}

} // namespace psycles::luisa_backend::detail
