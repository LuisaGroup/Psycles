/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_noise.h>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_object_info(Cursor &cursor, Stack &stack,
                      const InfoServices &services,
                      const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  Float data = 0.0f;

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_OB_LOCATION) {
      stack_store_float3(
          stack, output_offset, services.object_location(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_INFO_OB_COLOR) {
      stack_store_float3(
          stack, output_offset, services.object_color(shader_data.object));
    };
    PSYCLES_SVM_CASE(NODE_INFO_OB_ALPHA) {
      data = services.object_alpha(shader_data.object);
      stack_store_float(stack, output_offset, data);
    };
    PSYCLES_SVM_CASE(NODE_INFO_OB_INDEX) {
      data = services.object_pass_id(shader_data.object);
      stack_store_float(stack, output_offset, data);
    };
    PSYCLES_SVM_CASE(NODE_INFO_MAT_INDEX) {
      data = services.shader_pass_id(shader_data);
      stack_store_float(stack, output_offset, data);
    };
    PSYCLES_SVM_CASE(NODE_INFO_OB_RANDOM) {
      data = services.object_random_number(shader_data.object);
      stack_store_float(stack, output_offset, data);
    };
    $default { stack_store_float(stack, output_offset, 0.0f); };
  };
}

void node_particle_info(Cursor &cursor, Stack &stack,
                        const InfoServices &services,
                        const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_PAR_INDEX) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float(
          stack, output_offset,
          services.particle_index(particle).cast<float>());
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_RANDOM) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      const auto index = services.particle_index(particle);
      stack_store_float(
          stack, output_offset,
          cycles_noise::uint_to_float_inclusive(
              cycles_noise::hash_uint2(index, 0u)));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_AGE) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float(
          stack, output_offset, services.particle_age(particle));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_LIFETIME) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float(
          stack, output_offset, services.particle_lifetime(particle));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_LOCATION) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float3(
          stack, output_offset, services.particle_location(particle));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_SIZE) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float(
          stack, output_offset, services.particle_size(particle));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_VELOCITY) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float3(
          stack, output_offset, services.particle_velocity(particle));
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_ANGULAR_VELOCITY) {
      const auto particle =
          services.object_particle_id(shader_data.object);
      stack_store_float3(
          stack, output_offset,
          services.particle_angular_velocity(particle));
    };
    $default {};
  };
}

void node_hair_info(Cursor &cursor, Stack &stack,
                    const InfoServices *services,
                    const ShaderData &shader_data,
                    Bool &supported) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_IS_STRAND) {
      stack_store_float(
          stack, output_offset,
          select(0.0f, 1.0f,
                 (shader_data.type & primitive_curve) != 0u));
    };
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_INTERCEPT) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_LENGTH) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_RANDOM) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_THICKNESS) {
      if (services != nullptr) {
        stack_store_float(
            stack, output_offset, services->curve_thickness(shader_data));
      } else {
        supported = false;
      }
    };
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_TANGENT_NORMAL) {
      Float3 tangent_normal = make_float3(0.0f);
      $if ((shader_data.type & primitive_curve) != 0u) {
        const auto incoming = -shader_data.wi;
        tangent_normal =
            -(incoming - shader_data.dPdu *
                             (dot(shader_data.dPdu, incoming) /
                              dot(shader_data.dPdu, shader_data.dPdu)));
        tangent_normal = normalize_cycles(tangent_normal);
      };
      stack_store_float3(stack, output_offset, tangent_normal);
    };
    $default {};
  };
}

void node_point_info(Cursor &cursor, Stack &stack,
                     const InfoServices &services,
                     const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_POINT_POSITION) {
      stack_store_float3(
          stack, output_offset, services.point_position(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_INFO_POINT_RADIUS) {
      stack_store_float(
          stack, output_offset, services.point_radius(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_INFO_POINT_RANDOM) {};
    $default {};
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE
