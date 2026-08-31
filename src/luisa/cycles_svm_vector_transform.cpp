/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>
#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float3 normalize_cycles(Expr<luisa::float3> value) noexcept {
  // Cycles' unchecked NORMALIZE preserves a non-finite zero-vector boundary.
  // Use backend-native rsqrt rather than preserving a scalar sqrt/div path;
  // Luisa's generic normalize is zero-safe on some backends.
  return native_vector_math::normalize_unchecked(value);
}

[[nodiscard]] Float3 safe_normalize_cycles(Expr<luisa::float3> value) noexcept {
  const Float length_squared = dot(value, value);
  Float3 result = value;
  // The zero-vector guard is structural Cycles behavior; only the non-zero
  // arithmetic is delegated to the backend-native NORMALIZE operation.
  $if (length_squared != 0.0f) { result = normalize(value); };
  return result;
}

[[nodiscard]] Bool object_has_motion(const ShaderData &shader_data) noexcept {
  return (shader_data.object_flag & shader_data_object_motion) != 0u;
}

void object_position_transform(Float3 &value,
                               const TransformState &transform_state,
                               const ShaderData &shader_data,
                               bool object_motion_enabled) noexcept {
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = cycles_transform::point(shader_data.ob_tfm_motion, value);
    }
    $else {
      value = cycles_transform::point(transform_state.object_to_world, value);
    };
  } else {
    value = cycles_transform::point(transform_state.object_to_world, value);
  }
}

void object_inverse_position_transform(Float3 &value,
                                       const TransformState &transform_state,
                                       const ShaderData &shader_data,
                                       bool object_motion_enabled) noexcept {
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = cycles_transform::point(shader_data.ob_itfm_motion, value);
    }
    $else {
      value = cycles_transform::point(transform_state.world_to_object, value);
    };
  } else {
    value = cycles_transform::point(transform_state.world_to_object, value);
  }
}

void object_normal_transform(Float3 &value,
                             const TransformState &transform_state,
                             const ShaderData &shader_data,
                             bool object_motion_enabled) noexcept {
  const Bool is_object = shader_data.object != object_none;
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = normalize_cycles(cycles_transform::direction_transposed(
          shader_data.ob_itfm_motion, value));
    }
    $elif(is_object) {
      value = normalize_cycles(cycles_transform::direction_transposed(
          transform_state.world_to_object, value));
    };
  } else {
    $if(is_object) {
      value = normalize_cycles(cycles_transform::direction_transposed(
          transform_state.world_to_object, value));
    };
  }
}

void object_inverse_normal_transform(Float3 &value,
                                     const TransformState &transform_state,
                                     const ShaderData &shader_data,
                                     bool object_motion_enabled) noexcept {
  const Bool is_object = shader_data.object != object_none;
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      $if(is_object) {
        value = safe_normalize_cycles(cycles_transform::direction_transposed(
            shader_data.ob_tfm_motion, value));
      };
    }
    $elif(is_object) {
      value = safe_normalize_cycles(cycles_transform::direction_transposed(
          transform_state.object_to_world, value));
    };
  } else {
    $if(is_object) {
      value = safe_normalize_cycles(cycles_transform::direction_transposed(
          transform_state.object_to_world, value));
    };
  }
}

void object_dir_transform(Float3 &value,
                          const TransformState &transform_state,
                          const ShaderData &shader_data,
                          bool object_motion_enabled) noexcept {
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = cycles_transform::direction(shader_data.ob_tfm_motion, value);
    }
    $else {
      value =
          cycles_transform::direction(transform_state.object_to_world, value);
    };
  } else {
    value = cycles_transform::direction(transform_state.object_to_world, value);
  }
}

void object_inverse_dir_transform(Float3 &value,
                                  const TransformState &transform_state,
                                  const ShaderData &shader_data,
                                  bool object_motion_enabled) noexcept {
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = cycles_transform::direction(shader_data.ob_itfm_motion, value);
    }
    $else {
      value =
          cycles_transform::direction(transform_state.world_to_object, value);
    };
  } else {
    value = cycles_transform::direction(transform_state.world_to_object, value);
  }
}

void node_vector_transform(Cursor &cursor, Stack &stack,
                           const TransformState &transform_state,
                           const ShaderData &shader_data,
                           bool object_motion_enabled) noexcept {
  const auto transform_type = cursor.word();
  const auto convert_from = cursor.word();
  const auto convert_to = cursor.word();
  const auto vector_x = cursor.word();
  const auto vector_y = cursor.word();
  const auto vector_z = cursor.word();
  const auto vector_out_offset = cursor.word();

  Float3 value = stack_load_input_float3(stack, vector_x, vector_y, vector_z);
  const Bool is_object = shader_data.object != object_none;
  const Bool is_normal =
      transform_type ==
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_TYPE_NORMAL);
  const Bool is_direction =
      transform_type ==
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_TYPE_VECTOR);

  /* From world. */
  $if (convert_from ==
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD)) {
    $if (convert_to == static_cast<std::uint32_t>(
                          NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA)) {
      $if (is_normal) {
        value = normalize_cycles(cycles_transform::direction_transposed(
            transform_state.camera_to_world, value));
      }
      $else {
        $if (is_direction) {
          value = cycles_transform::direction(transform_state.world_to_camera,
                                              value);
        }
        $else {
          value =
              cycles_transform::point(transform_state.world_to_camera, value);
        };
      };
    }
    $else {
      $if ((convert_to == static_cast<std::uint32_t>(
                             NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT)) &
          is_object) {
        $if (is_normal) {
          object_inverse_normal_transform(value, transform_state, shader_data,
                                          object_motion_enabled);
        }
        $else {
          $if (is_direction) {
            object_inverse_dir_transform(
                value, transform_state, shader_data, object_motion_enabled);
          }
          $else {
            object_inverse_position_transform(
                value, transform_state, shader_data, object_motion_enabled);
          };
        };
      };
    };
  }

  /* From camera. */
  $elif (convert_from == static_cast<std::uint32_t>(
                            NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA)) {
    $if ((convert_to == static_cast<std::uint32_t>(
                           NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD)) |
        (convert_to == static_cast<std::uint32_t>(
                           NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT))) {
      $if (is_normal) {
        value = normalize_cycles(cycles_transform::direction_transposed(
            transform_state.world_to_camera, value));
      }
      $else {
        $if (is_direction) {
          value = cycles_transform::direction(transform_state.camera_to_world,
                                              value);
        }
        $else {
          value =
              cycles_transform::point(transform_state.camera_to_world, value);
        };
      };
    };
    $if ((convert_to == static_cast<std::uint32_t>(
                           NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT)) &
        is_object) {
      $if (is_normal) {
        object_inverse_normal_transform(value, transform_state, shader_data,
                                        object_motion_enabled);
      }
      $else {
        $if (is_direction) {
          object_inverse_dir_transform(
              value, transform_state, shader_data, object_motion_enabled);
        }
        $else {
          object_inverse_position_transform(value, transform_state, shader_data,
                                            object_motion_enabled);
        };
      };
    };
  }

  /* From object. */
  $elif (convert_from == static_cast<std::uint32_t>(
                            NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT)) {
    $if (((convert_to == static_cast<std::uint32_t>(
                            NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD)) |
         (convert_to == static_cast<std::uint32_t>(
                            NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA))) &
        is_object) {
      $if (is_normal) {
        object_normal_transform(value, transform_state, shader_data,
                                object_motion_enabled);
      }
      $else {
        $if (is_direction) {
          object_dir_transform(value, transform_state, shader_data,
                               object_motion_enabled);
        }
        $else {
          object_position_transform(value, transform_state, shader_data,
                                    object_motion_enabled);
        };
      };
    };
    $if (convert_to == static_cast<std::uint32_t>(
                          NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA)) {
      $if (is_normal) {
        value = normalize_cycles(cycles_transform::direction_transposed(
            transform_state.camera_to_world, value));
      }
      $else {
        $if (is_direction) {
          value = cycles_transform::direction(transform_state.world_to_camera,
                                              value);
        }
        $else {
          value =
              cycles_transform::point(transform_state.world_to_camera, value);
        };
      };
    };
  };

  $if (vector_out_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, vector_out_offset, value);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
