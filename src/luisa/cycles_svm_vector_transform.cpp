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
  const Float3 v = value;
  const Float length_squared = dot(v, v);
  // Branchless zero-vector guard: preserves zero vectors without NaN while
  // avoiding per-call CFG diamonds inside mega-kernels.
  const Float inv_len = rsqrt(max(length_squared, 1.0e-38f));
  return select(v, v * inv_len, length_squared != 0.0f);
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
      value = cycles_transform::point(transform_state.fetch_object_to_world(), value);
    };
  } else {
    value = cycles_transform::point(transform_state.fetch_object_to_world(), value);
  }
}

void object_position_transform(Dual3 &value,
                               const TransformState &transform_state,
                               const ShaderData &shader_data,
                               bool object_motion_enabled) noexcept {
  if (object_motion_enabled) {
    $if (object_has_motion(shader_data)) {
      value = transform_point(shader_data.ob_tfm_motion, value);
    }
    $else {
      value = transform_point(transform_state.fetch_object_to_world(), value);
    };
  } else {
    value = transform_point(transform_state.fetch_object_to_world(), value);
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
      value = cycles_transform::point(transform_state.fetch_world_to_object(), value);
    };
  } else {
    value = cycles_transform::point(transform_state.fetch_world_to_object(), value);
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
          transform_state.fetch_world_to_object(), value));
    };
  } else {
    $if(is_object) {
      value = normalize_cycles(cycles_transform::direction_transposed(
          transform_state.fetch_world_to_object(), value));
    };
  }
}

void object_normal_transform(Dual3 &value,
                             const TransformState &transform_state,
                             const ShaderData &shader_data,
                             bool object_motion_enabled) noexcept {
  const auto transform = [](Expr<luisa::float4x4> matrix,
                            const Dual3 &input) noexcept {
    return Dual3{
        .val = cycles_transform::direction_transposed(matrix, input.val),
        .dx = cycles_transform::direction_transposed(matrix, input.dx),
        .dy = cycles_transform::direction_transposed(matrix, input.dy)};
  };
  const Bool is_object = shader_data.object != object_none;
  if (object_motion_enabled) {
    $if(object_has_motion(shader_data)) {
      value = normalize_dual_cycles(transform(shader_data.ob_itfm_motion,
                                              value));
    }
    $elif(is_object) {
      value = normalize_dual_cycles(transform(transform_state.fetch_world_to_object(),
                                              value));
    };
  } else {
    $if(is_object) {
      value = normalize_dual_cycles(transform(transform_state.fetch_world_to_object(),
                                              value));
    };
  }
}

void object_inverse_normal_transform(Float3 &value,
                                     const TransformState &transform_state,
                                     const ShaderData &shader_data,
                                     bool object_motion_enabled) noexcept {
  if (!object_motion_enabled && transform_state.lazy_object_buffer &&
      *transform_state.lazy_object_buffer) {
    static auto callable = []() noexcept {
      Callable result{[](BufferVar<KernelObject> objects, UInt object,
                         Float3 n) noexcept {
        Float3 out = n;
        $if(object != object_none) {
          const auto t = objects.read(object).tfm;
          const Float3 r0 = make_float3(t.x.x, t.x.y, t.x.z);
          const Float3 r1 = make_float3(t.y.x, t.y.y, t.y.z);
          const Float3 r2 = make_float3(t.z.x, t.z.y, t.z.z);
          out = safe_normalize_cycles(r0 * n.x + r1 * n.y + r2 * n.z);
        };
        return out;
      }};
      result.set_name("cycles_obj_inv_normal_if_obj");
      return result;
    }();
    value = callable(*transform_state.lazy_object_buffer, shader_data.object,
                     value);
    return;
  }
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
          transform_state.fetch_object_to_world(), value));
    };
  } else {
    $if(is_object) {
      value = safe_normalize_cycles(cycles_transform::direction_transposed(
          transform_state.fetch_object_to_world(), value));
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
          cycles_transform::direction(transform_state.fetch_object_to_world(), value);
    };
  } else {
    value = cycles_transform::direction(transform_state.fetch_object_to_world(), value);
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
          cycles_transform::direction(transform_state.fetch_world_to_object(), value);
    };
  } else {
    value = cycles_transform::direction(transform_state.fetch_world_to_object(), value);
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

  stack_store_float3_if_valid(stack, vector_out_offset, value);
}

} // namespace psycles::luisa_backend::cycles_svm::detail
