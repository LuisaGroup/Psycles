/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_graph.h"

namespace psycles::compiler::cycles_svm {

class ConstantFolder {
public:
  CyclesGraph *const graph;
  GraphNode *const node;
  GraphOutput *const output;

  ConstantFolder(CyclesGraph *graph, GraphNode *node,
                 GraphOutput *output) noexcept;

  [[nodiscard]] bool all_inputs_constant() const noexcept;

  void make_constant(float value) const;
  void make_constant(Vec3f value) const;
  void make_constant(std::int32_t value) const;
  void make_constant_clamp(float value, bool clamp) const;
  void make_constant_clamp(Vec3f value, bool clamp) const;
  void make_zero() const;
  void make_one() const;

  void bypass(GraphOutput *new_output) const;
  void discard() const;
  void bypass_or_discard(GraphInput *input) const;
  [[nodiscard]] bool try_bypass_or_make_constant(GraphInput *input,
                                                 bool clamp = false) const;
  [[nodiscard]] bool is_zero(GraphInput *input) const noexcept;
  [[nodiscard]] bool is_one(GraphInput *input) const noexcept;

  void fold_mix(NodeMix type, bool clamp) const;
  void fold_mix_color(NodeMix type, bool clamp_factor, bool clamp) const;
  void fold_mix_float(bool clamp_factor, bool clamp) const;
  void fold_math(NodeMathType type) const;
};

[[nodiscard]] float svm_math(NodeMathType type, float a, float b,
                             float c) noexcept;

} // namespace psycles::compiler::cycles_svm
