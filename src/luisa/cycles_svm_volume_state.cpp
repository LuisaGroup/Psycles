/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_volume.h"

namespace psycles::luisa_backend::cycles_svm {
using namespace luisa::compute;
using namespace compiler::cycles_svm;

Float3 ClosurePool::volume_phase_parameters(Expr<unsigned> index) const noexcept {
  const auto type = volume_common(index).type;
  Float3 result = make_float3(0.0f);
  $switch(type) {
    $case(static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) {
      result.x = volume_henyey_greenstein_g(index);
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID)) {
      result = make_float3(volume_draine_parameters(index), 0.0f);
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_FOURNIER_FORAND_ID)) {
      result = volume_fournier_forand_coefficients(index);
    };
  };
  return result;
}

void ClosurePool::merge_volume_closures() noexcept {
  // Precondition: this is ShaderData from SHADER_TYPE_VOLUME with the Cycles
  // volume feature mask. Its allocated records are volume phases only; copy
  // their defined fields, never surface N or the uninitialized closure tail.
  UInt i = 0u;
  $while(i < _count) {
    const auto first = volume_common(i);
    const auto scatter =
        (first.type >= static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) &
        (first.type <= static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID));
    $if(scatter) {
      const auto parameters = volume_phase_parameters(i);
      UInt j = i + 1u;
      $while(j < _count) {
        const auto other = volume_common(j);
        // Canonical zeros stand only for unused parameters. Rayleigh never
        // reads them, HG reads g, Draine reads g/alpha, FF reads c1/c2/c3.
        const auto known_phase =
            (first.type == static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) |
            (first.type == static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID)) |
            (first.type == static_cast<unsigned>(CLOSURE_VOLUME_FOURNIER_FORAND_ID)) |
            (first.type == static_cast<unsigned>(CLOSURE_VOLUME_RAYLEIGH_ID));
        const auto equal = known_phase & (first.type == other.type) &
                           all(parameters == volume_phase_parameters(j));
        $if(equal) {
          add_weight(i, other.weight);
          add_sample_weight(i, other.sample_weight);
          UInt k = j;
          $while(k + 1u < _count) {
            const auto from = volume_common(k + 1u);
            const auto p = volume_phase_parameters(k + 1u);
            set_type(k, from.type);
            set_weight(k, from.weight);
            set_sample_weight(k, from.sample_weight);
            $switch(from.type) {
              $case(static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) {
                set_volume_henyey_greenstein(k, p.x);
              };
              $case(static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID)) {
                set_volume_draine(k, p.x, p.y);
              };
              $case(static_cast<unsigned>(CLOSURE_VOLUME_FOURNIER_FORAND_ID)) {
                set_volume_fournier_forand(k, p);
              };
            };
            k += 1u;
          };
          _count -= 1u;
        }
        $else { j += 1u; };
      };
    };
    i += 1u;
  };
}
} // namespace psycles::luisa_backend::cycles_svm
