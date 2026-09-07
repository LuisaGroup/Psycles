/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#include "cycles_svm_subsurface.h"

#include "cycles_svm_simple_closure.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::cycles_svm::detail {
using namespace luisa::compute;
namespace abi = compiler::cycles_svm;
namespace closure = ::psycles::luisa_backend::cycles_closure;

Bool surface_shader_material_eval_required(Expr<bool> subsurface_exit,
                                           Expr<std::uint32_t> flags) noexcept {
  return (!subsurface_exit) |
         ((flags & static_cast<std::uint32_t>(abi::SD_HAS_BSSRDF_BUMP)) != 0u);
}

void subsurface_shader_data_setup(ShaderData &sd) noexcept {
  auto &pool = *sd.closure;
  Float3 normal = sd.N;
  $if((sd.flag & static_cast<std::uint32_t>(abi::SD_HAS_BSSRDF_BUMP)) != 0u) {
    // Cycles surface_shader_bssrdf_normal: ordered weighted reduction over
    // BSSRDFs only. Use abs(average(weight)), not average(abs(weight)).
    Float3 sum = make_float3(0.0f);
    UInt index = 0u;
    $while(index < pool.count()) {
      const auto common = pool.common(index);
      $if(closure::is_bssrdf(common.type)) {
        sum += common.N * abs((common.weight.x + common.weight.y +
                               common.weight.z) / 3.0f);
      };
      index += 1u;
    };
    $if(any(sum != make_float3(0.0f))) { normal = normalize(sum); };
  };
  sd.flag &= ~static_cast<std::uint32_t>(abi::SD_CLOSURE_FLAGS);
  pool.reset();
  diffuse_setup(sd, normal, make_float3(1.0f));
}
} // namespace psycles::luisa_backend::cycles_svm::detail
