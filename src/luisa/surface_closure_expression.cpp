#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

SurfaceClosureExpression::SurfaceClosureExpression(
    const SurfaceClosureRecord &closure) noexcept
    : kind{closure.kind.expression()},
      lobe{closure.lobe.expression()},
      weight{closure.weight.expression()},
      allocation_weight{closure.allocation_weight.expression()},
      sample_weight{closure.sample_weight.expression()},
      setup_valid{closure.setup_valid.expression()},
      albedo{closure.albedo.expression()},
      reflection_albedo{closure.reflection_albedo.expression()},
      transmission_albedo{closure.transmission_albedo.expression()},
      color{closure.color.expression()},
      normal{closure.normal.expression()},
      roughness{closure.roughness.expression()},
      diffuse_roughness{closure.diffuse_roughness.expression()},
      metallic{closure.metallic.expression()},
      ior{closure.ior.expression()},
      specular_ior_level{closure.specular_ior_level.expression()},
      specular_tint{closure.specular_tint.expression()},
      sheen_transform_a{closure.sheen_transform_a.expression()},
      sheen_transform_b{closure.sheen_transform_b.expression()},
      evaluation_scale{closure.evaluation_scale.expression()},
      fresnel_f0{closure.fresnel_f0.expression()},
      fresnel_f90{closure.fresnel_f90.expression()},
      reflection_tint{closure.reflection_tint.expression()},
      transmission_tint{closure.transmission_tint.expression()},
      preserve_ggx_energy{closure.preserve_ggx_energy.expression()},
      beckmann{closure.beckmann.expression()} {}

}// namespace psycles::luisa_backend
