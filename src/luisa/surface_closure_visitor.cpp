#include <psycles/luisa/surface_closure_visitor.h>

#include <algorithm>

#include <psycles/luisa/cycles_closure.h>

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
      microfacet_tangent{closure.microfacet_tangent.expression()},
      microfacet_alpha_x{closure.microfacet_alpha_x.expression()},
      microfacet_alpha_y{closure.microfacet_alpha_y.expression()},
      diffuse_roughness{closure.diffuse_roughness.expression()},
      metallic{closure.metallic.expression()},
      ior{closure.ior.expression()},
      thin_film_thickness{closure.thin_film_thickness.expression()},
      thin_film_ior{closure.thin_film_ior.expression()},
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
      beckmann{closure.beckmann.expression()},
      bssrdf_method{closure.bssrdf_method.expression()},
      bssrdf_radius{closure.bssrdf_radius.expression()},
      bssrdf_albedo{closure.bssrdf_albedo.expression()},
      bssrdf_ior{closure.bssrdf_ior.expression()},
      bssrdf_roughness{closure.bssrdf_roughness.expression()},
      bssrdf_anisotropy{closure.bssrdf_anisotropy.expression()} {}

SurfaceClosureRecord SurfaceClosureExpression::reference() const noexcept {
    return {
        .kind = UInt{kind.expression()},
        .lobe = UInt{lobe.expression()},
        .weight = Float3{weight.expression()},
        .allocation_weight = Float{allocation_weight.expression()},
        .sample_weight = Float{sample_weight.expression()},
        .setup_valid = Bool{setup_valid.expression()},
        .albedo = Float3{albedo.expression()},
        .reflection_albedo = Float3{reflection_albedo.expression()},
        .transmission_albedo = Float3{transmission_albedo.expression()},
        .color = Float3{color.expression()},
        .normal = Float3{normal.expression()},
        .roughness = Float{roughness.expression()},
        .microfacet_tangent =
            Float3{microfacet_tangent.expression()},
        .microfacet_alpha_x =
            Float{microfacet_alpha_x.expression()},
        .microfacet_alpha_y =
            Float{microfacet_alpha_y.expression()},
        .diffuse_roughness = Float{diffuse_roughness.expression()},
        .metallic = Float{metallic.expression()},
        .ior = Float{ior.expression()},
        .thin_film_thickness = Float{thin_film_thickness.expression()},
        .thin_film_ior = Float{thin_film_ior.expression()},
        .specular_ior_level = Float{specular_ior_level.expression()},
        .specular_tint = Float3{specular_tint.expression()},
        .sheen_transform_a = Float{sheen_transform_a.expression()},
        .sheen_transform_b = Float{sheen_transform_b.expression()},
        .evaluation_scale = Float3{evaluation_scale.expression()},
        .fresnel_f0 = Float3{fresnel_f0.expression()},
        .fresnel_f90 = Float3{fresnel_f90.expression()},
        .reflection_tint = Float3{reflection_tint.expression()},
        .transmission_tint = Float3{transmission_tint.expression()},
        .preserve_ggx_energy = Bool{preserve_ggx_energy.expression()},
        .beckmann = Bool{beckmann.expression()},
        .bssrdf_method = UInt{bssrdf_method.expression()},
        .bssrdf_radius = Float3{bssrdf_radius.expression()},
        .bssrdf_albedo = Float3{bssrdf_albedo.expression()},
        .bssrdf_ior = Float{bssrdf_ior.expression()},
        .bssrdf_roughness = Float{bssrdf_roughness.expression()},
        .bssrdf_anisotropy = Float{bssrdf_anisotropy.expression()}};
}

SurfaceClosureExpressionVisitor::SurfaceClosureExpressionVisitor(
    std::size_t capacity) noexcept
    : _capacity{std::clamp(
          capacity,
          std::size_t{1u},
          static_cast<std::size_t>(
              maximum_surface_closure_capacity))} {}

Bool SurfaceClosureExpressionVisitor::retains(
    const SurfaceClosureExpression &closure,
    Expr<std::uint32_t> allocated_count) const noexcept {
    return
        (closure.kind != static_cast<std::uint32_t>(
                             SurfaceClosureKind::none)) &
        (closure.allocation_weight >=
            cycles_closure::closure_weight_cutoff) &
        (allocated_count < static_cast<std::uint32_t>(
                               _capacity));
}

void SurfaceClosureExpressionVisitor::begin(
    Expr<luisa::float3> shading_normal) noexcept {
    _closures.clear();
    _shading_normal = shading_normal.expression();
}

void SurfaceClosureExpressionVisitor::add(
    const SurfaceClosureRecord &closure) noexcept {
    _closures.emplace_back(closure);
}

void SurfaceClosureExpressionVisitor::finish() noexcept {
    if (_shading_normal != nullptr) {
        visit(
            Expr<luisa::float3>{_shading_normal},
            _closures);
    }
}

}// namespace psycles::luisa_backend
