#include <psycles/luisa/surface_closure_sampling.h>

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"
#include "thin_glass_component.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

#include <vector>

namespace psycles::luisa_backend {
namespace {

template<typename Closure>
[[nodiscard]] Bool has_kind(
    const Closure &closure,
    SurfaceClosureKind kind) noexcept {
    return closure.kind == static_cast<std::uint32_t>(kind);
}

template<typename Closure>
[[nodiscard]] Bool has_lobe(
    const Closure &closure,
    SurfaceClosureLobe lobe) noexcept {
    return closure.lobe == static_cast<std::uint32_t>(lobe);
}

[[nodiscard]] Bool has_property(
    Expr<std::uint32_t> properties,
    std::uint32_t property) noexcept {
    return (properties & property) != 0u;
}

}// namespace

Float3 make_surface_closure_sampling_incoming(
    const SurfaceClosurePoint &point) noexcept {
    return detail::safe_normalize(
        point.incoming, point.shading_normal);
}

SurfaceClosureSelectionInput
make_surface_closure_selection_input(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .kind = Expr<std::uint32_t>{closure.kind.expression()},
        .lobe = Expr<std::uint32_t>{closure.lobe.expression()},
        .bssrdf_method = Expr<std::uint32_t>{
            closure.bssrdf_method.expression()},
        .allocation_weight =
            Expr<float>{closure.allocation_weight.expression()},
        .sample_weight =
            Expr<float>{closure.sample_weight.expression()},
        .setup_valid =
            Expr<bool>{closure.setup_valid.expression()},
        .normal = Expr<luisa::float3>{closure.normal.expression()},
        .roughness = Expr<float>{closure.roughness.expression()},
        .preserve_ggx_energy = Expr<bool>{
            closure.preserve_ggx_energy.expression()},
        .beckmann = Expr<bool>{closure.beckmann.expression()}};
}

SurfaceClosureSelectionInput
make_surface_closure_selection_input(
    const SurfaceClosureExpression &closure) noexcept {
    return {
        .kind = closure.kind,
        .lobe = closure.lobe,
        .bssrdf_method = closure.bssrdf_method,
        .allocation_weight = closure.allocation_weight,
        .sample_weight = closure.sample_weight,
        .setup_valid = closure.setup_valid,
        .normal = closure.normal,
        .roughness = closure.roughness,
        .preserve_ggx_energy = closure.preserve_ggx_energy,
        .beckmann = closure.beckmann};
}

SurfaceClosureSelectionContext
make_surface_closure_selection_context(
    const SurfaceQuery &query) noexcept {
    return {
        .lobe_mask =
            Expr<std::uint32_t>{query.lobe_mask.expression()},
        .glossy_filter_roughness = Expr<float>{
            query.glossy_filter_roughness.expression()}};
}

luisa::compute::Var<SurfaceClosureSelectionCall>
surface_closure_selection(
    const SurfaceClosureSelectionContext &context,
    const SurfaceClosureSelectionInput &closure) noexcept {
    const auto identity = detail::SurfaceClosureIdentityExpression{
        .kind = closure.kind,
        .lobe = closure.lobe,
        .bssrdf_method = closure.bssrdf_method,
        .allocation_weight = closure.allocation_weight,
        .setup_valid = closure.setup_valid,
        .roughness = closure.roughness,
        .preserve_ggx_energy = closure.preserve_ggx_energy,
        .beckmann = closure.beckmann};
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_rough_translucent = has_kind(
        closure, SurfaceClosureKind::rough_translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto is_refraction = has_kind(
        closure, SurfaceClosureKind::refraction);
    const auto is_thin_glass_transmission = has_kind(
        closure, SurfaceClosureKind::thin_glass_transmission);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
    const auto is_bssrdf = has_kind(
        closure, SurfaceClosureKind::bssrdf);
    UInt lobe_mask{context.lobe_mask};
    const auto diffuse_enabled =
        (lobe_mask & static_cast<std::uint32_t>(event_diffuse)) != 0u;
    const auto glossy_enabled =
        (lobe_mask & static_cast<std::uint32_t>(event_glossy)) != 0u;
    const auto transparent_enabled =
        (lobe_mask & static_cast<std::uint32_t>(event_transparent)) !=
        0u;
    const auto transmission_enabled =
        (lobe_mask & static_cast<std::uint32_t>(event_transmission)) !=
        0u;
    Bool eligible = false;
    eligible = select(eligible,
        transparent_enabled,
        is_transparent);
    eligible = select(eligible,
        diffuse_enabled & transmission_enabled,
        is_translucent | is_rough_translucent);
    eligible = select(eligible,
        diffuse_enabled,
        is_diffuse | is_sheen | is_bssrdf);
    eligible = select(eligible,
        glossy_enabled,
        (is_principled & !is_sheen) | is_glossy);
    eligible = select(eligible,
        glossy_enabled | transmission_enabled,
        is_glass);
    eligible = select(eligible,
        glossy_enabled & transmission_enabled,
        is_refraction);
    eligible = select(eligible,
        glossy_enabled & transmission_enabled,
        is_thin_glass_transmission);
    eligible &= detail::closure_allocated(identity) &
                Bool{closure.setup_valid};

    luisa::compute::Var<SurfaceClosureSelectionCall> result;
    // Selection is a pure projection. Express its only conditional field as
    // data flow so the three consumers (measure, inversion, chosen closure)
    // do not each introduce a pair of control-flow blocks into the generated
    // sampler. The value is exactly 0 or sample_weight under the same
    // predicate; no closure operation is speculated by this select.
    result.weight = select(
        0.0f,
        Float{closure.sample_weight},
        eligible);
    result.glossy_normal = closure.normal;
    result.runtime_flags = detail::cycles_runtime_flags(
        identity, Float{context.glossy_filter_roughness});
    result.closure_type = detail::cycles_closure_type(identity);
    result.closure_sample_weight = Float{closure.sample_weight};
    return result;
}

luisa::compute::Var<SurfaceClosureConditionalSampleCall>
surface_closure_conditional_sample(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal_expression,
    const SurfaceClosurePhysicalRecord &closure,
    Expr<luisa::float3> incoming_expression,
    Expr<luisa::float3> glossy_normal_expression,
    Expr<luisa::float2> random_direction_expression,
    Expr<float> rescaled_lobe_expression,
    const SurfaceQuery &query,
    SurfaceClosureReachability reachability) noexcept {
    Float3 shading_normal{shading_normal_expression};
    Float3 incoming{incoming_expression};
    Float3 glossy_normal{glossy_normal_expression};
    Float2 random_direction{random_direction_expression};
    Float rescaled_lobe{rescaled_lobe_expression};

    const auto reachable_kind =
        [&reachability, &closure](SurfaceClosureKind kind) noexcept {
            if (!reachability.contains(kind)) { return Bool{false}; }
            return has_kind(closure, kind);
        };
    const auto is_translucent = reachable_kind(SurfaceClosureKind::translucent);
    const auto is_rough_translucent =
        reachable_kind(SurfaceClosureKind::rough_translucent);
    const auto is_principled = reachable_kind(SurfaceClosureKind::principled);
    const auto is_sheen =
        reachability.contains_principled_lobe(SurfaceClosureLobe::sheen)
            ? is_principled &
                  has_lobe(closure, SurfaceClosureLobe::sheen)
            : Bool{false};
    const auto is_glossy = reachable_kind(SurfaceClosureKind::glossy);
    const auto is_transparent = reachable_kind(SurfaceClosureKind::transparent);
    const auto is_glass = reachable_kind(SurfaceClosureKind::glass);
    const auto is_refraction = reachable_kind(SurfaceClosureKind::refraction);
    const auto is_thin_glass_transmission =
        reachable_kind(SurfaceClosureKind::thin_glass_transmission);
    const auto is_bssrdf = reachable_kind(SurfaceClosureKind::bssrdf);
    const auto is_dielectric = is_glass | is_refraction;
    const auto sample_glossy =
        is_glossy | (is_principled & !is_sheen);
    const auto principled_glossy_possible =
        reachability.contains(SurfaceClosureKind::principled) &&
        (reachability.principled_lobes &
         (all_surface_closure_lobes &
          ~surface_closure_lobe_bit(SurfaceClosureLobe::sheen))) != 0u;
    const auto transparent_possible =
        reachability.contains(SurfaceClosureKind::transparent);
    const auto dielectric_possible =
        reachability.contains(SurfaceClosureKind::glass) ||
        reachability.contains(SurfaceClosureKind::refraction);
    const auto thin_glass_possible =
        reachability.contains(SurfaceClosureKind::thin_glass_transmission);
    const auto sheen_possible =
        reachability.contains_principled_lobe(SurfaceClosureLobe::sheen);
    const auto glossy_possible =
        reachability.contains(SurfaceClosureKind::glossy) ||
        principled_glossy_possible;
    const auto cosine_possible =
        reachability.contains(SurfaceClosureKind::diffuse) ||
        reachability.contains(SurfaceClosureKind::translucent) ||
        reachability.contains(SurfaceClosureKind::rough_translucent);

    Float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    Float2 roughness = make_float2(1.0f);
    Float3 singular_evaluation = make_float3(0.0f);
    Float singular_pdf = 0.0f;
    Float eta = 1.0f;
    Bool singular = false;
    Bool glass_transmission = false;
    Bool valid = true;

    // This device branch executes only after the outer categorical inversion
    // has selected this closure. Exactly one conditional sampler is therefore
    // active for a surface sample.
    $if(is_bssrdf) {
        // Spatial transport consumes the payload after the categorical
        // closure choice. A BSSRDF has no conditional surface direction.
    }
    $elif(is_transparent) {
        if (transparent_possible) {
            direction = -point.incoming;
            roughness = make_float2(0.0f);
        }
    }
    $elif(is_dielectric) {
        if (dielectric_possible) {
            const detail::MicrofacetGlassComponent microfacet_glass{
                services, point};
            const auto glass = microfacet_glass.sample(
                closure,
                incoming,
                glossy_normal,
                random_direction,
                rescaled_lobe,
                (query.lobe_mask &
                 static_cast<std::uint32_t>(event_glossy)) != 0u,
                (query.lobe_mask &
                 static_cast<std::uint32_t>(event_transmission)) != 0u,
                query.glossy_filter_roughness);
            direction = glass.direction;
            roughness = make_float2(glass.alpha);
            singular_evaluation = glass.singular_evaluation;
            singular_pdf = glass.singular_pdf;
            eta = glass.eta;
            singular = glass.singular;
            glass_transmission = glass.transmission;
            valid = glass.valid;
        }
    }
    $elif(is_thin_glass_transmission) {
        if (thin_glass_possible) {
            const detail::ThinGlassComponent thin_glass{
                services, point};
            const auto thin = thin_glass.sample(
                closure,
                incoming,
                random_direction,
                query.glossy_filter_roughness);
            direction = thin.direction;
            roughness = make_float2(thin.alpha);
            singular_evaluation = thin.singular_evaluation;
            singular_pdf = thin.singular_pdf;
            eta = 1.0f;
            singular = thin.singular;
            glass_transmission = true;
            valid = thin.valid;
        }
    }
    $elif(is_sheen) {
        if (sheen_possible) {
            direction = detail::sample_sheen(
                closure, incoming, random_direction);
        }
    }
    $elif(sample_glossy) {
        if (glossy_possible) {
            const auto glossy =
                detail::sample_microfacet_reflection(
                    point,
                    shading_normal,
                    closure,
                    incoming,
                    random_direction,
                    glossy_normal,
                    query.glossy_filter_roughness);
            direction = glossy.direction;
            roughness = make_float2(glossy.alpha);
            singular_evaluation = glossy.singular_evaluation;
            singular_pdf = glossy.singular_pdf;
            singular = glossy.singular;
            valid = glossy.valid;
        }
    }
    $else {
        if (cosine_possible) {
            const auto normal = select(
                closure.normal,
                select(-glossy_normal,
                       closure.normal,
                       is_rough_translucent),
                is_translucent | is_rough_translucent);
            direction = detail::sample_cosine_hemisphere(
                normal, random_direction);
        }
    };

    using namespace surface_closure_sample_property;
    UInt properties = 0u;
    properties |= select(0u, transparent, is_transparent);
    properties |= select(
        0u, translucent, is_translucent | is_rough_translucent);
    properties |= select(0u, glossy, sample_glossy);
    properties |= select(
        0u, glass, is_dielectric | is_thin_glass_transmission);
    properties |= select(
        0u,
        surface_closure_sample_property::transmission,
        (is_dielectric | is_thin_glass_transmission) &
            glass_transmission);
    properties |= select(0u,
                         surface_closure_sample_property::singular,
                         singular);
    properties |= select(
        0u,
        surface_closure_sample_property::bssrdf,
        is_bssrdf);

    luisa::compute::Var<SurfaceClosureConditionalSampleCall> result;
    result.direction = direction;
    result.roughness = roughness;
    result.singular_evaluation = singular_evaluation;
    result.singular_pdf = singular_pdf;
    result.eta = eta;
    result.properties = properties;
    result.bssrdf_method = closure.bssrdf_method;
    result.bssrdf_radius = closure.bssrdf_radius;
    result.bssrdf_albedo = closure.bssrdf_albedo;
    result.bssrdf_normal = closure.normal;
    result.bssrdf_ior = closure.bssrdf_ior;
    result.bssrdf_roughness = closure.bssrdf_roughness;
    result.bssrdf_anisotropy = closure.bssrdf_anisotropy;
    result.valid = valid;
    return result;
}

SurfaceClosureSelectionMeasure::SurfaceClosureSelectionMeasure(
    Expr<bool> back_facing) noexcept
    : _runtime_flags{select(
          0u,
          cycles_closure::runtime_backfacing,
          Bool{back_facing})} {}

void SurfaceClosureSelectionMeasure::add(
    const luisa::compute::Var<
        SurfaceClosureSelectionCall> &selection) noexcept {
    add(selection, true);
}

void SurfaceClosureSelectionMeasure::add(
    const luisa::compute::Var<
        SurfaceClosureSelectionCall> &selection,
    Expr<bool> retained) noexcept {
    const Bool active{retained};
    _total_weight += select(0.0f, selection.weight, active);
    _runtime_flags |= select(
        0u, selection.runtime_flags, active);
    _retained_count += select(0u, 1u, active);
}

Expr<float> SurfaceClosureSelectionMeasure::total_weight()
    const noexcept {
    return Expr<float>{_total_weight.expression()};
}

Expr<std::uint32_t>
SurfaceClosureSelectionMeasure::runtime_flags() const noexcept {
    return Expr<std::uint32_t>{_runtime_flags.expression()};
}

Expr<std::uint32_t>
SurfaceClosureSelectionMeasure::retained_count() const noexcept {
    return Expr<std::uint32_t>{_retained_count.expression()};
}

SurfaceClosureCategoricalInversion::
    SurfaceClosureCategoricalInversion(
        Expr<float> random_lobe,
        const SurfaceClosureSelectionMeasure &measure) noexcept
    : _random_lobe{clamp(
          Float{random_lobe}, 0.0f, 0.99999994f)},
      _target{_random_lobe * measure.total_weight()},
      _retained_count{measure.retained_count()} {}

SurfaceClosureCategoricalChoice
SurfaceClosureCategoricalInversion::consider(
    const luisa::compute::Var<
        SurfaceClosureSelectionCall> &selection) noexcept {
    return consider(selection, true);
}

SurfaceClosureCategoricalChoice
SurfaceClosureCategoricalInversion::consider(
    const luisa::compute::Var<
        SurfaceClosureSelectionCall> &selection,
    Expr<bool> retained) noexcept {
    const auto weight = select(
        0.0f, selection.weight, Bool{retained});
    const auto next = _accumulated + weight;
    const auto choose =
        !_selected & (weight > 0.0f) &
        (_target < next);
    const auto rescaled = select(
        _random_lobe,
        (_target - _accumulated) /
            max(weight, 1.0e-20f),
        _retained_count > 1u);
    _selected |= choose;
    _accumulated = next;
    return {
        .choose = choose,
        .rescaled = rescaled};
}

Expr<bool> SurfaceClosureCategoricalInversion::selected()
    const noexcept {
    return Expr<bool>{_selected.expression()};
}

void SurfaceClosureSelectedSample::accept(
    Expr<std::uint32_t> closure_index,
    Expr<luisa::float3> closure_weight,
    Expr<luisa::float3> closure_normal,
    Expr<float> selection_rescaled,
    const luisa::compute::Var<
        SurfaceClosureSelectionCall> &selection,
    const luisa::compute::Var<
        SurfaceClosureConditionalSampleCall> &sample) noexcept {
    _selected = true;
    _closure_index = closure_index;
    _closure_type = selection.closure_type;
    _closure_sample_weight = selection.closure_sample_weight;
    _selection_rescaled = selection_rescaled;
    _closure_weight = closure_weight;
    _closure_normal = closure_normal;
    _selected_weight = selection.weight;
    _direction = sample.direction;
    _roughness = sample.roughness;
    _singular_evaluation = sample.singular_evaluation;
    _singular_pdf = sample.singular_pdf;
    _eta = sample.eta;
    _properties = sample.properties;
    _bssrdf_method = sample.bssrdf_method;
    _bssrdf_radius = sample.bssrdf_radius;
    _bssrdf_albedo = sample.bssrdf_albedo;
    _bssrdf_normal = sample.bssrdf_normal;
    _bssrdf_ior = sample.bssrdf_ior;
    _bssrdf_roughness = sample.bssrdf_roughness;
    _bssrdf_anisotropy = sample.bssrdf_anisotropy;
    _candidate_valid = sample.valid;
}

Expr<bool> SurfaceClosureSelectedSample::selected() const noexcept {
    return Expr<bool>{_selected.expression()};
}

Expr<std::uint32_t>
SurfaceClosureSelectedSample::closure_index() const noexcept {
    return Expr<std::uint32_t>{_closure_index.expression()};
}

Expr<luisa::float3>
SurfaceClosureSelectedSample::direction() const noexcept {
    return Expr<luisa::float3>{_direction.expression()};
}

SurfaceSampleTrace SurfaceClosureSelectedSample::finish(
    const SurfaceClosurePoint &point,
    const SurfaceClosureSelectionMeasure &measure,
    const SurfaceEvaluation &mixture_evaluation,
    bool trace_selection) const noexcept {
    using namespace surface_closure_sample_property;
    const auto selected_transparent =
        _selected & has_property(_properties, transparent);
    const auto selected_translucent =
        _selected & has_property(_properties, translucent);
    const auto selected_glossy =
        _selected & has_property(_properties, glossy);
    const auto selected_glass =
        _selected & has_property(_properties, glass);
    const auto selected_glass_transmission =
        selected_glass & has_property(_properties, transmission);
    const auto selected_singular =
        has_property(_properties,
            surface_closure_sample_property::singular);
    const auto selected_bssrdf =
        _selected & has_property(
            _properties,
            surface_closure_sample_property::bssrdf);

    const auto reflection_geometric_valid =
        dot(point.geometric_normal, _direction) > 0.0f;
    const auto transmission_geometric_valid =
        dot(point.geometric_normal, _direction) < 0.0f;
    const auto geometric_valid = select(
        reflection_geometric_valid,
        transmission_geometric_valid,
        selected_translucent | selected_glass_transmission);
    const auto sample_valid =
        _selected & _candidate_valid &
        (selected_bssrdf | selected_transparent | geometric_valid);

    auto regular = SurfaceEvaluation::zero();
    const auto regular_valid = sample_valid & !selected_bssrdf;
    regular.f = select(
        make_float3(0.0f), mixture_evaluation.f, regular_valid);
    regular.pdf = select(
        0.0f, mixture_evaluation.pdf, regular_valid);
    regular.diffuse_f = select(
        make_float3(0.0f),
        mixture_evaluation.diffuse_f,
        regular_valid);
    regular.glossy_f = select(
        make_float3(0.0f),
        mixture_evaluation.glossy_f,
        regular_valid);
    regular.diffuse_pdf = select(
        0.0f, mixture_evaluation.diffuse_pdf, regular_valid);
    regular.average_roughness_squared = select(
        0.0f,
        mixture_evaluation.average_roughness_squared,
        regular_valid);
    regular.events = select(
        0u, mixture_evaluation.events, regular_valid);

    const auto transparent_singular =
        sample_valid & selected_transparent;
    const auto glossy_singular =
        sample_valid & selected_glossy & selected_singular;
    const auto glass_singular =
        sample_valid & selected_glass & selected_singular;
    const auto transparent_delta = select(
        make_float3(0.0f),
        _closure_weight * 1.0e6f,
        transparent_singular);
    const auto glossy_delta = select(
        make_float3(0.0f),
        _singular_evaluation,
        glossy_singular);
    const auto glass_delta = select(
        make_float3(0.0f),
        _singular_evaluation,
        glass_singular);
    const auto delta_pdf_numerator =
        select(0.0f,
            1.0e6f * _selected_weight,
            transparent_singular) +
        select(0.0f,
            _singular_pdf * _selected_weight,
            glossy_singular) +
        select(0.0f,
            _singular_pdf * _selected_weight,
            glass_singular);

    auto trace = SurfaceSampleTrace::zero();
    auto &result = trace.sample;
    result.wi = _direction;
    result.runtime_flags = measure.runtime_flags();
    result.valid = sample_valid;
    const auto bssrdf_transport_weight = select(
        make_float3(0.0f),
        _closure_weight * measure.total_weight() /
            max(_closure_sample_weight, 1.0e-20f),
        selected_bssrdf & sample_valid);
    result.evaluation.f = select(
        regular.f + transparent_delta + glossy_delta + glass_delta,
        bssrdf_transport_weight,
        selected_bssrdf);
    result.evaluation.pdf = select(
        regular.pdf + delta_pdf_numerator /
                          max(measure.total_weight(), 1.0e-20f),
        1.0f,
        selected_bssrdf & sample_valid);
    result.evaluation.average_roughness_squared = select(
        0.0f,
        regular.average_roughness_squared * regular.pdf /
            max(result.evaluation.pdf, 1.0e-20f),
        result.evaluation.pdf > 0.0f);
    result.evaluation.diffuse_f = regular.diffuse_f;
    result.evaluation.glossy_f =
        regular.glossy_f + glossy_delta +
        select(glass_delta,
            make_float3(0.0f),
            selected_glass_transmission);
    result.evaluation.diffuse_pdf = regular.diffuse_pdf;

    auto sampled_surface_events = select(
        static_cast<std::uint32_t>(
            event_diffuse | event_reflection),
        static_cast<std::uint32_t>(
            event_glossy | event_reflection),
        selected_glossy);
    sampled_surface_events = select(
        sampled_surface_events,
        static_cast<std::uint32_t>(
            event_diffuse | event_transmission),
        selected_translucent);
    sampled_surface_events = select(
        sampled_surface_events,
        static_cast<std::uint32_t>(
            event_singular | event_reflection),
        selected_glossy & selected_singular);
    auto glass_events = select(
        static_cast<std::uint32_t>(event_glossy),
        static_cast<std::uint32_t>(event_singular),
        selected_singular);
    glass_events |= select(
        static_cast<std::uint32_t>(event_reflection),
        static_cast<std::uint32_t>(event_transmission),
        selected_glass_transmission);
    sampled_surface_events = select(
        sampled_surface_events, glass_events, selected_glass);
    result.evaluation.events = select(
        sampled_surface_events,
        static_cast<std::uint32_t>(
            event_transmission | event_transparent),
        selected_transparent);
    result.evaluation.events = select(
        0u, result.evaluation.events, sample_valid);
    result.evaluation.events = select(
        result.evaluation.events,
        static_cast<std::uint32_t>(event_subsurface),
        selected_bssrdf & sample_valid);
    result.eta = select(
        1.0f, _eta, selected_glass & sample_valid);
    result.roughness = select(
        make_float2(0.0f), _roughness, sample_valid);
    result.bssrdf_method = select(
        static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk),
        _bssrdf_method,
        selected_bssrdf & sample_valid);
    result.bssrdf_radius = select(
        make_float3(0.0f),
        _bssrdf_radius,
        selected_bssrdf & sample_valid);
    result.bssrdf_albedo = select(
        make_float3(0.0f),
        _bssrdf_albedo,
        selected_bssrdf & sample_valid);
    result.bssrdf_normal = select(
        make_float3(0.0f, 0.0f, 1.0f),
        _bssrdf_normal,
        selected_bssrdf & sample_valid);
    result.bssrdf_ior = select(
        1.4f, _bssrdf_ior, selected_bssrdf & sample_valid);
    result.bssrdf_roughness = select(
        1.0f, _bssrdf_roughness, selected_bssrdf & sample_valid);
    result.bssrdf_anisotropy = select(
        0.0f, _bssrdf_anisotropy, selected_bssrdf & sample_valid);

    if (trace_selection) {
        trace.closure_index = select(
            0u, _closure_index, _selected);
        trace.closure_type = select(
            0u, _closure_type, _selected);
        trace.closure_sample_weight = select(
            0.0f, _closure_sample_weight, _selected);
        trace.selection_rescaled = select(
            0.0f, _selection_rescaled, _selected);
        trace.closure_weight = select(
            make_float3(0.0f), _closure_weight, _selected);
        trace.closure_normal = select(
            make_float3(0.0f, 0.0f, 1.0f),
            _closure_normal,
            _selected);
        trace.closure_valid = _selected;
    }
    return trace;
}

DirectSurfaceClosureSamplingOperation::
    DirectSurfaceClosureSamplingOperation(
        const ShaderServices &services,
        const SurfaceClosurePoint &point,
        const SurfaceQuery &query,
        SurfaceClosureReachability reachability) noexcept
    : _services{services},
      _point{point},
      _query{query},
      _selection_context{make_surface_closure_selection_context(query)},
      _incoming{make_surface_closure_sampling_incoming(point)},
      _reachability{reachability} {}

luisa::compute::Var<SurfaceClosureSelectionCall>
DirectSurfaceClosureSamplingOperation::selection(
    const SurfaceClosureExpression &closure) const noexcept {
    return surface_closure_selection(
        _selection_context,
        make_surface_closure_selection_input(closure));
}

luisa::compute::Var<SurfaceClosureConditionalSampleCall>
DirectSurfaceClosureSamplingOperation::conditional_sample(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosureExpression &closure,
    Expr<luisa::float3> glossy_normal,
    Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe) const noexcept {
    return surface_closure_conditional_sample(
        _services,
        _point,
        shading_normal,
        closure.reference(),
        Expr<luisa::float3>{_incoming.expression()},
        glossy_normal,
        random_direction,
        rescaled_lobe,
        _query, _reachability);
}

SurfaceClosureSamplingVisitor::SurfaceClosureSamplingVisitor(
    std::size_t capacity,
    const SurfaceClosurePoint &point,
    const SurfaceClosureSamplingOperation &sampling,
    SurfaceClosureEvaluationOperation &evaluation,
    Expr<float> random_lobe,
    Expr<luisa::float2> random_direction,
    bool trace_selection) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _sampling{sampling},
      _evaluation{evaluation},
      _random_lobe{random_lobe},
      _random_direction{random_direction},
      _trace_selection{trace_selection},
      _result{SurfaceSampleTrace::zero()} {}

void SurfaceClosureSamplingVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression>
        &closures) noexcept {
    struct ScheduledSelection {
        Bool retained;
        UInt retained_index;
        luisa::compute::Var<SurfaceClosureSelectionCall>
            selection;
    };

    SurfaceClosureSelectionMeasure measure{_point.back_facing};
    // This is a host/JIT-stage AST schedule whose extent follows the scene's
    // reachable closure list. A vector avoids imposing a fixed closure-count
    // bound; revisit its host allocation only if profiling makes it material.
    std::vector<ScheduledSelection> schedule;
    schedule.reserve(closures.size());
    UInt retained_index = 0u;
    for (const auto &closure : closures) {
        const auto retained = retains(closure, retained_index);
        const auto selection = _sampling.selection(closure);
        measure.add(selection, retained);
        schedule.emplace_back(ScheduledSelection{
            .retained = retained,
            .retained_index = retained_index,
            .selection = selection});
        retained_index += select(0u, 1u, retained);
    }

    SurfaceClosureCategoricalInversion inversion{
        _random_lobe, measure};
    SurfaceClosureSelectedSample selected;
    UInt selected_program_index = ~std::uint32_t{0u};
    Float selected_rescaled = 0.0f;
    for (auto index = std::size_t{0u};
         index < closures.size(); ++index) {
        const auto &scheduled = schedule[index];
        const auto choice = inversion.consider(
            scheduled.selection,
            scheduled.retained);
        selected_program_index = select(
            selected_program_index,
            static_cast<std::uint32_t>(index),
            choice.choose);
        selected_rescaled = select(
            selected_rescaled,
            choice.rescaled,
            choice.choose);
    }

    // Categorical inversion selects at most one authored closure. Test that
    // single program index directly: the previous nested retained/choice
    // diamonds created two merge layers per candidate. A nested device switch
    // looks smaller in SPIR-V but is pathologically expensive for current RADV
    // when it sits below the scene's material switch, so keep this as one
    // explicit predicate layer until backend switch lowering is fixed.
    for (auto index = std::size_t{0u};
         index < closures.size(); ++index) {
        $if(selected_program_index ==
            static_cast<std::uint32_t>(index)) {
            const auto &closure = closures[index];
            const auto &scheduled = schedule[index];
            const auto sample =
                _sampling.conditional_sample(
                    shading_normal,
                    closure,
                    Expr<luisa::float3>{
                        scheduled.selection.glossy_normal.expression()},
                    _random_direction,
                    Expr<float>{selected_rescaled.expression()});
            selected.accept(
                Expr<std::uint32_t>{
                    scheduled.retained_index.expression()},
                closure.weight,
                closure.normal,
                Expr<float>{selected_rescaled.expression()},
                scheduled.selection,
                sample);
        };
    }

    _evaluation.set_outgoing(selected.direction());
    SurfaceClosureEvaluationAccumulator evaluation;
    retained_index = 0u;
    for (auto index = std::size_t{0u};
         index < closures.size(); ++index) {
        const auto &closure = closures[index];
        const auto &scheduled = schedule[index];
        $if(scheduled.retained) {
            evaluation.add(_evaluation.evaluate(
                shading_normal,
                closure,
                retained_index == selected.closure_index()));
        };
        retained_index += select(
            0u, 1u, scheduled.retained);
    }
    const auto mixture = evaluation.finish(true);
    const auto result = selected.finish(
        _point, measure, mixture, _trace_selection);

    _result.sample.evaluation.f = result.sample.evaluation.f;
    _result.sample.evaluation.pdf = result.sample.evaluation.pdf;
    _result.sample.evaluation.diffuse_f =
        result.sample.evaluation.diffuse_f;
    _result.sample.evaluation.glossy_f =
        result.sample.evaluation.glossy_f;
    _result.sample.evaluation.diffuse_pdf =
        result.sample.evaluation.diffuse_pdf;
    _result.sample.evaluation.average_roughness_squared =
        result.sample.evaluation.average_roughness_squared;
    _result.sample.evaluation.events =
        result.sample.evaluation.events;
    _result.sample.wi = result.sample.wi;
    _result.sample.eta = result.sample.eta;
    _result.sample.roughness = result.sample.roughness;
    _result.sample.runtime_flags = result.sample.runtime_flags;
    _result.sample.bssrdf_method = result.sample.bssrdf_method;
    _result.sample.bssrdf_radius = result.sample.bssrdf_radius;
    _result.sample.bssrdf_albedo = result.sample.bssrdf_albedo;
    _result.sample.bssrdf_normal = result.sample.bssrdf_normal;
    _result.sample.bssrdf_ior = result.sample.bssrdf_ior;
    _result.sample.bssrdf_roughness = result.sample.bssrdf_roughness;
    _result.sample.bssrdf_anisotropy =
        result.sample.bssrdf_anisotropy;
    _result.sample.valid = result.sample.valid;
    _result.closure_index = result.closure_index;
    _result.closure_type = result.closure_type;
    _result.closure_sample_weight =
        result.closure_sample_weight;
    _result.selection_rescaled = result.selection_rescaled;
    _result.closure_weight = result.closure_weight;
    _result.closure_normal = result.closure_normal;
    _result.closure_valid = result.closure_valid;
}

const SurfaceSampleTrace &SurfaceClosureSamplingVisitor::result()
    const noexcept {
    return _result;
}

}// namespace psycles::luisa_backend
