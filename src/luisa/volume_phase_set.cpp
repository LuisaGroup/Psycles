#include <psycles/luisa/volume_phase_set.h>

#include <algorithm>

#include <psycles/luisa/cycles_volume_phase.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

VolumePhaseSet::VolumePhaseSet(
    std::size_t capacity) noexcept
    : _capacity{std::max(capacity, std::size_t{1u})},
      _types{_capacity},
      _parameters{_capacity},
      _weights{_capacity},
      _count{0u} {
    _types.write(
        0u,
        static_cast<std::uint32_t>(
            cycles_volume_phase::Type::
                henyey_greenstein));
    _parameters.write(0u, make_float4(0.0f));
    _weights.write(0u, make_float4(0.0f));
}

void VolumePhaseSet::add(
    const cycles_volume_phase::Closure &phase,
    Float3 weight) noexcept {
    const auto allocated =
        max(weight, make_float3(0.0f));
    const auto closure_sample_weight =
        abs(
            (allocated.x +
             allocated.y +
             allocated.z) /
            3.0f);
    const auto active =
        (closure_sample_weight > 0.0f) &
        (_count <
         static_cast<std::uint32_t>(_capacity));
    $if(active) {
        _types.write(
            _count,
            static_cast<std::uint32_t>(
                phase.type));
        _parameters.write(
            _count,
            make_float4(phase.parameters, 0.0f));
        _weights.write(
            _count,
            make_float4(
                allocated,
                closure_sample_weight));
        _count += 1u;
    };
}

UInt VolumePhaseSet::count() const noexcept {
    return _count;
}

VolumePhaseEntry VolumePhaseSet::entry(
    UInt index) const noexcept {
    const auto valid = index < _count;
    const auto safe_index = select(
        0u, index, valid);
    const auto type = _types.read(safe_index);
    const auto parameters =
        _parameters.read(safe_index);
    const auto weight = _weights.read(safe_index);
    return {
        .type = select(0u, type, valid),
        .parameters = select(
            make_float3(0.0f),
            parameters.xyz(),
            valid),
        .weight = select(
            make_float3(0.0f),
            weight.xyz(),
            valid),
        .sample_weight =
            select(0.0f, weight.w, valid),
        .valid = valid};
}

void VolumePhaseSet::merge_equal() noexcept {
    UInt first = 0u;
    $while(first < _count) {
        UInt candidate = first + 1u;
        $while(candidate < _count) {
            const auto first_type =
                _types.read(first);
            const auto candidate_type =
                _types.read(candidate);
            const auto first_parameters =
                _parameters.read(first);
            const auto candidate_parameters =
                _parameters.read(candidate);
            const auto equal =
                (first_type == candidate_type) &
                all(
                    first_parameters.xyz() ==
                    candidate_parameters.xyz());
            $if(equal) {
                _weights.write(
                    first,
                    _weights.read(first) +
                        _weights.read(candidate));
                UInt shifted = candidate;
                $while(shifted + 1u < _count) {
                    _types.write(
                        shifted,
                        _types.read(shifted + 1u));
                    _parameters.write(
                        shifted,
                        _parameters.read(
                            shifted + 1u));
                    _weights.write(
                        shifted,
                        _weights.read(
                            shifted + 1u));
                    shifted += 1u;
                };
                _count -= 1u;
            }
            $else {
                candidate += 1u;
            };
        };
        first += 1u;
    };
}

void VolumePhaseSet::truncate(
    std::uint32_t maximum) noexcept {
    _count = min(_count, maximum);
}

VolumePhaseSetEvaluation VolumePhaseSet::evaluate(
    Float3 axis,
    Float3 outgoing) const noexcept {
    Float weighted_value = 0.0f;
    Float weighted_pdf = 0.0f;
    Float total_sample_weight = 0.0f;
    UInt index = 0u;
    const auto cosine = dot(axis, outgoing);
    $while(index < _count) {
        const auto type = _types.read(index);
        const auto parameters =
            _parameters.read(index).xyz();
        const auto sample_weight =
            _weights.read(index).w;
        const auto pdf =
            cycles_volume_phase::evaluate(
                type,
                parameters,
                cosine);
        $if(pdf != 0.0f) {
            weighted_value += pdf * sample_weight;
            weighted_pdf += pdf * sample_weight;
        };
        total_sample_weight += sample_weight;
        index += 1u;
    };
    const auto valid =
        total_sample_weight > 0.0f;
    Float inverse_weight = 0.0f;
    $if(valid) {
        inverse_weight =
            1.0f / total_sample_weight;
    };
    return {
        .value =
            weighted_value * inverse_weight,
        .pdf = weighted_pdf * inverse_weight,
        .sample_weight =
            total_sample_weight,
        .valid = valid};
}

VolumePhaseSetSample VolumePhaseSet::sample(
    Float3 axis,
    Float2 random) const noexcept {
    UInt selected = 0u;
    Float rescaled = random.x;
    Float sum = 0.0f;
    const auto has_closure = _count > 0u;
    $if(has_closure) {
        sum = _weights.read(0u).w;
        UInt index = 1u;
        $while(index < _count) {
            const auto sample_weight =
                _weights.read(index).w;
            sum += sample_weight;
            const auto threshold =
                sample_weight / sum;
            $if(rescaled < threshold) {
                selected = index;
                rescaled /= threshold;
            }
            $else {
                rescaled =
                    (rescaled - threshold) /
                    (1.0f - threshold);
            };
            index += 1u;
        };
    };
    const auto safe_selected = select(
        0u, selected, has_closure);
    const auto type = _types.read(safe_selected);
    const auto parameters =
        _parameters.read(safe_selected).xyz();
    const auto phase_sample =
        cycles_volume_phase::sample(
            type,
            parameters,
            axis,
            make_float2(rescaled, random.y));
    const auto valid =
        has_closure &
        (phase_sample.pdf > 0.0f);
    return {
        .direction = select(
            axis,
            phase_sample.direction,
            valid),
        .pdf =
            select(0.0f, phase_sample.pdf, valid),
        .sampled_roughness = select(
            1.0f,
            cycles_volume_phase::sampled_roughness(
                type, parameters),
            valid),
        .selection_rescaled = rescaled,
        .closure_index = selected,
        .closure_type = type,
        .valid = valid};
}

}// namespace psycles::luisa_backend
