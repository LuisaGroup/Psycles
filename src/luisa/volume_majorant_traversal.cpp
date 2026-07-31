#include <psycles/luisa/volume_majorant_traversal.h>

#include <limits>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

namespace {

inline constexpr std::uint32_t
    float_mantissa_bits = 23u;
inline constexpr float largest_below_two =
    1.9999999f;
inline constexpr float half_smallest_leaf =
    1.0f /
    static_cast<float>(
        2u << volume_majorant_maximum_depth);
inline constexpr float overlap_epsilon =
    5.0e-4f;

}// namespace

UInt VolumeMajorantTraversal::_octant()
    const noexcept {
    const auto x =
        (_current_position.x >> _scale) & 1u;
    const auto y =
        ((_current_position.y >> _scale) &
         1u)
        << 1u;
    const auto z =
        ((_current_position.z >> _scale) &
         1u)
        << 2u;
    return (x | y | z) ^ _octant_mask;
}

void VolumeMajorantTraversal::_descend()
    noexcept {
    auto node = _nodes.read(_node);
    $while(node.first_child != -1) {
        _scale -= 1u;
        _node =
            cast<uint>(node.first_child) +
            _octant();
        node = _nodes.read(_node);
    };
}

Float3 VolumeMajorantTraversal::_floor_position()
    const noexcept {
    const auto mask = ~0u << _scale;
    return as<luisa::float3>(
        _current_position &
        make_uint3(mask));
}

Float VolumeMajorantTraversal::_intersect_leaf()
    noexcept {
    const auto box_minimum =
        _floor_position();
    auto intersections =
        (box_minimum - _ray_origin) /
        _ray_direction;
    intersections = select(
        make_float3(
            std::numeric_limits<float>::max()),
        intersections,
        intersections >
            make_float3(_minimum));
    const auto maximum =
        min(
            intersections.x,
            min(
                intersections.y,
                intersections.z));
    const auto next =
        as<luisa::uint3>(
            select(
                _ray_direction * maximum +
                    _ray_origin,
                box_minimum -
                    make_float3(
                        half_smallest_leaf),
                intersections ==
                    make_float3(maximum)));
    const auto difference =
        (_current_position.x ^ next.x) |
        (_current_position.y ^ next.y) |
        (_current_position.z ^ next.z);
    _current_position = next;
    _next_scale = 32u - clz(difference);
    return min(maximum, _ray_maximum);
}

VolumeMajorantTraversal::
    VolumeMajorantTraversal(
        const BufferVar<
            VolumeMajorantNodeGpu> &nodes,
        const Var<
            VolumeMajorantRootGpu> &root,
        Float3 ray_origin,
        Float3 ray_direction,
        Float ray_minimum,
        Float ray_maximum) noexcept
    : _nodes{nodes},
      _root_node{root.node},
      _node{root.node},
      _ray_maximum{ray_maximum},
      _minimum{ray_minimum},
      _maximum{
          std::numeric_limits<float>::max()},
      _ray_origin{make_float3(0.0f)},
      _ray_direction{make_float3(0.0f)},
      _current_position{make_uint3(0u)},
      _scale{float_mantissa_bits},
      _next_scale{float_mantissa_bits},
      _octant_mask{0u},
      _hierarchical{false},
      _valid{false} {
    auto local_position =
        (ray_origin +
         ray_direction * ray_minimum) *
            root.scale +
        root.translation;
    auto local_direction =
        ray_direction * root.scale;
    const auto finite_scale =
        !any(isinf(root.scale)) &
        !any(isnan(root.scale));
    const auto positive =
        local_direction >
        make_float3(0.0f);
    _octant_mask =
        select(0u, 1u, positive.x) |
        select(0u, 2u, positive.y) |
        select(0u, 4u, positive.z);
    local_position =
        select(
            local_position,
            make_float3(3.0f) -
                local_position,
            positive);
    local_position =
        min(
            local_position,
            make_float3(
                largest_below_two));
    _current_position =
        as<luisa::uint3>(
            local_position);
    _ray_direction =
        -abs(local_direction);
    _ray_origin =
        local_position -
        _ray_direction * ray_minimum;
    _hierarchical =
        finite_scale &
        all(
            local_position >
            make_float3(1.0f));
    _valid =
        ray_minimum < ray_maximum;
    $if(_hierarchical) {
        _descend();
        _maximum = _intersect_leaf();
    }
    $else {
        _maximum = ray_maximum;
    };
}

VolumeMajorantLeaf
VolumeMajorantTraversal::current()
    const noexcept {
    const auto node = _nodes.read(_node);
    return {
        .minimum = _minimum,
        .maximum = _maximum,
        .sigma_minimum =
            node.sigma_minimum,
        .sigma_maximum =
            node.sigma_maximum,
        .node = _node,
        .valid =
            _valid &
            (_minimum < _maximum)};
}

Bool VolumeMajorantTraversal::advance()
    noexcept {
    Bool advanced = false;
    $if(_valid &
        (_maximum < _ray_maximum)) {
        $if(_next_scale >
            float_mantissa_bits) {
            $if(abs(
                    _maximum -
                    _ray_maximum) >
                overlap_epsilon) {
                _minimum = _maximum;
                _maximum = _ray_maximum;
                _node = _root_node;
                _hierarchical = false;
                advanced = true;
            };
        }
        $else {
            $while(_scale < _next_scale) {
                const auto node =
                    _nodes.read(_node);
                _node =
                    cast<uint>(node.parent);
                _scale += 1u;
            };
            _descend();
            _minimum = _maximum;
            _maximum = _intersect_leaf();
            advanced = true;
        };
    };
    return advanced;
}

}// namespace psycles::luisa_backend
