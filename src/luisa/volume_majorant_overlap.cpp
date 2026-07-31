#include <psycles/luisa/volume_majorant_overlap.h>

#include <limits>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

VolumeMajorantRuntimeExtrema
VolumeMajorantEntryProvider::extrema(
    const VolumeStackEntry &entry,
    const VolumeMajorantLeaf &leaf,
    Float object_density,
    Float shade_offset,
    Float3 world_ray_origin,
    Float3 world_ray_direction) const noexcept {
    static_cast<void>(entry);
    static_cast<void>(shade_offset);
    static_cast<void>(world_ray_origin);
    static_cast<void>(world_ray_direction);
    return {
        .minimum =
            leaf.sigma_minimum *
            object_density,
        .maximum =
            leaf.sigma_maximum *
            object_density};
}

VolumeMajorantOverlapTraversal::RootLookup
VolumeMajorantOverlapTraversal::_find_root(
    const VolumeStackEntry &entry)
    const noexcept {
    RootLookup result;
    result.root.scale =
        make_float3(0.0f);
    result.root.node = 0u;
    result.root.translation =
        make_float3(0.0f);
    result.root.shader =
        invalid_volume_identity;
    result.found = false;

    if (_node_count == 0u ||
        _root_count == 0u ||
        _range_count == 0u ||
        _world_range >= _range_count) {
        return result;
    }

    const auto object_entry =
        entry.instance_id !=
        invalid_volume_identity;
    const auto range_index =
        select(
            _world_range,
            entry.instance_id,
            object_entry);
    $if(entry.valid &
        (range_index < _range_count)) {
        Var<VolumeMajorantRootRangeGpu> range =
            _ranges.read(range_index);
        const auto safe_offset =
            min(
                range.offset,
                _root_count);
        const auto valid_range =
            (range.offset <= _root_count) &
            (range.count <=
             _root_count - safe_offset);
        $if(valid_range) {
            UInt cursor =
                range.offset + range.count;
            const auto shader =
                entry.shader &
                volume_majorant_cycles_shader_mask;
            Bool identity_matched = false;
            $while(
                (cursor > range.offset) &
                !identity_matched) {
                cursor -= 1u;
                Var<VolumeMajorantRootGpu>
                    candidate =
                        _roots.read(cursor);
                const auto matches =
                    candidate.shader == shader;
                $if(matches) {
                    identity_matched = true;
                    $if(candidate.node <
                        _node_count) {
                        result.root = candidate;
                        result.found = true;
                    };
                };
            };
        };
    };
    return result;
}

void VolumeMajorantOverlapTraversal::_select_entry(
    Bool condition,
    const VolumeStackEntry &entry,
    Float object_density) noexcept {
    $if(condition) {
        _active_entry.object =
            entry.object;
        _active_entry.shader =
            entry.shader;
        _active_entry.surface_tag =
            entry.surface_tag;
        _active_entry.parameter_block =
            entry.parameter_block;
        _active_entry.instance_id =
            entry.instance_id;
        _active_entry.sample_method =
            entry.sample_method;
        _active_entry.valid =
            entry.valid;
        _active_object_density =
            object_density;
    };
}

Bool VolumeMajorantOverlapTraversal::_setup(
    Float shade_offset) noexcept {
    Bool valid = false;
    $if(_no_overlap) {
        const auto leaf = _active.current();
        _active_valid =
            _active_valid & leaf.valid;
        valid =
            _active_valid &
            _lookup_complete;
    }
    $else {
        const auto skip = _active_entry;
        const auto active_leaf =
            _active.current();
        auto selected_maximum =
            select(
                std::numeric_limits<float>::max(),
                active_leaf.maximum,
                skip.valid);
        UInt index = 0u;
        $while(index < _stack.count()) {
            const auto entry =
                _stack.entry(index);
            const auto skip_entry =
                skip.valid &
                (entry.object == skip.object) &
                (entry.shader == skip.shader);
            $if(!skip_entry) {
                const auto lookup =
                    _find_root(entry);
                $if(lookup.found) {
                    const auto space =
                        _provider.entry_space(
                            entry,
                            _world_ray_origin,
                            _world_ray_direction);
                    VolumeMajorantTraversal candidate{
                        _nodes,
                        lookup.root,
                        space.ray_origin,
                        space.ray_direction,
                        active_leaf.minimum,
                        _ray_maximum};
                    const auto leaf =
                        candidate.current();
                    const auto extrema =
                        _provider.extrema(
                            entry,
                            leaf,
                            space.object_density,
                            shade_offset,
                            _world_ray_origin,
                            _world_ray_direction);
                    _sigma_minimum +=
                        select(
                            0.0f,
                            extrema.minimum,
                            leaf.valid);
                    _sigma_maximum +=
                        select(
                            0.0f,
                            extrema.maximum,
                            leaf.valid);

                    const auto select_candidate =
                        leaf.valid &
                        (leaf.maximum <=
                         selected_maximum);
                    selected_maximum =
                        select(
                            selected_maximum,
                            leaf.maximum,
                            select_candidate);
                    _active._select_from(
                        select_candidate,
                        candidate);
                    _select_entry(
                        select_candidate,
                        entry,
                        space.object_density);
                    _active_valid =
                        select(
                            _active_valid,
                            true,
                            select_candidate);
                }
                $else {
                    // A missing root is a malformed scene-resource domain.
                    // Do not run coordinate transforms or shader extrema
                    // evaluation before invalidating the whole reduction.
                    _lookup_complete = false;
                };
            };
            index += 1u;
        };
        _no_overlap =
            _stack.count() == 1u;
        const auto leaf = _active.current();
        _active_valid =
            _active_valid & leaf.valid;
        valid =
            _active_valid &
            _lookup_complete;
    };
    return valid;
}

VolumeMajorantOverlapTraversal::
    VolumeMajorantOverlapTraversal(
        Expr<Buffer<
            VolumeMajorantNodeGpu>> nodes,
        Expr<Buffer<
            VolumeMajorantRootGpu>> roots,
        Expr<Buffer<
            VolumeMajorantRootRangeGpu>> ranges,
        std::uint32_t node_count,
        std::uint32_t root_count,
        std::uint32_t range_count,
        std::uint32_t world_range,
        const VolumeStack &stack,
        const VolumeMajorantEntryProvider
            &provider,
        Float3 world_ray_origin,
        Float3 world_ray_direction,
        Float ray_minimum,
        Float ray_maximum,
        Float shade_offset) noexcept
    : _nodes{std::move(nodes)},
      _roots{std::move(roots)},
      _ranges{std::move(ranges)},
      _node_count{node_count},
      _root_count{root_count},
      _range_count{range_count},
      _world_range{world_range},
      _stack{stack},
      _provider{provider},
      _world_ray_origin{
          world_ray_origin},
      _world_ray_direction{
          world_ray_direction},
      _ray_maximum{ray_maximum},
      _active{_nodes, ray_maximum},
      _active_entry{
          VolumeStackEntry::none()},
      _active_object_density{0.0f},
      _sigma_minimum{0.0f},
      _sigma_maximum{0.0f},
      _active_valid{false},
      _no_overlap{false},
      _lookup_complete{
          node_count > 0u &&
          root_count > 0u &&
          range_count > 0u &&
          world_range < range_count} {
    // Cycles' global OctreeTracing begins at ray.tmin with FLT_MAX as its
    // endpoint. The inert traversal supplies exactly that common minimum to
    // the first setup without selecting a fictitious root.
    _active._minimum = ray_minimum;
    _active._maximum =
        std::numeric_limits<float>::max();
    static_cast<void>(_setup(shade_offset));
}

VolumeMajorantSegment
VolumeMajorantOverlapTraversal::current()
    const noexcept {
    const auto leaf = _active.current();
    return {
        .minimum = leaf.minimum,
        .maximum = leaf.maximum,
        .sigma_minimum =
            _sigma_minimum,
        .sigma_maximum =
            _sigma_maximum,
        .object = _active_entry.object,
        .shader = _active_entry.shader,
        .node = leaf.node,
        .valid =
            _active_valid &
            leaf.valid &
            _lookup_complete,
        .no_overlap = _no_overlap,
        .lookup_complete =
            _lookup_complete};
}

Bool VolumeMajorantOverlapTraversal::advance(
    Float shade_offset) noexcept {
    Bool advanced = false;
    const auto segment = current();
    $if(segment.valid) {
        const auto active_advanced =
            _active.advance();
        $if(active_advanced) {
            const auto leaf =
                _active.current();
            const auto extrema =
                _provider.extrema(
                    _active_entry,
                    leaf,
                    _active_object_density,
                    shade_offset,
                    _world_ray_origin,
                    _world_ray_direction);
            _sigma_minimum =
                extrema.minimum;
            _sigma_maximum =
                extrema.maximum;
            _active_valid = leaf.valid;
            advanced = _setup(
                shade_offset);
        };
    };
    return advanced;
}

}// namespace psycles::luisa_backend
