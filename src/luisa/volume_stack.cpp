#include <psycles/luisa/volume_stack.h>

#include <algorithm>

namespace psycles::luisa_backend {

using namespace luisa::compute;

VolumeStackEntry VolumeStackEntry::none() noexcept {
    return {
        .object = invalid_volume_identity,
        .shader = invalid_volume_identity,
        .surface_tag = invalid_volume_identity,
        .parameter_block = 0u,
        .instance_id = invalid_volume_identity,
        .valid = false};
}

VolumeStack::VolumeStack(
    std::size_t storage_size) noexcept
    : _storage_size{
          std::clamp(
              storage_size,
              std::size_t{1u},
              static_cast<std::size_t>(
                  maximum_volume_stack_size))},
      _identity{_storage_size},
      _instances{_storage_size},
      _count{0u} {
    _write(0u, VolumeStackEntry::none());
}

std::size_t VolumeStack::storage_size() const noexcept {
    return _storage_size;
}

std::size_t VolumeStack::maximum_entries() const noexcept {
    return _storage_size - 1u;
}

UInt VolumeStack::count() const noexcept {
    return _count;
}

Bool VolumeStack::empty() const noexcept {
    return _count == 0u;
}

void VolumeStack::_write(
    UInt index,
    const VolumeStackEntry &entry) noexcept {
    _identity.write(
        index,
        make_uint4(
            entry.object,
            entry.shader,
            entry.surface_tag,
            entry.parameter_block));
    _instances.write(index, entry.instance_id);
}

VolumeStackEntry VolumeStack::_read(
    UInt index) const noexcept {
    const auto identity = _identity.read(index);
    return {
        .object = identity.x,
        .shader = identity.y,
        .surface_tag = identity.z,
        .parameter_block = identity.w,
        .instance_id = _instances.read(index),
        .valid =
            identity.y != invalid_volume_identity};
}

VolumeStackEntry VolumeStack::entry(
    UInt index) const noexcept {
    const auto valid = index < _count;
    const auto safe_index =
        select(0u, index, valid);
    const auto value = _read(safe_index);
    const auto none = VolumeStackEntry::none();
    return {
        .object =
            select(none.object, value.object, valid),
        .shader =
            select(none.shader, value.shader, valid),
        .surface_tag = select(
            none.surface_tag,
            value.surface_tag,
            valid),
        .parameter_block = select(
            none.parameter_block,
            value.parameter_block,
            valid),
        .instance_id = select(
            none.instance_id,
            value.instance_id,
            valid),
        .valid = valid & value.valid};
}

Bool VolumeStack::_contains_pair(
    UInt object,
    UInt shader) const noexcept {
    Bool found = false;
    UInt index = 0u;
    $while((index < _count) & !found) {
        const auto candidate = _read(index);
        found =
            (candidate.object == object) &
            (candidate.shader == shader);
        index += 1u;
    };
    return found;
}

Bool VolumeStack::_contains_object(
    UInt object) const noexcept {
    Bool found = false;
    UInt index = 0u;
    $while((index < _count) & !found) {
        found = _read(index).object == object;
        index += 1u;
    };
    return found;
}

void VolumeStack::_append(
    const VolumeStackEntry &entry,
    Bool active,
    Bool duplicate) noexcept {
    const auto has_space =
        _count <
        static_cast<std::uint32_t>(
            maximum_entries());
    $if(active &
        entry.valid &
        !duplicate &
        has_space) {
        _write(_count, entry);
        _count += 1u;
        _write(_count, VolumeStackEntry::none());
    };
}

void VolumeStack::_exit(
    UInt object,
    UInt shader,
    Bool active) noexcept {
    Bool found = false;
    UInt exit_index = 0u;
    UInt index = 0u;
    $while((index < _count) & !found) {
        const auto candidate = _read(index);
        const auto matches =
            (candidate.object == object) &
            (candidate.shader == shader);
        exit_index =
            select(exit_index, index, matches);
        found = found | matches;
        index += 1u;
    };
    $if(active & found) {
        const auto last_index = _count - 1u;
        $if(exit_index != last_index) {
            _write(
                exit_index,
                _read(last_index));
        };
        _count -= 1u;
        _write(_count, VolumeStackEntry::none());
    };
}

void VolumeStack::clear() noexcept {
    _count = 0u;
    _write(0u, VolumeStackEntry::none());
}

void VolumeStack::initialize_background(
    const VolumeStackEntry &entry,
    Bool active) noexcept {
    _append(
        entry,
        active,
        _contains_pair(entry.object, entry.shader));
}

void VolumeStack::cross_boundary(
    const VolumeStackEntry &entry,
    Bool back_facing,
    Bool has_volume,
    Bool transmitted) noexcept {
    const auto active =
        has_volume & transmitted & entry.valid;
    $if(back_facing) {
        _exit(
            entry.object,
            entry.shader,
            active);
    }
    $else {
        _append(
            entry,
            active,
            _contains_pair(
                entry.object,
                entry.shader));
    };
}

void VolumeStack::copy_from(
    const VolumeStack &source) noexcept {
    const auto copied_count =
        min(
            source._count,
            static_cast<std::uint32_t>(
                maximum_entries()));
    UInt index = 0u;
    $while(index < copied_count) {
        _write(index, source._read(index));
        index += 1u;
    };
    _count = copied_count;
    _write(_count, VolumeStackEntry::none());
}

VolumeStackCameraInitializer::
    VolumeStackCameraInitializer(
        VolumeStack &stack,
        std::size_t enclosed_storage_size) noexcept
    : _stack{stack},
      _enclosed_storage_size{
          std::clamp(
              enclosed_storage_size,
              std::size_t{1u},
              static_cast<std::size_t>(
                  maximum_volume_stack_size))},
      _enclosed_objects{_enclosed_storage_size},
      _enclosed_count{0u} {
    _enclosed_objects.write(
        0u, invalid_volume_identity);
}

UInt VolumeStackCameraInitializer::
    enclosed_count() const noexcept {
    return _enclosed_count;
}

Bool VolumeStackCameraInitializer::_was_entered(
    UInt object) const noexcept {
    Bool found = false;
    UInt index = 0u;
    $while((index < _enclosed_count) & !found) {
        found =
            _enclosed_objects.read(index) == object;
        index += 1u;
    };
    return found;
}

Bool VolumeStackCameraInitializer::can_continue(
    UInt step) const noexcept {
    return (
        _stack._count <
        static_cast<std::uint32_t>(
            _stack.maximum_entries())) &
           (_enclosed_count <
            static_cast<std::uint32_t>(
                _enclosed_storage_size - 1u)) &
           (step <
            static_cast<std::uint32_t>(
                2u * _stack.storage_size()));
}

void VolumeStackCameraInitializer::observe(
    const VolumeStackEntry &entry,
    Bool back_facing,
    Bool has_volume) noexcept {
    const auto active =
        entry.valid & has_volume;
    const auto was_entered =
        _was_entered(entry.object);
    $if(active & back_facing) {
        const auto duplicate =
            was_entered |
            _stack._contains_object(entry.object);
        _stack._append(
            entry,
            true,
            duplicate);
    }
    $elif(active) {
        const auto has_space =
            _enclosed_count <
            static_cast<std::uint32_t>(
                _enclosed_storage_size - 1u);
        $if(has_space) {
            _enclosed_objects.write(
                _enclosed_count,
                entry.object);
            _enclosed_count += 1u;
            _enclosed_objects.write(
                _enclosed_count,
                invalid_volume_identity);
        };
    };
}

}// namespace psycles::luisa_backend
