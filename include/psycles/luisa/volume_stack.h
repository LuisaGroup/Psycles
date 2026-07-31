#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_stack.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/local.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

inline constexpr std::uint32_t invalid_volume_identity =
    ~std::uint32_t{0u};
inline constexpr std::uint32_t maximum_volume_stack_size = 32u;
inline constexpr luisa::float3 camera_volume_probe_direction{
    0.0f, 0.0f, 1.0f};

struct VolumeStackEntry {
    luisa::compute::UInt object;
    luisa::compute::UInt shader;
    luisa::compute::UInt surface_tag;
    luisa::compute::UInt parameter_block;
    luisa::compute::UInt instance_id;
    luisa::compute::Bool valid;

    [[nodiscard]] static VolumeStackEntry none() noexcept;
};

// Device-local path state with Cycles' volume-stack transition semantics.
// storage_size includes the mandatory SHADER_NONE terminator, so at most
// storage_size - 1 media can be active.
class VolumeStack {

  private:
    std::size_t _storage_size;
    luisa::compute::Local<luisa::uint4> _identity;
    luisa::compute::Local<luisa::uint> _instances;
    luisa::compute::UInt _count;

    void _write(
        luisa::compute::UInt index,
        const VolumeStackEntry &entry) noexcept;
    [[nodiscard]] VolumeStackEntry _read(
        luisa::compute::UInt index) const noexcept;
    [[nodiscard]] luisa::compute::Bool _contains_pair(
        luisa::compute::UInt object,
        luisa::compute::UInt shader) const noexcept;
    [[nodiscard]] luisa::compute::Bool _contains_object(
        luisa::compute::UInt object) const noexcept;
    void _append(
        const VolumeStackEntry &entry,
        luisa::compute::Bool active,
        luisa::compute::Bool duplicate) noexcept;
    void _exit(
        luisa::compute::UInt object,
        luisa::compute::UInt shader,
        luisa::compute::Bool active) noexcept;

    friend class VolumeStackCameraInitializer;

  public:
    explicit VolumeStack(std::size_t storage_size) noexcept;

    VolumeStack(const VolumeStack &) = delete;
    VolumeStack(VolumeStack &&) = delete;
    VolumeStack &operator=(const VolumeStack &) = delete;
    VolumeStack &operator=(VolumeStack &&) = delete;

    [[nodiscard]] std::size_t storage_size() const noexcept;
    [[nodiscard]] std::size_t maximum_entries() const noexcept;
    [[nodiscard]] luisa::compute::UInt count() const noexcept;
    [[nodiscard]] luisa::compute::Bool empty() const noexcept;
    [[nodiscard]] VolumeStackEntry entry(
        luisa::compute::UInt index) const noexcept;

    void clear() noexcept;
    void initialize_background(
        const VolumeStackEntry &entry,
        luisa::compute::Bool active) noexcept;
    // A miss may retain only the world entry. Object volumes are closed
    // manifolds; keeping leaked object entries to infinity creates the same
    // artifacts that Cycles' volume_stack_clean() explicitly prevents.
    void clean_for_background(
        bool keep_background) noexcept;
    void cross_boundary(
        const VolumeStackEntry &entry,
        luisa::compute::Bool back_facing,
        luisa::compute::Bool has_volume,
        luisa::compute::Bool transmitted) noexcept;
    void copy_from(const VolumeStack &source) noexcept;
};

// Camera rays require a one-time +Z probe to discover enclosing media.
// Front-facing hits mark objects entered by the probe; a back-facing hit with
// no preceding front hit means that the camera started inside that object.
class VolumeStackCameraInitializer {

  private:
    VolumeStack &_stack;
    std::size_t _enclosed_storage_size;
    luisa::compute::Local<luisa::uint> _enclosed_objects;
    luisa::compute::UInt _enclosed_count;

    [[nodiscard]] luisa::compute::Bool _was_entered(
        luisa::compute::UInt object) const noexcept;

  public:
    explicit VolumeStackCameraInitializer(
        VolumeStack &stack,
        std::size_t enclosed_storage_size =
            maximum_volume_stack_size) noexcept;

    VolumeStackCameraInitializer(
        const VolumeStackCameraInitializer &) = delete;
    VolumeStackCameraInitializer(
        VolumeStackCameraInitializer &&) = delete;
    VolumeStackCameraInitializer &operator=(
        const VolumeStackCameraInitializer &) = delete;
    VolumeStackCameraInitializer &operator=(
        VolumeStackCameraInitializer &&) = delete;

    [[nodiscard]] luisa::compute::UInt
    enclosed_count() const noexcept;
    [[nodiscard]] luisa::compute::Bool can_continue(
        luisa::compute::UInt step) const noexcept;
    void observe(
        const VolumeStackEntry &entry,
        luisa::compute::Bool back_facing,
        luisa::compute::Bool has_volume) noexcept;
};

}// namespace psycles::luisa_backend
