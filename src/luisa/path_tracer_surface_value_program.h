#pragma once

#include "path_tracer_attribute_lookup.h"
#include "path_tracer_internal.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_texture_sampling.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <luisa/dsl/local.h>

namespace psycles::luisa_backend::detail {

struct PathSurfaceAmbientOcclusionContext;

using SurfaceValueStackBank =
    std::array<float, SurfaceValueRuntime::stack_capacity>;

struct SurfaceValueLocalScalarView {
    luisa::compute::detail::Ref<SurfaceValueStackBank> storage;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        return mutable_storage[std::forward<Index>(index)];
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        mutable_storage[std::forward<Index>(index)] =
            std::forward<Value>(value);
    }
};

struct SurfaceValueLocalVectorView {
    luisa::compute::detail::Ref<SurfaceValueStackBank> storage;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        auto base = std::forward<Index>(index);
        return make_float3(mutable_storage[base],
                           mutable_storage[base + 1u],
                           mutable_storage[base + 2u]);
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        auto base = std::forward<Index>(index);
        auto stored = std::forward<Value>(value);
        mutable_storage[base] = stored.x;
        mutable_storage[base + 1u] = stored.y;
        mutable_storage[base + 2u] = stored.z;
    }
};

struct SurfaceValueLocalUnsignedIntegerView {
    luisa::compute::detail::Ref<SurfaceValueStackBank> storage;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        auto base = std::forward<Index>(index);
        return make_float2(mutable_storage[base],
                           mutable_storage[base + 1u])
            .template bitcast<luisa::ulong>();
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<SurfaceValueStackBank> &>(storage);
        auto base = std::forward<Index>(index);
        auto stored = std::forward<Value>(value)
                          .template bitcast<luisa::float2>();
        mutable_storage[base] = stored.x;
        mutable_storage[base + 1u] = stored.y;
    }
};

struct SurfaceValueLocalsView {
    SurfaceValueLocalScalarView scalars;
    SurfaceValueLocalVectorView vectors;
    SurfaceValueLocalUnsignedIntegerView unsigned_integers;
};

struct SurfaceValueLocals {
    luisa::compute::Local<float> stack{SurfaceValueRuntime::stack_capacity};

    [[nodiscard]] SurfaceValueLocalsView view() const noexcept;
    void define_all() const noexcept;
};

template<typename T>
[[nodiscard]] auto surface_value_runtime_buffer(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot slot) noexcept {
    return runtime.device_view->buffer<T>(
        surface_value_runtime_buffer_slot(slot), false, true);
}

// Host/JIT dispatcher for exactly one unified value record. It owns the
// scene-pruned typed evaluator callables while leaving PC/control sequencing
// to the surface SVM interpreter. The exact instruction-variant side stream is
// read only when more than one evaluator inhabits the same handler fiber.
class SurfaceValueInstructionDispatcher {

  public:
    struct Impl;

  private:
    std::shared_ptr<const Impl> _impl;

  public:
    explicit SurfaceValueInstructionDispatcher(
        std::shared_ptr<const Impl> impl) noexcept;

    [[nodiscard]] bool requires_ambient_occlusion() const noexcept;

    void operator()(
        Expr<Buffer<float>> scalar_parameters,
        Expr<Buffer<luisa::float3>> vector_parameters,
        Expr<Buffer<float>> cycles_bsdf_tables,
        Expr<BindlessArray> textures,
        Expr<BindlessArray> geometry_heap,
        Var<SurfacePointCall> &point,
        Float3 transaction_shading_normal,
        Bool use_undisplaced_geometry,
        Var<luisa::uint4> instruction,
        UInt instruction_index,
        luisa::compute::detail::Ref<SurfaceValueStackBank> stack,
        const PathSurfaceAmbientOcclusionContext
            *ambient_occlusion) const noexcept;
};

[[nodiscard]] Float read_scalar_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept;

[[nodiscard]] Float3 read_vector_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept;

[[nodiscard]] SurfaceValueInstructionDispatcher
make_surface_value_instruction_dispatcher(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries = false) noexcept;

} // namespace psycles::luisa_backend::detail
