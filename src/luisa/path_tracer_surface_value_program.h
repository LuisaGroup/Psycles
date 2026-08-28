#pragma once

#include "path_tracer_attribute_lookup.h"
#include "path_tracer_internal.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_texture_sampling.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <luisa/dsl/local.h>

namespace psycles::luisa_backend::detail {

struct PathSurfaceAmbientOcclusionContext;

using SurfaceValueScalarBank =
    std::array<float, SurfaceValueRuntime::scalar_capacity>;
using SurfaceValueVectorBank =
    std::array<luisa::float3, SurfaceValueRuntime::vector_capacity>;

template<typename T, std::size_t Size>
struct SurfaceValueLocalArrayView {
    using Storage = std::array<T, Size>;
    luisa::compute::detail::Ref<Storage> storage;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<Storage> &>(storage);
        return mutable_storage[std::forward<Index>(index)];
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<Storage> &>(storage);
        mutable_storage[std::forward<Index>(index)] =
            std::forward<Value>(value);
    }
};

template<typename T>
struct SurfaceValueLocalScalarView {
    luisa::compute::detail::Ref<T> storage;

    template<typename Index>
    [[nodiscard]] auto read(Index &&) const noexcept {
        return Var<T>{storage.expression()};
    }

    template<typename Index, typename Value>
    void write(Index &&, Value &&value) const noexcept {
        auto &mutable_storage = const_cast<
            luisa::compute::detail::Ref<T> &>(storage);
        mutable_storage = std::forward<Value>(value);
    }
};

struct SurfaceValueLocalsView {
    SurfaceValueLocalArrayView<
        float, SurfaceValueRuntime::scalar_capacity>
        scalars;
    SurfaceValueLocalArrayView<
        luisa::float3, SurfaceValueRuntime::vector_capacity>
        vectors;
    SurfaceValueLocalScalarView<luisa::ulong> unsigned_integers;
};

struct SurfaceValueLocals {
    luisa::compute::Local<float> scalars{
        SurfaceValueRuntime::scalar_capacity};
    luisa::compute::Local<luisa::float3> vectors{
        SurfaceValueRuntime::vector_capacity};
    luisa::compute::Local<luisa::ulong> unsigned_integers{
        SurfaceValueRuntime::unsigned_integer_capacity};

    [[nodiscard]] SurfaceValueLocalsView view() const noexcept;
    void define_all() const noexcept;
};

enum class SurfaceValueBankDefinition {
    program_prefix,
    full_bank,
};

template<typename T>
[[nodiscard]] auto surface_value_runtime_buffer(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot slot) noexcept {
    return runtime.device_view->buffer<T>(
        surface_value_runtime_buffer_slot(slot), false, true);
}

// A value program observes the immutable surface point and produces only the
// transaction's shading-normal projection. Passing the point by reference is
// the device analogue of Cycles' ShaderData pointer: it preserves field-wise
// demand loads instead of forcing every SurfacePointCall member through the
// callable ABI. No other SurfacePoint field is an output of this program.
using SurfaceValueProgramCallable = Callable<luisa::float3(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, luisa::uint, SurfacePointCall &, SurfaceValueScalarBank &,
    SurfaceValueVectorBank &, luisa::ulong &)>;

// AO-aware counterpart. The suffix is present only in a semantic domain that
// actually contains an AO instruction. The two packed integer vectors are
// (sequence_size, sample, pixel_hash, rng_offset) and
// (Cycles object, Cycles primitive); authored distance remains scene data.
using SurfaceValueAmbientOcclusionProgramCallable =
    Callable<luisa::float3(
        Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
        BindlessArray, luisa::uint, SurfacePointCall &,
        SurfaceValueScalarBank &, SurfaceValueVectorBank &, luisa::ulong &,
        Buffer<luisa::float4>, luisa::uint4, luisa::uint2)>;

// Host/JIT sum type over the two proven callable ABIs. It never becomes a
// device variant: construction chooses exactly one member from the exact
// value-program domain, and invocation emits only that callable.
class SurfaceValueProgram {

private:
    std::optional<SurfaceValueProgramCallable> _ordinary;
    std::optional<SurfaceValueAmbientOcclusionProgramCallable>
        _ambient_occlusion;

public:
    explicit SurfaceValueProgram(
        SurfaceValueProgramCallable callable) noexcept;
    explicit SurfaceValueProgram(
        SurfaceValueAmbientOcclusionProgramCallable callable) noexcept;

    [[nodiscard]] bool requires_ambient_occlusion() const noexcept;
    [[nodiscard]] luisa::compute::Function function() const noexcept;

    [[nodiscard]] Float3 operator()(
        Expr<Buffer<float>> scalar_parameters,
        Expr<Buffer<luisa::float3>> vector_parameters,
        Expr<Buffer<float>> cycles_bsdf_tables,
        Expr<BindlessArray> textures,
        Expr<BindlessArray> geometry_heap,
        UInt surface_tag,
        Var<SurfacePointCall> &point,
        luisa::compute::detail::Ref<SurfaceValueScalarBank> scalar_bank,
        luisa::compute::detail::Ref<SurfaceValueVectorBank> vector_bank,
        luisa::compute::detail::Ref<luisa::ulong> unsigned_integer_bank,
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

// Builds one Cycles-style value-program transaction for the selected
// endpoint. The exact value, SetNormal, and transitive Bump-height domains
// are derived from the same formal call-graph view; callers cannot pair a
// program with an unrelated handler domain.
[[nodiscard]] SurfaceValueProgram
make_surface_value_program_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries = false) noexcept;

} // namespace psycles::luisa_backend::detail
