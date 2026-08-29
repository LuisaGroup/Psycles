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

// Cycles exposes a 255-lane semantic address space, but a scene-pruned Luisa
// shader need not materialize all 255 private lanes when the verified program
// image names a strict prefix. A 32-lane floor limits the number of generated
// callable ABIs; above that floor, the finite geometric bucket set bounds
// physical waste below 2x while preserving one final bucket for the complete
// Cycles domain. The selected bucket is host/JIT state, never device control
// flow.
inline constexpr std::array<std::uint32_t, 4u>
    surface_value_stack_lane_buckets{32u, 64u, 128u,
                                     SurfaceValueRuntime::stack_capacity};
static_assert(surface_value_stack_lane_buckets[0u] <
                  surface_value_stack_lane_buckets[1u] &&
              surface_value_stack_lane_buckets[1u] <
                  surface_value_stack_lane_buckets[2u] &&
              surface_value_stack_lane_buckets[2u] <
                  surface_value_stack_lane_buckets[3u] &&
              surface_value_stack_lane_buckets.back() ==
                  SurfaceValueRuntime::stack_capacity);

[[nodiscard]] constexpr std::uint32_t
surface_value_stack_storage_lanes(std::uint32_t required_lanes) noexcept {
    for (const auto capacity : surface_value_stack_lane_buckets) {
        if (required_lanes <= capacity) { return capacity; }
    }
    return 0u;
}

template<std::size_t Capacity>
using SurfaceValueStackBankFor = std::array<float, Capacity>;

// Full-domain alias retained for tests which exercise the Cycles capacity
// boundary directly. Production selects SurfaceValueStackBankFor<N> from the
// proven scene maximum through the dispatcher below.
using SurfaceValueStackBank =
    SurfaceValueStackBankFor<SurfaceValueRuntime::stack_capacity>;

struct SurfaceValueLocalScalarView {
    const luisa::compute::Expression *storage{};

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto i = luisa::compute::def(std::forward<Index>(index));
        return luisa::compute::Expr<float>{builder->access(
            luisa::compute::Type::of<float>(), storage, i.expression())};
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto i = luisa::compute::def(std::forward<Index>(index));
        auto element = builder->access(
            luisa::compute::Type::of<float>(), storage, i.expression());
        auto &reference = *builder->create_temporary<
            luisa::compute::Var<float>>(element);
        reference = std::forward<Value>(value);
    }
};

struct SurfaceValueLocalVectorView {
    SurfaceValueLocalScalarView scalars;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto base = std::forward<Index>(index);
        return make_float3(scalars.read(base),
                           scalars.read(base + 1u),
                           scalars.read(base + 2u));
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto base = std::forward<Index>(index);
        auto stored = std::forward<Value>(value);
        scalars.write(base, stored.x);
        scalars.write(base + 1u, stored.y);
        scalars.write(base + 2u, stored.z);
    }
};

struct SurfaceValueLocalUnsignedIntegerView {
    SurfaceValueLocalScalarView scalars;

    template<typename Index>
    [[nodiscard]] auto read(Index &&index) const noexcept {
        auto base = std::forward<Index>(index);
        return make_float2(scalars.read(base),
                           scalars.read(base + 1u))
            .template bitcast<luisa::ulong>();
    }

    template<typename Index, typename Value>
    void write(Index &&index, Value &&value) const noexcept {
        auto base = std::forward<Index>(index);
        auto stored = std::forward<Value>(value)
                          .template bitcast<luisa::float2>();
        scalars.write(base, stored.x);
        scalars.write(base + 1u, stored.y);
    }
};

struct SurfaceValueLocalsView {
    SurfaceValueLocalScalarView scalars;
    SurfaceValueLocalVectorView vectors;
    SurfaceValueLocalUnsignedIntegerView unsigned_integers;

    explicit SurfaceValueLocalsView(
        const luisa::compute::Expression *storage) noexcept
        : scalars{storage}, vectors{scalars}, unsigned_integers{scalars} {}

    [[nodiscard]] const luisa::compute::Expression *
    storage_expression() const noexcept {
        return scalars.storage;
    }
};

struct SurfaceValueLocals {
    luisa::compute::Local<float> stack;

    explicit SurfaceValueLocals(std::uint32_t capacity) noexcept;
    [[nodiscard]] SurfaceValueLocalsView view() const noexcept;
};

// One immutable description of the buffers addressed by a unified value
// record. Passing this aggregate into family emitters keeps bytecode decoding
// independent of physical bindless slot numbers.
struct SurfaceValueBytecodeSlots {
    SurfaceValueRuntimeBufferSlot operand;
    SurfaceValueRuntimeBufferSlot metadata_static_u0;
    SurfaceValueRuntimeBufferSlot metadata_parameter;
    SurfaceValueRuntimeBufferSlot metadata_static_range;
    SurfaceValueRuntimeBufferSlot static_data;
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
// to the surface SVM interpreter. The instruction control word selects exactly
// one handler; material data never participates in device dispatch identity.
class SurfaceValueInstructionDispatcher {

  public:
    struct Impl;

  private:
    std::shared_ptr<const Impl> _impl;

  public:
    explicit SurfaceValueInstructionDispatcher(
        std::shared_ptr<const Impl> impl) noexcept;

    [[nodiscard]] bool requires_ambient_occlusion() const noexcept;
    [[nodiscard]] std::uint32_t stack_capacity() const noexcept;

    void
    operator()(Expr<Buffer<float>> scalar_parameters,
               Expr<Buffer<luisa::float3>> vector_parameters,
               Expr<Buffer<float>> cycles_bsdf_tables,
               Expr<BindlessArray> textures, Expr<BindlessArray> geometry_heap,
               Var<SurfacePointCall> &point, Float3 transaction_shading_normal,
               Bool use_undisplaced_geometry, Var<luisa::uint4> instruction,
               const SurfaceValueLocalsView &locals,
               const PathSurfaceAmbientOcclusionContext *ambient_occlusion)
        const noexcept;
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

[[nodiscard]] ULong read_unsigned_integer_dynamic(
    const ShaderServices &services, const SurfacePoint &point,
    const SurfaceValueLocalsView &locals, UInt address) noexcept;

[[nodiscard]] SurfaceValueInstructionDispatcher
make_surface_value_instruction_dispatcher(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries = false) noexcept;

} // namespace psycles::luisa_backend::detail
