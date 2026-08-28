#pragma once

#include "path_tracer_ambient_occlusion.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_surface_value_program.h"
#include "path_tracer_surfaces.h"

#include "graph_surface_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct SurfaceSvmExecutionResources {
    Expr<Buffer<float>> scalar_parameters;
    Expr<Buffer<luisa::float3>> vector_parameters;
    Expr<Buffer<float>> cycles_bsdf_tables;
    Expr<BindlessArray> textures;
    Expr<BindlessArray> geometry_heap;
};

// Host/JIT eliminator for closure leaves. Implementations own only their
// reduction state; the interpreter owns the PC, typed locals, closure decode,
// and exact source ordering. Virtual dispatch happens while constructing the
// shader AST and therefore introduces no device-side vtable or weakly typed
// payload.
class SurfaceSvmClosureConsumer {

  public:
    virtual ~SurfaceSvmClosureConsumer() noexcept = default;

    // SetNormal is the sole transaction boundary. The compiler proves that it
    // precedes every structured closure record, so consumers can update their
    // ShaderData::N projection once without retaining prefix locals.
    virtual void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept = 0;

    virtual void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t> instruction_index) noexcept = 0;
};

// One Cycles-style PC loop over the replacement scene image. Value records,
// closure-weight SSA, guards, SetNormal and closure leaves share one typed
// local bank and one execution order; no value or closure suffix is replayed
// implicitly by this component.
class SurfaceSvmInterpreter {

  public:
    struct Impl;

  private:
    std::shared_ptr<const Impl> _impl;

  public:
    explicit SurfaceSvmInterpreter(
        std::shared_ptr<const Impl> impl) noexcept;

    [[nodiscard]] bool requires_ambient_occlusion() const noexcept;

    [[nodiscard]] Float3 execute(
        const SurfaceSvmExecutionResources &resources,
        const ShaderServices &services,
        UInt surface_tag,
        const SurfacePoint &base_point,
        SurfaceSvmClosureConsumer &consumer,
        const PathSurfaceAmbientOcclusionContext
            *ambient_occlusion = nullptr) const noexcept;
};

[[nodiscard]] SurfaceSvmInterpreter make_surface_svm_interpreter(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries = false) noexcept;

} // namespace psycles::luisa_backend::detail
