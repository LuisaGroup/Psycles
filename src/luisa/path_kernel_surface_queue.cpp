#include "path_kernel_surface_queue.h"

#include "cycles_shader_identity.h"
#include "cycles_surface_sort.h"
#include "path_kernel_primitive_material.h"

#include <psycles/luisa/cycles_svm.h>

#include <limits>
#include <utility>

#include <luisa/coro/radix_sort.h>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto surface_sort_schema = "org.psycles.cycles.surface_sort";

class SurfaceSortHandler final
    : public luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler {
  using Prepare = luisa::compute::coro::WavefrontCoroExtensionPrepareContext;
  using Stage = luisa::compute::coro::WavefrontCoroExtensionStage;
  using Dispatch = luisa::compute::coro::WavefrontCoroExtensionDispatchContext;
  luisa::compute::Buffer<luisa::uint> _keys[2u];
  luisa::compute::Buffer<luisa::uint> _indices;
  luisa::compute::coro::radix_sort::temp_storage _temporary;
  unsigned _range;
  unsigned _partition;
  luisa::compute::coro::radix_sort::instance<luisa::compute::Buffer<luisa::uint>,
                                             luisa::compute::ByteBuffer, luisa::uint,
                                             luisa::uint, luisa::uint>
      _sort;

public:
  SurfaceSortHandler(Prepare &context, const Stage &stage, unsigned range,
                     unsigned partition, unsigned composite_range)
      : _range{range}, _partition{partition} {
    using namespace luisa::compute;
    using namespace luisa::compute::coro;
    _keys[0] = context.device.create_buffer<luisa::uint>(context.frame_capacity);
    _keys[1] = context.device.create_buffer<luisa::uint>(context.frame_capacity);
    _indices = context.device.create_buffer<luisa::uint>(context.frame_capacity);
    auto digit = std::min(composite_range, radix_sort::hist_block_size);
    _temporary = radix_sort::temp_storage{context.device, context.frame_capacity, digit};
    const auto *binding = &stage.binding("shader");
    const auto *desc = &context.frame_desc;
    auto fields = binding->reconstruct_slots();
    Callable<luisa::uint(luisa::uint, Buffer<luisa::uint>, ByteBuffer, luisa::uint,
                         luisa::uint, luisa::uint)>
        key = [binding, desc, fields, layout = context.frame_layout,
               soa = context.global_memory_soa](UInt slot, BufferUInt indices,
                                                ByteBufferVar storage, UInt capacity,
                                                UInt range, UInt partition) {
          auto index = indices.read(slot);
          auto frame = CoroFrame::create(desc);
          coro_frame_load_into(frame, storage, index, capacity, layout, soa, fields,
                               false, false);
          auto shader = binding->read<luisa::uint>(frame);
          UInt locality = 0u;
          $if(partition != 0u) { locality = index / partition; };
          return shader + locality * range;
        };
    Callable<luisa::uint(luisa::uint, Buffer<luisa::uint>, ByteBuffer, luisa::uint,
                         luisa::uint, luisa::uint)>
        index = [](UInt slot, BufferUInt indices, ByteBufferVar, UInt, UInt, UInt) {
          return indices.read(slot);
        };
    auto high_bit = 0u;
    while ((composite_range >> high_bit) != 1u) {
      ++high_bit;
    }
    _sort = radix_sort::instance<Buffer<luisa::uint>, ByteBuffer, luisa::uint,
                                 luisa::uint, luisa::uint>{
        context.device,
        context.frame_capacity,
        _temporary,
        &key,
        &index,
        &key,
        composite_range <= radix_sort::hist_block_size ? 1u : 0u,
        digit,
        0u,
        high_bit,
        "psycles_surface_sort"};
    LUISA_INFO("Psycles surface-sort Handler: boundary={} shader_keys={} "
               "partition_size={} composite_keys={}.",
               stage.boundary->index, range, partition, composite_range);
  }
  [[nodiscard]] luisa::string_view name() const noexcept override {
    return surface_sort_schema;
  }
  [[nodiscard]] luisa::compute::coro::WavefrontCoroExtensionExecution
  execution() const noexcept override {
    return luisa::compute::coro::WavefrontCoroExtensionExecution::before_resume;
  }
  [[nodiscard]] luisa::compute::BufferView<luisa::uint>
  dispatch_queue(const Dispatch &context) noexcept override {
    luisa::compute::BufferView<luisa::uint> keys[]{_keys[0].view(), _keys[1].view()};
    luisa::compute::BufferView<luisa::uint> indices[]{
        context.frame_indices, _indices.view().subview(0u, context.frame_count)};
    auto result = _sort.sort_switch(context.stream, keys, indices, context.frame_count,
                                    context.frame_indices, context.frame_buffer,
                                    context.frame_capacity, _range, _partition);
    return indices[result];
  }
};

class SurfaceQueueKeyStageImpl final : public SurfaceQueueKeyStage {

private:
  ScenePrimitiveStagePlan _plan;
  std::shared_ptr<const PrimitiveMaterialComponent> _material{
      make_primitive_material_component()};

public:
  explicit SurfaceQueueKeyStageImpl(ScenePrimitiveStagePlan plan) noexcept
      : _plan{plan} {}

  [[nodiscard]] UInt
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit) const noexcept override {
    const UInt instance_id = hit->inst;
    Var<InstanceGpu> instance = scene->instance_buffer->read(instance_id);
    Var<GeometryGpu> geometry = scene->geometry_buffer->read(instance.geometry_index);

    // Cycles intersection_get_shader_from_isect_prim(): the native
    // interpreter's identity is the shader-table index, not the deduped
    // graph topology. Material overrides are already resolved in this
    // geometry image. Do not reintroduce the legacy binding indirections.
    if (scene->native_cycles_svm_surface) {
      LUISA_ASSERT(scene->cycles_svm && scene->cycles_svm->geometry,
                   "Native surface sorting requires finalized Cycles geometry.");
      const auto &native = *scene->cycles_svm->geometry;
      UInt shader = 0u;
      const auto triangle_shader = [&] noexcept {
        shader = native.triangle_shader_buffer->read(geometry.cycles_primitive_offset +
                                                     hit->prim);
      };
      const auto curve_shader = [&] noexcept {
        const auto segment =
            scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base).read(hit->prim);
        shader = native.curve_buffer->read(segment.cycles_curve_index)
                     .shader_id.cast<std::uint32_t>();
      };
      if (_plan.mixed()) {
        $if(hit->is_procedural()) { curve_shader(); }
        $else { triangle_shader(); };
      } else if (_plan.curves) {
        curve_shader();
      } else {
        triangle_shader();
      }
      return shader & cycles_shader_identity::shader_mask;
    }
    UInt material_slot = 0u;

    const auto resolve_triangle = [&] noexcept {
      material_slot = _material->triangle_material_slot(scene, geometry, hit->prim);
    };
    const auto resolve_curve = [&] noexcept {
      const auto segment =
          scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base).read(hit->prim);
      material_slot = _material->curve_material_slot(scene, geometry, segment);
    };

    if (_plan.mixed()) {
      $if(hit->is_procedural()) { resolve_curve(); }
      $else { resolve_triangle(); };
    } else if (_plan.curves) {
      resolve_curve();
    } else {
      resolve_triangle();
    }

    // The legacy diagnostic evaluator still expands graph implementations
    // by SurfaceDispatch::surface_tag. Keep its own coherence key.
    auto binding = _material->resolve_binding(scene, instance, geometry, material_slot);
    return std::move(binding.surface_tag);
  }
};

} // namespace

std::unique_ptr<SurfaceQueueKeyStage>
make_surface_queue_key_stage(ScenePrimitiveStagePlan plan) {
  return std::make_unique<SurfaceQueueKeyStageImpl>(plan);
}

std::uint32_t surface_queue_key_range(const LuisaSceneData &scene) noexcept {
  LUISA_ASSERT(!scene.native_cycles_svm_surface || scene.cycles_svm,
               "Native surface sorting requires a Cycles shader table.");
  const auto count = scene.native_cycles_svm_surface
                         ? scene.cycles_svm->compilation.kernel_shaders.size()
                         : scene.surfaces.size();
  LUISA_ASSERT(count <= std::numeric_limits<std::uint32_t>::max(),
               "Surface queue key range {} exceeds the uint32 scheduler ABI.", count);
  return static_cast<std::uint32_t>(count);
}

luisa::compute::CoroSuspendExtensionPtr
make_surface_sort_annotation(UInt shader, std::uint32_t shader_count,
                             bool native_cycles) noexcept {
  auto annotation = luisa::compute::coro_annotation(surface_sort_schema, 1u);
  annotation.read("shader", shader)
      .attribute("shader_count", shader_count)
      .attribute("native_cycles", native_cycles);
  return std::move(annotation).build();
}

luisa::unique_ptr<luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>
make_surface_sort_handler(
    luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
    const luisa::compute::coro::WavefrontCoroExtensionStage &stage) noexcept {
  using namespace luisa::compute;
  namespace sort = luisa::compute::coro::radix_sort;
  if (stage.extension->schema() != surface_sort_schema) {
    return nullptr;
  }
  LUISA_ASSERT(stage.extension->version() == 1u,
               "Unsupported Psycles surface-sort schema version.");
  std::uint64_t range = 0u;
  bool native = false, has_native = false;
  for (const auto &attribute : stage.extension->attributes()) {
    if (attribute.name == "shader_count") {
      range = luisa::get<std::uint64_t>(attribute.value);
    }
    if (attribute.name == "native_cycles") {
      native = luisa::get<bool>(attribute.value);
      has_native = true;
    }
  }
  LUISA_ASSERT(range != 0u && range <= std::numeric_limits<unsigned>::max() &&
                   has_native &&
                   stage.binding("shader").type() == Type::of<luisa::uint>() &&
                   stage.binding("shader").readable(),
               "Invalid Psycles surface-sort typed binding or attributes.");
  auto shader_count = static_cast<unsigned>(range);
  auto partition =
      native ? cycles_surface_sort_partition_size(context.frame_capacity, shader_count)
             : 0u;
  auto partitions = partition == 0u
                        ? 1u
                        : context.frame_capacity / partition +
                              unsigned(context.frame_capacity % partition != 0u);
  auto composite = range * partitions;
  LUISA_ASSERT(composite <= std::numeric_limits<unsigned>::max(),
               "Surface-sort composite key overflows uint.");
  if (composite > sort::hist_block_size &&
      context.device.compute_warp_size() != sort::warp_size) {
    LUISA_WARNING(
        "Psycles surface-sort radix path needs {}-lane subgroups; annotation ignored.",
        sort::warp_size);
    return nullptr;
  }
  return luisa::make_unique<SurfaceSortHandler>(context, stage, shader_count, partition,
                                                static_cast<unsigned>(composite));
}

} // namespace psycles::luisa_backend::detail
