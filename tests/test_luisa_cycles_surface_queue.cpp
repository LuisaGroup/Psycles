#include "path_kernel_surface_queue.h"
#include "cycles_shader_identity.h"
#include "cycles_surface_sort.h"

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/wavefront.h>

#include <array>
#include <fstream>
#include <iostream>
#include <string>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

struct Oracle {
  unsigned shaders{};
  unsigned partition_size{};
  std::array<luisa::uint4, 6u> rows{};
};

bool read_oracle(Oracle &oracle) {
  std::ifstream input{PSYCLES_SURFACE_QUEUE_ORACLE};
  std::string kind;
  if (!(input >> kind >> oracle.shaders >> oracle.partition_size) ||
      kind != "configuration") {
    return false;
  }
  for (auto i = 0u; i < oracle.rows.size(); ++i) {
    auto &row = oracle.rows[i];
    if (!(input >> kind >> row.x >> row.y >> row.z >> row.w) || kind != "key") {
      return false;
    }
    if (row.x != i || row.z >= oracle.shaders || row.y >= 1048576u ||
        oracle.partition_size !=
            cycles_surface_sort_partition_size(1048576u, oracle.shaders) ||
        row.w != row.z + oracle.shaders * (row.y / oracle.partition_size)) {
      return false;
    }
  }
  return !(input >> kind) && input.eof();
}

bool check_partition_policy() {
  // Boundary values from DeviceQueue::num_sort_partitions and the following
  // divide_up in PathTraceWorkGPU::alloc_integrator_sorting (Cycles 5.2.1).
  // In particular, capacity / 65536 is floor, not ceiling, and 300 disables
  // partitioning. Nonmultiples must not silently round the divisor down.
  constexpr std::array cases{luisa::uint3{1u, 39u, 1u},
                             luisa::uint3{65535u, 39u, 65535u},
                             luisa::uint3{65536u, 39u, 65536u},
                             luisa::uint3{131071u, 39u, 131071u},
                             luisa::uint3{131072u, 39u, 65536u},
                             luisa::uint3{131073u, 39u, 65537u},
                             luisa::uint3{1000000u, 39u, 66667u},
                             luisa::uint3{1048576u, 299u, 65536u},
                             luisa::uint3{1048576u, 300u, 1048576u},
                             luisa::uint3{1048576u, 301u, 1048576u}};
  for (const auto c : cases) {
    if (cycles_surface_sort_partition_size(c.x, c.y) != c.z) {
      std::cerr << "Cycles surface partition divisor mismatch\n";
      return false;
    }
  }
  return true;
}

bool check_scheduler(Device &device, Stream &stream, const Oracle &oracle) {
  using namespace luisa::compute::coro;
  constexpr auto capacity = 131073u;
  const auto divisor = cycles_surface_sort_partition_size(capacity, oracle.shaders);
  auto records = device.create_buffer<luisa::uint4>(oracle.rows.size());
  auto order = device.create_buffer<unsigned>(capacity);
  std::vector<unsigned> actual(capacity, ~0u);
  stream << records.copy_from(oracle.rows.data()) << order.copy_from(actual.data())
         << synchronize();
  Coroutine<void(Buffer<luisa::uint4>, Buffer<unsigned>)> coroutine{
      [](BufferUInt4 rows, BufferUInt physical_order) {
        const auto id = dispatch_x();
        const auto hint = rows.read(id % 6u).z;
        $suspend("shade_surface", coro_frame_export("coro_hint", hint));
        // dispatch_x is restored from the frame; block/thread IDs are the
        // physical launch position after the scheduler applies its indices.
        physical_order.write(block_x() * 128u + thread_x(), id);
      }};
  WavefrontCoroSchedulerConfig config;
  config.thread_count = capacity;
  config.hint_range = oracle.shaders;
  config.hint_fields = {"shade_surface"};
  config.execution_block_size = 128u;
  config.largest_continuation_first = true;
  config.incremental_continuation_counts = true;
  config.hint_partition_size = divisor;
  WavefrontCoroScheduler<Buffer<luisa::uint4>, Buffer<unsigned>> scheduler{
      device, coroutine, config};
  scheduler(records, order).dispatch(capacity)(stream);
  stream << order.copy_to(actual.data()) << synchronize();
  std::vector<bool> seen(capacity);
  unsigned previous_key = 0u;
  for (const auto id : actual) {
    if (id >= capacity || seen[id]) {
      std::cerr << "Sorted continuation must preserve the full frame permutation\n";
      return false;
    }
    seen[id] = true;
    const auto key = oracle.rows[id % 6u].z + oracle.shaders * (id / divisor);
    if (key < previous_key) {
      std::cerr << "Continuation is not ordered by Cycles locality/shader key\n";
      return false;
    }
    previous_key = key;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  Oracle oracle;
  if (!read_oracle(oracle)) {
    std::cerr << "Invalid Cycles HIP surface-queue oracle\n";
    return 1;
  }
  Context context{argv[0]};
  auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();

  // Two materials share a topology tag, but not Cycles shader identity.
  // The second instance uses a geometry image with material overrides baked
  // into its canonical Cycles primitive range, as the scene uploader does.
  const std::array geometries{GeometryGpu{.bindless_base = 0u, .material_count = 2u},
                              GeometryGpu{.bindless_base = 0u,
                                          .material_count = 2u,
                                          .cycles_primitive_offset = 2u},
                              GeometryGpu{.bindless_base = geometry_bindless_stride,
                                          .material_count = 2u,
                                          .primitive_kind = geometry_kind_curve}};
  const std::array instances{InstanceGpu{.geometry_index = 0u},
                             InstanceGpu{.geometry_index = 1u, .override_count = 2u},
                             InstanceGpu{.geometry_index = 2u}};
  const std::array materials{
      MaterialBindingGpu{.surface_tag = 1u, .cycles_shader_index = 12u},
      MaterialBindingGpu{.surface_tag = 1u, .cycles_shader_index = 19u}};
  const std::array overrides{
      MaterialBindingGpu{.surface_tag = 7u, .cycles_shader_index = 12u},
      MaterialBindingGpu{.surface_tag = 9u, .cycles_shader_index = 38u}};
  const std::array material_slots{0u, 1u};
  const std::array segments{
      CurveSegmentGpu{.curve_index = 0u, .cycles_curve_index = 1u},
      CurveSegmentGpu{.curve_index = 1u, .cycles_curve_index = 0u}};
  const std::array<unsigned, 4u> triangles{
      12u | cycles_shader_identity::smooth_normal | cycles_shader_identity::cast_shadow,
      19u | cycles_shader_identity::cast_shadow,
      12u | cycles_shader_identity::cast_shadow, 38u | cycles_shader_identity::use_mis};
  const std::array curves{
      abi::KernelCurve{.shader_id =
                           static_cast<int>(3u | cycles_shader_identity::cast_shadow)},
      abi::KernelCurve{.shader_id =
                           static_cast<int>(6u | cycles_shader_identity::use_mis)}};
  const std::array hits{CommittedHit{.inst = 0u, .prim = 0u, .hit_type = 1u},
                        CommittedHit{.inst = 0u, .prim = 1u, .hit_type = 1u},
                        CommittedHit{.inst = 1u, .prim = 0u, .hit_type = 1u},
                        CommittedHit{.inst = 1u, .prim = 1u, .hit_type = 1u},
                        CommittedHit{.inst = 2u, .prim = 1u, .hit_type = 2u},
                        CommittedHit{.inst = 2u, .prim = 0u, .hit_type = 2u}};

  scene->geometry_buffer = device.create_buffer<GeometryGpu>(geometries.size());
  scene->instance_buffer = device.create_buffer<InstanceGpu>(instances.size());
  scene->geometry_material_buffer =
      device.create_buffer<MaterialBindingGpu>(materials.size());
  scene->override_material_buffer =
      device.create_buffer<MaterialBindingGpu>(overrides.size());
  auto slot_buffer = device.create_buffer<unsigned>(material_slots.size());
  auto segment_buffer = device.create_buffer<CurveSegmentGpu>(segments.size());
  auto hit_buffer = device.create_buffer<CommittedHit>(hits.size());
  scene->heap = device.create_bindless_array(2u * geometry_bindless_stride);
  scene->heap.emplace_on_update(4u, slot_buffer);
  scene->heap.emplace_on_update(geometry_bindless_stride + 4u, slot_buffer);
  scene->heap.emplace_on_update(geometry_bindless_stride, segment_buffer);
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->cycles_svm->compilation.kernel_shaders.resize(oracle.shaders);
  scene->cycles_svm->geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  auto &geometry = *scene->cycles_svm->geometry;
  geometry.triangle_shader_buffer = device.create_buffer<unsigned>(triangles.size());
  geometry.curve_buffer = device.create_buffer<abi::KernelCurve>(curves.size());
  stream << scene->geometry_buffer.copy_from(geometries.data())
         << scene->instance_buffer.copy_from(instances.data())
         << scene->geometry_material_buffer.copy_from(materials.data())
         << scene->override_material_buffer.copy_from(overrides.data())
         << slot_buffer.copy_from(material_slots.data())
         << segment_buffer.copy_from(segments.data())
         << hit_buffer.copy_from(hits.data())
         << geometry.triangle_shader_buffer.copy_from(triangles.data())
         << geometry.curve_buffer.copy_from(curves.data()) << scene->heap.update()
         << synchronize();

  auto output = device.create_buffer<unsigned>(hits.size());
  std::array<unsigned, hits.size()> actual{};
  const std::array<unsigned, hits.size()> legacy{1u, 1u, 7u, 9u, 1u, 1u};
  bool ok = check_partition_policy();
  for (const auto native : {false, true}) {
    scene->native_cycles_svm_surface = native;
    if (native && surface_queue_key_range(*scene) != oracle.shaders) {
      std::cerr << "Native key range must preserve sparse shader-table identity\n";
      ok = false;
    }
    for (const auto plan :
         {ScenePrimitiveStagePlan{true, false}, ScenePrimitiveStagePlan{false, true},
          ScenePrimitiveStagePlan{true, true}}) {
      const auto stage = make_surface_queue_key_stage(plan);
      Kernel1D kernel = [&](Var<Buffer<CommittedHit>> input, BufferUInt out,
                            UInt first) {
        const auto row = first + dispatch_x();
        const Var<CommittedHit> hit = input.read(row);
        out.write(row, stage->emit(scene, hit));
      };
      auto shader = device.compile(kernel);
      const auto first = plan.triangles ? 0u : 4u;
      const auto count = plan.mixed() ? 6u : (plan.triangles ? 4u : 2u);
      stream << shader(hit_buffer, output, first).dispatch(count)
             << output.copy_to(actual.data()) << synchronize();
      for (auto row = first; row < first + count; ++row) {
        const auto expected = native ? oracle.rows[row].z : legacy[row];
        if (actual[row] != expected) {
          std::cerr << "native=" << native << " row=" << row << " key=" << actual[row]
                    << " expected=" << expected << '\n';
          ok = false;
        }
      }
    }
  }
  ok = check_scheduler(device, stream, oracle) && ok;
  if (ok) {
    std::cout << "Cycles surface queue shader identity passed\n";
  }
  return ok ? 0 : 1;
}
