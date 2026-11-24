#include "cycles_shader_identity.h"
#include "cycles_surface_sort.h"
#include "path_kernel_surface_buckets.h"
#include "path_kernel_surface_queue.h"
#include "path_kernel_transitions.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/luisa-compute.h>

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

bool check_scheduler(Device &device, Stream &stream, const Oracle &oracle,
                     unsigned capacity, bool annotations) {
  using namespace luisa::compute::coro;
  const auto divisor = cycles_surface_sort_partition_size(capacity, oracle.shaders);
  auto records = device.create_buffer<luisa::uint4>(oracle.rows.size());
  auto order = device.create_buffer<unsigned>(capacity);
  std::vector<unsigned> actual(capacity, ~0u);
  stream << records.copy_from(oracle.rows.data()) << order.copy_from(actual.data())
         << synchronize();
  Coroutine<void(Buffer<luisa::uint4>, Buffer<unsigned>)> coroutine{
      [annotations, shaders = oracle.shaders](BufferUInt4 rows,
                                              BufferUInt physical_order) {
        const auto id = dispatch_x();
        const auto hint = rows.read(id % 6u).z;
        if (annotations) {
          $suspend("shade_surface", make_surface_sort_annotation(hint, shaders, true));
        } else {
          $suspend("shade_surface", coro_frame_export("coro_hint", hint));
        }
        // dispatch_x is restored from the frame; block/thread IDs are the
        // physical launch position after the scheduler applies its indices.
        physical_order.write(block_x() * 128u + thread_x(), id);
      }};
  WavefrontCoroSchedulerConfig config;
  config.thread_count = capacity;
  if (!annotations) {
    config.hint_range = oracle.shaders;
    config.hint_fields = {"shade_surface"};
    config.hint_partition_size = divisor;
  }
  config.execution_block_size = 128u;
  config.largest_continuation_first = true;
  config.incremental_continuation_counts = true;
  WavefrontCoroScheduler<Buffer<luisa::uint4>, Buffer<unsigned>> scheduler{
      device, coroutine, config};
  auto handlers = 0u;
  const auto partitions = capacity / divisor + unsigned(capacity % divisor != 0u);
  const auto can_sort = oracle.shaders * partitions <= radix_sort::hist_block_size ||
                        device.compute_warp_size() == radix_sort::warp_size;
  if (annotations) {
    if (coroutine.frame().field("coro_hint") != nullptr ||
        !scheduler.config().hint_fields.empty()) {
      std::cerr << "Typed surface-sort annotation must not rely on the legacy hint ABI\n";
      return false;
    }
    scheduler.register_extension_handler(stream, [&](auto &context, auto &stage) {
      auto handler = make_surface_sort_handler(context, stage);
      handlers += unsigned(handler != nullptr);
      return handler;
    });
    if ((handlers != 0u) != can_sort) {
      std::cerr << "Surface-sort handler must honor the radix subgroup precondition\n";
      return false;
    }
  }
  stream << scheduler(records, order).dispatch(capacity);
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
    if (can_sort && key < previous_key) {
      std::cerr << "Continuation is not ordered by Cycles locality/shader key\n";
      return false;
    }
    previous_key = key;
  }
  return true;
}

bool check_joint_surface_queue(Device &device, Stream &stream, const Oracle &oracle) {
  using namespace luisa::compute::coro;
  constexpr unsigned capacity = 15u, surface_count = 8u;
  auto records = device.create_buffer<luisa::uint4>(oracle.rows.size());
  auto routes = device.create_buffer<unsigned>(capacity);
  auto order = device.create_buffer<unsigned>(surface_count);
  auto completion = device.create_buffer<unsigned>(capacity + 1u);
  std::array<unsigned, capacity> inputs{};
  std::array<unsigned, surface_count> actual{};
  std::array<unsigned, capacity + 1u> ranks{};
  for (auto i = 0u; i < capacity; ++i) { inputs[i] = capacity - 1u - i; }
  actual.fill(~0u);
  ranks.fill(~0u);
  ranks[capacity] = 0u;
  stream << records.copy_from(oracle.rows.data()) << routes.copy_from(inputs.data())
         << order.copy_from(actual.data()) << completion.copy_from(ranks.data());
  Coroutine<void(Buffer<luisa::uint4>, Buffer<unsigned>, Buffer<unsigned>, Buffer<unsigned>)>
      coroutine{[shaders = oracle.shaders, capacity](BufferUInt4 rows, BufferUInt input,
                                                   BufferUInt physical_order, BufferUInt done) {
        const auto route = input.read(dispatch_x());
        // Initial queues: surface 4, producer 8, rival 3. Producer then adds
        // surface 4 and rival 4: the complete surface queue must beat rival 7.
        $if((route >= 4u) & (route < 12u)) { $suspend("producer"); };
        $if(route < 8u) {
          const auto shader = rows.read(route % 6u).z;
          $suspend("shade_surface", make_surface_sort_annotation(shader, shaders, true));
          physical_order.write(block_x() * 32u + thread_x(), route);
        }
        $else { $suspend("rival"); };
        done.write(route, done.atomic(capacity).fetch_add(1u));
      }};
  const auto *surface = coroutine.graph().node_by_name("shade_surface");
  if (surface == nullptr) { return false; }
  WavefrontCoroSchedulerConfig config;
  config.thread_count = capacity; // Admit every runtime route before any resume.
  config.execution_block_size = 32u;
  config.largest_continuation_first = true;
  config.incremental_continuation_counts = true;
  config.fused_continuation_counts = true;
  config.frame_buffer_compaction = false;
  config.report_stats = true;
  WavefrontCoroScheduler<Buffer<luisa::uint4>, Buffer<unsigned>, Buffer<unsigned>,
                        Buffer<unsigned>> scheduler{device, coroutine, config};
  auto handlers = 0u;
  scheduler.register_extension_handler(stream, [&](auto &context, auto &stage) {
    auto handler = make_surface_sort_handler(context, stage);
    handlers += unsigned(handler != nullptr);
    return handler;
  });
  stream << scheduler(records, routes, order, completion).dispatch(capacity);
  stream << order.copy_to(actual.data()) << completion.copy_to(ranks.data()) << synchronize();
  const auto &stats = scheduler.last_dispatch_stats();
  std::uint64_t sort_dispatches = 0u;
  for (const auto &stage : stats.extensions) { sort_dispatches += stage.dispatch_count; }
  const auto &target = stats.continuations[surface->index];
  if (handlers != 2u || sort_dispatches != 1u || target.dispatch_count != 1u ||
      target.executed_count != surface_count || target.peak_queued_count != surface_count ||
      ranks[capacity] != capacity) {
    std::cerr << "Both surface entries must share one real sort Handler and one resume\n";
    return false;
  }
  std::array<bool, surface_count> seen{};
  auto previous_key = 0u;
  for (const auto route : actual) {
    if (route >= surface_count || seen[route]) { return false; }
    seen[route] = true;
    const auto key = oracle.rows[route % oracle.rows.size()].z;
    if (key < previous_key) {
      std::cerr << "Joint surface queue is not ordered by the original Cycles shader IDs\n";
      return false;
    }
    previous_key = key;
  }
  for (auto route = 0u; route < capacity; ++route) {
    if (ranks[route] >= capacity || (ranks[route] < surface_count) != (route < surface_count)) {
      std::cerr << "Logical surface population 4+4 must be scheduled before rival 7\n";
      return false;
    }
  }
  return true;
}
bool check_multi_bucket_surface_routing(Device &device, Stream &stream) {
  using namespace luisa::compute::coro;
  constexpr unsigned shader_count = 78u;
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = device;
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &comp = scene->cycles_svm->compilation;
  comp.table.valid = true;
  comp.table.shader_count = shader_count;
  comp.table.words.resize(shader_count * 4u, 0u);
  comp.table.ir.valid = true;
  comp.table.ir.shaders.resize(shader_count);
  comp.shader_metadata.resize(shader_count);
  comp.shader_surface_entry_usage.resize(shader_count);
  comp.kernel_shaders.resize(shader_count);

  for (unsigned i = 0u; i < shader_count; ++i) {
    comp.table.words[i * 4u] = i + 1u;
    abi::SvmOpIr op{};
    op.instruction.opcode = abi::NODE_CLOSURE_BSDF;
    op.instruction.payload_offset = i * 4u;
    op.instruction.payload_word_count = 4u;
    comp.table.ir.shaders[i].routines[0u].push_back(op);
    comp.shader_metadata[i].surface_num_closures = (i < 38u) ? 1u : 4u;
    comp.shader_metadata[i].has_surface_bssrdf = (i >= 70u);
    if (i == 60u) {
      comp.kernel_shaders[i].flags = static_cast<int>(abi::SD_HAS_EMISSION);
      comp.shader_surface_entry_usage[i]
          .node_types_used[abi::NODE_CLOSURE_EMISSION] = true;
    }
  }
  scene->cycles_svm->kernel_features =
      psycles::luisa_backend::cycles_svm::kernel_feature_subsurface;

  const auto plan =
      make_path_surface_bucket_plan(*scene, true, true, false, true);
  if (plan.buckets.size() != 2u || plan.buckets[0u].max_surface_closures != 1u ||
      plan.buckets[0u].may_have_emission || plan.buckets[0u].may_have_bssrdf ||
      plan.buckets[1u].max_surface_closures != 4u ||
      !plan.buckets[1u].may_have_emission || !plan.buckets[1u].may_have_bssrdf) {
    std::cerr << "Unexpected 78-shader surface bucket specialization\n";
    return false;
  }

  auto out_bucket = device.create_buffer<unsigned>(shader_count);
  auto out_key = device.create_buffer<unsigned>(shader_count);
  Kernel1D check_map = [&](BufferUInt buckets_out, BufferUInt keys_out) {
    const auto sid = dispatch_x();
    const auto b = plan.emit_bucket_index(sid);
    UInt k = 0u;
    $if(b == 0u) { k = plan.emit_dense_sort_key(0u, sid); }
    $else { k = plan.emit_dense_sort_key(1u, sid); };
    buckets_out.write(sid, b);
    keys_out.write(sid, k);
  };
  auto map_shader = device.compile(check_map);
  std::vector<unsigned> host_b(shader_count, ~0u), host_k(shader_count, ~0u);
  stream << map_shader(out_bucket, out_key).dispatch(shader_count)
         << out_bucket.copy_to(host_b.data()) << out_key.copy_to(host_k.data())
         << synchronize();

  std::array<std::vector<bool>, 2u> seen{
      std::vector<bool>(plan.buckets[0u].key_count, false),
      std::vector<bool>(plan.buckets[1u].key_count, false)};
  for (unsigned i = 0u; i < shader_count; ++i) {
    const unsigned expected_b = (i < 38u) ? 0u : 1u;
    if (host_b[i] != expected_b || host_k[i] >= plan.buckets[expected_b].key_count ||
        seen[expected_b][host_k[i]]) {
      std::cerr << "Dense sort key bijection failed at shader " << i << '\n';
      return false;
    }
    seen[expected_b][host_k[i]] = true;
  }

  // Also verify 4-bucket + background fused continuation count conservation.
  constexpr unsigned n_threads = 256u;
  auto visited = device.create_buffer<unsigned>(n_threads);
  std::vector<unsigned> host_visited(n_threads, 0u);
  stream << visited.copy_from(host_visited.data()) << synchronize();
  Coroutine<void(Buffer<unsigned>)> coro4{[](BufferUInt out) {
    const auto id = dispatch_x();
    $suspend(path_transition::intersect_closest);
    const auto mode = (id * 7u + (id >> 4u)) % 5u;
    $if(mode == 4u) {
      $suspend(path_transition::shade_background);
      out.write(id, 100u);
    }
    $elif(mode == 0u) {
      $suspend(path_transition::shade_surface,
               make_surface_sort_annotation(id % 7u, 7u, true));
      out.write(id, 1u);
    }
    $elif(mode == 1u) {
      $suspend(path_transition::shade_surface_1,
               make_surface_sort_annotation(id % 5u, 5u, true));
      out.write(id, 2u);
    }
    $elif(mode == 2u) {
      $suspend(path_transition::shade_surface_2,
               make_surface_sort_annotation(id % 6u, 6u, true));
      out.write(id, 3u);
    }
    $else {
      $suspend(path_transition::shade_surface_3,
               make_surface_sort_annotation(id % 4u, 4u, true));
      out.write(id, 4u);
    };
  }};
  WavefrontCoroSchedulerConfig cfg4;
  cfg4.thread_count = n_threads;
  cfg4.execution_block_size = 32u;
  cfg4.largest_continuation_first = true;
  cfg4.incremental_continuation_counts = true;
  cfg4.fused_continuation_counts = true;
  cfg4.frame_buffer_compaction = true;
  WavefrontCoroScheduler<Buffer<unsigned>> sched4{device, coro4, cfg4};
  sched4.register_extension_handler(stream, make_surface_sort_handler);
  stream << synchronize() << sched4(visited).dispatch(n_threads)
         << visited.copy_to(host_visited.data()) << synchronize();
  for (unsigned i = 0u; i < n_threads; ++i) {
    const unsigned m = (i * 7u + (i >> 4u)) % 5u;
    const unsigned expected = (m == 4u) ? 100u : (m + 1u);
    if (host_visited[i] != expected) {
      std::cerr << "4-bucket fused continuation count mismatch at lane " << i
                << ": got " << host_visited[i] << " expected " << expected << '\n';
      return false;
    }
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
                             InstanceGpu{.geometry_index = 1u,
                                         .override_count = 2u,
                                         .cycles_primitive_offset = 2u},
                             InstanceGpu{.geometry_index = 2u}};
  const std::array materials{
      MaterialBindingGpu{.surface_tag = 1u, .cycles_shader_index = 12u},
      MaterialBindingGpu{.surface_tag = 1u, .cycles_shader_index = 19u}};
  const std::array overrides{
      MaterialBindingGpu{.surface_tag = 7u, .cycles_shader_index = 12u},
      MaterialBindingGpu{.surface_tag = 9u, .cycles_shader_index = 38u}};
  const std::array material_slots{0u, 1u};
  const std::array segments{CurveSegmentGpu{.curve_index = 0u, .cycles_curve_index = 1u},
                            CurveSegmentGpu{.curve_index = 1u, .cycles_curve_index = 0u}};
  const std::array<unsigned, 4u> triangles{
      12u | cycles_shader_identity::smooth_normal | cycles_shader_identity::cast_shadow,
      19u | cycles_shader_identity::cast_shadow,
      12u | cycles_shader_identity::cast_shadow, 38u | cycles_shader_identity::use_mis};
  const std::array curves{abi::KernelCurve{.shader_id = static_cast<int>(
                                               3u | cycles_shader_identity::cast_shadow)},
                          abi::KernelCurve{.shader_id = static_cast<int>(
                                               6u | cycles_shader_identity::use_mis)}};
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
         << segment_buffer.copy_from(segments.data()) << hit_buffer.copy_from(hits.data())
         << geometry.triangle_shader_buffer.copy_from(triangles.data())
         << geometry.curve_buffer.copy_from(curves.data()) << scene->heap.update()
         << synchronize();

  auto output = device.create_buffer<unsigned>(hits.size());
  std::array<unsigned, hits.size()> actual{};
  bool ok = check_partition_policy();
  if (surface_queue_key_range(*scene) != oracle.shaders) {
    std::cerr << "Native key range must preserve sparse shader-table identity\n";
    ok = false;
  }
  for (const auto plan :
       {ScenePrimitiveStagePlan{true, false}, ScenePrimitiveStagePlan{false, true},
        ScenePrimitiveStagePlan{true, true}}) {
    const auto stage = make_surface_queue_key_stage(plan);
    Kernel1D kernel = [&](Var<Buffer<CommittedHit>> input, BufferUInt out, UInt first) {
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
      const auto expected = oracle.rows[row].z;
      if (actual[row] != expected) {
        std::cerr << "row=" << row << " key=" << actual[row]
                  << " expected=" << expected << '\n';
        ok = false;
      }
    }
  }
  ok = check_scheduler(device, stream, oracle, 131073u, false) && ok;
  for (auto capacity : {65u, 131073u, 1048576u}) {
    ok = check_scheduler(device, stream, oracle, capacity, true) && ok;
  }
  ok = check_joint_surface_queue(device, stream, oracle) && ok;
  ok = check_multi_bucket_surface_routing(device, stream) && ok;
  if (ok) {
    std::cout << "Cycles surface queue shader identity passed\n";
  }
  return ok ? 0 : 1;
}
