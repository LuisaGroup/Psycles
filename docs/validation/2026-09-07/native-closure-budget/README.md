# Native Cycles graph closure capacity

Native surface allocation now takes its closure capacity from the finalized
Cycles graph image, not from the legacy `SurfaceProgram` estimator. This is
scene-load/JIT metadata; it does not specialize on an observed ray, change a
closure opcode or payload, or discard a possible runtime closure.

## Original Cycles contract

Authority: Cycles 5.2.1 `cb168525138fecc792cc393f94afc39582b0103c` at
`/home/mike/Projects/blender-cycles-trace-5.2`:

- `scene/shader_graph.cpp::ShaderGraph::get_num_closures()` counts finalized
  graph nodes, once each, using the node's virtual `get_closure_type()`.
- `scene/shader_nodes.h` supplies closure types, including distribution and
  method properties. Emission and Background inherit `CLOSURE_NONE_ID`.
- `scene/scene.cpp::Scene::get_max_closure_count()` takes the maximum over
  referenced shader graphs, retains the scene's previous high watermark,
  and caps it at `MAX_CLOSURE` (64).
- `scene/shader.cpp::ShaderManager::add_default()` permanently references
  the default Principled graph. A fresh background-render scene therefore
  has a minimum capacity of 12, even with only an emission material.

The graph count is: none 0, BSSRDF 3, multiscatter 2, Principled 12,
volume 32, physical/F82 conductor and Beckmann/GGX glass and Chiang/Huang
hair 2, other closures 1. The volume contribution is Cycles'
`MAX_VOLUME_STACK_SIZE`, not a sampled active depth. Multiple graph nodes
can contribute more than 64 before the scene cap is applied.

Psycles currently builds a fresh immutable compiled scene. If persistent
scene updates are added, they must preserve Cycles' previous high watermark
rather than shrinking allocations while old compiled work remains live.
OSL is not involved in this native SVM path.

## Independent observations and regressions

The checked-in `probe.py` creates original Blender graphs. `observe.gdb`
reads the original graph/scene function return values without modifying the
inferior or substituting an evaluator. The trace Blender is
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`, with its
original HIP renderer on RX 9070 XT, at 8x8 / 1 spp.

For example, from the Psycles worktree:

```sh
gdb -q -batch \
  -x docs/validation/2026-09-07/native-closure-budget/observe.gdb \
  --args /home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  --background --factory-startup --threads 32 --python-exit-code 1 \
  --python docs/validation/2026-09-07/native-closure-budget/probe.py \
  -- diffuse-emission
```

`tests/data/cycles_svm_closure_budget.txt` retains all 20 observed graph/scene
pairs. Notable structural cases are Diffuse+Emission `1/12`, a constant Mix
that removes Principled `1/12`, a shared Diffuse node `1/12`, and three
distinct volume nodes `96/64`. They distinguish graph counting from opcode
counting, mix traversal occurrences, and the old legacy capacity estimator.

The standalone compiler regression reads that external fixture strictly and
checks node families, distribution variants, finalization, shared nodes,
inert dense shader holes, duplicate indices, the scene cap, and an empty
scene. No CPU renderer is introduced. The backend regression calls the real
native `compile_scene()` for Diffuse+Emission and a folded Principled branch
and verifies the allocation input actually becomes 12. Before the change,
the compiler reported no graph count and production used legacy capacity 2
(or 14 for the folded graph).

The graph count and maximum are separate metadata; the existing SVM words,
stack offsets, PC loop, feature masks and closure setup remain unchanged.
Unrepresented shader-index holes remain inert. This does not claim to
restore missing built-in shader word streams or complete volume/SSS node
support.

## Evidence and current validation

Original logs and red/green regressions are retained in
`/var/tmp/psycles-native-material-domain-BucnLr`. For Diffuse+Emission the
valid complete observation is `cycles-diffuse-emission-4.log`; earlier
attempts hit GDB failures and are not oracle evidence. The other 19 final
`cycles-CASE.log` observations completed successfully.

The active-worktree all-target build uses 32 threads. Full HIP 168/168 and
fallback 170/170 pass (`final-all-*.log` under
`/var/tmp/psycles-svm-local-extents-SlzH2v`). The host/compiler/policy suite
passes 146/146 (`host-compiler-policy.log`), excluding the existing source-size
violations and the unrelated Blender probe-registry synchronization failure.
Strict native Vulkan focused coverage passes 12/12, including the new scene
fixture.

The independent application projection is an archive of `1444110c` plus only
this closure metadata change, the regressions, and the compile/upload
diagnostic. Its all-target 32-thread build completes 929 steps. Compiler
tests pass 5/5; focused HIP, fallback and strict native Vulkan pass 6/6 each
(`isolated-core.log`, `isolated-hip.log`, `isolated-fallback.log`,
`isolated-native-vk.log`). It uses clean Luisa `8e2b0ac78` headers and the
active SDK binary libraries, not an independently rebuilt clean Luisa SDK.

Whole-program Map Range canaries on that isolated application, in HIP,
fallback, then Vulkan order, all complete at 16x16 / 4 spp with 46 finite EXR
channels. Vulkan runs with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`
and `LD_DEBUG=libs`. Its log confirms successful SPIR-V compilation without
loading DXC/DXIL (`map-isolated-vk.log`). These are correctness canaries, not
performance measurements.

The production native compile/upload diagnostic, without a render, reports
Lone Monk as 39 shader slots / 54308 words / 25 stack lanes / 24 closure slots,
Monster as 32 / 10948 / 22 / 24, and Classroom as 91 / 68180 / 23 / 12.
Logs are `static-monk-domain.log`, `static-monster-domain.log` and
`static-classroom-domain.log` in the evidence directory. The slots are Cycles'
80-byte records: 24 slots account for the observed 1920 B local pool. The
12-slot default-shader floor is not these scenes' final maximum.

This change is not a claimed speedup for those scenes. It corrects the
source of the capacity and removes a native allocation dependency on legacy
surface lowering. The legacy renderer continues to use its own estimator.
