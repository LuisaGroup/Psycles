# Typed surface sorting through Coro Ext/Handler

## Contract and ownership

Cycles 5.2.1 remains the reference, revision
`cb168525138fecc792cc393f94afc39582b0103c` in
`/home/mike/Projects/blender-cycles-trace-5.2`. This change moves the existing
shader-identity/locality sorting policy into Psycles' typed annotation and
Handler. It changes no SVM words, node dispatch, closure state, feature masks,
scalar/vector initialization, or light transport algorithms. No application
`noinline` marking is added.

The staged host-wavefront path emits `org.psycles.cycles.surface_sort`, version
1, with a read-only `uint shader` binding and shader-count/native-Cycles
attributes. The Handler uses the existing Cycles partition-divisor projection
and generic radix sort. Its selected index view is handed directly to the
continuation. The application no longer configures `coro_hint`, `hint_fields`,
or `hint_partition_size` for this path. Graph/persistent compatibility paths
still use their existing export ABI; they are not claimed migrated here.

The shader count and partition divisor are runtime arguments of the key reader;
their host values select sort storage/pass configuration. Typed bindings come
from the compiler-proved reconstruction plan, not a hard-coded frame offset.
The existing subgroup capability restriction remains: the small bucket path
works on all tested backends, while unsupported large-radix annotations are
ignored. Ignoring this advisory ordering must preserve every frame and result.

The renderer-neutral SDK protocol is published on `origin/next` as
`c0dd7d444`. An ordinary independent Extension does not promise order after a
later gather. A read-only `before_resume` suffix instead hands an exact
permutation through its handlers and directly to its target continuation,
without gathering or relocating it again. Boundary-specific typed bindings
remain distinct. Queue accounting, admission, refill eligibility, and equal-
population priority use the correct physical queue/logical continuation owner.

The same SDK checkpoint fixes descriptor ownership: a scheduler must retain
immutable boundary/frame metadata after its factory-local source Coroutine
dies. The original whole-scene fallback test exposed the borrowed descriptor;
a permanent factory-lifetime witness reproduced it independently. This owns
host metadata, not another device frame allocation. The SDK's formal analysis,
red witnesses, and regression results are recorded in
`docs/validation/2026-09-07/resume-annotations/README.md` in LuisaCompute.

## Regression and backend evidence

Main evidence directory: `/var/tmp/psycles-coro-resume-8fNIXm`.

- The production surface-queue test retains the Cycles HIP shader-identity
  oracle and tests the actual Handler at capacities 65, 131073, and 1048576.
  It checks the entire frame permutation, physical continuation order,
  partition boundaries, sparse shader IDs, overrides, triangle/curve sources,
  and that the typed path has no legacy `coro_hint` field or hint config.
- Full original build uses all 32 threads (`whole-build.log`). Full HIP CTest:
  157/157; fallback: 159/159; compiler/adapter selection: 100/100. The existing
  `blender_export_render_settings` test is excluded from that last selection;
  this is not a claim that all repository CTests pass.
- The clean SDK headers/test sources pass all 13 affected coroutine test
  executables on HIP and fallback: 156 cases per backend, 13463/13462 assertions.
  Four new pre-resume tests cover permutation handoff, source lifetime,
  equal-count priority, multiple incoming boundaries, self edges, snapshot
  versus writable binding semantics, ignored annotations, frame reuse,
  SoA/AoS, compaction, token sorting, and all count-accounting modes.
- `sort-native-vulkan-loader.log` records 26 native SPIR-V compilations for
  the application oracle. `clean-native-vulkan.log` records 94 for the SDK
  regressions. Both set `LUISA_VULKAN_USE_XIR=1`,
  `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`, and
  `LD_DEBUG=libs`; neither loads DXC/DXIL libraries.
- Legacy dispatch spellings were separately migrated to
  `stream << scheduler(args...).dispatch(size)`: Psycles `4d035079`, Luisa
  `411f63cb2`. The existing SoA test's incorrect payload-relative physical
  field indices were separately corrected in Luisa `b41f64572`; no default
  initialization or backend alignment rule was changed to accommodate it.

The clean SDK surface tests link the active compiler/backend libraries, not a
clean rebuild of unrelated pending Local/scope/XIR passes. The Psycles gitlink
is intentionally not advanced from the dirty SDK checkout. The published APIs
above are required; the old pinned revision alone is not sufficient.

An exact snapshot of the staged Psycles source was separately built in
`/var/tmp/psycles-handler-candidate-a2CxC6` (355 steps, all 32 threads), using
published SDK headers and the active SDK libraries. It includes none of the
protected Psycles dirt. Its surface/shadow tests pass 2/2 HIP and 3/3 fallback
(including the full scene film test); the latter film executable also passes
when invoked directly with HIP. Its native-XIR Vulkan surface-queue test passes
with loader tracing and no DXC/DXIL loads. `ldd` confirms both Psycles libraries
come from this isolated build. This verifies the staged application code, not
an old SDK pin or a clean rebuild of the pending compiler changes.
After restoring the production path, `final-build.log` and
`final-hip.log`/`final-fallback.log`/`final-core.log` repeat the full 32-thread
build and the same 157/159/100 passing test selections.

## Lone Monk paired HIP measurement

Frame 4, 1440x1080, 256 fixed samples, seed 0, RX 9070 XT. Export:
`/var/tmp/psycles-lone-monk-f297ec53-20260904/export`. Each run uses the normal
runtime library path, disables shader caching, and records kernel trace, LLVM
IR, and ISA. The independent shadow pipeline remains enabled in all runs.
Only the surface-sort representation changes: typed annotation/Handler versus
the previous hint export/configuration. The control is built in place and the
production source is restored before the repeat; no relocated-library harness.

| Sequence | Surface sorting | Render seconds | Evidence directory |
| --- | --- | ---: | --- |
| B | Handler | 14.3337 | `/var/tmp/psycles-handler-monk-pLesvD` |
| A | legacy hint | 14.3137 | `/var/tmp/psycles-handler-control-iASbnf` |
| B | Handler | 14.3325 | `/var/tmp/psycles-handler-repeat-KOAArk` |

Mean Handler time is 14.3331 seconds, 0.14% longer than the control. This is
essentially neutral in this limited B/A/B experiment, not a claimed speedup.
It remains 3.63% longer than the previously matched Cycles HIP sample interval,
13.831431 seconds. The independent-shadow gain from the preceding checkpoint
must not be attributed to this Handler migration.

Paired control versus Handler repeat, GPU kernel durations:

| Stage | Dispatches, both | Control seconds | Handler seconds | VGPR / scratch bytes / block, both |
| --- | ---: | ---: | ---: | --- |
| surface | 1402 | 6.839490 | 6.849388 | 256 / 3328 / 512 |
| closest intersection | 1403 | 3.030989 | 3.034207 | 128 / 240 / 32 |
| light NEE | 1072 | 0.361207 | 0.359571 | 256 / 1360 / 32 |
| shadow intersection | 1071 | 0.812630 | 0.813770 | 160 / 400 / 32 |
| shadow shading | 384 | 0.219410 | 0.220718 | 256 / 1856 / 32 |

Both representations retain a 55-field / 220-byte main frame. This is not the
total state allocation: main plus equal-capacity shadow task/batch storage is
still 520 bytes per capacity slot, excluding queue buffers, versus the measured
Cycles 172+224 bytes. Shadow phase reuse/native packing remain open.

The two Handler surface kernels are `kernel_37b88ea9e42458fb`; their final IR
is byte-identical (SHA-256
`d4af830f570835b9cb5608e8fe91842c95ea4edd2835f5de343c2cf250e5c193`).
Map dumps by kernel name, not compilation sequence index. The tested repeat
runtime SHA-256 is
`072bb256b8a68b49f25fd7a7d5b9b0cc80b100726ac39c768311557e437e0e78`.
These active-tree timings retain the inherited surface block-size/fused-count
settings and pending path-tracer changes; those unrelated changes are not
included in this commit.

All images use multilayer `render.exr`, not separate per-pass files. Comparisons
against `/var/tmp/psycles-shadow-queue-cycles-aBpy7U/cycles.exr` verify the exact
Blender build metadata. That reference disables denoising/adaptive sampling.
Combined relative RMSE is 0.0111407085 / 0.0111380349 / 0.0111333974 (B/A/B).
All eight compared passes have zero invalid actual pixels. The repeated
Handler's DiffInd / GlossInd errors remain 0.1410593846 / 0.1619810285; sorting
does not fix the existing indirect-light discrepancies. Full pass metrics are
in each run's `compare.json`.

The complete SVM, image-parity, and performance goals remain open, including
global main/shadow queue policy, Cycles' compaction heuristic, native state
packing, outstanding geometry/domains, and clean dependency integration.
