# Statically sized native SVM local stacks

The light-emission and shadow-surface JIT entrypoints now allocate their SVM
stacks from the compiled scene's static stack bound. They no longer fall back
to the interpreter API's conservative 255-float default. No profile,
pre-render, sampled path, per-scene constant, or GPU usage readback computes
this size.

## Static proof and implementation

`src/compiler/cycles_svm_compiler.cpp::Stack::assign(width)` finds a free
contiguous range and updates `peak = max(peak, offset + width)`. Its users
and releases are compiler-side socket lifetimes, not executed shader paths.
The recorded high watermark therefore bounds every allocated lane, including
temporary values, vector derivatives, and the ten-lane BOTH bump state.
Resetting a shader domain clears live allocation users but deliberately does
not clear the high watermark. All compiled surface, volume and displacement
domains contribute, even if the current camera never executes them.

`ShaderImage::peak_stack_usage` stores this static bound. The scene linker
rejects a shader bound above `SVM_STACK_SIZE` and takes the maximum across the
statically used shader domain. The JIT allocation is therefore

```
M = max over shaders and their allocations of (offset + width)
physical float lanes = max(1, M)
```

The one-lane minimum represents an otherwise empty local object; it is not a
shader-dependent heuristic. Stack offsets, words, typed payloads, PC control
flow, closure state and Cycles feature-mask semantics do not change. A caller
without a compiled bound keeps the safe maximum-sized API default.

The internal `Stack` is a thin `Local<float>` wrapper whose default capacity
also keeps direct node fixtures source-compatible. SVM scratch remains
uninitialized, like Cycles' `float stack[SVM_STACK_SIZE]`; ordinary Luisa
scalar/vector default zero initialization is unchanged. This patch adds no
compiler pass, `noinline`, software floating-point path, or renderer policy
inside Luisa.

## Regression before any render

`test_luisa_cycles_svm_stack_extent.cpp` compiles real scene snapshots through
the production native scene loader. One scene has constant emission; the
second adds a triangle behind the camera with a Geometry/Vector Math material.
It immediately records both production auxiliary entrypoints and inspects
their AST local types. It does not write the size metadata, render, evaluate
a shading kernel, profile a path, or read back stack usage. The larger static
shader must increase the bound despite being behind the camera. Current
compiler results are 1 and 9 lanes; neither number is encoded as an expected
size. The old implementation fails on its remaining 255-lane array.

The existing native shadow and NEE oracle fixtures additionally record
conservative bounds 7, 23 and 255 to detect a forgotten argument or a
hardcoded replacement. Those are test inputs, not scene-sizing policy.
They then run their unchanged original-Cycles emission/transparency oracles
with their exact one-lane fixture bound. The scene linker and compiler suites
retain their original Cycles word-stream and static-stack regressions.

An initial end-to-end fixture used Camera Data, exposing the already known
full scene loader's legacy-lowering prerequisite. The final test uses
Geometry/Vector Math, supported by both existing loading paths. No shader
values or compiler outputs were substituted to hide that separate blocker.

## Whole-scene observations

RX 9070 XT / gfx1201, HIP, full scene bundles, fast math enabled:

| Scene | Static lanes | Light/shadow local array, each | Main coroutine frame |
| --- | ---: | ---: | ---: |
| Lone Monk | 25 | 1020 B -> 100 B | 220 B, unchanged |
| Monster | 22 | 1020 B -> 88 B | 284 B, unchanged |

These are observations of generated LLVM types, not inputs to the allocator.
The active worktree's main surface entrypoint already used the static bound;
all three SVM entries now do. The inherited main-surface/status-machine work
is not part of this selective patch. In the isolated committed-code projection
that entrypoint retains the maximum-sized default until its separate work is
promoted. Local scratch and a coroutine's persistent frame are distinct.

The native closure pool is separately scene-sized. Its generated local
storage is 1920 B in these scenes. Volume-stack and phase capacities already have
scene-owned bounds; hair quadrature and SSS hit-reservoir sizes are algorithmic
contracts, not safe candidates for shrinking to an observed usage count.

The large validation renders are Lone Monk 1440x1080 / 256 spp and Monster
1080x1080 / 256 spp. All 46 EXR channels are finite. Against unchanged original
Cycles HIP references, Combined / Diff Ind relative RMSE is
`0.01240462 / 0.12881742` and `0.00547921 / 0.02552785`, respectively.
Residuals are essentially unchanged. Single sampling times, 13.8884 s and
15.3433 s, are canaries, not paired benchmarks or evidence of a speedup.

`full-aux-stack.gdb` is an optional diagnostic counterfactual for x86-64 System
V: it restores only the auxiliary host argument to Cycles' old maximum while
keeping the same executable, scene words, main surface and device semantics.
It does not infer or set the production scene bound. Its ISA dumps are used
only to check resource effects, never to size an allocation or time a render.

The successful `monk-full-isa-r2` counterfactual records two shadow callable
constructions and one light-emission construction. LLVM confirms only the
auxiliary stacks return to 255 lanes. Matched final AMDGPU kernel metadata
shows NEE private storage `1360 -> 464 B` and shadow shading
`1920 -> 992 B`; both still use 256 VGPRs. Main-kernel resources are unchanged.
These are actual private-segment sizes, including other temporaries/spills,
not the C++ array size or a rendering-throughput measurement. The first
diagnostic attempt rendered successfully but its harness incorrectly expected
two rather than three recordings; only the corrected complete run is used.

## Validation and scope

Evidence is retained in `/var/tmp/psycles-svm-local-extents-SlzH2v`:

- `red-build-2.log`, `red-hip.log`: the two production oracle fixtures fail
  their new AST-size checks before the auxiliary caller fix. An earlier
  stale-binary run is explicitly named `stale-binary-not-validation.log` and
  is not used as evidence.
- `full-build.log`, `all-hip.log`, `all-fallback.log`: 32-thread all-target
  build, 167/167 existing HIP tests and 169/169 existing fallback tests pass.
- `final-full-build.log`, `final-all-hip.log`, `final-all-fallback.log`:
  the final 32-thread all-target rebuild and complete suites, including the
  added static scene test, pass 168/168 HIP and 170/170 fallback.
- `static-scene-final-red-hip.log`: the final real-scene fixture fails the
  old implementation. `static-scene-hip-fallback.log` passes both backends.
- `focused-native-vk.log`: 12/12 pass, including the real-scene fixture and
  original native shadow/NEE/bump oracles. All three strict gates are set:
  `LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
  `LUISA_VULKAN_DISABLE_DXC=1`.
- `source` is an application archive of `302831b9` with only this change;
  its 32-thread all-target build succeeds. It uses clean Luisa headers from
  `8e2b0ac78` and the active SDK binaries, not a clean independent SDK build.
  Executable RUNPATHs select its own application libraries first.
- The isolated full HIP suite passes 160/166. All six failures reproduce
  before this change, with identical reported values: surface population,
  compact preparation/tail, native closure, default Principled, and
  Principled transmission. They are not counted as successes. The final new
  real-scene test passes separately on isolated HIP and fallback.
- The isolated full fallback suite passes 163/169; the same six failures
  reproduce on the baseline with identical reported values.
- `isolated-native-vk.log`: 11/11 focused native Vulkan regressions pass,
  including the new scene-driven bound and the two auxiliary evaluators.
- The isolated original Map Range scene completes a full 16x16 / 4-spp render
  on HIP, fallback and strict native Vulkan. All 46 channels are finite.
  Vulkan logs show 25 successful SPIR-V compilations and `LD_DEBUG=libs`
  shows no DXC or DXIL library load.

Only this bounded API/local-storage change, its auxiliary callers, tests and
report belong to the patch. The inherited path-tracer changes, incomplete
Luisa work and parent gitlink remain unstaged.
