# Sparse XIR ray-query scratch effects

## Scope and result

The compact surface program makes Barbershop's staged path practical at run
time, but exposed a host-compiler complexity defect in Luisa's coroutine
ray-query normalization. On the same Blender 5.2 Barbershop export, RX 9070 XT,
8x8 image, 256 fixed spp, and one 64-sample dispatch, the complete shader-JIT
interval fell from `99.7033--100.821 s` to `29.1316 s`. The final coroutine
compilation interval is `27.9451 s`; ray-query normalization remains its largest
phase at `16.8576 s` and is the next target.

The optimization changes host analysis only. The generated coroutine graph is
unchanged at 9 subroutines, 212 frame fields, and 3,184 frame bytes. Render-only
time remained in the same narrow range (`0.359723 s` before and `0.364140 s`
after). The all-pass EXR comparison has RMS `1.98955e-8` and maximum absolute
error `4.76838e-7`, with no pixel above `1e-6`; this is the normal
non-deterministic floating-point accumulation envelope, not a structured image
change.

## Root cause and formal model

For each candidate handler-local aggregate `A`, the original analysis attached
two dense `AggregateFieldBitmask(A)` values to every XIR instruction, including
instructions unrelated to `A`. With `L` primitive leaves and `N` instructions,
the identity effects alone therefore performed `Theta(NL/64)` allocation,
zeroing, copying, and destruction. Barbershop's profile placed most host time
in aggregate-mask construction/size operations and
`RayQueryHandlerScratchAnalyzer::instruction_effect`; this work did not depend
on the number of actual accesses.

The dataflow domain for a path is the pair `(need, define)`:

```text
need(A; B)   = need(A) union (need(B) - define(A))
define(A; B) = define(A) union define(B)

need(join)   = union of predecessor needs
define(join) = intersection of predecessor defines
```

`need` means a leaf may be read before a must-definition, while `define` means
the leaf is defined on every represented path. The lattice identity is
`(empty, empty)`. Luisa now represents that identity with two disengaged
optionals and materializes a dense mask only at a related load, store, or
reference-call boundary. The equations and fixed point are unchanged; only the
physical representation of the bottom element is sparse. This changes the
unrelated-instruction cost from `Theta(NL/64)` to `Theta(N)`, while relevant
mask operations retain their exact aggregate-leaf semantics.

Pointer provenance was the next measured hotspot. Resolution now uses one
memo table with the conventional white/gray/black DFS states: absence is
white, a pessimistic invalid cache entry is gray, and the final summary is
black. A back edge through a malformed cyclic GEP remains conservatively
invalid. Pure SSA operands are rejected before provenance recursion, and the
analysis validates only the operand roles admitted by each XIR instruction
kind. This removes one temporary active-set allocation per lookup without
weakening the accepted language.

`AggregateFieldBitmask` copy and move construction were also corrected to
reuse the already interned field tree. Copy allocates and copies exactly once;
move steals the large backing allocation rather than allocating, clearing, and
swapping a second one.

## Measured decomposition

| Revision | Complete shader JIT | Ray-query normalization | Pre-distill | CFG distill | Render-only |
|---|---:|---:|---:|---:|---:|
| dense effects | `99.7033--100.821 s` | not separately instrumented | not separately instrumented | not separately instrumented | `0.359723 s` |
| sparse identity masks | `32.9784 s` | `20.9157 s` | `6.56713 s` | `1.64452 s` | `0.360227 s` |
| cached provenance, pre-rebase | `28.7626 s` | `16.3784 s` | `6.59192 s` | `1.66676 s` | `0.364793 s` |
| final `b1a969e23` on current `next` | `29.1316 s` | `16.8576 s` | `6.49122 s` | `1.69597 s` | `0.364140 s` |

The final whole-JIT reduction relative to the two dense observations is
`3.42--3.46x` (`70.8--71.1%`). Frame layout takes `0.106 ms`, so frame layout
is not being mistaken for this host-analysis bottleneck.

The compact staged path is also a meaningful runtime boundary. Against the
same compact-populate megakernel (`2.08996 s` render-only), the staged path at
`0.359723 s` is `5.81x` faster. A prior one-sample dispatch took `6.743 s`
because it issued 256 host dispatch chunks; the accepted result uses
`samples_per_dispatch=64`, as required by the per-(pixel, sample) topology.

## Rejected register-bank experiment

A diagnostic scalarized the compact value banks behind generated switch
accessors. It was semantically correct but structurally worse: Barbershop HIP
JIT grew from `18.8898 s` to `95.4506 s`, render-only grew from `2.08996 s` to
`2.89015 s`, generated AMD code grew to `1,961,268 B`, and peak host RSS was
`7,283,544 KiB`. Strict native Vulkan spent more than three minutes before
emitting its first SPIR-V module. The experiment was fully reverted and was
not committed.

This result is consistent with Cycles 5.2 rather than contrary to it. Cycles'
`intern/cycles/kernel/svm/svm.h` deliberately uses a 255-float local SVM stack
and relies on same-shader wave coherence and separate, sorted shade-surface
wavefront kernels. Replacing indexed storage with a large control-flow mux is
not the right abstraction. Psycles should continue to improve scheduling,
storage lifetime, and fixed interpreter structure instead.

## Regression coverage

Luisa commit `b1a969e23` adds a complexity regression with a 4,096-leaf
aggregate and 8,192 unrelated arithmetic instructions before DCE. The full
lowering must still localize the one real scratch aggregate. This makes the
intended complexity property observable without timing thresholds.

The following checks passed after rebasing onto and pushing current Luisa
`next`:

```text
test_xir_aggregate_field_bitmask
  42 assertions in 8 tests

test_xir_pass_lower_ray_query_to_pipeline
  308 assertions in 31 tests

cmake --build build --parallel

./build/bin/psycles_luisa_scene_traversal_tests fallback
./build/bin/psycles_luisa_scene_traversal_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
./build/bin/psycles_luisa_scene_traversal_tests vk
```

All three traversal runs exited successfully. The Vulkan log contains native
SPIR-V optimization/codegen records and does not load DXC.

Reproduction of the final Barbershop compiler workload:

```text
LUISA_CORO_PROFILE_COMPILATION=1 \
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
./build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  output.exr hip 8 8 256 64 - 0 0 0 0 256 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

Raw logs, EXRs, and `perf` captures remain under
`/var/tmp/psycles-compact-populate-20260821` and
`/var/tmp/psycles-barbershop-coro-host*.perf.data`. This is a compiler-focused
8x8 canary, so no enlarged visual triptych is presented as a scene-quality
claim. The previously committed full-resolution scene triptychs remain the
quality oracle; the numerical all-pass comparison here is stronger evidence
for this analysis-only change.
