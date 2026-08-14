# HIP resumable RayQuery frontier validation

This checkpoint redesigns LuisaCompute's gfx12 resumable RayQuery traversal
state. It does not add renderer policy to the HIP backend: source filtering,
alpha evaluation, material closures, and transparent-shadow batching remain
expressed by ordinary RayQuery handlers in Psycles. The backend change only
changes how the native HIPRT traversal frontier survives a candidate callback.

Machine: Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.53211.

LuisaCompute commits:

- `3a556dd47` -- formally localize invocation-local RayQuery handler scratch;
- `d98735f55` -- persist resumable RayQuery on the gfx12 hardware frontier;
- `8a7a62162` -- evaluate the handler-scratch proof as sparse reachability.

## Traversal-state model

Let `U` be the set of reachable but not yet visited BVH nodes. At every
candidate transition the implementation represents `U` as the disjoint union

```text
U = H disjoint-union S
```

where `H` is the bounded `ds_bvh_stack` frontier in LDS and `S` is either empty
or one parent-link subtree continuation `(root, parent, completed_child)`. A
complete BVH8 expansion may add eight entries, so traversal enters `S` before
expanding whenever the hardware entry count is at least `capacity - 8`.
Within `S`, descent clears `completed_child`; ascent follows the node parent and
uses the completed child as the skip token. Exhausting `root` resumes `H`.

A BLAS traversal saves the corresponding TLAS continuation separately. On a
candidate return, the RayQuery state persists both packed hardware-stack
addresses and both parent-link continuations. The candidate leaf has already
been removed from `U`, so resumption cannot replay it. An unexpected hardware
overflow traps because it would violate the partition invariant instead of
silently dropping or repeating candidates.

The former implementation persisted a 64-entry per-thread private software
stack. The new state is 224 B instead of 448 B and uses a nine-entry resumable
hardware frontier. Nine is the minimum capacity that leaves one ordinary
hardware entry plus the complete eight-child expansion reserve. Synchronous
RayQuery retains its measured 16-entry hot frontier; ordinary static trace
retains eight entries.

The deep regression constructs 1024 overlapping BLAS triangles and 256
overlapping TLAS instances. It verifies that every candidate callback is
observed exactly once, covering repeated `proceed`, BLAS/TLAS continuation,
paired triangles, rejection, commit, and early termination.

## Handler-local storage proof

XIR outlining previously treated every entry-block `alloca` referenced by a
candidate handler as captured state, even when the allocation was only
invocation-local scratch. The new analysis localizes an allocation only when
all of the following hold:

1. every use is a direct whole-object load or store;
2. its address never escapes through GEP, cast, call, pointer store, or return;
3. exactly one candidate-handler region owns all loads;
4. every handler load is preceded by a full store on every path from that
   handler's entry.

The implementation evaluates the fourth condition through its equivalent
counterexample graph. Starting at handler entry in the uninitialized state,
each path is cut immediately after its first full store. Localization is valid
exactly when no load is reachable in the remaining graph:

```text
unsafe(alloca) iff
    exists path(entry, load) containing no earlier full_store(alloca)
```

The traversal scans instructions through the first store and visits each block
at most once per candidate allocation. Diamond joins and cycles need no dense
fixed-point iteration: both written arms are cut, while any unwritten arm still
reaches and rejects a load. Anything not proven safe remains captured.
Regression coverage includes load-before-store, a conditional single-arm
store, stores on both diamond arms, an outside initializer killed by a
must-store, and genuinely observable cross-candidate state.

The first implementation computed a separate dense predecessor fixed point for
every captured allocation. A cold Psycles mixed triangle/curve shadow shader
exposed its pathological complexity: it remained in XIR for more than 58 s,
and a five-second `perf` sample attributed 32.45% directly to predecessor
traversal plus 24.69% to associated hashing. The sparse equivalent completes
the entire cache-disabled HIP scene-traversal regression in 1.25 s; its two
HIP LLVM codegen intervals are 443.9 ms and 340.7 ms. The regression now
explicitly disables persistent shader cache so this compiler path is exercised
on every run.

The cutout kernel's projected callback environment fell from 48 B to 16 B and
its linked code object from approximately 46,836 B to 45,992 B. In the current
Psycles split kernels, the remaining projected environments are 96 B and
192 B; the analysis correctly rejects all of their cross-stage allocations.

## Renderer transparent-shadow integration

Psycles now expresses the Cycles GPU shadow traversal shape with one ordinary
RayQuery per batch. The handler maintains this order-independent invariant:

```text
hits[0:count) = nearest min(total, 4) accepted candidates, sorted by ray t
```

An opaque candidate or exhausted transparent-bounce budget terminates the
query. Otherwise the shade stage evaluates the four raw material closures in
distance order. It launches another query only when `total > count`, starting
after the farthest shaded hit. No opacity or closure is baked during scene
export; the material flag is only a conservative proof that a shader cannot be
transparent and therefore may terminate without deferred closure evaluation.

The batch storage is explicitly initialized before traversal. A Luisa DSL
`Var<T>` allocates storage but does not implicitly establish a value, and the
fixed-capacity reduction eagerly reads inactive lanes before masking them with
`count`. This initialization requirement was exposed by the cache-disabled HIP
test and is now covered on fallback, HIP, and native XIR-to-SPIR-V Vulkan.

Curve candidates use the affine inverse uploaded with the Cycles instance
instead of recomputing a general matrix inverse inside every candidate. Exact
ribbon quad construction is control-dependent on the coarse cylinder test and
the subdivision loop exits immediately after the same first accepted interval
as before. XIR-shape regressions verify both properties.

## Performance

### Cutout microbenchmark

Command:

```bash
./build-tests/bin/example_path_tracing_cutout hip --offline --spp 64
```

The image is 1024x1024 and timing is render-only. The first run is treated as a
cold/cache warm-up; the following five runs determine the median.

| Configuration | Median FPS | Relative |
|---|---:|---:|
| Previous independent default-path checkpoint | 447.302 | 1.000x |
| New frontier plus XIR scratch localization | 459.962 | 1.028x |
| Ordinary no-RayQuery trace | 913.061 | 2.041x throughput |

The five final warm cutout measurements were 464.227, 458.487, 458.441,
462.802, and 459.962 FPS. The remaining cutout/direct time ratio is 1.998x.
That ratio includes real alpha fetch/evaluation and candidate callback work;
it is not interpreted as traversal overhead alone.

A separate forced-resumable A/B isolates the traversal-state redesign from
the default synchronous cost choice. The old private-stack implementation was
approximately 327.45 FPS and the new persistent hardware frontier was
approximately 378.29 FPS, a 15.5% gain.

### Frontier-capacity A/B

Scene: current Lone Monk export, 640x480, 64 spp, staged wavefront. These runs
are only a capacity sensitivity test: the renderer source also contains the
new transparent-shadow batching work, so they must not be compared directly
with older whole-render checkpoints.

| Resumable capacity | Five-run warm median | Versus 9 entries |
|---|---:|---:|
| 9 | 1.77029 s | baseline |
| 12 | 1.77192 s | 0.09% slower |
| 16 | 1.78296 s | 0.72% slower |

Increasing the frontier does not improve render time and consumes more LDS.
The retained nine-entry capacity is therefore both the formal lower bound and
the measured optimum. Parent-link backtracking is not the current production
bottleneck.

### Lone Monk renderer validation

The final transparent-shadow implementation was rendered at 640x480, 64 spp,
with a 32-sample staged-wavefront dispatch on HIP:

```bash
./build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-all-scenes-5.2-20260814/exports/lone-monk \
  /var/tmp/psycles-shadow-batch-final.exr hip 640 480 64 64 - \
  0 0 0 0 64 - 1 0 wavefront-staged 32 32768 32 1 1 0 4 2 \
  4096 131072 0 0 1
```

The scene contains 348 geometries, 87,534 instances, and 37 materials. Scene
upload took 4.397 s, cold JIT compilation took 11.544 s, and render-only time
was 1.799 s. The Blender source enables adaptive sampling and denoising, while
this Psycles run uses fixed sampling; official Cycles comparisons must disable
both features as well.

Against the immediately preceding initialized-batch checkpoint, Combined has
RMSE 0.0017703, relative RMSE 0.0011346, MAE 0.00001570, and a mean-luminance
ratio of 1.000022. The 95th percentile is exactly zero and the 99th percentile
is 6.88e-8; the sparse maximum of 0.6978 lies on stochastic/geometry edges.
Emission, Environment, Transmission, and volume passes are unchanged where
present.

The triptych below compares the earlier single-hit transparent-shadow
checkpoint with the four-hit batch. Its larger Combined RMSE of 0.03480
(relative RMSE 0.02231, MAE 0.00525) reflects changed sample consumption and
Monte Carlo noise. Visual inspection finds no coherent geometry, material,
grass, roof, or illumination displacement: the amplified difference is
noise-like and follows the high-variance lit surfaces. Emission and
Environment remain bit-exact.

![Lone Monk single-hit shadow, four-hit batch, and amplified difference](triptychs/lone-monk-single-hit-vs-four-hit-shadow.png)

Machine-readable reports are retained at
`/var/tmp/psycles-shadow-batch-final-comparison.json` and
`/var/tmp/psycles-shadow-batch-vs-single-hit.json`; the rendered EXR is
`/var/tmp/psycles-shadow-batch-final.exr`.

## Remaining bottleneck

The resumable kernel still spills and reloads values across every candidate
callback. LLVM inspection shows that the dominant residual cost is the
continuation ABI and the live resource/private-reference set around
`luisa_ray_query_proceed`, not repeated TLAS restarts or parent-link traversal.

A whole-object struct-frame experiment reduced the apparent callback product
but regressed render time by about 31%, because it destroyed LLVM lifetime
separation and stack coloring. It was discarded and is not present in either
commit. The next design must preserve independent object lifetimes. The
candidate direction is an address-only table for proven private references,
combined with interprocedural rematerialization of immutable kernel arguments
when every call site has one identical kernel-argument provenance. Ambiguous,
external, escaping, or address-sensitive values must fail closed.

## Validation

All checks were rebuilt with 32 parallel jobs after the final source change:

```text
test_xir_pass_lower_ray_query_loop : 212 assertions / 18 tests
test_hip_llvm_pipeline            :  39 assertions / 12 tests
test_hip_ray_query_pipeline hip   : 1482 assertions / 8 tests
test_hip_ray_query_pipeline fallback: 42 assertions / 8 tests
psycles_luisa_scene_traversal_tests: pass
psycles_luisa_curve_ribbon_tests   : pass
```

The fallback run exercises shared semantic cases and explicitly skips only the
HIP implementation-specific stack tests. No CPU reference renderer is used;
Cycles remains the renderer reference.
