# HIP resumable RayQuery frontier validation

This checkpoint redesigns LuisaCompute's gfx12 RayQuery traversal state and
callback transport. It does not add renderer policy to the HIP backend: source
filtering, alpha evaluation, material closures, and transparent-shadow batching
remain expressed by ordinary RayQuery handlers in Psycles. The backend changes
only how the native HIPRT frontier and callback state are represented.

Machine: Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.53211.

LuisaCompute commits:

- `3a556dd47` -- formally localize invocation-local RayQuery handler scratch;
- `d98735f55` -- persist resumable RayQuery on the gfx12 hardware frontier;
- `8a7a62162` -- evaluate the handler-scratch proof as sparse reachability;
- `fb2427984` -- separate RayQuery identity from callback captures;
- `e6c65fb90` -- compact the synchronous hardware frontier to the proven
  nine-entry minimum.

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
stack. The new state is 224 B instead of 448 B. Both resumable and synchronous
RayQuery now use a nine-entry hardware frontier. Nine is the minimum capacity
that leaves one ordinary hardware entry plus the complete eight-child
expansion reserve; both paths use the same exact parent-link continuation when
that reserve would be crossed. Ordinary static trace retains eight entries and
its one-shot overflow fallback.

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

At the scratch-localization checkpoint, the cutout kernel's projected callback
environment fell from 48 B to 16 B and its linked code object from
approximately 46,836 B to 45,992 B. The corresponding Psycles split-kernel
environments were 96 B and 192 B; identity separation removes another 8 B
from each, leaving 88 B and 184 B. The analysis correctly retains all of their
genuinely cross-stage allocations.

## Callback identity and environment ABI

The follow-up redesign models a synchronous candidate callback as the product

```text
CallbackABI = QueryIdentity x UserEnvironment
```

`QueryIdentity` is not a user capture. It is the identity of the active
per-invocation traversal state and is always required by the candidate object.
AMDGPU private pointers are exactly 32 bits in the target data layout, so the
HIP LLVM RayQuery representation is now the exact `i32` state-address token
instead of the historical five-`i64`, 40 B source-layout surrogate. The token
is stored in an existing four-byte alignment hole in the 128 B synchronous
state; neither the state size nor the following ray offset changes. The native
wrapper passes a pointer to that token as a dedicated dispatcher operand.

Only `UserEnvironment` is subject to the interprocedural least-fixed-point
demand proof described above. An empty product is represented by `null`. A
single retained generic pointer is transported directly using the exact
one-field-product isomorphism `{p : ptr} <-> p`; non-pointer scalars are not
encoded as pointers. Larger products retain ordinary typed storage. This does
not merge object lifetimes or infer aliasing: it only removes representation
that has no semantic observer.

The structural regression compiles a cache-disabled RayQuery and checks the
final LLVM IR for all four properties: no 40 B query surrogate, no escaped
standalone query-token allocation, no materialized empty user environment,
and direct decoding of the 32-bit token from the dedicated dispatcher
identity operand. The semantic RayQuery suite additionally covers rejection,
commit, early termination, paired triangles, deep TLAS/BLAS continuation, and
reentrant fallback selection.

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
| Compact identity/environment ABI | 485.045 | 1.084x |
| Compact nine-entry synchronous frontier | 499.724 | 1.117x |
| Opaque-hit semantic projection | 501.687 | 1.122x |
| Ordinary no-RayQuery trace | 913.061 | 2.041x throughput |

The final ABI's five warm cutout measurements were 487.461, 482.681, 485.045,
486.078, and 483.040 FPS. Its median is 5.45% above the frontier-plus-XIR
checkpoint and 8.44% above the original independent checkpoint. The remaining
cutout/direct time ratio is 1.882x, down from 1.998x. That ratio includes real
alpha fetch/evaluation and candidate callback work; it is not interpreted as
traversal overhead alone. The generated kernel uses 150 VGPRs instead of the
pre-compaction 162, and the cutout's two static user environments collapse
from 128 B total to zero.

The final nine-entry frontier's five warm measurements were 498.612, 501.135,
494.660, 499.724, and 500.938 FPS. Its 499.724 FPS median is another 3.03%
above the compact-ABI/16-entry result and 11.72% above the original checkpoint.
The remaining cutout/direct time ratio is 1.827x.

The subsequent opaque-hit projection's five warm measurements were 505.410,
501.687, 499.664, 499.931, and 503.002 FPS. Its 501.687 FPS median is 0.39%
above the nine-entry result and 12.16% above the original checkpoint. The
remaining cutout/direct time ratio is 1.820x. The 64 spp output is pixel-exact
against the retained pre-projection 64 spp outputs; a low-spp image produced
by an earlier ISA probe was explicitly rejected as an invalid comparison
baseline.

This is a semantics-preserving projection, not a scene-specific shortcut. For
a surface candidate `h`, the generic transition is
`candidate := h; callback(candidate); commit(candidate)`. Once the instance's
opaque bit proves that the RayQuery language does not expose `h` to the
callback, the observable transition is exactly `committed := h`. The gfx12
flat traversal now performs that transition directly. Non-opaque and
procedural candidates retain the complete arbitrary DSL callback semantics,
including rejection, explicit commit, termination, candidate-ray access, and
reentrant tracing. The opacity proof is evaluated at the candidate, so it is
not incorrectly promoted across an observable handler invocation.

Static code-object metadata explains why this helps and rules out private
scratch as the dominant difference:

| Kernel | VGPR | SGPR spills | Private bytes | LDS bytes |
|---|---:|---:|---:|---:|
| Ordinary closest/any trace | 140 | 24 | 2,800 | 16,384 |
| RayQuery, 16-entry frontier | 150 | 0 | 144 | 32,768 |
| RayQuery, 9-entry frontier | 150 | 0 | 144 | 18,432 |
| RayQuery, 9-entry plus opaque projection | 151 | 2 | 144 | 18,432 |

The old synchronous policy doubled LDS despite using less private scratch than
ordinary trace. Reducing the frontier therefore removes an occupancy limit
without increasing VGPRs, spills, private state, or code-object size.

The opaque projection reduces the final code-object text by 128 B, but costs
one VGPR and two SGPR spills on this gfx1201 kernel. Its small measured speedup
therefore does not justify attributing the remaining gap to the old opaque
callback boundary. This mixed resource result is retained in the evidence
rather than hidden behind the favorable timing median.

A separate forced-resumable A/B isolates the traversal-state redesign from
the default synchronous cost choice. The old private-stack implementation was
approximately 327.45 FPS and the new persistent hardware frontier was
approximately 378.29 FPS, a 15.5% gain.

### Resumable frontier-capacity A/B

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

### Synchronous frontier-capacity A/B

The same 1024x1024/64 spp cutout benchmark was rebuilt under a fresh cache
revision for every capacity. Five warm runs determine each median.

| Synchronous capacity | Median FPS | Versus 9 entries |
|---|---:|---:|
| 9 | 499.724 | baseline |
| 12 | 498.410 | 0.26% slower |
| 16 | 485.045 | 2.94% slower |

At this checkpoint nine entries were both the formal minimum and the measured
optimum. The deep semantic test with 1,024 overlapping BLAS triangles and 256
overlapping TLAS instances passes at this capacity, so the gain did not rely on
the shallow Cornell-box benchmark. A later occupancy-policy change altered the
block size and LDS tradeoff; the fresh matched A/B below retains 16 entries.

`rocprofv3 --kernel-trace` successfully measured the ordinary-trace kernel,
but consistently stalled the cutout process during the final HIPRT geometry
build, before the RayQuery kernel was loaded or dispatched. Kernel-name
filtering and PMC-only collection reproduced the stall; the legacy profiler is
unsupported on gfx1201. No incomplete profiler trace is used as evidence here.
The A/B uses in-application synchronized render timing, while the resource
counts above come directly from dumped AMDGPU code-object metadata.

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
upload took 4.397 s, cold JIT compilation took 11.544 s, and the pre-ABI
render-only time was 1.799 s. With the compact ABI and a warm shader cache,
scene upload took 4.167 s, session creation took 0.670 s, and render-only time
was 1.780 s, a 1.08% reduction in this matched full-scene check. The Blender
source enables adaptive sampling and denoising, while this Psycles run uses
fixed sampling; official Cycles comparisons must disable both features as
well.

With the final nine-entry synchronous frontier, five warm render-only times
were 1.78276, 1.78158, 1.77928, 1.77682, and 1.78139 s, for a 1.78139 s median.
This is within 0.08% of the preceding compact-ABI run: RayQuery is only one
part of this full renderer workload, so the microbenchmark gain is not
misreported as a whole-render speedup.

With direct opaque-hit projection, five warm render-only times were 1.77464,
1.78079, 1.77676, 1.78441, and 1.78107 s. The 1.78079 s median is 0.034% below
the nine-entry checkpoint and therefore neutral at whole-scene scale.

Against the immediately preceding initialized-batch checkpoint, Combined has
RMSE 0.0017703, relative RMSE 0.0011346, MAE 0.00001570, and a mean-luminance
ratio of 1.000022. The 95th percentile is exactly zero and the 99th percentile
is 6.88e-8; the sparse maximum of 0.6978 lies on stochastic/geometry edges.
Emission, Environment, Transmission, and volume passes are unchanged where
present.

The compact-ABI render versus the immediately preceding four-hit render has
Combined RMSE 0.0007384, relative RMSE 0.0004732, MAE 4.77e-6, luminance ratio
0.9999940, and no invalid pixels. Its p95 pixel error is zero and p99 is
4.30e-9. Normal, Albedo, and every light pass were also checked: Emission,
Environment, all Transmission passes, and both Volume passes are bit-exact;
all changed Diffuse/Glossy passes have a zero p99 pixel error and at most
0.002501 relative RMSE. Visual inspection of the triptych below finds no
coherent geometry, grass, material, shadow, or lighting change. The amplified
difference consists of sparse atomic-accumulation noise.

![Lone Monk before and after the compact RayQuery ABI, with amplified difference](triptychs/ray-query-abi/combined.png)

The nine-entry render versus the preceding 16-entry render has Combined RMSE
0.001615, relative RMSE 0.001035, MAE 1.51e-5, luminance ratio 0.9999830, and
no invalid pixels. Its p95 error is zero and p99 is 6.88e-8. Emission,
Environment, every Transmission pass, and both Volume passes are bit-exact;
all remaining differences are sparse atomic-order variation. Visual inspection
again finds no structured change.

![Lone Monk 16-entry and 9-entry RayQuery frontiers, with amplified difference](triptychs/ray-query-frontier9/combined.png)

The opaque-projection render versus the nine-entry checkpoint has Combined
RMSE 0.001780, relative RMSE 0.001141, MAE 1.60e-5, luminance ratio 1.0000234,
and no invalid pixels. Its p95 error is zero and p99 is 6.88e-8. Emission,
Environment, every Transmission pass, and both Volume passes are bit-exact;
all remaining differences are sparse atomic-order variation. The original
640x480 triptych was inspected visually: geometry, grass, materials, shadows,
and illumination remain structurally unchanged.

![Lone Monk nine-entry frontier and opaque-hit projection, with amplified difference](triptychs/ray-query-opaque-fastpath/combined.png)

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
`/var/tmp/psycles-shadow-batch-vs-single-hit.json`; the compact-ABI comparison
is `/var/tmp/psycles-rq-abi-lone-monk-all-pass-comparison.json` and its rendered
EXR is `/var/tmp/psycles-rq-abi-lone-monk-warm.exr`. The frontier comparison is
`/var/tmp/psycles-rq-frontier9-lone-monk-all-pass-comparison.json`; its final
render is `/var/tmp/psycles-rq-frontier9-lone-monk-warm-5.exr`. The opaque-hit
comparison is `/var/tmp/psycles-rq-opaque-lone-monk-all-pass-comparison.json`;
its final render is `/var/tmp/psycles-rq-opaque-lone-monk-warm-e.exr`.

### Native candidate transaction and stable-opacity snapshot

The subsequent native synchronous work removes candidate costs under explicit
semantic proofs rather than scene-name or material-name tests:

1. A surface candidate emitted by the gfx12 traversal has already passed a
   valid scene instance node. The candidate-domain lookup therefore does not
   repeat the null-instance and invalid-ID checks used by defensive public
   wrapper entry points.
2. Both triangles reported by one immutable hardware triangle-packet
   intersection are consumed from that result. Before consuming the second
   triangle, traversal rechecks its interval and the current opacity. It does
   not restart the intrinsic on the same leaf.
3. Parent resolution is an overflow-only path. Marking the source helper cold
   records that frequency fact, while the HIP optimizer deliberately leaves the
   final inline decision to LLVM's cost model.
4. Stable instance opacity is projected into the HIPRT instance node only when
   the complete kernel-reachable call graph contains no
   `RAY_TRACING_SET_INSTANCE_OPACITY` operation.

The fourth rule has a simple refinement proof. Let `V` be the public eight-bit
visibility mask and `O` the instance opacity. The backend stores
`P = V | (O << 31)` in `CodegenInstance::visibility_mask`, and a scene
build/refit copies `P` to the HIPRT instance node. If whole-module analysis
proves absence of a device opacity write, `O` is invariant for the dispatch.
The traversal transports
`tag = hardware_instance_id | (node_mask & (1 << 31))`; gfx12 HIPRT instance
IDs use 24 bits, so decoding the public ID by masking those bits is injective
and cannot alter `O`. Ray visibility is masked to the public low eight bits,
so the private tag bit cannot affect culling. If the proof fails, every
candidate instead reads the authoritative `CodegenInstance::flags`, preserving
callback-visible mutation even between the two triangles of one packet.

Visibility and opacity are independently device-mutable. A host mirror may be
stale after either device write, so host updates copy only disjoint bytes: the
low byte for visibility and the high byte for packed opacity. Regressions cover
both hostile orders: device opacity followed by host visibility, and device
visibility followed by host opacity. A paired-triangle regression changes
opacity in the first callback and proves that the second triangle observes it;
the stable specialization is therefore licensed by absence, not by an assumed
material convention.

The current 1024x1024, 64 spp microbenchmark uses a 16-entry synchronous stack.
Each row is the median of five warm independent processes; FPS is the example's
reported samples-per-second rate.

| Trace mode | Before packed snapshot | Packed snapshot | Change |
|---|---:|---:|---:|
| Direct trace | 1294.509 | 1288.819 | -0.44% (noise) |
| Opaque RayQuery | 1084.672 | 1100.501 | +1.46% |
| Accept callback | 964.640 | 988.424 | +2.47% |
| Alpha cutout callback | 799.371 | 809.069 | +1.21% |

The packed cutout runs were 803.599, 802.728, 809.069, 814.227, and
812.706 FPS. The accept runs were 987.919, 987.833, 988.424, 989.279, and
989.072 FPS; the opaque runs were 1099.937, 1097.579, 1107.189, 1100.501,
and 1101.770 FPS. The direct trace varied more widely, so its -0.44% delta is
not treated as a regression. Relative to current direct throughput, the
opaque-query time gap is 17.1%, accept-query is 30.4%, and cutout-query is
59.3%.

An earlier candidate kept opacity in a separate Boolean for the entire BLAS
traversal. Although semantically valid, it expanded the live set and regressed
cutout by 1.7%, accept by 1.3%, and opaque by 1.9%; it was removed. Carrying the
bit in the already-live instance-ID register is the retained representation.
Current code-object metadata confirms the resulting tradeoff:

| Kernel | VGPR | SGPR spills | Private bytes | LDS bytes |
|---|---:|---:|---:|---:|
| Direct | 143 | 25 | 2,800 | 16,384 |
| Opaque RayQuery | 144 | 16 | 176 | 16,384 |
| Accept RayQuery | 144 | 24 | 176 | 16,384 |
| Cutout RayQuery | 144 | 22 | 192 | 16,384 |

After the occupancy-policy change, a clean Lone Monk A/B measured a 1.77695 s
median with 16 entries and 1.77935 s with nine entries. The current policy
therefore retains 16 entries; the 0.14% difference is small, but it no longer
justifies the older nine-entry choice.

The final packed-opacity Lone Monk run used the same 640x480, 64 spp staged
wavefront command shown above. Its five warm render-only times were 1.77575,
1.77993, 1.77848, 1.77859, and 1.77806 s, for a 1.77848 s median. This is only
0.09% above the immediately preceding 16-entry median and is classified as
whole-scene noise rather than a claimed speedup.

Against the retained pre-snapshot EXR, Combined has RMSE 0.00085777, relative
RMSE 0.00054975, MAE 3.60e-6, luminance ratio 1.00000139, zero invalid pixels,
and p99 pixel RMSE 4.30e-9. Emission, Environment, all Transmission passes, and
both Volume passes are exact. Normal, Diffuse, and Glossy passes have no p99
error and at most 0.001448 relative RMSE. Combined, Normal, Diffuse Color, and
the light-pass triptychs were inspected at native resolution: geometry,
silhouettes, grass, materials, normals, and shadows have no coherent change;
the amplified Combined difference is sparse atomic-order noise.

![Lone Monk 16-entry baseline and packed stable opacity, with amplified difference](triptychs/packed-opacity/combined.png)

The machine-readable all-pass report is
`/var/tmp/psycles-rq-packed-all-pass-comparison.json`; the final EXR is
`/var/tmp/psycles-rq-packed.exr`. All pass triptychs are retained under
`triptychs/packed-opacity/`.

## Remaining bottleneck

The compact ABI, occupancy-targeted 16-entry frontier, paired-hit consumption,
and stable-opacity snapshot reduce the current cutout/direct time ratio to
1.593x. Opaque RayQuery remains 1.171x direct trace, proving there is still a
backend traversal/transaction floor even without an observable callback. The
remaining structural delta for accept and cutout is the handling of genuinely
observable non-opaque/procedural candidates: flat traversal state stays live
around a generic candidate-state transaction and an out-of-line DSL
dispatcher, while the actual alpha predicate is a small part of that
transaction.

Final LLVM IR shows that all four candidate sites call one out-of-line
dispatcher, but that dispatcher still switches on a pipeline ID which is
constant at each post-optimization call site. The next experiment is a general
post-inlining constant-argument specialization: clone once per distinct
constant pipeline identity, replace the argument inside the clone, simplify,
and redirect calls. This preserves one handler body per pipeline and avoids the
rejected alternative of inlining the handler into every candidate site.

Cycles 5.2 uses a HIPRT function-table filter inside one any-hit traversal.
Luisa cannot copy that renderer-specific filter, but the backend can adopt the
same native boundary: lower each RayQuery handler to a typed device filter
thunk, carry query identity separately from its exact projected environment,
and communicate `reject`, `commit(t)`, and `terminate` directly to the active
traversal. Opaque hits remain an identity transition and procedural commits
retain their caller-supplied distance. Candidate fields should be materialized
only when demanded by the handler's fixed-point use set. Reentrancy requires
per-query identity rather than global callback state.

Merely switching back to HIPRT's generic resumable `getNextHit()` class is not
the redesign: the retained flat gfx12 traversal was introduced after that
route measured roughly 2x slower. Nor is restarting closest-hit traversal
after every rejection valid or efficient. The next prototype must preserve a
single exact frontier and integrate the generated handler at its candidate
boundary; it will be accepted only if the deep-overlap, paired-triangle, ANY,
termination, procedural, and reentrant semantic suites pass and the cutout
median materially improves.

A whole-object struct-frame experiment reduced the apparent callback product
but regressed render time by about 31%, because it destroyed LLVM lifetime
separation and stack coloring. It was discarded and is not present in the
retained history. The next design must preserve independent object lifetimes. The
candidate direction is an address-only table for proven private references,
combined with interprocedural rematerialization of immutable kernel arguments
when every call site has one identical kernel-argument provenance. Ambiguous,
external, escaping, or address-sensitive values must fail closed.

## Validation

All checks were rebuilt with 32 parallel jobs after the final source change:

```text
test_xir_pass_lower_ray_query_loop : 212 assertions / 18 tests
test_hip_callable_boundary hip     :  72 assertions / 4 tests
test_hip_llvm_pipeline            :  39 assertions / 12 tests
test_hip_ray_query_pipeline hip   : 1492 assertions / 8 tests
test_accel_visibility hip         :  162 assertions
test_hip_ray_query_pipeline fallback: 42 assertions / 8 tests
psycles_luisa_scene_traversal_tests fallback/hip/vk: pass
psycles_luisa_curve_ribbon_tests fallback/hip/vk   : pass
```

The fallback run exercises shared semantic cases and explicitly skips only the
HIP implementation-specific stack tests. No CPU reference renderer is used;
Cycles remains the renderer reference.

The Vulkan runs set `LUISA_VULKAN_DISABLE_DXC=1`; both scene kernels were
compiled by the native XIR-to-SPIR-V route (32,484 and 37,566 optimized SPIR-V
words), with no DXC load or invocation. The HIP wrapper bitcode was rebuilt for
gfx1030, gfx1100, gfx1200, and gfx1201 before running the semantic tests.
