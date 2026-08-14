# HIP resumable RayQuery frontier validation

This checkpoint redesigns LuisaCompute's gfx12 resumable RayQuery traversal
state. It does not add renderer policy to the HIP backend: source filtering,
alpha evaluation, material closures, and transparent-shadow batching remain
expressed by ordinary RayQuery handlers in Psycles. The backend change only
changes how the native HIPRT traversal frontier survives a candidate callback.

Machine: Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.53211.

LuisaCompute commits:

- `3a556dd47` -- formally localize invocation-local RayQuery handler scratch;
- `d98735f55` -- persist resumable RayQuery on the gfx12 hardware frontier.

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
4. a forward must-definition fixed point proves every handler load is preceded
   by a full store on every path from that handler's entry.

For block `B`, `D_out(B)` means that a full definition exists on every path to
the end of `B`:

```text
D_in(entry) = false
D_in(B)     = intersection(D_out(P) for P in pred(B))
D_out(B)    = D_in(B) or contains_full_store(B)
```

Non-entry blocks start at `true` and converge downward, computing the greatest
fixed point while the false entry boundary prevents cyclic self-justification.
A second instruction-order scan rejects a load before a later store in the same
block. Anything not proven safe remains captured. Regression coverage includes
load-before-store, a conditional single-arm store, stores on both diamond arms,
an outside initializer killed by a must-store, and genuinely observable
cross-candidate state.

The cutout kernel's projected callback environment fell from 48 B to 16 B and
its linked code object from approximately 46,836 B to 45,992 B. In the current
Psycles split kernels, the remaining projected environments are 96 B and
192 B; the analysis correctly rejects all of their cross-stage allocations.

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
