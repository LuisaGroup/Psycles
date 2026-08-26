# Metal scheduler cache and Classroom validation

Date: 2026-08-21

## Scope

This record validates the Metal scheduler-cache correction and the compact
surface/populate-once renderer path on an Apple M1 Max. Production image runs
use the Blender 5.2 Classroom export at 1920x1080, 64 fixed samples, Metal,
fast math, no adaptive sampling, and no denoising. Both compact experiments
are enabled explicitly:

```text
PSYCLES_COMPACT_SURFACE_VALUES=1
PSYCLES_POPULATE_SURFACE_ONCE=1
```

The cache and staged-render measurements below were produced at:

- Psycles `23786b055700b1ea5e6b4845484d8d070e3d37cc`
- LuisaCompute `a317b9e1fff987fb1fe4e8646da2a492f229f7f6`

Before the formal cross-scheduler runs, Luisa `next` advanced to
`41d676501bdfdbb7d66e541e8a9e4da7fe00d24b`. That revision contains the two
Metal fixes above and adds indexed XIR ray-query handler analysis. Psycles was
rebuilt against it; the formal scheduler rows appended below use that newer
revision.

## Configuration boundary

Scheduler configuration is classified by whether it changes the shader AST,
control flow, frame layout, or compiled-kernel set:

- `frame_buffer_compaction`, SoA versus AoS, execution block size, hint-sort
  kernel shape, and the presence or absence of an optional tail kernel remain
  host-side structural specializations;
- frame-pool capacity, graph worker count, readback/refill policy, the
  magnitude of a nonzero tail threshold, and persistent worker/fetch sizes are
  runtime arguments or host scheduling policy and do not enter shader cache
  identity.

In particular, frame-buffer compaction is deliberately **not** converted to a
runtime device branch: it changes generate-kernel control flow. The Luisa unit
regression constructs compact and sparse schedulers and requires different
shader-structure hashes.

The original real-scene cache split was not caused by an embedded capacity.
Metal code generation traversed an unordered custom-callable set, so equivalent
coroutine programs could acquire different source order and cache keys. Metal
now emits directly reachable callables in deterministic `Function::hash()`
order. A regression builds the same callable DAG in opposite insertion order
and requires the second compilation to reuse the first in-memory shader.

## Luisa validation

The following focused suites passed at the recorded Luisa revision. The two
XIR rows are host-only; the scheduler/runtime rows use Metal:

| Target | Reported result |
| --- | ---: |
| `test_xir_pass_lower_ray_query_to_pipeline` | 319 assertions / 32 tests |
| `test_xir_passes` | 2,457 assertions / 392 tests |
| `test_metal_codegen_regressions metal` | 6 cases passed |
| `test_timeline_event metal` | 11 cases passed |
| `test_coro_soa_layout metal` | 122 assertions / 12 tests |
| `test_coro_wavefront metal` | 782 assertions / 28 tests |
| `test_coro_persistent_opt metal` | 90 assertions / 23 tests |
| `test_coro_radix_sort metal` | 32 assertions / 7 tests |
| `test_coro_graph_wavefront_policy metal` | 46 assertions / 5 tests |
| `test_coro_scheduler_base metal` | 39 assertions / 10 tests |

The timeline-event suite covers the Metal backend correction that makes event
completion wait for both the GPU fence and all preceding stream host
callbacks. Graph-wavefront counter readback no longer observes a completed
event before its host copy callback has published the corresponding snapshot.

## Real-scene cache validation

The same built executable rendered a 64x64, 1-spp staged smoke followed
immediately by a 1920x1080, 1-spp run. Their internal frame capacities were
4096 and 131072 respectively.

| Run | Capacity | Shader JIT | Render-only | Process wall |
| --- | ---: | ---: | ---: | ---: |
| cold 64x64 smoke | 4,096 | 909.666 s | 0.171577 s | 915.15 s |
| immediate 1920x1080 | 131,072 | 2.69777 s | 4.38946 s | 11.48 s |

The 338x JIT-time reduction on the second run is the production evidence that
changing only queue capacity no longer creates another giant Metal shader.

## One sample versus four samples per dispatch

Both staged-wavefront images cover the exact absolute sample range `[0, 64)`.
Only `max_samples_per_dispatch` differs.

| Dispatch batch | Scene compile | Shader JIT | Render-only | Process wall | Maximum RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 spp | 1.54275 s | 2.48400 s | 242.992 s | 249.68 s | 3,086,860,288 B |
| 4 spp | 1.75290 s | 2.66934 s | 254.579 s | 261.68 s | 3,018,276,864 B |

One sample per dispatch is 4.55% lower in render-only time on this workload and
is selected for the remaining scheduler comparisons. At Metal's 131072
pixel-sample watchdog cap, host row partitioning makes both settings issue a
similar number of bounded work chunks, so this is a modest scheduling effect,
not a fourfold launch-count result.

The two Psycles images are numerically equivalent apart from floating-point
accumulation order: Combined RMSE `1.42155e-5`, p99 pixel RMSE `9.73e-8`, and
maximum absolute error `0.030159`.

## Formal scheduler and Cycles Metal comparison

The formal scheduler rows use Psycles `f68c88f` with Luisa `41d676501`, the two
compact environment flags above, one sample per dispatch, a warm shader cache,
and no scheduler-stat logging. Cycles is a fresh Blender 5.2.0 LTS process on
the same Apple M1 Max with the same image dimensions, 64 fixed samples,
tabulated Sobol, seed 1, adaptive sampling disabled, and denoising disabled.

| Renderer / scheduler | Scene compile | Shader JIT | Render-only | Process wall | Maximum RSS | Psycles / Cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cycles Metal | - | - | 15.5552 s | 17.53 s | 3,182,723,072 B | 1.000x |
| Psycles tuned graph, selective + hint sort + tail | 1.43832 s | 2.31652 s | 199.188 s | 207.44 s | 3,037,200,384 B | 12.805x |
| Psycles ordinary wavefront | 1.44286 s | 3.30121 s | 239.207 s | 246.45 s | 3,010,101,248 B | 15.378x |
| Psycles staged wavefront | 1.54275 s | 2.48400 s | 242.992 s | 249.68 s | 3,086,860,288 B | 15.621x |
| Psycles default graph | 1.44229 s | 2.33971 s | 247.616 s | 253.90 s | 3,047,505,920 B | 15.919x |
| Psycles tuned graph, tail disabled | 1.45357 s | 2.21276 s | 248.393 s | 254.58 s | 3,016,359,936 B | 15.969x |

Tuned graph is the fastest tested Psycles Metal scheduler on this workload. It
is 18.03% lower than staged, 16.73% lower than ordinary wavefront, and 19.56%
lower than default graph. The nonzero tail state-machine drain reduces the
tuned graph render by 19.81% relative to the otherwise matched `tail=0` run.

The tradeoff is cold compilation. The first graph run took about 1,693 seconds
to construct the graph scheduler and compile its large Metal kernels; the
optional tail kernel accounted for the later long compiler phase. Once cached,
all graph formal runs report about 2.2-2.3 seconds of total shader JIT. This is
why zero versus nonzero tail remains structural, while the magnitude of a
nonzero threshold remains host policy.

Persistent was also exercised rather than silently omitted. Even the 64x64,
1-spp smoke failed while creating the giant persistent Metal pipeline:
`MTLCompilerService` ran for about 97 seconds, repeatedly interrupted its XPC
connection, and the process exited with status 134 before rendering. This is
recorded as an Apple Metal compiler scalability failure, not a performance
number and not a Psycles-side fallback.

## Image inspection

Combined comparison metrics are:

- no invalid pixels;
- RMSE `0.00909157`, relative RMSE `0.0210225`;
- mean luminance ratio `0.9993810`;
- p99 pixel RMSE `0.0459147`;
- maximum absolute error `0.884907`.

The tuned graph and staged Psycles Combined images have identical channel and
luminance means. Their RMSE is `3.25234e-6`, with zero p99 pixel RMSE; the
sparse maximum absolute difference is `0.00598046` from atomic accumulation
order. Thus the scheduler speed ranking does not exchange image semantics for
throughput.

Native-resolution inspection of the tuned graph image confirms that the Classroom
window panes have no coherent purple lines, the clock face/digits/hands are
visible, and the transom above the door is translucent rather than black. The
amplified difference panel contains sparse magenta/green residuals around
high-contrast window edges; those residuals are not present as purple lines in
the unamplified Psycles render.

Generated EXRs, reports, crops, and the 12 MiB triptych remain in the ignored
build benchmark directory rather than being committed as source assets:

```text
build-macos-shader-translation-plan/benchmarks/2026-08-21/
  classroom-capacity-cache-validation/
  classroom-1080p64-compact-metal/
```

## Reproduction

From `build-macos-shader-translation-plan`, the staged production command is:

```bash
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
/usr/bin/time -lp ./bin/psycles_render_blender_scene \
  ../build-macos-wavefront/scene-exports/classroom-5.2 \
  benchmarks/2026-08-21/classroom-1080p64-compact-metal/staged-dispatch1.exr \
  metal 1920 1080 64 1 - 960 540 0 0 64 - 1 0 \
  wavefront-staged 32
```

The direct 1-versus-4 image comparison uses the project comparator:

```bash
python3 ../tools/compare_cycles.py \
  benchmarks/2026-08-21/classroom-1080p64-compact-metal/staged-dispatch4.exr \
  benchmarks/2026-08-21/classroom-1080p64-compact-metal/staged-dispatch1-vs-dispatch4.json \
  --allow-unverified-build-identity \
  --triptych-dir benchmarks/2026-08-21/classroom-1080p64-compact-metal/triptychs/dispatch-ab \
  Combined=benchmarks/2026-08-21/classroom-1080p64-compact-metal/staged-dispatch1.exr
```

These are single-process measurements on an interactive workstation. Formal
scheduler ranking uses cache-warm runs without scheduler-stat logging and does
not overlap Metal workloads.
