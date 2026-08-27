# Value-callable closure evaluator: rejected

## Result

Passing each populated physical closure to one Luisa callable by value avoids
the private-storage escape of the earlier reference experiment, but it does
not create a shared native evaluator. The experiment was removed after a cold
LLVM audit and two complete Barbershop HIP profiles. It preserves the 864 B
coroutine frame, yet enlarges the surface code object by 2.02%, adds 8 B of
fixed private storage, and changes normalized `shade_surface` time by +0.197%
(noise/slightly slower). There is no performance benefit to retain.

Both forms use the accepted `eab23be` renderer baseline, LuisaCompute
`ca29d37d`, the official Blender 5.2 Barbershop export, RX 9070 XT (`gfx1201`),
640x480, 64 fixed spp, Tabulated Sobol, compact surface values, one surface
population, and the staged wavefront scheduler. Adaptive sampling and
denoising are disabled.

| Metric | Exact baseline | Value-callable candidate | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| `shade_surface` fixed private segment | 3,096 B | 3,104 B | +8 B |
| `shade_surface` HIP object | 340,184 B | 347,056 B | +6,872 B (+2.020%) |
| `shade_surface` median | 26.7645 ns/item | 26.8172 ns/item | +0.197% |
| Render-only mean | 2.52278 s | 2.52450 s | +0.068% |

## Formal representation result

Let the physical closure arena be the private sequence `A`, and let `E` be the
per-closure evaluator. The candidate changed the consumer shape from two
source expansions of `E(load(A, i))` to calls of one value function

```text
E_v : Block0 x Block1 x Query -> Contribution.
```

Because `load(A, i)` occurs before the call and only its two matrix values are
passed, no storage identity or pointer into `A` crosses the boundary. The
unchanged 864 B frame is the measured witness for that property. This fixes
the lifetime defect of the reference-argument proposal.

It does not, however, imply native code sharing. In an uncached build the final
surface module is `hip_kernel_final_5.ll`, 54,865 lines and 3,040,081 B. It
contains the root kernel and four retained Luisa callables, none with the two
physical-matrix evaluator signature. `E_v` was fully inlined by LLVM O3 at its
NEE and BSDF-mixture call sites. Therefore the generated program is

```text
inline(E_v at use 0) + inline(E_v at use 1) + value-call ABI/setup residue,
```

not one shared evaluator. The 6,872 B object increase and 8 B private increase
are consistent with this structural observation, and both candidate timing
samples are above the three-sample baseline median. No `noinline` annotation
was added: forcing a function boundary would substitute a backend heuristic
for the missing representation and contradict the policy that LLVM decides
ordinary callable inlining.

This experiment also leaves Psycles' expensive selected-lobe replay intact:
the conditional sampler still chooses a direction, after which the generic
mixture evaluator reconstructs that selected closure. The next viable design
must instead keep the sampled conditional contribution family-local and seed
the existing mixture fold without widening a universal result across the
family merge, or execute the closure consumer through a compact interpreter.

## HIP samples

The baseline values are the three retained identity-quotient traces. Candidate
work counts differ by at most one 32-lane block, so duration is normalized by
the actual launched work.

| Build / run | Calls | Work-items | `shade_surface` ns/item |
|---|---:|---:|---:|
| Baseline 1 | 293 | 53,660,800 | 26.7558394 |
| Baseline 2 | 293 | 53,660,864 | 26.7644853 |
| Baseline 3 | 293 | 53,660,864 | 26.8457060 |
| Value callable 1 | 293 | 53,660,896 | 26.8189860 |
| Value callable 2 | 293 | 53,660,864 | 26.8153510 |

## Numerical and visual check

All 15 named film passes contain zero invalid pixels. Against the exact
baseline, Combined has RMSE `8.58008e-4` and mean-luminance ratio
`0.99999902`; Normal has RMSE `6.85829e-5`. Native-resolution inspection found
the same geometry, UV/texture placement, floor, ceiling, cabinets, lighting,
materials, and normal orientation. The amplified residuals are sparse
sample/atomic-arrival differences rather than a coherent image change.

- [All-pass report](all-pass-report.json)
- [Combined triptych](triptychs/combined.png)
- [Normal triptych](triptychs/normal.png)

## Reproduction

```sh
cmake --build build --parallel 32 --target \
  psycles_render_blender_scene \
  psycles_luisa_surface_closure_collection_tests \
  psycles_luisa_surface_population_tests

ctest --test-dir build --output-on-failure -j2 \
  -R '^psycles\.luisa_surface_(closure_collection|population)_(fallback|hip)$'

PSYCLES_DISABLE_SHADER_CACHE=1 \
LUISA_DUMP_LLVM_IR=1 \
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
LUISA_CORO_SHADER_MAP=1 \
rocprofv3 --kernel-trace --scratch-memory-trace --stats \
  -f rocpd -d PROFILE_DIR -o trace_results -- \
  build/bin/psycles_render_blender_scene \
    BARBERSHOP_5_2_EXPORT PROFILE_DIR/out.exr hip \
    640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
    32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

The candidate source is intentionally absent from production; this directory
retains the negative result and its machine-readable evidence.
