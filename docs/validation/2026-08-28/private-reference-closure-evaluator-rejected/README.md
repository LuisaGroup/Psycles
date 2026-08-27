# Private-reference closure evaluator: rejected

## Result

Passing the populated physical-closure records to a shared Luisa callable by
reference is semantically correct, but catastrophically unprofitable in the
coroutine path. The experiment was removed from production after the complete
Barbershop HIP profile. It is documented here as a negative structural result
so the same representation is not rediscovered and judged only by source-level
deduplication.

The accepted `eab23be` baseline and the candidate used the same Blender 5.2
export, RX 9070 XT (`gfx1201`), 640x480, 64 fixed spp, Tabulated Sobol,
compact surface values, one surface population, and the staged wavefront
scheduler. Both launches made 293 `shade_surface` calls over essentially the
same logical work (53,660,864 versus 53,660,896 launched items). The normalized
profile is therefore the relevant comparison.

| Metric | `eab23be` baseline | reference candidate | Change |
|---|---:|---:|---:|
| Coroutine frame | 864 B | 2,272 B | +1,408 B |
| `shade_surface` fixed private segment | 3,096 B | 7,072 B | +128.42% |
| `shade_surface` HIP object | 340,184 B | 413,616 B | +21.59% |
| `shade_surface` | 26.8457 ns/item | 72.8820 ns/item | +171.48% |
| Render only | 2.52529 s | 5.06264 s | +100.48% |

## Formal root cause

The physical closure arena has capacity 11 for this export and consists of two
`Local<float4x4>` arrays. A value load does not expose either array's storage
identity. The candidate instead passed an element of each array through a
`float4x4 &` callable argument. This makes both arrays address-taken at the XIR
call boundary. The coroutine analysis cannot prove that the resulting private
pointer is confined to the current resume function, so it must retain the
complete referenced objects in the frame:

```text
2 arrays * 11 records * sizeof(float4x4)
  = 2 * 11 * 64 B
  = 1,408 B.
```

The measured frame increase is exactly 1,408 B. This equality identifies the
representation error independently of any material case or backend timing.
The larger frame then raises scratch traffic and occupancy pressure; it also
prevents the hoped-for compact shared evaluator after LLVM optimization. No
inlining attribute was added or changed.

Consequently, a profitable shared consumer must preserve value semantics at
the coroutine boundary or execute over an SVM-owned value state. It must not
take the address of the per-path closure arena. Backend heuristics cannot
repair this loss of lifetime information after the address has escaped.

## Numerical and visual check

The candidate completed the full render with zero invalid pixels in all 15
named passes. Against the baseline, Combined RMSE is `8.58008e-4`, mean
luminance ratio is `0.99999902`, and Normal RMSE is `6.85829e-5`. These are
consistent with the existing HIP atomic/queue-order variation; the experiment
does not reveal a semantic or structural image change.

The triptychs were inspected at native resolution. The baseline and candidate
panels have the same camera, geometry, textures, material placement, lighting,
and normal orientation. The amplified differences are sparse stochastic
highlights and edge/sample residuals, not a coherent surface change.

- [Combined triptych](triptychs/combined.png)
- [Normal triptych](triptychs/normal.png)

## Reproduction

```sh
cmake --build build --parallel 32 --target \
  psycles_render_blender_scene \
  psycles_luisa_surface_closure_collection_tests \
  psycles_luisa_surface_population_tests

ctest --test-dir build --output-on-failure -j1 \
  -R '^psycles\.luisa_surface_(closure_collection|population)_(fallback|hip)$'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
LUISA_CORO_SHADER_MAP=1 \
rocprofv3 --kernel-trace --stats -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene BARBERSHOP_5_2_EXPORT out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

The full machine-readable comparison and profiler summary are in
[`metrics.json`](metrics.json). The candidate source was intentionally
reverted; only this negative-result record is retained.
