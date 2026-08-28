# Materialized surface-handler key rejected

## Outcome

Two related surface-SVM dispatch changes were implemented, formally checked,
and removed after an interleaved Barbershop HIP profile:

1. materialize the Image-BOX execution-family quotient in a reserved bytecode
   bit, replacing the runtime key reconstruction with one mask;
2. additionally fold the internal surface-normal transition into the ordinary
   handler switch, replacing the existing predicate followed by a switch with
   one larger switch.

Neither is profitable.  Key materialization alone is performance-neutral
(`+0.032%` mean) and the fused switch is consistently slower (`+0.183%` mean)
while increasing the main HIP object by 16,472 B.  The production bytecode and
interpreter remain unchanged.

## Formal model

For a regular instruction `i`, the primary handler quotient is

```text
Q(i) = (operation(i), result_bank(i), image_box_family(i)).
```

`image_box_family` is one exactly when the operation is Image Color or Image
Alpha and its projection immediate is BOX.  All other image projections share
the one-sample family.  The candidate placed that Boolean in control bit 8,
which is disjoint from the opcode, result-bank, and 14-bit immediate fields.
The lowerer constructed it from the immutable semantic fields and the image
verifier independently recomputed `Q`; a missing BOX bit or a foreign BOX bit
was rejected by permanent candidate regressions.

The fused form extended the dispatch sum with the unique internal tag
`T = 0xff`.  Every authored operation is strictly below `T`, so

```text
{T} disjoint-union image(Q)
```

is a valid one-switch representation.  The transition case performed the
same state update `(use_undisplaced, shading_normal) := (false, result)` and
ordinary cases were unchanged.  The transformation was therefore semantic,
not an ad hoc opcode shortcut.

The negative result is a backend-layout result: the larger switch changed the
optimized HIP CFG/code layout enough to outweigh the removed predicate.  It
does not invalidate the quotient proof, but the proof alone does not establish
profitability.

## HIP A/B/B/A evidence

All runs used the official Blender 5.2 Barbershop export, fixed 640x480/64 spp,
compact populate-once surface execution, fast math, staged wavefront, warm
shader caches, and `rocprofv3 --kernel-trace --scratch-memory-trace --stats`.
Times are normalized by each trace's exact launched surface work.

| Form | Run 1 ns/item | Run 2 ns/item | Mean | Change |
|---|---:|---:|---:|---:|
| Existing derived key | 26.790123 | 26.794782 | 26.792453 | baseline |
| Materialized key + fused transition | 26.831577 | 26.851316 | 26.841447 | +0.183% |
| Materialized key only | 26.786991 | 26.814850 | 26.800921 | +0.032% |

| Form | surface hash | HIP object | private/thread | VGPR / SGPR |
|---|---|---:|---:|---:|
| Existing derived key | `fd1f1c5285842623` | 340,184 B | 3,096 B | 256 / 128 |
| Materialized key + fused transition | `9a36bf98e39f96a9` | 356,656 B | 3,096 B | 256 / 128 |
| Materialized key only | `4b641acacdbbc54e` | 340,056 B | 3,096 B | 256 / 128 |

All six surface traces issued 293 calls and approximately 53.66 million work
items.  Coroutine layout stayed 177 fields / 864 B.  The key-only object is
128 B smaller but has no timing or resource-pressure improvement, so consuming
a bytecode ABI bit would add complexity without a measured renderer gain.

## Reproduction

```text
cmake --build build --parallel 32 --target \
  psycles_surface_svm_record_immediate_tests \
  psycles_luisa_compact_surface_preparation_tests \
  psycles_render_blender_scene

ctest --test-dir build -R \
  'psycles.surface_svm_record_immediates|psycles.luisa_compact_surface_preparation_fallback' \
  --output-on-failure

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
rocprofv3 --kernel-trace --scratch-memory-trace --stats \
  -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene BARBERSHOP_EXPORT out.ppm hip \
    640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
    32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```
