# Surface SVM superinstruction overlay: rejected

## Outcome

A formally validated, scene-specialized adjacent-handler overlay was built and
removed after HIP measurement. On Blender 5.2 Barbershop, the eight most common
static exact pair classes cover only 1,610 of 8,556 legal immutable adjacent
edges (18.82%). The device implementation increases the surface HIP object by
13.27%, fixed private storage by 216 B, and normalized `shade_surface` time by
0.214%. Three interleaved hot renders are 0.36% slower at the median.

This rejects the representation, not merely K=8. The overlay adds a second
dynamic pair switch and duplicates handler call sites while removing at most
one loop backedge, instruction fetch, and primary dispatch per matched pair.
Its realized coverage is too small to amortize the increased code and register
pressure. Increasing K would monotonically add switch cases and handler call
sites, while the raw top-K transition census is only an overlap-ignoring upper
bound. No production code or environment switch is retained.

## Formal experiment

An exact candidate class was

```text
p = (source_variant, target_variant, direct_dependency),

direct_dependency = exists target operand q:
                    address(q) = result_address(source).
```

Surface-normal commits were hard segment boundaries. The planner selected the
K classes with highest static occurrence count, with lexicographic ties. For a
fixed dictionary it marked edges left-to-right within each straight-line
segment. This greedy rule is maximum-cardinality on a path: for the first
available edge, an optimum either uses it, uses the only overlapping next edge
which can be exchanged for it, or can add it; induction applies to the suffix.

The bytecode verifier required a parallel one-byte id stream, an in-segment
ordinary successor for every marked head, exact agreement with the dictionary,
and no instruction belonging to two pairs. A permanent-style mutation test
confirmed that overlapping marks were rejected. At runtime the id occupied the
canonical control word's reserved bits 8..15, so instructions stayed 16 B and
no device buffer was added. The canonical compiler image continued to require
those bits to be zero.

For every marked pair `(a,b)`, the device executed the original exact handler
for `a`, then the original exact handler for `b`, with the original instruction
records and typed-local state. Partitioning a segment into singleton and paired
blocks therefore proves semantic equivalence by induction. The fallback and
HIP full per-sample film regressions passed with K=8, including Combined,
Normal, Albedo, all light passes, volume passes, split dispatch, and every
scheduler mode exercised by that test.

## Generated structure

Both cold runs use the same current worktree, RX 9070 XT, Barbershop export,
320x240 one-sample AST, disabled shader cache for the newly generated stage,
and the same staged-wavefront configuration. The one-sample work changes only
execution count, not shader structure.

| Metric | Canonical K=0 | Pair overlay K=8 | Change |
| --- | ---: | ---: | ---: |
| Static candidate edges | 8,556 | 8,556 | unchanged |
| Selected pair heads | 0 | 1,610 | 18.82% coverage |
| Selected exact classes | 0 | 8 | +8 |
| Surface HIP object | 339,800 B | 384,888 B | +45,088 B (+13.27%) |
| Main HIPRTC link | 2,967.65 ms | 3,362.77 ms | +13.31% |
| Complete shader JIT | 9.73222 s | 10.4425 s | +7.30% |
| Fixed private storage | 3,096 B | 3,312 B | +216 B (+6.98%) |
| VGPR / SGPR | 256 / 128 | 256 / 128 | unchanged |

The object increase is not an artifact of an extra side stream: there is none.
It is native code created by the pair branch/switch and its additional handler
call sites.

## HIP performance

The hot sequence alternated K=0/K=8 three times. Every run rendered 640x480 at
64 fixed samples with Tabulated Sobol and staged wavefront.

| Run | K=0 | K=8 |
| --- | ---: | ---: |
| 1 | 2.50893 s | 2.50043 s |
| 2 | 2.48991 s | 2.49884 s |
| 3 | 2.48882 s | 2.49829 s |
| Median | 2.48991 s | 2.49884 s |
| Candidate change | | **+0.359%** |

A back-to-back `rocprofv3 --kernel-trace` pair maps the surface continuations
through the renderer's shader-map hashes:

| Variant | Calls | Work items | GPU time | ns/item |
| --- | ---: | ---: | ---: | ---: |
| K=0 `kernel_4201432322c51ca1` | 355 | 53,568,704 | 1,157.628 ms | 21.610160 |
| K=8 `kernel_1f2425e36cf1c64d` | 355 | 53,596,256 | 1,160.704 ms | 21.656438 |
| Candidate change | | | | **+0.214%** |

The work difference is 0.051%, so normalization removes it. Total mapped
renderer-kernel time happens to move in the opposite direction in this single
trace, but the modified surface stage and the three-run render median both
regress. The decision therefore does not attribute unrelated scheduler noise
to the SVM change.

## Historical measurement command

The candidate source and environment switch are intentionally absent from
`main`; this records the exact command shape used for the rejected worktree,
not a command that current `main` accepts.

```sh
cmake --build build --parallel 32 --target \
  psycles_surface_program_metadata_tests \
  psycles_luisa_sample_dispatch_film_tests \
  psycles_render_blender_scene

PSYCLES_SURFACE_SUPERINSTRUCTION_COUNT=8 \
  build/bin/psycles_luisa_sample_dispatch_film_tests fallback
PSYCLES_SURFACE_SUPERINSTRUCTION_COUNT=8 \
  build/bin/psycles_luisa_sample_dispatch_film_tests hip

rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  env PSYCLES_SURFACE_SUPERINSTRUCTION_COUNT=K \
      LUISA_CORO_SHADER_MAP=1 \
  build/bin/psycles_render_blender_scene BARBERSHOP_EXPORT out.exr hip \
    640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
    32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The retained transition census remains useful: it proves that adjacent direct
data flow is common and that a Bump-only patch is insufficient. This rejected
experiment adds the missing backend evidence that dispatch fusion alone is not
the right way to exploit it. A future producer-to-consumer optimization must
remove semantic loads/stores or narrow live state, not merely wrap two unchanged
handlers in another dispatch layer.
