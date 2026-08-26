# Structured closure Mix control

## Result

The compact surface runtime no longer stores and re-evaluates the complete
ancestor Mix path on every closure leaf. The host compiler now lowers the
closure tree to one structured instruction stream:

```text
mix_begin; left-region; mix_right; right-region; mix_end
```

Every contributing Mix factor is evaluated once per surface execution. Real
Blender 5.2 exports prove an exact bijection between contributing source Mix
nodes and dynamic factor evaluations:

| Scene | Closure leaves | Contributing Mix nodes | Factor evaluations | Maximum depth | `float2` frames |
|---|---:|---:|---:|---:|---:|
| Barbershop Interior | 377 | 184 | 184 | 3 | 3 |
| Classroom | 91 | 33 | 33 | 2 | 2 |
| Monster Under the Bed | 24 | 8 | 8 | 1 | 1 |

The previous flattened Barbershop representation stored 412 ancestor terms,
including 228 repeated terms. The new representation evaluates 184 factors and
uses 24 bytes of maximum dynamic Mix-frame storage.

## Formal model

Let `w` be the current closure weight at entry to a closure region. A Mix with
clamped factor `f` stores the immutable frame

```text
(parent, right) = (w, w * f)
```

and enters the left region with `w * (1 - f)`. `mix_right` selects the stored
right component; `mix_end` restores the stored parent. The inductive invariant
is therefore:

> Every well-formed closure region exits with exactly its entry weight.

Leaves preserve the invariant directly. AddClosure concatenates two regions,
so the restored exit weight of its first child is the entry weight of its
second child. MixClosure follows from restoration at its paired end. This is
why `Add(Mix(A, B), C)` gives `C` the Add parent's weight rather than leaking
the Mix right-branch weight.

Open Mix intervals are laminar. Greedy allocation of the lowest free frame
slot is consequently an exact interval coloring: the number of slots equals
the maximum number of simultaneously open Mix regions.

The serialized-image verifier independently checks:

- exact `mix_begin` / `mix_right` / `mix_end` pairing and strict nesting;
- relative offsets and containment within the enclosing branch;
- factor type, control-bit domain, and leaf operand types;
- write-before-read frame lifetime with no overlapping slot reuse;
- exact slot count and maximum structured depth;
- parallel feature metadata and aggregate operation/feature masks.

Relative offsets remain valid when programs are concatenated into the scene
image; only leaf operand bases are relocated.

## Regression coverage

`psycles_surface_closure_execution_plan_tests` contains explicit coverage for
simple Mix, one-sided endpoint projection, nested Mix, malformed/crossing
regions, exact slot coloring, scene relocation, and the original
`Add(Mix(A, B), C)` state-leak counterexample.

The compact-vs-expanded device oracle was strengthened to a depth-three tree:

```text
Add(Mix(Mix(Mix(diffuse, glass), glossy), transparent), emission)
```

This simultaneously exercises three live frames, Add continuation state,
transparent closure replay, emission, and physical closure population. It
passes on fallback, HIP, and strict native Vulkan XIR-to-SPIR-V. The complete
focused surface matrix passes 44/44.

```sh
cmake --build build --parallel "$(nproc)"

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure \
  --parallel "$(nproc)" -R surface
```

The enlarged device fixture was moved into
`compact_surface_program_test_support.cpp`; the main compact-surface test
translation unit remains below 2,000 lines.

## HIP profile

Hardware was an AMD Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.4. Both runs used
640x480, 64 fixed samples, Tabulated Sobol, staged wavefront, 32-thread
continuation blocks, and the same Blender 5.2 Barbershop export. Adaptive
sampling was not used.

```sh
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene SCENE PROFILE_DIR/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

| Measurement | Flattened baseline | Structured | Change |
|---|---:|---:|---:|
| Render-only | 2.62672 s | 2.58736 s | -1.50% |
| Mapped renderer kernels | 2235.699 ms | 2205.505 ms | -1.35% |
| Warm shader JIT | 2.63029 s | 2.63222 s | +0.07% |
| `shade_surface` | 1203.176 ms | 1252.199 ms | +4.07% |
| `shade_surface` object payload | 351,608 B | 357,536 B | +1.69% |
| `shade_surface` private / VGPR | 3,072 B / 256 | 3,152 B / 256 | +80 B / unchanged |
| `intersect_shadow` | 118.652 ms | 59.005 ms | -50.27% |
| `intersect_shadow` object payload | 236,080 B | 58,416 B | -75.26% |
| `intersect_shadow` private / VGPR | 1,024 B / 216 | 288 B / 192 | -736 B / -24 |

The result is already profitable overall and removes the large transparent
shadow duplication. It also exposes the next optimization target precisely:
the unconditional `mix_end` and `float2` parent component slightly increase
the hot `shade_surface` continuation's private state and control traffic. A
continuation-sensitive tail-region lowering can omit restoration where no Add
sibling observes the exit weight; that refinement must retain the same formal
region invariant at every observable continuation boundary.

## Output validation and nondeterminism

Classroom's one-sample baseline and structured outputs pass exact 46-channel
`idiff`. Barbershop cannot use a cross-process exact hash as a semantic oracle:
even the unchanged flattened binary differs against itself. At one sample,
two flattened megakernel runs differed in 8.94% of pixels; at 64 samples, two
flattened staged runs had mean channel error 0.004904. Sparse high-energy
indirect samples move between runs, including the same 728.138-valued outlier.

The structured 64-sample output versus a fresh flattened run had lower mean
error, 0.001779, than the flattened self-comparison. These noisy full-scene
comparisons are retained as distributional and visual checks, not presented as
exact evidence. Exact semantics come from the deterministic compact-vs-expanded
device oracle above.

Local artifacts:

- baseline profile: `/var/tmp/psycles-structured-mix-control/profile-barbershop-baseline`;
- structured profile: `/var/tmp/psycles-structured-mix-control/profile-barbershop-corrected-20260827`.
