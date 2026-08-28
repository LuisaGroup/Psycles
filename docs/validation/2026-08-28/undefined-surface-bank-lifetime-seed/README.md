# Undefined surface-bank lifetime seed

## Outcome

Accepted. The compact surface-value interpreter no longer executes 44 zero
stores at the start of every surface evaluation. Its three typed local banks
instead receive one whole-bank `undefined<T>()` seed each. The seed is a
fresh-lifetime witness for XIR's alloca-scope analysis, not an observable
shader value.

On the official Blender 5.2 Barbershop export, the final HIP surface module
loses exactly 44 stores. The code object is 384 B smaller, its disassembly has
14 fewer scratch stores and 52 fewer global loads, and matched HIP profiling
improves normalized `shade_surface` time by 2.406%. Mean render-only time
improves by 1.144%. The coroutine frame remains 177 fields / 864 B and fixed
private storage remains 3,096 B; this change removes work inside the surface
stage rather than claiming a frame-size improvement.

This is a typed lifetime-semantics change. It does not special-case a material,
opcode, scene, or backend, and it adds no inline/noinline attributes.

## Reference identity

- Psycles baseline: `5c62d055942a093c5c92cbec1f74c8a9dd7e029a`.
- LuisaCompute baseline: `ca29d37d3458bd44c4d7bbc55f69c7bf047f83ae`.
- LuisaCompute undefined semantics: `bd05221638494269261418f0d14478aef4fc7680`.
- LuisaCompute integration (including the direct-C regression):
  `004096a9bb8b246ba0872df0ecc6d9ac2a01325a` on `origin/next`.
- Cycles source reference: Blender 5.2 release at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Device: AMD Radeon RX 9070 XT (`gfx1201`), HIP backend.
- Scene: official Blender 5.2 Barbershop export, 640x480, 64 fixed samples,
  Tabulated Sobol, adaptive sampling disabled, staged wavefront scheduler.

The A/B in this document compares the candidate with its exact Psycles parent.
It is a compiler/performance validation, not a new Psycles-versus-Cycles image
fidelity claim.

## Formal model

For one published surface bytecode program, let the ordered instruction stream
be

```text
P = (i_0, i_1, ..., i_n).
```

The host verifier computes definite initialization as a forward must-property.
Let `D_t` be the set of typed local addresses definitely initialized before
`i_t`. Parameters are immutable inputs and do not belong to this local set.
For every ordinary instruction the verifier checks

```text
local_operands(i_t) subseteq D_t
```

before applying the transfer function

```text
D_(t+1) = D_t union { result(i_t) }.
```

The surface-normal transition is a read-only transition and is accepted only
when its local vector operand is already in `D_t`. Operands are checked before
the result is inserted, so reuse of a dying physical slot cannot manufacture a
self-definition. Every composed program and every program in the immutable
scene image passes this verifier before publication. Permanent negative tests
reject both an ordinary local read-before-write and an uninitialized normal
transition.

Each invocation gives the three physical banks a fresh logical lifetime. Let
`Z_B` be the previous all-zero root definition for bank `B`, and `U_B` an
arbitrary well-typed value. The verifier establishes that every device-visible
local read is dominated in program order by a producer belonging to the same
invocation. Therefore neither `Z_B` nor `U_B` reaches an observable read:

```text
for every legal execution E and observable output O,
eval(E, B := Z_B, O) = eval(E, B := U_B, O).
```

Replacing `Z_B` by `U_B` is consequently valid for every value chosen by
`undefined`, including a backend's legal refinement to zero. The aggregate
store remains a complete-definition witness, allowing `coro_alloca_scope` to
move the bank allocation and its first definition together. Native backends
may then erase that store because the value has no semantic constraint.

This proof relies on the published bytecode verifier, not on a scene census or
on the probability of taking a branch. If a future opcode introduces a local
read that is not dominated by a producer, scene construction fails before any
kernel is built.

## Luisa semantic chain

LuisaCompute now preserves this concept end to end:

- `CallOp::UNDEFINED` is appended to the AST enum, preserving all existing
  numeric values, and requires zero arguments plus a non-void result.
- `luisa::compute::undefined<T>()` records the typed AST expression.
- AST-to-XIR creates an XIR `Undefined`; XIR-to-AST recreates the same AST
  operation instead of refining it to zero.
- HIP, fallback, CUDA-XIR, and SPIR-V-LLVM lower it to the corresponding
  undefined SSA value.
- Native XIR-to-SPIR-V emits `OpUndef` for the value type. A Vulkan 1.2
  validator-backed aggregate regression requires `OpUndef` whose result type
  is `OpTypeArray`.
- Direct AST C, CUDA, Metal, and HLSL routes choose zero, which is a valid
  refinement of an unobserved arbitrary value and preserves their existing
  working code paths.
- `coro_alloca_scope` recognizes XIR `Undefined` as a globally available SSA
  operand when moving a first definition; it does not introduce a new IR
  declaration entity.

Permanent Luisa regressions cover AST-to-XIR aggregate preservation,
XIR-to-AST-to-XIR aggregate preservation, native SPIR-V aggregate `OpUndef`,
movement of an undefined full definition with its fresh-lifetime alloca, and
valid direct-C aggregate syntax under the system STL configuration.
Psycles additionally records a callable and requires exactly three typed,
argument-free `UNDEFINED` roots: scalar bank, vector bank, and unsigned bank.

## Cold generated structure

Both rows use the same one-sample Barbershop scene with shader caching disabled.
`hip_kernel_final_5.ll` is the final surface-stage LLVM module. Code-object
metadata comes from `llvm-readelf --notes`; instruction categories come from
`llvm-objdump` over the complete surface code object.

| Metric | Zero seed | Undefined seed | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| Final LLVM | 53,989 lines / 3,008,050 B | 53,903 / 3,001,514 B | -86 lines / -6,536 B |
| LLVM stores | 1,319 | 1,275 | **-44** |
| LLVM allocas / loads | 17 / 2,527 | 17 / 2,527 | unchanged |
| LLVM phi / branches / selects | 2,281 / 1,589 / 2,666 | same | unchanged |
| LLVM calls / switches | 3,351 / 38 | same | unchanged |
| HIP surface object | 340,184 B | 339,800 B | -384 B (-0.113%) |
| Disassembled instructions | 60,719 | 60,706 | -13 |
| Scratch loads / stores | 678 / 543 | 678 / 529 | stores -14 |
| Global loads / stores | 1,239 / 277 | 1,187 / 277 | loads -52 |
| Fixed private storage | 3,096 B | 3,096 B | unchanged |
| VGPR / SGPR metadata | 256 / 107 | 256 / 107 | unchanged |

The exact `-44` LLVM store delta equals 8 scalar leaves plus 12 three-component
vector leaves. LLVM already erased the unsigned bank's scalar zero in the
baseline, so it contributes no additional final-IR delta. Unrelated LLVM
transformations can change how the final object accounts memory operations;
the complete opcode counts above are therefore reported instead of attributing
every ISA delta to one source statement.

Cold JIT time is deliberately not claimed: cache state and process startup
were not matched tightly enough for a compiler-time conclusion.

## HIP measurement

Each `rocprofv3` trace renders the same 640x480 image at 64 fixed samples. The
surface metric is normalized by the actual sum of `grid_x * grid_y * grid_z`.
This accounts for small scheduler-dependent differences in call count and
work-item count rather than treating wall-clock kernel totals as equal work.

| Build / run | Calls | Work-items | GPU duration (ns) | ns/work-item |
|---|---:|---:|---:|---:|
| Zero seed 1 | 365 | 53,572,256 | 1,180,051,937 | 22.027295938 |
| Zero seed 2 | 355 | 53,557,568 | 1,179,782,042 | 22.028297513 |
| Undefined seed 1 | 356 | 53,605,312 | 1,153,878,070 | 21.525442665 |
| Undefined seed 2 | 358 | 53,604,160 | 1,151,813,401 | 21.487388311 |
| Undefined seed 3 | 358 | 53,590,368 | 1,151,165,919 | 21.480836239 |
| Zero-seed mean | | | | 22.027796726 |
| Undefined-seed mean | | | | 21.497889072 |
| Candidate change | | | | **-2.405632%** |

Render-only wall times are 2.53212 s and 2.53125 s for the zero seed, versus
2.50699 s, 2.50154 s, and 2.49962 s for the undefined seed. The means are
2.531685 s and 2.502717 s, a **1.144231%** improvement. An unchanged
direct-light kernel remains near 8 ns/work-item across both builds, providing a
same-run control against a whole-device speed shift.

Megakernel render-only time is 4.28316 s for the zero seed and 4.26834 s for
the undefined seed. This single pair is used only to obtain a scheduler-matched
visual diagnostic, not as the primary performance claim.

## Numerical and visual inspection

All 15 Combined, data, light, and volume passes in the megakernel A/B contain
zero invalid pixels. The complete reports are
[`candidate-vs-zero.json`](candidate-vs-zero.json) and
[`zero-repeat.json`](zero-repeat.json).

The current parallel sample path is not bitwise deterministic across repeated
processes. It was measured explicitly before interpreting the candidate A/B:

| Pass | Candidate vs zero RMSE | Zero run 2 vs zero run 1 RMSE | Candidate/zero mean result |
|---|---:|---:|---:|
| Combined | 0.01486346 | 0.01882981 | luminance ratio 1.012389 |
| Normal | 0.001113101 | 0.002133163 | luminance ratio 0.9999990 |
| DiffCol | 0.002789730 | 0.003037665 | luminance ratio 0.9999336 |

Emission agrees to `4.532e-9` RMSE and Environment is exact. In each primary
pass the candidate-versus-zero error is below the same zero-seed binary's own
repeat error. These statistics are a stochastic smoke test only; the
read-before-write proof and backend regressions are the semantic gate.

The following native-resolution triptychs place the zero seed on the left,
undefined seed in the center, and an explicitly amplified absolute difference
on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Visual inspection found matching geometry, UV placement, material regions,
surface orientation, and lighting structure. The amplified difference panels
are concentrated on stochastic highlights, fine geometry, and subpixel edges;
there is no new coherent outline, texture transform, material partition, or
normal-direction error.

## Regression results

The project was built with all 32 host threads. Eighteen host/fallback/HIP
surface tests and seven strict native Vulkan surface tests pass. The Vulkan
tests set all of

```text
LUISA_VULKAN_USE_XIR=1
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1
LUISA_VULKAN_DISABLE_DXC=1
```

so success cannot come from a DXC runtime path.

The complete Psycles suite reports 309/315 passing. Its six failures are the
same pre-existing floating-threshold fixtures as the parent:

```text
psycles.luisa_stacked_volume_fallback
psycles.luisa_homogeneous_volume_fallback
psycles.luisa_area_light_forward_vk
psycles.luisa_volume_path_fallback
psycles.luisa_volume_path_vk
psycles.luisa_volume_triangle_fallback
```

Luisa's focused translator, alloca-scope, native-SPIR-V, and direct-C
regressions pass. Its XIR/SPIR-V/direct-C suite passes 79/79 after excluding
the existing `test_spirv_target_feature_codegen` failure.

## Reproduction

```sh
cmake --build build --parallel 32

ctest --test-dir build --output-on-failure -j2 \
  -R 'psycles\.(surface_(program_metadata|svm_(math_immediate|vector_math_immediate|record_immediates))|luisa_(microfacet_anisotropy|surface_closure_(collection|reachability|physical)|surface_population|compact_surface_(preparation|tail))_(fallback|hip))$'

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.luisa_(microfacet_anisotropy|surface_closure_(collection|reachability|physical)|surface_population|compact_surface_(preparation|tail))_vk$'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene \
  BARBERSHOP_5_2_EXPORT PROFILE_DIR/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1 1048576
```

The visual reports use `tools/compare_cycles.py` with
`--allow-unverified-build-identity` because both inputs are locally identified
Psycles A/B artifacts rather than Blender/Cycles outputs. The override is not
used for a Cycles compatibility claim.
