# Surface kernel shape refactor

Date: 2026-08-09

## Failure model

Several closure regressions constructed one large `SurfaceDispatch`, recorded
many calls to the same surface operation for different test cases, and then
compiled all of them as one kernel. This is pathological for a multistage DSL.

Let `D` be the number of material topologies in a dispatcher and let `C_o` be
the number of host-side recordings of surface operation `o`. Before backend
optimization, the recorded code contains approximately

```text
sum(o) C_o * dispatch(D, topology(o)).
```

Even a tag that later becomes constant does not make the duplicated AST free:
the dispatcher and every operation call have already emitted their candidate
graphs. Backend common-subexpression elimination is too late to prevent LLVM
or SPIR-V optimizer memory growth.

## Structural correction

The tests now follow the same multistage rules expected from production shader
construction:

- unrelated material fixtures own separate dispatch domains;
- semantic cases are device inputs selected by `dispatch_x`, so one operation
  graph handles all cases;
- closure trace, evaluation, sampling, light evaluation, AOV, and extinction
  have explicit compilation boundaries;
- when one operation feeds another, the dependency is materialized in a small
  device buffer instead of replaying the producer graph;
- endpoint layouts use named case/record constants rather than positional
  magic numbers;
- Principled Sheen moved out of the 1,927-line closure test into a real,
  independent translation unit. The remaining file is 1,582 lines and the new
  test is 540 lines.

This changes only how test shader ASTs are scheduled. The tested inputs remain
raw Blender/Cycles closure graphs and all existing Cycles-derived numeric
oracles remain active.

## Cold fallback results

Measurements used a 32-core Ryzen host. Incomplete runs were terminated once
they had already demonstrated runaway growth; successful runs used a 24 GiB
virtual-memory ceiling.

| Regression | Before | After | Peak RSS after |
| --- | --- | --- | ---: |
| Combined closure / Sheen | exceeded 24 GiB; an earlier uncapped run reached 90.6 GiB after 2:48 without completing | Sheen 8.48 s; remaining closure suite 12.36 s | 375 MiB / 420 MiB |
| Principled Transmission | 1:37, 17.9 GiB, incomplete | 8.86 s, pass | 377 MiB |
| Principled Thin Wall | 0:34, 6.86 GiB, incomplete | 11.38 s, pass | 409 MiB |
| Standalone Refraction | proactively refactored after the same pattern was found | 0.28 s, pass | 193 MiB |
| Principled Coat | three aggregate kernels recorded 3,522,270 XIR instructions | 10.34 s, pass; eight semantic kernels record 655,845 instructions | 379 MiB |
| Standalone Beckmann Glossy | one aggregate kernel recorded 143,240 XIR instructions | 0.27 s, pass; seven semantic kernels record 62,012 instructions | 187 MiB |

The successful runs preserved every previous semantic assertion. Cached runs
are intentionally excluded from the before/after table.

## HIP and Vulkan verification

Device: AMD Radeon RX 9070 XT (`gfx1201`). Vulkan logs show native
XIR-to-SPIR-V compilation and SPIR-V optimization; DXC is not loaded.

| Regression | HIP | Vulkan XIR/SPIR-V |
| --- | ---: | ---: |
| Principled Sheen | 14.20 s, pass | 13.27 s, pass |
| Remaining closure suite | 16.34 s, pass | 19.26 s, pass |
| Principled Transmission | 11.76 s, pass | 13.32 s, pass |
| Principled Thin Wall | 13.95 s, pass | 15.04 s, pass |
| Standalone Refraction | 0.49 s, pass | 0.44 s, pass |
| Principled Coat | 12.89 s, pass | 14.61 s, pass |
| Standalone Beckmann Glossy | 0.55 s, pass | 0.30 s, pass |

These are cold shader-regression wall times, not scene-render throughput.

## Code-shape regression

Fallback test runs translate every refactored kernel to unoptimized XIR before
backend compilation and assert a kernel-specific instruction ceiling. The
ceilings retain roughly 20--25% headroom over this baseline, so a repeated
surface traversal or topology-multiplication regression fails before invoking
LLVM. Representative largest kernels are:

| Kernel | XIR instructions | Ceiling |
| --- | ---: | ---: |
| `principled_sheen_sample` | 309,482 | 390,000 |
| `transmission_sample_ggx` | 309,441 | 390,000 |
| `thin_wall_sample_closures` | 309,516 | 390,000 |
| `cycles_closure_subsurface` | 361,813 | 455,000 |
| `principled_coat_sample` | 309,391 | 390,000 |
| `refraction_sample` | 22,784 | 30,000 |
| `beckmann_glossy_sample` | 22,132 | 30,000 |

Principled Coat previously recorded three aggregate kernels containing
1,255,564, 2,144,563, and 122,143 XIR instructions. Its largest new kernel is
309,391 instructions (85.6% smaller), and the sum across all compilation units
is 81.4% smaller. Standalone Beckmann's largest new kernel is 22,132
instructions (84.5% smaller); its total recorded XIR is 56.7% smaller. These
reductions come from scheduling each operation once, not from weakening any
numeric assertion or changing a material graph.

The Coat and Beckmann refactors were verified together with:

```sh
ctest --test-dir build \
  -R 'psycles\\.luisa_(beckmann_glossy|principled_coat)_(fallback|hip|vk)$' \
  --output-on-failure
```

All six backend combinations passed. The combined cached run took 28.64 s.

All kernel counts can be reproduced with:

```sh
PSYCLES_REPORT_SHADER_SHAPES=1 \
  build/bin/psycles_luisa_principled_sheen_tests fallback
```

The named compilation boundary can be logged independently with
`PSYCLES_TRACE_SHADER_COMPILATION=1`.
