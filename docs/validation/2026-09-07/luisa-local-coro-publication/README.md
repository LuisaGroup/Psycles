# Published Local/coroutine integration

The Luisa dependency is advanced to
`53aa4bf02f1e557590473909a7b311105880067f` on `origin/next`. Its unpublished
Local/frame/scope fixes were integrated onto `3485bd89d`, preserving the
already-published scheduler extensions, HIP ABI and late-inlining fixes,
native SPIR-V changes, and subsequent Tile work. The original dirty submodule
was not committed wholesale over its old base.

The Luisa proof, two integration counterexamples, permanent regressions and
validation qualifications are documented in
[Local storage and frame reconstruction](../../../../third_party/LuisaCompute/docs/validation/2026-09-07/local-and-coro-lifetimes/README.md).
Ordinary scalar/vector/Array initialization is still zero; Local lexical
epochs and exact frame reconstruction do not manufacture initialized values.
The old application-side SurfaceValueLocals lifetime seed is now redundant
and removed.

## Renderer changes published with the dependency

- Compiler-validated SVM uses the Cycles release PC loop and direct NODE_END
  return; the diagnostic evaluator retains status/PC reporting. Tests check
  both against the same Cycles closure oracles and forbid forced noinline.
- The surface interpreter consumes the statically proved scene stack extent
  and Cycles feature masks. Hair, principled-hair and BSSRDF consumers obey
  the corresponding host-time feature guards.
- Surface closure initialization observes terminating, emission-only and
  shadow evaluation. Empty/non-scattering populations cannot enter the
  picker; BSSRDF payloads are read only in the selected BSSRDF branch.
- Film AOV reduction runs only on eligible paths and after closure setup.
  Transparent extinction survives allocation overflow; retained transparent
  closures still participate in normal/roughness reductions. Native and
  transitional consumers share the Cycles microfacet albedo factors.
- Contract visibility is explicitly permuted to the Cycles ABI, including
  expansion/folding of the two Cycles shadow bits.
- Scheduler tuning remains in Psycles through the generic continuation
  configuration. Optional frame logging uses
  `PSYCLES_DUMP_COROUTINE_FRAME`; no renderer-specific Luisa pass is added.

The remaining legacy SurfaceProgram dependencies have **not** been removed by
this commit. This is a publication of the jointly verified runtime changes,
not a claim that the broader native SVM migration or benchmark campaign is
complete.

## Verification

Evidence directory: `/var/tmp/luisa-publish-coro-bQIjua`.

The SDK and the whole renderer were independently rebuilt with 32 threads.
The renderer snapshot contained all the implementation/test changes published
here, including the new microfacet files; file contents were compared with
the active worktree before publication. It used the new SDK headers,
generated configuration and libraries, not the previous build's binaries.
Profiling dumps and the unfinished volume oracle were excluded.

| Run | Result |
| --- | --- |
| Psycles host CTest | 120/120; excludes source-size and external Blender checks |
| Psycles HIP CTest | 168/168, serial GPU execution |
| Psycles fallback CTest | 170/170 |
| Psycles strict native Vulkan focused CTest | 12/12 |
| SDK HIP / fallback runtime suites | 7/7 on each backend |
| SDK strict native Vulkan runtime suites | 6/6 |
| SDK focused host CTest | 151/151; qualifications in the Luisa report |

The initial parallel HIP test run was interrupted with active test processes;
it is not counted as successful. The complete serial rerun passed. The full
SDK host run also has three documented unrelated baseline/configuration
failures; this report does not describe that run as fully green.

Vulkan uses native XIR-to-SPIR-V, disables AST/LLVM-to-SPIR-V and DXC
compatibility, and sets `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
`LUISA_VULKAN_DISABLE_DXC=1`. `LD_DEBUG=libs` confirms no DXC/DXIL load. The AMD
build explicitly disables the independent `LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP`
option to avoid requiring the absent NVIDIA driver.

The whole-scene map-range canary passes at 16x16 / 4 spp on fallback and
strict native Vulkan, with all 46 output channels finite. Uncached HIP
staged-wavefront scene canaries also pass:

| Scene | Resolution / samples | Render time | Frame |
| --- | --- | --- | --- |
| Lone Monk | 1440x1080 / 256 spp | 13.7828 s | 55 fields / 220 B |
| Monster | 1080x1080 / 256 spp | 14.9817 s | 71 fields / 284 B |

Both output 46 finite channels. These are single canary timings, not paired
benchmark results or a measured speedup. Lone Monk versus the recorded
Cycles 5.2.1 HIP reference has Combined relative RMSE 0.0124035 and DiffInd
relative RMSE 0.128819. That remaining structural/parity investigation is open.

## Recoverability

The original unpublished child work is preserved in stash commit
`26bee45c309ba0f6f6303bc99169a021a82658a2` in the local Luisa repository and as
`original-tracked.patch` / `original-untracked.tar` in the evidence directory.
Tracked content and untracked archive contents were verified unchanged before
stashing. The child checkout was then fast-forwarded to the published commit;
the stash is deliberately not reapplied, because that would reintroduce old
versions of fixes already integrated upstream. No profiling artifacts or
unfinished volume implementation were included in the publication.
