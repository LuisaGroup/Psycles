# Cycles SVM microfacet tagged-union dispatch

## Source relation

Cycles 5.2.1 `bsdf_microfacet_eval/sample` performs shared geometry/VNDF
work first and dispatches `MicrofacetBsdf::fresnel_type` inside
`microfacet_fresnel`. Psycles instead dispatched conductor, F82, and the
remaining payloads around the entire sampling/evaluation algorithm.

A pool-backed `MicrofacetClosure` now retains the storage/index handle
corresponding to Cycles' tagged extra-payload pointer. Only the matching
Fresnel branch loads that payload. The direct typed overloads remain
available for standalone closure probes. The inline generalized-Schlick
payload is still supported; pool-backed albedo consumers load it lazily too.
No SVM word, stack address, closure allocator transition, or feature-mask
semantics changes. No application `noinline` is introduced.

## Reproduction and regression

`tools/cycles_svm_microfacet_union_oracle.hip` includes and invokes the
external Cycles device routines on HIP. It contains no reference shading
implementation. The source used was
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`; the SHA-256 of
`intern/cycles/kernel/closure/bsdf_microfacet.h` was
`be2f90dccd1498915e95c6db56c911fa581710d469c2d18b0e7d0dfe87e24965`.

The new independent test freezes four HIP oracle cases:
GGX/Beckmann crossed with conductor/F82. Runtime input-buffer values
interleave the cases across 64 lanes. It records sampled direction, PDF,
evaluation, roughness, eta, label, and direct evaluation/PDF. Only the
active tagged payload is initialized.

Before the change the unified pool-view regression fails on HIP:
`scenario 0 row 0 channel 0: 0 != Cycles 0.0538458`.
Previously the renderer avoided this unsupported generic-view path through
its outer Fresnel dispatch; this is not evidence that the old renderer
rendered all metals black.

## Validation

All builds used 32 threads.

- Active worktree: 39 compiler/SVM tests, 55 HIP SVM tests, and 55 fallback
  SVM tests passed.
- The three microfacet-union/scattering/BSDF-dispatch tests passed on native
  XIR Vulkan with USE_XIR=1, REQUIRE_NATIVE_XIR_SPIRV=1, DISABLE_DXC=1.
- An isolated candidate containing only the eight implementation/test/tool
  files built the complete Psycles libraries (352 build steps) against the
  existing Luisa libraries. All three focused tests passed on HIP and
  fallback. Unrelated renderer changes and the Luisa gitlink are excluded.

Active-tree evidence is under
`/var/tmp/psycles-microfacet-union-profile-BlyG7X` and
`/var/tmp/psycles-microfacet-union-clean-QdCDE9`.
The performance measurement includes the pre-existing uncommitted renderer
work; it is not a benchmark of the isolated commit alone.

Lone Monk, frame 4, 1440x1080, 256 spp, HIP:

| Measurement | Before | After |
| --- | ---: | ---: |
| Render elapsed | 19.9789 s | 19.9471 s |
| Surface kernel total | 12.460404 s | 12.385482 s |
| Surface code object | 1,096,984 B | 1,080,728 B |
| Surface private scratch | 5,284 B | 5,252 B |
| VGPR count | 256 | 256 |
| Coroutine frame fields / struct size | 105 / 440 B | 105 / 440 B |

Timings overlap the previous run range: there is no established end-to-end
speedup. Combined relative RMSE against the Cycles HIP oracle is 1.1138%,
with zero nonfinite pixels. The Cycles baseline remains 16.4504 s total and
5.9491 s in shade-surface; the overall performance goal is not reached.

The 440 B value above is the frame's struct size, not the complete scheduler
state allocation. Current runtime SoA uses 436 B per slot and the separate
four-hit shadow buffer uses 108 B per slot; queue/sort storage and private
kernel scratch are additional.
