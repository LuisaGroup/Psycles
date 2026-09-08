# Native descriptor sampler alignment

Barbershop remains substantially slower than Cycles. This correction removes
an unnecessary shader branch product and gives a **1.0% lower Psycles render
median** in the follow-up canaries. It does not explain or close the main gap.
The [matched kernel profile](../matched-pass-hip/README.md) still supplies the
diagnosis: surface shading, not volume, dominates the measured excess.

## Structural correction and original oracle

Cycles binds filter/address state into each ImageManager texture object.
Psycles previously generated all 3 interpolation families x 4 extensions in
its image adapter. It now binds a distinct sampler descriptor per native
handle while sharing the underlying decoded image. Only ordinary versus
four-tap bicubic algebra remains in the shader.

The shape regression changes **24 to 5 native sample call sites**. These are
generated call sites, not texture fetches executed per path or an instruction
count. The old shader already selected only one interpolation/extension path.
SVM words, PC, stack, closure and feature-mask semantics are unchanged, as are
fast math, UV orientation, bicubic arithmetic, coroutine policy and inlining.

The complete scene-upload regression exercises sixteen handles sharing one
image, with 320 native-control component checks per backend. HIP additionally
checks 320 components against the original Cycles GPU `svm_image_texture`.
The [source notes](source-notes.md) document the original source revision,
true red regression, sampler mapping and the storage-coordinate boundary of
that oracle. It is not a CPU reference renderer or a claim of bitwise texture
filtering parity across backends.

## Six full-resolution HIP canaries

All cases use 256 spp, the same scene exports, seeds and exact fifteen-pass
commands as [the paired baseline](../matched-pass-hip/README.md). Only output
paths change. The retained Cycles run-1 EXRs are checked against their schema-v3
provenance; **these are not new paired Cycles timing measurements**.

| Scene | Resolution | Old Psycles median, s | New Psycles render, s | Repeats |
| --- | --- | ---: | ---: | ---: |
| Lone Monk | 1440x1080 | 13.6826 | 13.4104 | 1 |
| Monster | 1080x1080 | 15.1328 | 14.8652 | 1 |
| Classroom | 1920x1080 | 18.7626 | 17.9502 | 1 |
| Barbershop | 2048x858 | 39.2753 | **38.8902 median** | 3 |

Barbershop repeats are 38.6666, 38.8902 and 38.9089 s. Relative to the retained
Cycles median of 25.3775 s, the new median is still 1.5325x. The other scenes'
single observations are correctness/performance canaries, not sufficiently
repeated evidence for a small performance lead over Cycles.

All six outputs contain exactly 46 finite channels, all fifteen pass
comparisons complete, and frames remain 220 / 284 / 264 / 416 B for
Monk / Monster / Classroom / Barbershop. The first Combined triptych for each
scene was visually inspected at the viewer's scaled resolution. Existing
residuals remain; no full-image bitwise agreement is claimed.

| Scene | Combined relative RMSE | DiffCol relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: | ---: |
| Lone Monk | 0.01240781 | 0.000545938 | 0.1288152 |
| Monster | 0.00547872 | 0.000109916 | 0.0255250 |
| Classroom | 0.00353320 | 0.000102324 | 0.1782033 |
| Barbershop, first repeat | 0.01079846 | 0.001597277 | 0.0745404 |

Original Classroom still contains invalid DiffDir/GlossDir pixels. The
comparator excludes the invalid union explicitly; all Psycles pixels are
finite. Image differences have not materially changed from the paired
baseline, but neither similarity nor these metrics establish global path/RNG
parity.

### Initialization is cache-sensitive

The main Psycles shader cache is disabled throughout; auxiliary/downstream/OS
caches retain normal policy. First-use session initialization (JIT plus setup,
not compiler-only) is 78.1307 s for Barbershop, 62.2088 for Monk, 73.1555 for
Monster and 55.8885 for Classroom. Subsequent Barbershop values are 25.4336 and
25.5426 s. The first Barbershop large HIP bitcode link takes 34.984 s; therefore
the first-use increase must not be presented as a new steady JIT regression,
nor may the warm result be advertised as a fully cold compile time.

No build, test, other campaign render or profiler overlaps render timing.
Six executable/library hashes remain fixed across all cases. Ordinary desktop
activity is not isolated. Raw EXRs, triptychs and logs are retained under
/var/tmp/psycles-native-volume-svm-06XnDX with the bound-sampler prefix.

The checked-in [canary summary](canary-summary.json) preserves exact commands,
hashes and primary metrics for every pass; the raw record's hash identifies the retained
full metadata. [run_canaries.py](run_canaries.py) reproduces the six cases and
[summarize_canaries.py](summarize_canaries.py) performs the documented reduction.

## Validation at the sampler checkpoint

- All-target build: passed with all 32 threads.
- HIP: **180/180**, 50.78 s (`bound-sampler-final-full-hip.log`).
- Fallback: **181/182**, 92.22 s (`bound-sampler-final-full-fallback.log`).
  The outstanding dispatch-trace assertion is a separate follow-up, not
  silently waived here.
- Host: **155/156**, 5.49 s (`bound-sampler-full-host.log`); only the four
  previously recorded source-size violations fail.
- Strict native Vulkan image gate: **5/5**, 0.68 s
  (`bound-sampler-native-vk.log`). All three native-XIR/no-DXC guards are set;
  emitted SPIR-V and dynamic-loader logs show no loaded DXC/DXIL library.

The child stays at published Luisa 9ea3b720f. No child or toolchain change is
part of this correction. The small production fix and original-GPU fixtures
are intended to preserve the Cycles structure even where end-to-end gains
are modest.
