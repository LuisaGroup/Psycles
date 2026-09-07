# Native failed-image state and Barbershop admission

Base: Psycles `c653c872`; Luisa unchanged at `fb315d27d`. Original Cycles
source: `cb168525138fecc792cc393f94afc39582b0103c`. Rendering/export Blender
build: `9e2066aef7ef`; the instrumented build is used only for diagnostic
word dumps/path traces. Evidence: `/var/tmp/psycles-native-volume-svm-06XnDX`.

## Cause and original semantics

The exporter omitted image datablocks with zero dimensions. Import then
collapsed their references to resource zero, and native scene admission
correctly rejected an undefined resource. The original ImageManager instead
retains an assigned image's identity even if its external loader fails.

Cycles `kernel/device/gpu/image.h::kernel_image_interp` checks both an absent
image handle and a failed full-image record before wrapping UVs or accessing
a texture. Both return `IMAGE_MISSING_RGBA`. `KERNEL_IMAGE_NONE` is `INT_MAX`,
not minus one. A 1x1 magenta replacement is not equivalent: a loaded CLIP
texture returns black outside its bounds. Tiled-image wrapping has different
control flow; this change does not claim UDIM/tile-streaming support.

Export/import now preserve failed-image identity, dimensions, color/alpha
metadata and an explicit `load_failed` state without a fabricated payload.
The ordinary image contract still requires dimensions and encoded pixels;
failed state plus encoded pixels is rejected. Runtime bindings remain eight
bytes, with the failed full-image predicate in a spare metadata bit. Failed
images allocate no GPU texture and return before the fixed sampler dispatch.
No SVM opcode, typed payload, word, stack offset or shader jump is rewritten.

Image export and its existing packed/generated/linked-file helpers are now
in `tools/blender_image_export.py`, included in the exporter content identity.
The extraction keeps the main exporter below the existing source-size limit;
the limit was not relaxed. The benchmark identity regression checks this new
semantic dependency, preventing reuse after its implementation changes.

## Permanent regressions and oracle

`tools/cycles_svm_missing_image_oracle.hip` directly calls the original
Cycles HIP `svm_image_texture` with real `KernelImageTexture` failed records.
It is not a reference sampler. Sixty original GPU outputs cover four
extensions plus the absent handle, three UV positions and four image flags.
The committed fixture SHA-256 is
`88b99124cb771a17b8be128df08f661d977bcc8eb1a8487f02dc98712a2b86e2`.

The Luisa regression compiles 48 high-level image/emission graphs, executes
their complete native word streams, checks status/final PC, and delegates to
the production resource sampler. Deliberately unbound texture slots prove
that failed sampling cannot touch hardware storage. It repeats the checks
for closest/linear/cubic/smart interpolation and retains a loaded CLIP
texture as the contrasting black-result control. Color comparisons allow
one ULP: fallback's fast alpha-unassociation reciprocal yields
`0x1.fffffep-1` instead of `1.0`. Native fast math remains enabled and no
renderer/compiler arithmetic was changed to force bit parity.

Separate exporter/importer tests cover distinct missing datablocks and a
FILE image with cached positive dimensions after its external file vanishes.
The exporter regression failed before the fix. The complete renderer also
failed before JIT on the original Blender probe's missing resource zero.
Afterward, the linked outside-UV 64x64/1-spp probe has zero Combined and Emit
RMSE against the matching production Cycles HIP build.

```sh
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC \
  -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_missing_image_oracle.hip -o /a/new/oracle
cmake --build build --parallel 32
```

Full HIP: **176/176**, 126.08 s. Host: **151/152**, 12.84 s; only the four
previously recorded over-limit source files fail `source_size`, not the
extracted exporter. Full fallback: **177/178**, 186.01 s; the missing-image
regression passes. The fallback film test retains its known `light_ng.z`
trace difference; its tolerance and implementation are unchanged.

Strict native Vulkan: **4/4**, 11.76 s (missing-image, image, scene admission,
volume path), with all three fail-closed environment flags set. An uncached
missing-image loader audit produces native SPIR-V, 13,973 -> 12,363 words,
and succeeds without loading DXC/DXIL. Logs use `native-missing-image-*`.

## Barbershop: admission fixed, image parity not yet achieved

The subsequent [shared-closure correction](../shared-closure-weights/README.md)
fixes the missing transparent closure identified below and records the large
reduction in image residual. The timings and images here remain the exact
before-correction checkpoint, not the current best Barbershop result.

Fresh export from the original `barbershop_interior.blend` retains 197 image
datablocks, including seven failed images. The original source is unchanged.
The main shader cache is disabled; fast math and the staged surface/NEE
queues are enabled. No profiler, concurrent GPU test or CPU build runs during
this single timing canary. Fixed samples [0,256), TABULATED_SOBOL, raw seed 0,
animated frame 1, effective seed 1,267,069,554 match the reference.

| Extent / spp | Scene compile s | Cold main JIT s | Render-only s | Coro frame |
| --- | ---: | ---: | ---: | --- |
| 2048x858 / 256 | 15.6293 | 95.2809 | 48.0181 | 896 B, 153 fields, 6 stages |

All 46 output channels are finite. Combined / DiffCol / DiffInd relative
RMSE is **0.13641207 / 0.12181950 / 0.26861247**. Combined mean luminance is
0.962264 of Cycles. These are not correctness/performance success criteria.
The older Cycles 40.379 s value is a Python render-call duration, not a
matched main-loop baseline, and must not be used for a claimed speed ratio.

A same-sample trace at film pixel (1260,616), sample 0, finds the first
structural divergence: both renderers hit object 14, triangle 20011, shader
27 (`cobwebs.001`); Cycles creates one transparent closure while Psycles
creates none and terminates the path. This is separate from the corrected
failed-image state. Its shared-closure/Mix compilation and runtime control
flow require reduction against the captured original word image before a
fix. Neither an RNG mismatch nor harmless indirect noise has been established
as its cause. Frame-field lifetime and the remaining scene residuals also
remain open work.

Artifacts: `barbershop-native-missing-image-{hip.log,hip.exr,metrics.json}`,
`barbershop-native-oracle.svm52`, `barbershop-*-1260-616-s0.*`, and
`barbershop-triptychs/`. The canary bundle predates only the mechanical image
module extraction; it is retained as the exact input of that measurement.
