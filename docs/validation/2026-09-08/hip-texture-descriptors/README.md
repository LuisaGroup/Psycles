# HIP texture descriptors are already native

## Technical summary

Luisa already implements the proposed ROCclr descriptor extraction: it copies
real image and sampler hardware descriptors into device memory and directly
issues AMD image-sampling instructions. Native HIP supports **point and linear**
filtering; original Cycles uses four bilinear samples for CUBIC/SMART, not a
native cubic filter. The remaining bindless pointer hop is real, but removing
it did not produce a consistent improvement in the controlled GPU experiment.
Neither a combined descriptor layout nor a hand-written grouping loop is
adopted: both regress divergent sampling.

The new original-scene profile still places the large Barbershop gap in surface
shading. This investigation changes no production renderer, Luisa ABI, SVM
semantics, frame layout, fast-math setting or inlining policy.

## Original HIP objects and device-resident SRDs are different

A shader resource descriptor (SRD) is the hardware state consumed by an AMD
image instruction. ROCm 7.2's texture-object prefix contains 12 image dwords
followed by 8 sampler dwords; the instructions used here consume the leading
8 and 4 respectively. The [ROCm source](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/hipamd/src/hip_texture.cpp)
copies these from the image and sampler runtime objects. Device sampling
does not chase the later `amd::Image*` and `amd::Sampler*` C++ members.

Current Luisa `HIPTexture::copy_image_descriptors` and
`copy_sampler_descriptors` already query that prefix through HIP-created
objects. They retain 32-byte image SRDs per mip and a shared table of sixteen
16-byte sampler SRDs. `HIPBindlessArray::Slot` is 72 bytes; its image pointer
addresses the compact mip array directly, not another texture-object pointer
table. Scene/update-time allocation is not per-thread or per-path allocation.

The source requests `CL_MEM_SVM_FINE_GRAIN_BUFFER` for the native object.
The independent live allocation probe on this host reports:

| Allocation | HIP memory type | HSA pool owner | Reported HSA flags |
| --- | --- | --- | --- |
| Native HIP texture object | Host | Ryzen 9 9950X3D CPU | Coarse-grained, 4 |
| `hipMalloc` compact descriptor | Device | gfx1201 GPU | Coarse-grained, 4 |

These are observations of this runtime, not a universal residency or
fine-grain guarantee. In particular, the source's requested SVM flag must not
be substituted for the HSA query's effective flags. Luisa already moves the
sampled descriptor bytes into the GPU-owned allocation.

## Fewer pointer hops did not guarantee faster sampling

The synthetic experiment uses 1,048,576 GPU threads, sixteen dependent samples
per thread, 4,096 descriptors, 256 distinct 64x64 BYTE4 images, linear filtering
and all four address modes. It uses two warmups and nine timed HIP-event
measurements per variant, rotating variant order. Every variant samples the
same pixels with the same coordinates; original HIP `tex2D` is the GPU control.
No software sampler or CPU reference renderer is introduced.

The final run's median kernel times, in milliseconds, are:

| Descriptor implementation | Same ID within wave | Neighboring IDs per lane | Hashed IDs per lane |
| --- | ---: | ---: | ---: |
| Indirect compact SRD, shared sampler table | 0.1235 | 1.1455 | 1.2242 |
| Dense image SRD plane, shared sampler table | 0.1247 | 1.1419 | 1.2245 |
| Combined 64-byte image/sampler slot | 0.1025 | 1.7548 | 1.5192 |
| Original HIP object, native `tex2D` | 0.1842 | 3.8547 | 4.3890 |
| Device-memory prefix copy, native `tex2D` | 0.1386 | 1.1478 | 1.2280 |
| Original HIP prefix, direct image intrinsic | 0.1756 | 3.8669 | 4.4021 |
| Indirect SRD, explicit index-grouping loop | 0.1337 | 2.6462 | 2.9033 |

The dense image plane removes a dependent pointer load without a stable
general gain. Across the two preceding independent process runs it saves
about 3-4% for wave-coherent IDs and is essentially neutral for divergent IDs;
the final run is neutral in all three patterns. The combined layout is
smaller in machine code (564 versus 620 bytes) yet much slower for divergent
IDs. Its waterfall-loop ordering also changes, so this is not a pure count of
loads or bytes. None of these local measurements is a renderer speedup.

The native-object controls separate residency from the HIP device wrapper:
copying the complete 80-byte prefix to device memory makes native `tex2D`
roughly match compact SRDs, while bypassing the wrapper but retaining the
original object does not. Native-object timing varies substantially between
processes; all samples, including outliers, are retained. The two prior runs
and the final run support the direction, not a universal speedup factor.

The final run verifies **75,497,472 output-component comparisons** against the
native GPU control with maximum absolute error zero. Non-finite results are
rejected by integer exponent-bit checks, which remain valid with fast math.
Earlier six-variant runs each verify another 62,914,560 comparisons.

## Divergent texture selection needs grouping on both implementations

Disassembly of both the actual Luisa integration test and the original Cycles
GPU image oracle contains `image_sample_lz` with `readfirstlane`, execution-mask
comparisons and backedges. AMD instructions consume scalar SRDs, so divergent
resources are grouped before sampling; seeing this loop is not uniquely a
Luisa defect. Same descriptor pointers would imply the same immutable SRDs,
but grouping on that narrower key is not automatically faster.

The explicit-key experiment groups by the immutable descriptor index before
loading SRDs, without assuming uniform texture selection. It passes the GPU
control but is about 2.3x slower in the divergent cases. It moves loads into
the serialized groups and loses the original gather-before-grouping behavior.
This is a rejected experiment, not a new compiler strategy or production patch.

## Barbershop remains dominated by the surface gap

The follow-up uses the same 2048x858, 64-spp, fifteen-pass instrumented command
as the retained matched profile, after the published sampler-binding fix.
It rechecks the 46-channel output contract and all six renderer/library hashes.
The original Cycles profile is retained, not rerun simultaneously.

| Stage | Retained Cycles GPU seconds | Current Psycles GPU seconds | Psycles minus Cycles |
| --- | ---: | ---: | ---: |
| Surface shading | 2.9628 | 5.6336 | +2.6708 |
| Closest intersection | 1.3669 | 1.5397 | +0.1728 |
| Volume shading | 0.3536 | 0.4289 | +0.0753 |

Psycles surface time was 5.7269 seconds before the sampler-binding change;
the follow-up is about 1.6% lower, consistent with its small end-to-end canary
change. This single kernel profile is not a statistically controlled speedup
estimate. Surface resource counts remain 256 VGPR / 2432-byte scratch versus
the retained Cycles 192 / 6976; neither count alone measures achieved occupancy.
Dispatch counts remain 1120 versus 797 and are **not active shading-event counts**.
Instrumented render wall time is 10.1749 seconds, not a new 256-spp benchmark.
All 46 actual channels are finite; the fifteen-pass image comparison uses the
original 64-spp Cycles output and matching exact Blender build metadata.
Combined / DiffCol / DiffInd relative RMSE is 0.02055 / 0.001893 / 0.11273.
These 64-spp errors must not replace the earlier 256-spp comparison numbers.

## Boundaries and next checks

Keep the current device-resident compact SRD scheme. No ROCclr patch or runtime
object ABI expansion is needed to obtain these descriptors. Keep version/layout
checks, original resources alive while their SRDs are consumed, and proper
in-flight update ownership; cloned prefixes must never enter HIP host object
destruction APIs. Do not create a device allocator in the render path.

The experiment models descriptor access, not Luisa's complete sampling lowering:
its indirect pointers share a contiguous backing array, whereas current Luisa
allocates per binding; it does not model mixed formats, mip/gradient sampling,
3D textures or the generic R10G10B10A2 software-format branch. The original
Cycles [image-binding regression](../bound-image-sampler/README.md) remains the
separate SVM/upload/filter correctness evidence. A successful linear microbench
is not proof of whole-scene texture parity.

The unresolved work is to measure surface **active work**, material/texture
divergence and whole-kernel register/code pressure. In particular, the generic
packed-format branch appears in generated code but these scenes do not execute
that format; its whole-kernel cost is not yet established. These experiments
do not prove that texture-related code is irrelevant, only that the proposed
SRD extraction is already implemented and further layout simplifications need
whole-kernel evidence before adoption.

Exact commands, source identities and raw data are in the
[source notes](source-notes.md) and [audited record](audit-summary.json).
