# Native surface state, geometry and image inputs

These are three independently observed structural differences from Cycles
5.2.1, not an attempt to force floating-point bit identity. Original source:
`/home/mike/Projects/blender-cycles-trace-5.2`,
`cb168525138fecc792cc393f94afc39582b0103c`.

## State and work mapping

| Boundary | Previous native route | Cycles and this change |
| --- | --- | --- |
| SSS exit, no BSSRDF bump | Evaluated the material again | Skip SVM |
| SSS exit, BSSRDF bump | SVM plus legacy BSSRDF-normal evaluation | SVM once, reduce its retained BSSRDF normals |
| Exit closure consumers | Ordinary BSDF work followed by a unit-Lambert override | Reset the pool, allocate one unit Diffuse, use the ordinary Cycles BSDF |
| Main-hit geometry | Legacy normals, eager UV/generated/tangent work and per-hit inverse | Same typed geometry and precomputed object transforms as native shadow setup |
| Static baked normals | Transform raw normal, then pack | Pack, decode, transform, normalize, pack |
| JPEG input | STB first | Existing libjpeg-backed OIIO loader first |

`kernel/integrator/shade_surface.h` evaluates the SVM at an SSS exit only
when `SD_HAS_BSSRDF_BUMP` is set. `subsurface_shader_data_setup` in
`kernel/integrator/subsurface.h` clears `SD_CLOSURE_FLAGS`, resets both
closure allocation counters, and allocates a unit Diffuse. The BSSRDF
normal uses `abs(average(weight))`, not `average(abs(weight))`, and falls
back to `sd.N` if the weighted sum is zero. It does not replace `sd.N`.
Retaining that distinction is necessary for generic BSDF bump shadowing.
`SD_TRANSPARENT` is not part of `SD_CLOSURE_FLAGS` and is not newly cleared.

The reset changes only allocation metadata. Old ordinary and extra-payload
bytes become unreachable; there is no added arena clear, changed scalar or
vector default, lifetime workaround, noinline, or software float path.
The SSS path flag remains set through population and is cleared at the
Cycles post-setup boundary. Optional material-execution counters observe
whether SVM actually ran, rather than counting every surface hit as SVM.

Main geometry reuses `setup_cycles_svm_ray_shader_data` and preserves
backfacing, packed smooth/corner normals, static transforms and compact ray
differentials. The admitted native image is still static triangles. This
does not pretend to support motion/curve setup by projecting it onto triangles.
Attributes required by SVM are fetched at their original nodes.

The normal packing order follows `blender/mesh.cpp` and
`scene/mesh.cpp::Mesh::apply_transform`. A mathematically equivalent
single-pack sequence is not the same geometry payload.

## JPEG root cause

Blender `source/blender/imbuf/intern/format_jpeg.cc` uses libjpeg. Although
that code assigns JDCT_FLOAT before reading the header, libjpeg's header
initialization resets the effective mode to the default JDCT_ISLOW. Forcing
FLOAT on the basis of the earlier assignment would be incorrect.

On Monster's original 2048-square Wood color JPEG, STB differs from the
original Blender byte buffer in 1,023,218 channels (maximum byte difference
3). OIIO differs in zero of all 16,777,216 RGBA bytes. Direct libjpeg ISLOW
also matches; FLOAT and IFAST do not. This is a decoded-resource mismatch,
not floating-point BSDF rounding or a different RNG.

JPEG is recognized by its encoded magic, including resources with empty or
non-JPEG name hints. No fake texture file is created. Other formats retain
their previous path. Exact JPEG parity here requires the existing
`PSYCLES_ENABLE_OPENIMAGEIO=ON` build; STB remains the optional-build/error
fallback and is not claimed to be byte-identical.

## Independent regressions

- `test_luisa_cycles_svm_subsurface_exit.cpp`: 144 cases against the
  original Cycles HIP functions. Flags, closure count/capacity, selected
  type, material-evaluation predicate, BSSRDF weighted/cancelled/empty
  normals, BSDF eval/PDF, sampling and bump shadowing are covered.
- `test_luisa_cycles_svm_surface_geometry.cpp`: 25 original Cycles HIP
  geometry cases, using the existing native-shadow oracle. The test
  deliberately supplies no Accel or legacy UV/generated/tangent buffers.
- `test_cycles_static_normal.cpp`: 32 original Cycles host packing cases;
  seven witnesses distinguish the wrong single-pack sequence.
- `test_scene_jpeg_decode.cpp`: all 3,072 original Blender decoded bytes
  match exactly for three resource hints. The old STB-first path fails.

Oracle generators are `cycles_svm_subsurface_exit_oracle.hip`,
`cycles_static_normal_oracle.cpp`, and
`export_blender_jpeg_decode_oracle.py` under `tools/`. They call the
original Cycles/Blender implementation, not a Psycles CPU renderer.
The JPEG fixture records its Blender identity and both SHA-256 hashes.

All builds use 32 threads. Active-worktree validation:
HIP 164/164; fallback 166/166, repeated after Luisa's generic fallback ABI
temporary-alignment repair (`origin/next` `8e2b0ac78`); core 141/142.
The remaining core failure is the known source-size gate on unrelated
oversized files, not an image/state regression. The existing render-settings
test was excluded from that core run. The new HIP/fallback/native-Vulkan
surface regressions and JPEG test pass.

Isolation exports the exact staged Psycles index, without the inherited
closure guards, ABI assumptions, other feature-mask work, frame logging,
scratch workaround removal, or dirty Luisa gitlink. It compiles against
clean Luisa `origin/next` public headers and the current local runtime
libraries; it is not a claim of a fully clean independent SDK rebuild.
Isolated focused HIP, fallback and strict native Vulkan tests pass.
The exact-index whole renderer also passes Lone Monk 32x32/4 spp on HIP,
then the Map Range probe 16x16/4 spp on fallback and strict native Vulkan.
All shader-cache reuse is disabled. Vulkan has all three native-XIR,
require-native-SPIR-V and disable-DXC gates set; the log records successful
SPIR-V generation through the complete renderer.

The isolated Monster import stops at an existing aliased-light shader-name
collision (shader 6), which the inherited uncommitted light-import work
addresses. That unrelated patch is deliberately not added to this commit.
Monster full-image and SSS path-trace results below use the active worktree;
they are not mislabeled as clean-index Monster results.

## Full images and limits

Same original Cycles HIP references, 256 spp, no adaptive sampling or denoising:

| Scene / image | Before these fixes | SSS only | SSS + geometry + JPEG |
| --- | ---: | ---: | ---: |
| Monster 1080x1080, Combined relative RMSE | 0.0326730 | 0.00815114 | 0.00590316 |
| Monster, Diff Ind relative RMSE | 0.142609 | 0.0352252 | 0.0275206 |
| Lone Monk 1440x1080, Diff Ind relative RMSE | 0.141058 | no SSS | 0.146092 |

Final Monster Diff Ind mean ratio is 1.0002529; Combined is 1.0000271.
Both final images have zero invalid pixels. Lone Monk is unresolved:
the Monster improvement is not evidence that all material/path work or the
remaining indirect discrepancy is aligned.

The actual Monster pixel (540,540), sample 7/256, now has the original
SSS-exit evaluation 0.3101585805 while its PDF remains 0.3102987111. Wood's
green BSDF evaluation becomes 0.0759745985 versus Cycles 0.0759745836 after
JPEG decoding is aligned.

Single-run sampling times are not a controlled speedup result: Monster
SSS-only changed from 15.5508 to 15.0281 seconds; Lone Monk geometry-only
was 13.5272 seconds, while the final codec run had CPU-test contention.
Main coroutine frames remain Monster 284 B and Lone Monk 220 B. Do not
compare those main-only sizes with Cycles' complete main-plus-shadow state.

## Trace validity and continuing diagnosis

Native diagnostics now record actual ShaderData flags, not legacy runtime
bits. BSSRDF selection records the closure pick but not an ordinary BSDF
sample/post-bounce record that original Cycles never writes. Optional
`--align-by-rng-offset` matches unique observed surface events across SSS
dimension gaps; unmatched/ambiguous events still fail. Derived closure
selection rescaling has the float policy, while original random draws remain
exact gates. Eight comparison and eight schema tests pass.

The four-event trace horizons differ: Cycles indexes by RNG offset, Psycles
by observed event ordinal. An additional Psycles record beyond the original
Cycles observation horizon is not by itself an additional physical bounce.
Likewise padded kernel-launch lanes are not actual active-path counts.

An original-input HIP replay of Lone Monk pixel (720,360), sample 7/256,
separates input conditioning from material evaluation. Replacing only the
intersection barycentrics/distance with Cycles' values reduces the first
closure normal's largest error from about 4.8e-4 to 3e-7 and matches the BSDF
PDF (1.07055068). The original random draws are unchanged. This identifies
the leading difference on this particular path before shading; it does not
prove that the entire image gap or work distribution has the same cause.
Primary-ray projection and active-path work remain under investigation.

Evidence: `/var/tmp/psycles-native-sss-exit-nGfyEt`; exact-index build and
GPU-input replay: `/var/tmp/psycles-native-surface-candidate-eqsvzP`.
Raw traces, EXRs, comparison JSON, original-code probes and build/test logs
are retained. No inherited path-tracer edits or Luisa gitlink are bundled
into this checkpoint.
