# Lone Monk residual: separate path semantics from coincident geometry

This is a diagnosis, not a renderer fix or a claim of complete image parity.
Original source scene, renderer geometry and fast-math settings remain intact.
The temporary scene ablation below must not become an importer deduplication
rule or a replacement for the original-scene benchmark.

Evidence directory: `/var/tmp/psycles-path-structure-3ahR3G`.
Blender 5.2.1 production HIP build `9e2066aef7ef`; trace build `cb168525138f`.
Both production renderers use hardware ray tracing. The Cycles log explicitly
reports HIPRT layout and Hardware Ray-Tracing On; the comparison is not HIPRT
against a CPU reference or software-only Cycles BVH.

## Locate the contribution, not just the largest relative pass error

1440x1080, 256 fixed samples, seed 0. A separate original-Cycles seed-1 render
provides a noise/correlation control, not a proof that residuals are unbiased.
The following RGB relative RMSEs use the seed-0 reference as denominator.
Block averaging crops to complete blocks and averages before computing RMSE.

| Pass / block width | Cycles seed 1 vs seed 0 | Psycles seed 0 vs Cycles seed 0 |
| --- | ---: | ---: |
| Combined / 1 | 0.056653 | 0.012405 |
| Combined / 16 | 0.004010 | 0.005185 |
| Combined / 64 | 0.001363 | 0.002478 |
| DiffInd / 1 | 0.656653 | 0.128816 |
| DiffInd / 16 | 0.051963 | 0.011506 |
| DiffInd / 64 | 0.015889 | 0.004476 |

Thus the raw DiffInd discrepancy alone is not a count of extra paths, and the
coherent Combined residual cannot simply be dismissed as noise. Multiplying
light passes back by their color passes locates the largest coherent residual
in direct diffuse radiance: at 32x32 blocks its RMSE, normalized by Combined
RMS, is 0.002285 versus 0.001538 in the independent-seed Cycles pair. Direct
diffuse plus direct glossy explains the dominant bright vegetation strip.

## Same-sample trace

At Cycles film pixel (277,394), sample 93/256, events 0 and 1 traverse fully
transparent leaf cards. Event 2 reaches object 83478, material `bush`, shader
36. The same light is selected, with nearly equal unshadowed contributions:

| Event 2 RGB | Cycles | Psycles |
| --- | --- | --- |
| Unshadowed NEE | (31.4005,34.9389,14.6327) | (31.4837,35.0093,14.6830) |
| Final NEE | (0,0,0) | (31.4837,35.0093,14.6830) |

This is a visibility discrepancy, not a missing random-number dimension or
an energy-normalization error at this event. Cycles then intersects another
triangle of the same object about 3.35e-6 along the continuation ray; Psycles
reaches a different object 0.279 units away. Later state differences are
downstream of this geometry divergence, not independent RNG failures.

The receiving Cycles triangle is global 1572052, local 13; Psycles picks global
1572044, local 5. They are **exact coincident duplicates**, not an eight-triangle
import offset: geometry 344 starts at 1572039 and contains 16 triangles. The
exported positions, normals and UVs of its two halves are identical. Blender's
original `leaf.001` mesh itself has 20 vertices and eight quads, no modifiers,
with quads 4..7 duplicating 0..3. Both import and BVH input retain both halves.

The origins differ by only a few float32 steps. Their source-self exclusion
and origin-offset control flow map to Cycles. At exact coincident surfaces,
the signed near-zero intersection distance is ill-conditioned. An isolated
native HIPRT replay of the leaf misses for both recorded origins; merely
swapping origins does not reproduce the Cycles hit. That experiment does not
identify a particular arithmetic instruction as the cause. Do not turn this
into a slow software intersection, strict-FP mode, or bit-matching workaround.

Enabling trace instrumentation (which disables the independent shadow queue)
and rendering all 256 samples gives the same pixel as the uninstrumented
production render. A windowed independent-queue run also agrees. The side
queue is therefore not a necessary trigger for this discrepancy.

## Controlled scene ablation

`leaf_ablation.py` verifies exact duplicate coordinates, face ordering,
material/smooth flags and UVs before removing only the redundant leaf quads
in a new .blend. The original scene is untouched. A fresh original-scene
export with the same current exporter controls for exporter-version changes.
All renders below use 1440x1080, 256 spp, seed 0, normal native SVM and fast math.

| Relative RMSE | Fresh original export | Duplicate-free diagnostic copy |
| --- | ---: | ---: |
| Combined | 0.0123919975 | 0.0052862299 |
| DiffCol | 0.0005459054 | 0.0005439478 |
| DiffDir | 0.0136429763 | 0.0040994183 |
| DiffInd | 0.1288172839 | 0.1278904324 |
| GlossInd | 0.1359731043 | 0.1324497900 |

At (277,394), original Cycles/Psycles RGB is approximately
(0.725,0.962,0.316)/(4.561,5.812,1.978). In the ablation it becomes
(4.892,6.282,2.134)/(4.914,6.285,2.143). The duplicated-surface configuration
is a causal amplifier of the direct-light residual. It does not explain all
indirect differences, and it does not justify changing the authored scene in
production. No speedup is claimed from these diagnostic runs.

Artifacts: `monk-277-394-s93.{raw,chunks,compare}.json`,
`monk-cycles-277-394-s93.exr`, `leaf-original-current-compare.json`,
`leaf-compare.json`, `leaf-{cycles,psycles}1080.exr`, and
`monk-cycles-seed1.exr`. Original trace commands use
`tools/render_cycles_path_trace.py` with the full 1440x1080 extent, pixel
(277,394), `--total-samples 256 --sample 93 --seed 0 --cycles-device HIP`.

```sh
blender lone-monk_cycles_and_exposure-node_demo.blend --background --threads 32 \
  --python-exit-code 1 \
  --python docs/validation/2026-09-07/lone-monk-residual/leaf_ablation.py -- \
  /a/new/temporary/monk-leaf-ablation.blend
# Export takes an output DIRECTORY, not a JSON filename.
blender /a/new/temporary/monk-leaf-ablation.blend --background --threads 32 \
  --python-exit-code 1 --python tools/export_psycles_scene.py -- \
  /a/new/temporary/export
```

The separate sampler-property fix does not explain this result: Lone Monk's
actual automatic-scrambling setting was already false. Continue validating
discrete paths/work and other scenes. Do not equate either a single trace or
an independent-seed RMSE control with whole-renderer structural completion.
