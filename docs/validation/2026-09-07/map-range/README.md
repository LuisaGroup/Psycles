# Cycles Map Range and multi-scene validation

Monster under the Bed exposed an unmigrated `psycles.math.map_range` in
shader 5 (`Area.001`). This checkpoint preserves both original SVM nodes
and fixes the generic import of unavailable Blender sockets.

## Semantic analysis

Reference: Cycles 5.2.1 at `cb168525138fecc792cc393f94afc39582b0103c`:

- `intern/cycles/blender/shader.cpp`: unavailable inputs are skipped before
  copying defaults; unavailable endpoints are absent from socket maps.
- `intern/cycles/scene/shader_nodes.cpp`: MapRangeNode and
  VectorMapRangeNode prototypes, expansion, linear-operation classification
  and compilation.
- `intern/cycles/kernel/svm/map_range.h`: both runtime functions.
- `intern/cycles/util/math_base.h`: scalar smoothstep's boundary branches.
- `intern/cycles/kernel/svm/svm.h`: original interpreter/domain dispatch.

The manifest previously preserved hidden authored defaults without recording
socket availability. For example, LINEAR scalar Map Range with hidden
Steps=3 incorrectly replaced the Cycles prototype's Steps=4. Filtering links
alone cannot repair this. The manifest now records `available`; the generic
normalizer skips unavailable defaults and link endpoints. Old manifests
without the optional field retain their previous interpretation. There is
no Map-Range-specific default substitution in the importer.

| Property | Scalar | Vector |
| --- | --- | --- |
| Typed payload after opcode | 8 words | 21 words |
| Arguments | Six SVMInputFloat values | Six SVMInputFloat3 values |
| Equal From bounds | Result zero | Factor zero, result To Min |
| Clamp | Separate NODE_CLAMP_RANGE expansion, all modes | Payload byte; ignored in smooth modes |
| Reversed smooth bounds | Swap and complement scalar smoothstep | Clamp component factors before polynomial |
| Linear-operation test | To Min/Max unlinked | All four bounds unlinked |

Both retain LINEAR, STEPPED, SMOOTHSTEP and SMOOTHERSTEP. STEPPED maps
nonpositive steps to factor zero. Unknown runtime mode values use LINEAR,
as in Cycles. Invalid source properties fail compilation. The vector graph
alias only selects the original VectorMapRangeNode schema; it is not a new
SVM opcode. Clamp expansion shares linked bounds and retains the Map Range
node's inputs. Ordinary scalar/vector zero initialization, coroutine policy,
feature-mask specialization and the PC loop are unchanged. No noinline or
software floating-point path is introduced.

## Independent regressions

Evidence: `/var/tmp/psycles-native-map-range-gw6Cv6`.
`tools/cycles_shader_probe/map_range.py`, registered as
`svm_map_range_matrix`, creates 24 live Blender graphs: scalar/vector,
four interpolations, both Clamp states, degenerate bounds, nonpositive
steps and all-stack scalar inputs. Geometry supplies runtime inputs.

The observer-enabled original Cycles executable
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender` produced
`svm-oracle.svm52`; the separate production HIP executable is not an SVM
dumper. A 24x16/1-spp original Cycles CPU render only triggers serialization,
not a Psycles CPU reference. The earlier `oracle.*` artifacts in this evidence
directory belong to a different existing probe and are not used here.

`tools/extract_cycles_svm_shader.py` extracts shader IDs 5..28. Only the
three global jump targets are relocated. The compiler test compares every
word in every imported image exactly, including unused typed fields and
Clamp expansion. A second fixture contains Monster's actual Area.001 light
graph and its complete 43-word image, jump `(1,4,41,42)`. No expected image
is synthesized from Psycles output.

The runtime oracle includes the original `svm_eval_nodes` and runs it on
HIP for 24 images x 16 positions x three shader domains = 1,152 cases.
Only input positions and dumped words are shared with the Luisa regression.
RGB is external Cycles GPU output; flags, END status and final PC are checked
exactly. Volume/displacement programs must retain the initial accumulator.

```sh
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC \
  -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_map_range_oracle.hip -o /path/to/oracle-hip
/path/to/oracle-hip tests/data/cycles_map_range_words.txt
```

The initial compiler failure and unavailable-default witness are retained in
`compiler-red.log` and `availability-red.log`. `runtime-red-hip.log` records
the missing-opcode runtime failure before adding either handler.

### Discontinuous floor and one ULP

The unmodified HIP oracle passes on Luisa HIP. The first fallback failure
is scalar STEPPED at Value=-1, From=(2,-1), Steps=3. The minimal device
reductions `step_luisa.cpp` and `step_cycles.hip` observe:

| Backend | Quotient bits | Input to floor | Input bits | floor |
| --- | --- | --- | --- | --- |
| Cycles HIP compiler flags | `3f7fffff` | 3.99999976 | `407fffff` | 3 |
| Luisa HIP | `3f7fffff` | 3.99999976 | `407fffff` | 3 |
| Luisa fallback | `3f800000` | 4 | `40800000` | 4 |

Cycles HIP itself enables `-ffast-math`. This is a one-ULP quotient/product
difference across a discontinuity, not a graph, stack or dispatch defect.
The production implementation is not changed to emulate a device's rounding.

Two separate original HIP interpreter runs use
`-DPSYCLES_MAP_RANGE_FLOOR_ULP=-1` and `=1`. After pre-including all
dependencies, a wrapper affects only the input to `floorf` in the original
`map_range.h`; every other node, output conversion and closure stays original.
The central oracle remains unmodified. The regression accepts only one of
these three discrete output candidates per channel (3e-5 relative/absolute
arithmetic allowance), not any value in a widened interval. The policy is
identical on HIP, fallback and Vulkan; non-STEPPED results are unchanged.

SHA-256 provenance:

| Artifact | Hash |
| --- | --- |
| Probe SVM dump | `9c89402691fd0e955f812a050ec99b4305f7331dd5a07172193c53d3b67c44f1` |
| Monster SVM dump | `d42864f16a53349323b18bc3df56f54598ec15453b5961fbfa2d8c76a428abb1` |
| Map Range word fixture | `c4562a4693b43631ff0dde99b3bddb1091f3025a1d1f3b0a48ad306e41cd2568` |
| Central runtime fixture | `dd7523d5806b167f6d2134f96a20f35ff19e0c639b03fdca83621a75ef5270a9` |
| Rounding fixture | `a481828be4c2cbf3830917dc42ad92a19c61a4d99ef9b1ae3a633bb31b7bb68f` |
| Monster light word fixture | `47fa81a6ec22f98baf4f1b35f1fdd15ce49bb3cdc82479856f8b70efbfcf2c76` |

## Whole Monster scene, HIP

Source is the unmodified official `monster_under_the_bed_sss_demo_by_metin_seven.blend`.
After re-export with socket availability, native compilation/upload succeeds:
32 shaders, 10,948 words; 34 meshes, 36 instances. The full staged renderer
passes its 16x16/4-spp canary, then renders 1080x1080/256 spp. The square
aspect ratio is intentional and matches the authored camera. Frame 1,
seed 0, adaptive sampling and denoising disabled match the original HIP run.

Artifacts are `monster1080.exr`, `monster1080.log` and
`monster1080-compare.json` in the evidence directory; the Cycles reference
is `/var/tmp/psycles-multiscene-52-Vw3BkE/monster/cycles1080/cycles.exr`.

| Pass | Relative RMSE | Mean luminance ratio | Invalid pixels |
| --- | --- | --- | --- |
| Combined | 0.03267304 | 1.00289128 | 0 |
| DiffCol | 0.00037594 | 0.99994030 | 0 |
| DiffDir | 0.00901832 | 1.00244746 | 0 |
| DiffInd | 0.14260939 | 1.00616614 | 0 |
| Emit | 0.00014055 | 0.99999762 | 0 |
| GlossCol | 0.00029270 | 0.99999231 | 0 |
| GlossDir | 0.00691334 | 0.99993624 | 0 |
| GlossInd | 0.13260235 | 1.00189496 | 0 |

This is not sample-identical transport and is not claimed as complete image
parity. The larger indirect differences remain to be investigated against
the original Cycles kernels. Psycles reports scene compilation 1.50004 s,
JIT 21.5446 s and sampling 15.5508 s. Cycles' 15.518 s is the complete
`bpy.ops.render` interval, not the same sampling interval, so these are not
a speedup comparison. This active-worktree run also includes pre-existing
closure/scheduler work and is not commit-only performance evidence.

Monster's main coroutine frame is 71 fields / 284 bytes versus Lone Monk's
55 fields / 220 bytes. Neither is a like-for-like comparison with Cycles'
complete wavefront state plus shadow/task storage. No frame-policy changes
are part of this checkpoint.

Classroom's native compiler/upload already passes after the preceding Sky
checkpoint, but its full pipeline still encounters legacy ObjectIndex
lowering. Barbershop still has missing source texture resources plus that
legacy dependency. Those failures are retained rather than substituting
materials or reporting the scenes as fully passing.

## Final validation and isolated checkpoint

The isolated application candidate at
`/var/tmp/psycles-map-range-candidate-6zPrSK/source` contains only this
checkpoint on `0590af6a`, excluding the inherited closure, path-tracer and SDK
work. Its 25 exact compiler images and 1,152 original-HIP runtime cases pass
on HIP, fallback and strict native Vulkan. The 16x16/4-spp whole-film HIP and
native XIR-to-SPIR-V Vulkan canaries also pass. All Vulkan runs set
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
`LUISA_VULKAN_DISABLE_DXC=1`; they are not DXC canaries.

The whole-film fallback canary initially exposed an independent Luisa ABI
defect: matrix-valued native-call temporaries had four-byte allocation
alignment while the embedded device library required sixteen. The original
311,835-instruction material kernel crashed on an aligned vector store.
Luisa commit `8e2b0ac78`, pushed to `origin/next`, fixes the generic temporary
alignment rule and adds a 53-contract permanent LLVM regression. With that
backend fix, the unchanged isolated application binary completes its original
whole-film fallback render. Evidence:
`/var/tmp/psycles-fallback-codegen-policy-io0puc/fixed-candidate-film.log`.
No application noinline, scratch-lifetime or arithmetic workaround was added.

The active worktree's all-thread build passes; its latest HIP suite is
164/164 and fallback suite 166/166. Core tests are 141/142 with only the
previously known source-size check failing on unrelated oversized files;
the existing Blender render-settings test was excluded from this core run.
The active-worktree counts include later dedicated SSS/geometry tests and are
not misrepresented as the isolated checkpoint's test inventory. The SDK
gitlink and unrelated dirty files are excluded from this commit.
