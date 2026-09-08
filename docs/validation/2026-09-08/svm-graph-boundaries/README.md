# SVM graph boundaries and redundant texture evaluation

This is the frozen `5096a41f` checkpoint. The later
[native socket/declaration repair](../svm-socket-declarations/README.md)
supersedes its current word-discrepancy counts, source-size failures and
follow-up timings. Its original probes and measurements remain unchanged.

## Technical summary

Psycles `5096a41f` (published to `origin/main`), with unchanged published
Luisa `85e5300f1`, repairs three host graph differences against Blender Cycles
5.2.1: automatic bump rewrites no longer redirect surface-owned aliases,
BUMP materials retain their displacement entry, and legacy output hints no
longer split one procedural texture into separate Color/Factor evaluations.
The full Barbershop used-shader comparison now has 274 equal-length images
out of 279, up from 215. Its statically analyzed stack bound is 33 floats,
down from 36. These are structural results, not an assumed speedup.

No Luisa compiler policy, device arithmetic, texture filtering, RNG dimension,
coroutine boundary, register cap or inlining directive changes in this work.
The prior full-scene inlining intervention remains rejected: retaining more
callable boundaries made surface time worse, not better.

## Original words expose three distinct graph defects

The comparison population is the production collector's 280 compile units,
279 uniquely named used shaders and 490 dense indices. The original Cycles
dump also includes unused shaders (541 indices). Compare corresponding local
images, not global table sizes. Only the three ShaderJump addresses are
relocated; resource IDs, constants, padding, stack addresses and all other
words are retained. Equal lengths do not establish equal semantics.

| Used-shader image comparison | Before this repair | After |
| --- | ---: | ---: |
| Entire raw image equal | 100 | 112 |
| Same length, different raw words | 115 | 162 |
| Different length | 64 | 5 |
| Static stack / closure capacity | 36 floats / 12 | 33 floats / 12 |

The [per-shader audit](scene-words.json) retains all domain counts, original
jumps and first mismatches. Resource-manager numbering and small host
constant differences remain unclassified raw differences, not automatically
bugs. Global Psycles storage grows from 815,999 to 895,618 words because
previously missing displacement entries are restored; this is not additional
surface execution. Per-entry static specialization remains enabled.

### Automatic bump rewrote a shared conversion, not just its own edge

Let `N` be the original Geometry.Normal, `C` a NORMAL-to-VECTOR conversion,
`D` the automatic displacement dot products and `S` a surface consumer.
Before refinement, cleanup may legally share `C(N)` between `D` and `S`.
Original `ShaderGraph::bump_from_displacement()` then creates a new Geometry
node `Nb` and connects only the three new dots and Bump.Normal to it.
The required invariant is that every pre-existing consumer outside the bump
reconstruction boundary keeps its producer.

Psycles instead disconnected `C`'s input and connected `Nb`: `S` now depended
on the newly created bump basis too. In `black_metal.001`, the implicit BSDF
normal still used `N`, so surface emitted two Geometry.Normal records.
The fix reconnects each dot's Vector2 input with its own original-style
autoconversion. It never mutates the shared conversion's input. This is edge
ownership restoration, not an extra CSE pass after bump finalization.

The original sources are `scene/shader_graph.cpp` (`simplify`, `finalize`,
`bump_from_displacement`, `connect`) and `scene/shader_nodes.cpp` (SetNormal).
The original reference order is cleanup, authored bump refinement, automatic
displacement bump, closure transformation and domain compilation.

### BUMP policy must not erase the displacement graph domain

Original `SVMCompiler::compile()` emits surface, volume and displacement
entries. `compile_type(SHADER_TYPE_DISPLACEMENT)` reads the retained
Output.Displacement link even for BUMP-only materials. Geometry evaluation
is a separate material-policy decision.

The importer had retained this root only for DISPLACEMENT/BOTH. It now keeps
the authored displacement root for all three valid policies. Mesh planning
and private displacement-prepass admission already require
`uses_true_displacement`, so BUMP-only geometry remains unevaluated. Existing
import tests now check both retained roots and the unchanged BUMP policy.
Original `kernel/features.h` explicitly aliases the BUMP node mask to the
DISPLACEMENT mask; using that common mask is not a discrepancy.

### Legacy output hints prevented original texture deduplication

The contract importer creates per-output projections with a `NeedsColor`
property. That property belongs to the old evaluator; none of the original
Noise, White Noise, Magic or Wave node declarations contains it. Retaining
it in `GraphNode::equals` makes otherwise identical Color and Factor
projections unequal. Both survive cleanup and execute the texture twice.

Native projection now removes that non-Cycles property, like the already
removed `NormalLinked` hint. It retains every actual node parameter and uses
the original output links to determine valid stack outputs. The existing
Cycles deduplication phase then reconstructs one multi-output node. No
device-side texture or sampler optimization is involved.

This removes the extra Wave invocation in `pages_mat` and the three extra
mapping/noise sequences in each wall-tile bump graph. All 48 three-word
surface excesses are gone too. Remaining length discrepancies are `bricks`
(+6) and four damaged-label shaders (-9 each); their conversion boundaries
are follow-up work, not normalized away by the comparator.

## Twenty-four original-Cycles graphs constrain the fixes

The two checked-in fixtures are raw exported graphs plus exact word images
from the observer-only Cycles build `cb168525138f` (5.2.1 LTS). HIP renders
the original probe scenes; the fixture extractor only copies material/group
payloads and relocates ShaderJump. It does not generate expected words.

| Probe family | Original cases | Before repair | After |
| --- | ---: | --- | --- |
| Surface aliases with NONE/BUMP/BOTH displacement | 12 | 7 fail | 12 exact |
| Noise/White Noise/Magic/Wave multi-output, including authored bump | 12 | 8 fail | 12 exact |

The alias-only intermediate repair leaves precisely the four missing BUMP
displacement tails failing. Restoring the domain makes all 12 images exact.
For textures, the four Color-only controls already pass before the fix;
each dual-surface and dual-bump case fails before it. All 24 tests compare
every word with no floating-point tolerance. The separate earlier Vector
Math literal tolerance is not expanded.

Reproduction uses `tools/create_cycles_bump_alias_probe.py` and
`tools/create_cycles_texture_outputs_probe.py`, then the original observer
with `PSYCLES_CYCLES_SVM_DUMP`, `tools/render_cycles_golden.py`, HIP and 32
threads. Export with `tools/export_psycles_scene.py`; retain fixtures with
the preceding report's `extract_hidden_socket_fixture.py` using the
`cycles_bump_alias` / `cycles_texture_outputs` stems and `--count 12`.
The full Barbershop graph is recompiled after both repairs, not just the
reduced probes.

## Smaller graph state does not change this surface machine code

The full-scene 64-spp profiler control uses identical exported scene bytes,
seed 0 and all 15 passes / 46 finite channels. It compares the prior published
host-folding checkpoint with this repair, without concurrent builds or other
agent renders. These timings are sums of measured kernel intervals, not
end-to-end paired Cycles measurements.

| Stage / measure | Before | After |
| --- | ---: | ---: |
| Surface GPU seconds | 5.867683420 | 6.033289805 |
| Closest-intersection GPU seconds | 1.672008508 | 1.699354368 |
| Volume GPU seconds | 0.419151551 | 0.430163245 |
| Surface invocations | 332,307,894 | 332,307,894 |
| Surface VGPR / SGPR | 256 / 107 | 256 / 107 |
| Surface fixed private bytes | 2,496 | 2,496 |

This run does not show a speedup. All three measured stages are slower than
the retained profile; one temporal follow-up is insufficient to attribute
that movement to a particular change. The known surface symbol changed from
`kernel_fe65567e53db5725` to `kernel_9bbb1beb16d04044`, but extraction of their
complete ELF `.text` sections produces identical bytes and SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
The pre-optimization LLVM stack declarations really are `[36 x float]` and
`[33 x float]`; the final code/resources do not improve from that reduction.

The main function still has 185,614 static instructions, 1,469 scratch-load
sites, 610 scratch-store sites and 49 call sites. The only outlined functions
remain 1D noise, 4D signed noise and OCML tangent. Static sites are not dynamic
memory-traffic or stall measurements, and the GPU does not expose the needed
PC sampling here. The original Cycles selected ordinary surface kernel's
2.962829 s retained GPU total remains well below Psycles, but that selected
main function's ISA size must not be confused with a transitive total that
includes its outlined BSDF/noise functions.

The repair changes the interpreted program data and removes repeated work
inside a shader invocation. It does not remove surface invocations, alter
coroutine scheduling or change the broad interpreter codegen cost. Control
image relative RMSE against the preceding Psycles render is 0.00001872 for
Combined and 0.00005846 for DiffInd; this is an intervention control, not the
original-Cycles image-parity metric.

Static opcode pruning is still active. Removing repeated instructions from
some material programs does not remove an opcode case still required by
another used material. A smaller program-data stream and a smaller compiled
interpreter are distinct outcomes; the latter did not occur here.

## Six full-resolution 256-spp follow-ups

All six runs use frozen `5096a41f` binaries, main shader caching disabled,
native fast math, unchanged authored seeds, the earlier geometry/image bytes
and fresh socket metadata. No profiler, build, CPU-heavy test or other agent
render overlaps them. The reference is the retained equal-pass Cycles
campaign, not six freshly paired Cycles runs. Both sides have 15 passes and
46 channels. Psycles channels are all finite in all six runs.

| Scene / run | Extent / seed | Render seconds | Session init seconds | Frame bytes |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 40.8982 | 29.8318 | 416 |
| Lone Monk 1 | 1440x1080 / 0 | 13.6448 | 22.2264 | 220 |
| Monster 1 | 1080x1080 / 0 | 15.0251 | 26.2560 | 280 |
| Classroom 1 | 1920x1080 / 1 | 18.5797 | 20.9987 | 260 |
| Barbershop 2 | 2048x858 / 0 | 41.0184 | 29.0572 | 416 |
| Barbershop 3 | 2048x858 / 0 | 40.2199 | 27.0611 | 416 |

Barbershop's median is 40.8982 s, 1.49% above the previous 40.2962 s;
the observed interval is 40.2199–41.0184 s. There is no demonstrated
speedup. Its ratio to the retained Cycles median of 25.3775 s is 1.6116.
The other scenes have only one new follow-up each, not new repeated estimates.
Session initialization is the CLI's `shader_jit_seconds`: JIT plus setup and
baking, not pure compiler time. The separate 64-spp profile initialized in
88.5084 s with IR/profiler output and ran before these canaries, warming
downstream caches. The 29.8318 s first listed initialization must not be
described as a cold-JIT improvement.

| First-run image comparison | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: |
| Lone Monk | 0.01240371 | 0.12881677 |
| Monster | 0.00547872 | 0.02552497 |
| Classroom | 0.00353320 | 0.17820333 |
| Barbershop | 0.01052113 | 0.07126167 |

These residuals remain unresolved. Original Cycles Classroom has 25 invalid
DiffDir and 27 invalid GlossDir pixels; the comparator explicitly excludes
the invalid union for those passes. All other metrics retain their reference
signal scale and absolute errors in [results.json](results.json). All four
first-run Combined triptychs were inspected at resized viewing resolution,
with no obvious new layout/material regression; that is not pixel parity.

## Verification and reproducibility

| Gate | Result | Scope |
| --- | --- | --- |
| Full build | Passed | All 32 hardware threads |
| HIP | 182/182 | Complete registered suite, run before fallback |
| Fallback | 184/184 | Complete parallel suite |
| Host | 160/161 | Only the existing source-size gate fails |
| Strict native Vulkan | 2/2 | Lamp routing and bump state, native SPIR-V, no DXC/DXIL load |
| Luisa child | 155/155 retained | Unchanged published `85e5300f1`; not rerun in this checkpoint |

The four unwaived size violations are `cycles_svm_nodes.cpp` (2026 lines),
`test_cycles_svm_compiler.cpp` (2088),
`test_luisa_compact_surface_preparation.cpp` (2114), and
`test_luisa_cycles_svm.cpp` (2038). The full host suite is not claimed green.
The new 24 exact original word images join the preceding 81 hidden-input
and Vector Math images. Native Vulkan sets all three required flags:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
`LUISA_VULKAN_DISABLE_DXC=1`; the retained loader trace contains no DXC/DXIL load.

The evidence root is `/var/tmp/psycles-svm-surface-tail-AWvSlZ`.
`archive_results.py` validates the complete six-run order, binary hashes,
pass/finite contracts, original red/green probes and complete gate totals
before writing [results.json](results.json). That archive retains original
source/image hashes, exact subprocess commands, full per-pass measurements,
profile/code-object identities and the `.text` comparison. Regenerate it with:

```bash
python docs/validation/2026-09-08/svm-graph-boundaries/archive_results.py \
  /var/tmp/psycles-svm-surface-tail-AWvSlZ \
  docs/validation/2026-09-08/svm-graph-boundaries/results.json
```

## Remaining questions

The five unequal-length images still require source-level conversion
analysis. Matching all lengths will not by itself resolve resource identity,
payload, scheduling or path-state differences. The remaining surface-stage
performance gap and DiffInd residual must continue to be measured with
original Cycles, keeping fast math and native texture sampling enabled.
There is no new CPU reference renderer, no bit-matching device fallback and
no claim that the entire Cycles compatibility goal is complete.
