# Native Cycles volume SVM nodes

The native host compiler and Luisa interpreter now implement Cycles 5.2.1
Absorption/Scatter Volume, Volume Coefficients, and Principled Volume. This is
the original word-stream/stack/PC model, not the legacy VolumeInstruction
evaluator. Scene-level opcode usage still removes unused case labels before
recording device code. Fast math remains enabled.

This checkpoint does **not** yet replace the renderer's stacked-volume
consumer. That remaining consumer must share one closure budget across the
entire stack, retain Cycles' per-entry merge ordering, and only then copy the
active phase prefix. Per-material calls with freshly reset allocators would
not implement `volume_shader_eval`. Legacy SurfaceProgram removal and the
full multi-scene performance/parity campaign remain open.

## Reference and field mapping

Reference source: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`. The relevant original definitions
are `scene/shader_nodes.cpp`, `scene/shader_nodes.h`, `kernel/svm/closure.h`,
`kernel/closure/volume.h`, `kernel/svm/svm.h`, and `kernel/features.h`.

| Opcode | Typed payload after the opcode | State transition |
| --- | --- | --- |
| `NODE_CLOSURE_VOLUME` | 5 words | Density/object scale, absorption or phase allocation, extinction |
| `NODE_VOLUME_COEFFICIENTS` | 10 words | Signed absorption/emission coefficients, scatter allocation, shadow emission gate |
| `NODE_PRINCIPLED_VOLUME` | 18 words | Density/color/temperature attributes, HG scatter, absorption, emission and blackbody |

All payload types are the retained Cycles structs. Host serialization retains
linked versus literal input encoding, mix-weight stack offsets, padding and
input evaluation order. The three Principled string sockets become symbolic
contract properties (`DensityAttribute`, `ColorAttribute`,
`TemperatureAttribute`), never SVM stack entries. The importer now preserves
their authored strings. Defaults are `density`, empty, and `temperature`.
Attribute requests remain conditional on linked/positive Density and
Blackbody Intensity, and named IDs are allocated by the original compiler
order, independently of geometry attribute requests.

The volume node factory is separated from surface closures. Its `equals`
returns false, matching this version's `VolumeNode::equals`. Cycles' comment
explicitly says volume nodes could be deduplicated with care; this is current
compiler behavior, not a semantic prohibition on all possible deduplication.

The runtime preserves these distinctions:

- Extinction can change when no scatter closure can be allocated. Zero mix
  exits before density/attribute evaluation; zero density is not a substitute
  for that branch.
- Allocation uses the existing Cycles `bsdf_alloc` counterpart, including
  nonnegative scatter weights and the volume-specific cutoff behavior.
  HG/Draine/FF parameter inputs are evaluated only after successful allocation.
- Mie fitting precedes two independent allocation attempts, HG then Draine.
  Capacity one retains the first phase; the pair is not allocated atomically.
- World density is one. An unavailable object-density service does not
  silently give an object world semantics; the diagnostic evaluator rejects
  that unsupported service path.
- Extinction/emission flags control initialization versus accumulation.
  Coefficients and Principled emission are skipped for shadow visibility;
  Principled emission/blackbody do not depend on positive density.
- Unused opcode: no case. Disabled node-volume feature: empty case, no payload
  consumption. Enabled node feature in another shader domain, or with the
  global volume feature disabled: consume the typed payload, without volume
  state changes. These are different Cycles control-flow rules.

`ShaderData::closure[]` retains its 80-byte slot stride. Volume closures use
only a 32-byte prefix: weight at byte 0, type at 12, sample weight at 16,
HG/Draine g at 20, Draine alpha at 24, and FF coefficients at 20/24/28.
The common volume accessor deliberately does not read surface normal `N`:
its bytes overlap the phase parameters. No inactive phase field or trailing
padding is read to construct a surface-shaped temporary.

## Independent compiler fixtures

`tools/cycles_shader_probe/volume_closures.py` creates four Blender probes.
The observer-enabled executable is
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`.
The production HIP Blender does not implement `PSYCLES_CYCLES_SVM_DUMP`;
a successful render with that executable is not evidence of a word dump.

| Probe | Original global words / shaders | Tested local images |
| --- | --- | --- |
| `volume_coefficients_svm` | 286 / 12 | Five phases: 22 words each; linked/mixed: 54; repeated: 37 |
| `principled_volume_svm` | 115 / 6 | Default symbolic attributes: 30 words |
| `principled_volume_named_svm` | 115 / 6 | Three authored attribute names: 30 words |
| `principled_volume_linked_svm` | 144 / 6 | Dynamic color/density/blackbody and mix: 59 words |

`tests/test_cycles_svm_volume.cpp` imports the original exported materials and
compares all ten complete images. Only the three global jump targets are
relocated by `tools/extract_cycles_svm_shader.py`; expected payloads are not
rewritten to fit Psycles. JSON fixtures omit unrelated geometry and scene
resources, but retain the original camera, materials, images and node groups.
The test also checks volume flags, the graph closure count, symbolic attribute
requests and attribute-dependency metadata. Before the port, it failed on the
first Volume Coefficients node with `Cycles SVM node family is not migrated`.

Reproduction, repeated for each probe above:

```sh
blender --background --threads 32 --python-exit-code 1 \
  --python tools/create_cycles_shader_probe.py -- /var/tmp/probe.blend PROBE
blender /var/tmp/probe.blend --background --threads 32 --python-exit-code 1 \
  --python tools/export_psycles_scene.py -- /var/tmp/probe-export
PSYCLES_CYCLES_SVM_DUMP=/var/tmp/probe.svm52 trace-blender \
  /var/tmp/probe.blend --background --threads 32 --python-exit-code 1 \
  --python tools/render_cycles_golden.py -- /var/tmp/probe.exr 4 4 1 20903
python tools/extract_cycles_svm_shader.py /var/tmp/probe.svm52 5
```

The original Cycles CPU render here only triggers its host serialization. It
does not introduce a Psycles CPU reference renderer.

## Independent HIP state oracle and coroutine regression

`tools/cycles_svm_volume_oracle.hip` constructs 20 legal streams with original
Cycles typed structs and executes the original `svm_eval_nodes` on HIP. The
tool also emits device-side closure size/offsets. It does not implement an
alternative evaluator or manufacture expected results.

```sh
hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
  -I /path/to/blender-cycles/intern/cycles \
  tools/cycles_svm_volume_oracle.hip -o /var/tmp/volume-svm-oracle
/var/tmp/volume-svm-oracle
```

The output is retained byte-for-byte in `tests/data/cycles_svm_volume.txt`.
It contains 2,880 state rows: 20 streams, capacities 0/1/64, three shader
domains, and 16 object/visibility/initial-state combinations. Inputs cover
all five scatter phases, four Mie diameter intervals, negative/zero density,
zero/implicit mix, repeated closures, signed coefficients, existing emission
and extinction, mesh attributes, blackbody-off and negative temperature.

`tests/test_luisa_cycles_svm_volume.cpp` performs 7,680 comparisons:

- 5,760 through diagnostic and compiler-validated native SVM entrypoints.
- 960 with the global volume feature off, checking the no-op state against
  the original non-volume-domain rows and checking final PC/status.
- 960 after an actual wavefront suspend/resume, using the original volume
  rows. An eight-frame pool handles sixteen tasks, forcing frame reuse.

The coroutine test reads the active volume payload only after suspension.
It uses `stream << scheduler(...).dispatch(...)` and supplies no frame
exports, custom lifetime seeds, forced noinline, or Psycles-specific Luisa
handler. Scalar/vector default initialization is unchanged. The test keeps
the closure pool itself live intentionally: frame sizes for capacities 0/1/64
are 144/144/5184 bytes, not production renderer frame sizes or an optimization
claim. Status/PC, flags and allocator counts are exact; finite float results
use a native-fast-math tolerance, not slow bit-matching operations.

Before dispatch was connected, the first runtime row failed with
`invalid_node` and retained the initial extinction instead of the Cycles
result. The final native test passes on HIP, fallback and native-XIR Vulkan. The AST pruning
regression now checks 99 implemented opcode handlers and 25 whole-body feature
families, each enabled/disabled in all three domains. Nine executable nodes
remain unimplemented; NONE/PAD1 are the other two enum entries.

## Validation evidence

Evidence directory: `/var/tmp/psycles-native-volume-svm-06XnDX`.

- All-target Release build: 32 threads, passed; focused final test rebuilds
  also use 32 threads.
- Host CTest: 121/121, including adapter import tests. External Blender
  processes and the separate source-size policy are outside this selection.
- HIP CTest: 169/169. The final global-feature-off addition also passes in
  the direct 7,680-case HIP rerun.
- Fallback CTest: 171/171, including the final 7,680-case regression.
- Vulkan focused volume regression: 1/1 with
  `LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
  `LUISA_VULKAN_DISABLE_DXC=1` (`volume-native-vk.log`).
- Uncached HIP staged-wavefront scene canaries at 256 spp: Lone Monk
  1440x1080, 13.9261 s, 220-byte frame; Monster 1080x1080, 15.1739 s,
  284-byte frame. Both used native SVM and the direct-light queue. These are
  single-run integration canaries, not paired performance benchmarks, and
  do not validate a native stacked-volume consumer.

The scene logs are `monk-native-staged-hip.log` and
`monster-native-staged-hip.log`. Earlier logs in the same evidence directory
used the legacy default or omitted the direct-light queue; their different
frame sizes must not be interpreted as a native-frame regression. Monster
uses the freshly exported `monster-export` bundle: the older bundle has
ambiguous source shader identities and is correctly rejected.

Original fixture hashes (SHA-256):

```text
volume_coefficients_svm.svm52
35e438ef99b145fb3cef69d9fd2c0bda32d8b4a5f7331b69781079fd508be6b2
principled_volume_svm.svm52
8504bf46caa17729d304fc737c3ee149416173dc2321bd26a72dd4e2db665c93
principled_volume_named_svm.svm52
aacddea4c111691bc2caa29e3ccaa1605766ea8e9ce5dd1b5d303ff78658169b
principled_volume_linked_svm.svm52
52833dcd8dacdf6c72fd9623b45874b54555fb66a7e485f7cef4a551359f1fe2
cycles-volume-svm.txt
4b82195fd8af4045ec0ad82d5aa318eebb1b573bb022cc5f54ddf4a61170a7cd
```
