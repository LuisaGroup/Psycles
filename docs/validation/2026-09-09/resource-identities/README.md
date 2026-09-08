# Barbershop SVM resource identities

The remaining 164 resource-field-only shader differences are resolved:
all **279 used shader images** have the same typed instruction layout, stack
addresses, jump/domain boundaries, non-resource payload and bound identities.
The audit checks **600 image references and 795 attribute references**, including
755 numerically equal IDs. There are zero mismatches. No SVM words are rewritten
or normalized. This does **not** establish complete shader or render parity.

Psycles compilation is `076cf9f5` (render implementation `ffb3f7f2`), Luisa
`da8fff856`. The fresh original Cycles reference is
`cb168525138fecc792cc393f94afc39582b0103c`. Full evidence and hashes are in
[results.json](results.json); replay uses [audit_bindings.py](audit_bindings.py)
and [archive_results.py](archive_results.py).

## Observation and proof boundary

Both original and actual compilers allocate 174 image handles and 21 named
attributes. Allocation order differs. Comparing different integers cannot
prove a wrong binding; comparing equal integers cannot prove a correct one.

The original observer records ImageManager's actual pool after its existing
image tasks and descriptor copies finish. It records assigned IDs, loader
names, native parameters, finalized metadata and UDIM handles. Attribute
insertion records the native registry under its existing lock. The hooks do
not load, finalize, assign, sample or convert resources. The SVM word dump and
both registries come from the **same original HIP session**, avoiding races
between unrelated captures' ID-assignment orders.

The production Psycles table compiler supplies the actual words, image
handles and named attributes. Its imported image descriptions are captured in
an optional sidecar from the existing
[scene compiler diagnostic](../../2026-09-08/lamp-routing-and-surface/dump_scene_svm.cpp).
The stdout image is byte-identical to the earlier production dump
(`1d179d2dad2cc91237f4301a1be6d0f1dd4357dcb817a476cecd3578fea03d10`).
No alternate shader evaluator or compiler is used.

All 197 exported image names and all 174 native loader names are unique in
this scene. The join checks width/height, filtering, extension, alpha and
declared/final color spaces. Two assigned-but-failed images retain zero-sized
native metadata and the actual missing-image state. Empty native requested
color space can be resolved by native metadata queries; it is not assumed to
mean data/linear pixels. All 21 named attribute strings agree. Standard
attribute IDs remain the pinned native ABI rather than name-remapped values.

Every declared resource-reference word is checked, not just differing words.
Every other word remains exact. Original ShaderJump global-to-local
relocation is the only transformation. Unknown names, duplicate identities,
unsupported animated/tiled/UDIM/sky inputs and inconsistent metadata fail
closed. The audit's admitted domain does not imply general UDIM or motion
support. This proves binding identity, **not** equality of decoded image
pixels, attribute values, shader-state transitions or render outputs.

## Reproduction and restoration

Evidence directory: `/var/tmp/psycles-resource-identities-kS7pwB`.
The native hooks and header are retained in
[cycles_binding_oracle.patch](../../../../tools/cycles_binding_oracle.patch) and
[cycles_binding_oracle.h](../../../../tools/cycles_binding_oracle.h).
In an isolated checkout of the pinned original, place the header at
`intern/cycles/scene/psycles_binding_oracle.h` and apply the two-file patch.
Use the existing version-pinned SVM dump observer as well. Build with all
32 threads. The captured command runs the original Barbershop `.blend` via
`tools/render_cycles_golden.py` at 2048x858/1 spp on HIP, with:

```text
BLENDER_SYSTEM_RESOURCES=/home/mike/Projects/blender-install-psycles-trace-5.2/5.2
PSYCLES_CYCLES_SVM_DUMP=<evidence>/barbershop-original.svm52
PSYCLES_CYCLES_IMAGE_BINDING_DUMP=<evidence>/barbershop-images.bin
PSYCLES_CYCLES_ATTRIBUTE_BINDING_DUMP=<evidence>/barbershop-attributes.bin
```

These are child-process variables, not system configuration. Two initial
build-tree launches failed before rendering because Blender's resource base
was missing; those logs are retained. Supplying the complete resource base
fixed startup. The diagnostic render is **not a benchmark**.

The instrumented binary is retained separately as `blender-resource-observer`.
The temporary image/shader hooks and header were then removed and the original
build restored successfully with 32 threads. SHA-256 checks confirm that all
four touched/observed original source files match their pre-probe bytes.
The preexisting dirty `scene/light.cpp` and `scene/svm.cpp` remain untouched.
Neither installed Blender binary was overwritten.

Build the host diagnostic using the evidence directory's `CMakeLists.txt` and
32 threads. Run it with the unchanged control bundle and sidecar path:

```bash
<build>/dump_scene_svm <barbershop-bundle> <evidence>/psycles-bindings.bin \
  > <evidence>/barbershop-actual.svm52
```

Decode the original registries with
[decode_cycles_bindings.py](../../../../tools/decode_cycles_bindings.py).
Run the existing raw/typed image auditors, then `audit_bindings.py`; its
positional inputs and hashed paths are recorded in `results.json`.

## Permanent checks and outcome

[The regression](../../../../tests/test_cycles_binding_oracle.py) retains the
exact captured registry bytes and a complete original/actual 1,212-word
`shaving_brush_wood` image in a provenance-bearing fixture. Eight test methods
cover positive decoding and identity checks, malformed bounds/version/flags,
non-finite frames, unknown/ambiguous resources, mismatched image parameters,
literal-word changes and numerically equal but incorrectly bound IDs.

The all-thread build and host suite pass **173/173**. Focused image-binding and
missing-image tests pass **2/2 each on HIP and fallback**. Strict native Vulkan
passes **2/2**, with two observed native SPIR-V compilations and no DXC/DXIL
library load. The larger 184-HIP/186-fallback campaign remains the preceding
unchanged-renderer checkpoint; it was not rerun for metadata-only tools.

No rendering/compiler optimization is introduced by this checkpoint, and no
new speedup is claimed. The latest Barbershop 256-spp median remains 39.9800 s
versus the retained Cycles 25.3775 s. The major surface-cost gap and residual
DiffInd/shadow differences remain open; resource renumbering does not explain
them. Continue with runtime state and final transitive machine-code analysis.
