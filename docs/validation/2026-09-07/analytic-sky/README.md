# Cycles 5.2.1 analytic Sky: native words and evaluation

This checkpoint restores the two analytic branches of `NODE_TEX_SKY`.
Classroom exposed the missing Hosek-Wilkie compiler node. Auditing the same
import path found that Preetham was silently replaced with Nishita.

## Semantic cause and mapping

The Sky opcode does not have a single fixed-size payload. Preetham (type 0)
and Hosek-Wilkie (type 1) consume the same **32 float words**: phi, theta,
three radiances, and three nine-coefficient arrays. Nishita consumes a
different **12-word typed payload**, including an integer texture handle.
Rejecting the analytic types or treating their payload as Nishita changes
both radiance and the interpreter PC. A graph-level model substitution is
also invalid even if the substituted graph renders a plausible sky.

The implementation follows these Cycles sources at
`cb168525138fecc792cc393f94afc39582b0103c`:

- `intern/cycles/scene/shader_nodes.cpp`: `sky_spherical_coordinates`,
  `sky_texture_precompute_preetham`, `sky_texture_precompute_hosek`, and
  `SkyTextureNode::compile`.
- `intern/cycles/kernel/svm/sky.h`: typed payload selection, both analytic
  radiance functions, horizon clamp, and RGB conversion.
- `intern/cycles/kernel/svm/svm.h`: the original interpreter and feature
  masks used by the HIP oracle.

Host spherical conversion remains float `acos(z), atan2(x,y)` without an
extra normalization. Only Hosek clamps turbidity and solar theta and calls
the existing vendored Blender sky library in double, as Cycles does.
Preetham retains its own float precompute and zeros the four unused
coefficients in each channel. At runtime, the angular separation from the
sun is computed **before** clamping the query theta to the horizon.
Preetham uses xyY -> XYZ -> clamped RGB; Hosek uses XYZ -> clamped RGB and
the original `2*pi/683` scale. The existing Nishita branch is retained.

No new SVM opcode, stack model, noinline annotation, or software floating
point emulation is introduced. Ordinary scalar/vector initialization is
unchanged. All branches remain inside the same PC-loop node dispatch.

## Independent oracles

Evidence directory: `/var/tmp/psycles-native-analytic-sky-0Nncdf`.
The existing `hosek_wilkie_diffuse_transport` Blender probe supplies the
Hosek world. The Preetham probe changes only the Sky node's `sky_type` to
`PREETHAM`; it does not replace shader connections, sun direction, or
atmospheric parameters.

Bytecode comes from the observer-enabled **original Cycles** executable
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`.
The production HIP renderer at `blender-install-5.2-hiprt/blender` does not
contain that dump observer. A tiny original Cycles CPU render was used
only to trigger serialization; there is no Psycles CPU reference renderer.

Each dump contains 152 words and six shaders. Shader 3, `default_background`,
is extracted with `tools/extract_cycles_svm_shader.py`; only its three jump
targets are relocated. The resulting image has 52 words and jump
`(1,4,50,51)`. Words 10..41 are the analytic payload. Compiler regression
compares the **entire imported shader image**, with at most 1 ULP permitted
only in that float payload; opcodes, offsets, and all other words are exact.

Fixture hashes (SHA-256):

| Fixture | Hash |
| --- | --- |
| Hosek `.svm52` | `bdccd69ef6eacf656ebb8fc04093b30b0a45ffadd926bbaddd4742368fbecb9b` |
| Preetham `.svm52` | `249a3310a541563dfa6389906e309d830c35dcefa45bc2474f4fcc38435e428e` |
| `cycles_hosek_sky_words.txt` | `f6e1ff24079fbbbace34d67232bfeb2eea86fc213d4232de76f9a20d53cc4f89` |
| `cycles_preetham_sky_words.txt` | `0ed8cf5840df5b1990020ab4ae6c1fb458ba81b8af0bb72ad7497738d744c263` |
| `cycles_analytic_sky_runtime.txt` | `27f7a9812b56165e4a6a8e2f1d1150ac873428ed9b40da2ffe1d2398bf445eab` |
| `cycles_analytic_sky_rounding.txt` | `d7994f06fe982a3dbc42cc5d2c79669fba5ce875f3d65b72307193df56292f19` |

The runtime fixture is produced by `tools/cycles_svm_analytic_sky_oracle.hip`
executing the original `svm_eval_nodes` on those independently dumped words.
The only shared data are inputs (query directions, color-space rows, words).
Expected radiance is never computed by a copied host formula or Psycles.

```sh
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC \
  -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_analytic_sky_oracle.hip -o /path/to/oracle-hip
/path/to/oracle-hip tests/data/cycles_hosek_sky_words.txt \
  tests/data/cycles_preetham_sky_words.txt
```

`-ffast-math` is intentional: both Cycles' HIP CMake rule and runtime HIP
compiler options enable it. A diagnostic build without that option was
also retained; the largest `abs(fast-strict)/max(1,abs(fast))` difference
over these queries is `5.53e-7`, not a structural discrepancy.

The permanent runtime regression checks 16 directions x two models x
three shader domains: poles, horizon directions, both sides of the horizon,
upper/lower diagonals, and sun/opposite-sun directions. It checks finite
RGB against original HIP results, unchanged emission flags, `NODE_END`
status, and final PC 50/51/52 for surface/volume/displacement respectively.
The non-surface images preserve the initial accumulator through their
empty programs. The flags, status and PC must agree exactly.

### One-ULP angular conditioning, without a runtime workaround

The initial fallback run failed at the Preetham sun direction (red channel
`1.99985` vs HIP `2.00115`). The reduced device programs `angle_luisa.cpp`
and `angle_cycles.hip` in the evidence directory observe identical theta
and phi, but different cosine of angular separation:

| Backend | cosine bits | angle in radians |
| --- | --- | --- |
| Original Cycles HIP | `0x3f800000` | `0` |
| Luisa HIP | `0x3f800000` | `0` |
| Luisa fallback | `0x3f7fffff` | `0.000345266977` |

This is the conditioning of `acos`, whose derivative is unbounded at
`+/-1`, not a PC, payload, or formula mismatch. At the opposite-sun query,
the same adjacent-float issue occurs at -1. No production code is changed
to force cross-device agreement.

The central fixture remains the **unmodified** Cycles HIP result. Two
additional HIP runs compile the same oracle with
`-DPSYCLES_SKY_COSINE_ULP=-1` and `=1`. A narrowly scoped observer wrapper
only substitutes the adjacent representable argument to `safe_acosf` in
`sky_angle_between`; the original SVM, Sky radiance functions, coefficients,
and color conversion then propagate that perturbation. All dependencies
are included before the macro, so other math functions are not rewritten.
Both runs visit every model/query/domain, without backend-specific or
case-specific thresholds.

`cycles_analytic_sky_rounding.txt` stores per-channel min/max of the central
and these two externally evaluated outputs, with columns
`model query domain low_R low_G low_B high_R high_G high_B`. Tests check
against that envelope plus the unchanged `3e-5 * max(1,abs(ref))` arithmetic
tolerance. The central result must itself lie in its envelope. For the
Preetham sun, the minus-one-ULP oracle returns `1.9998492` red, explaining
the original fallback result without relaxing unrelated cases or changing
their expected radiance.

## Regression evidence and scope

- `red.log`: the Hosek world is rejected as an unmigrated native node.
- `preetham-red.log`: the complete Preetham image has the wrong length
  because of the importer substitution.
- `runtime-red.log`: the new interpreter regression fails its exact
  flag/status/PC check against the isolated previously published runtime.
- `analytic-green.log`: both complete compiler images pass.
- `runtime-hip.log`: all 96 query/domain cases pass before the conditioning
  allowance. `runtime-fallback.log` retains the initial sun-direction failure.
- `rounding-hip.log`, `rounding-fallback.log`: all 96 query/domain cases pass
  with the independently propagated one-ULP envelope.

The optional original-scene mode of the scene-compilation test now checks
native compile and device upload directly. It no longer applies the
Camera Data fixture's material names and identities to Classroom or other
unrelated exports. Its normal CTest mode retains all Camera Data assertions.

## Complete validation and scene canaries

- Full 32-thread build: pass (`final-build.log`).
- HIP suite: **161/161** (`final-hip.log`).
- Fallback suite: **163/163** (`final-fallback.log`). The first suite run's
  sun-direction failure is retained in `full-fallback.log`.
- Host suite excluding backend variants and the previously failing Blender
  render-settings test: **137/138**. The only failure is the pre-existing
  source-size gate: `cycles_svm_nodes.cpp`, `test_cycles_svm_compiler.cpp`,
  `test_luisa_compact_surface_preparation.cpp`, `test_luisa_cycles_svm.cpp`.
  None is changed by this checkpoint.
- Isolated Psycles candidate at
  `/var/tmp/psycles-analytic-sky-candidate-la0tLf`: published `7f7fa26c`
  plus only the staged Sky changes, without the pending closure/pool,
  path-tracer, or gitlink changes. It uses published SDK headers from
  `88b5a03ad` and the active SDK shared libraries, not a claim that the
  entire SDK dirty worktree has been integrated. Compiler regression and
  the 96-case runtime regression pass HIP, fallback, and strict Vulkan.
- Strict Vulkan uses `LUISA_VULKAN_USE_XIR=1`,
  `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and `LUISA_VULKAN_DISABLE_DXC=1`.
  `LD_DEBUG=libs` confirms no DXC/DXIL library load. The isolated runtime
  test compiles three native SPIR-V kernels; the full staged-coroutine
  sky canary compiles 25 (`isolated-rounding-vk.log`,
  `isolated-vk-canary.log`). Fallback and Vulkan full-scene canaries render
  16x16 at 4 spp; they are functional checks, not speed measurements.

The full Hosek diffuse-transport scene is also rendered by original Cycles
HIP and the isolated Psycles staged-coroutine renderer. Its authored square
aspect ratio is preserved, including the 1080-pixel-high / 256-spp run.

| Resolution / spp | Combined relative RMSE | Combined luminance ratio | DiffDir relative RMSE |
| --- | --- | --- | --- |
| 128x128 / 256 | 0.0144382 | 1.0015648 | 0.0114548 |
| 1080x1080 / 256 | 0.0144484 | 1.0013569 | 0.0114535 |
| 64x64 / 4096 | 0.00131637 | 1.0000321 | 0.00126921 |

All four compared passes (Combined, DiffCol, DiffDir, DiffInd) contain zero
invalid pixels. DiffCol is exact and DiffInd is zero in both renderers.
The 256-spp runs do **not** meet the probe runner's stricter 0.05% mean-ratio
gate; the 4096-spp run does. The convergence study does not establish
sample-by-sample transport equivalence, nor a performance gain. Forward
environment evaluation and its sampling distribution still have legacy
dependencies outside this node-family checkpoint.

Other original scenes were freshly exported and checked under
`/var/tmp/psycles-multiscene-52-Vw3BkE`:

- **Classroom**: native compilation plus device upload now passes, 91
  shaders / 68,180 words. Full rendering remains blocked by the outer
  legacy material compiler's unsupported Object Index output.
- **Barbershop**: the next native blocker is unavailable image resource 0.
  The import also reports source files already missing in original Cycles:
  `guilder_ornament.png` and `generic_scratches.png`. No texture or shader
  replacement was made. The outer Object Index dependency also remains.
- **Monster under the Bed**: fresh export has 34 meshes, 36 instances,
  19 materials, and 22 images. Native shader 5 exposes the unmigrated
  Map Range family. An original Cycles `.svm52` dump is retained for the
  next family regression.
- **Preetham full-scene probe**: the native compiler and interpreter now
  work, but the outer legacy world compiler rejects the newly preserved
  Preetham type. The former silent Nishita substitution is deliberately
  not restored to conceal that separate dependency.

Compiling and uploading an entire native SVM scene is not the same as
completing the renderer's remaining legacy dependency migration. The
overall multi-scene correctness and Cycles performance-parity goal remains
unfinished.
