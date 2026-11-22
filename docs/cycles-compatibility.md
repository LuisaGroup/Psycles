# Cycles Compatibility Specification

Psycles targets semantic parity with **Blender Cycles 5.2.1 / 5.3.0** across shader compilation, closure evaluation, path transport, sampling, and linear render passes. All execution goes through native Cycles SVM — there is no legacy shader fallback.

For benchmark verification across all 6 backends (`hip`, `dx`, `vk-xir`, `vk-dxc`, `fallback`, `simd`) and 5 canonical scenes (`0` invalid pixels across all 30 runs), see [optimization.md](optimization.md).

---

## 1. Supported Execution Domains & Features

### Shader & Material Domains

- **Surface & Bump/Displacement:** Full native Cycles SVM `surface` and `displacement` execution (`TRUE`, `BUMP`, and `BOTH` methods with native displacement prepass and bump fallthrough).
- **World Background & Importance Baking:** Native SVM world shader evaluation and camera-conditioned environment importance map construction.
- **Lights & Emission:** Emissive triangles, analytic lamps (`point`, `spot`, `area`, `distant`/sun), MIS with LightTree or flat light distributions, and portal lights (`use_mis`, portal continuation, persistent `portal_depth`).
- **Volumes:** Homogeneous and heterogeneous SVM volumes, `Volume Absorption`, `Volume Scatter`, `Principled Volume`, ordered volume stacks (up to 8 merged phases), majorant prepass/hierarchy, and shadow volume transmittance.
- **Closures & Integrator Transport:**
  - Principled BSDF (metallic, transmission, coat, sheen, subsurface, thin-film iridescence tables, thin wall), Diffuse, Glossy/Conductor, Glass, Refraction, Translucent, Transparent, Emission, Background, Holdout (object & node), Hair (Chiang/Huang), Toon, Ashikhmin-Shirley, Ray Portal BSDF, Burley & Random-Walk BSSRDF.
  - Direct-light MIS, Russian roulette, sample clamping (`sample_clamp_direct` / `sample_clamp_indirect`), caustic visibility suppression (`caustics_reflective` / `caustics_refractive`), glossy filter (`filter_glossy`), shading/shadow terminator geometry offsets.

### Geometry & Textures

- Static triangle meshes with corner/point/face attributes, explicit tangent/normal spaces, and displaced mesh generation (`path_tracer_displacement_scene.cpp`).
- Static ribbon curves (packed segment indices + containing curve attributes).
- Procedural textures (Noise, Voronoi F1/Smooth F1/Distance-to-Edge/N-Sphere, Wave, Gradient, Checker, Brick, Magic, Gabor, White Noise), Nishita & Hosek-Wilkie sky nodes, IES profiles, color ramps, RGB/vector/float curves, bump/normal maps, packed/external images (including Cycles missing-image error color vs unassigned socket distinction).

### Passes, Sampling & EXR Output

- Exact **Tabulated Sobol** sequence keyed by full `SampleRange.total` and scene scrambling/seed contract.
- **15 common linear render passes** (46 channels total including Combined alpha) exported as single-part 32-bit float multilayer OpenEXR (`ViewLayer.<pass>.<component>`, `oiio:ColorSpace = lin_rec709_scene`) plus per-pass PFMs.

---

## 2. Specialization & Precision Contracts

1. **Immutable Compile-Time Specialization Only:**
   - Opcode switches, `ClosureUsage` bitmasks, `Stack` capacities, closure pool caps, attribute buffer residency, and shadow-terminator paths specialize strictly from compiled scene/session metadata at host recording time — never from scene names, heuristics, or warmup runs.
2. **Native Fast Math & Hardware Filtering:**
   - Production shaders compile with `enable_fast_math = true`. Nearest, linear, and 4-tap Cycles bicubic textures use hardware texture instructions; normalizations use native `rsqrt`.
   - CPU backends preserve IEEE edge-case predicates (`ISNAN`, `ISINF`, `FMOD`, `exp` overflow selects) by omitting unsafe `nnan`/`ninf` fast-math flags in LLVM.
3. **Explicit Diagnostics Over Silent Fallbacks:**
   - Unimplemented nodes, malformed graphs, missing attributes, or unsupported geometry produce explicit compiler/session diagnostics rather than silently rendering incorrect defaults.

---

## 3. Known Compatibility Gaps & Platform Caveats

- **Unimplemented SVM Opcodes (8 out of 110 tags in `NODE_NUM`):**
  - `NODE_RADIAL_TILING`
  - `NODE_BEVEL`
  - `NODE_AMBIENT_OCCLUSION`
  - `NODE_RAYCAST`
  - `NODE_AOV_START`, `NODE_AOV_COLOR`, `NODE_AOV_VALUE`
  - `NODE_SCENE_TIME`
  *(Additionally `NODE_NONE` and `NODE_PAD1` are non-executable markers; the other 100 opcodes have AST/device handlers.)*
- **Unadmitted Features:**
  - Open Shading Language (OSL) custom scripts
  - OpenVDB / sparse external volume grids (volumes currently admit procedural/SVM density fields)
  - Deforming / rigid object motion blur and non-ribbon curve shapes / point clouds

---

## 4. Compatibility Verification Policy

1. **External Cycles Oracles:** Node and render acceptance compares against official Blender Cycles outputs or versioned oracle fixtures (such as the Blender 4.5.10 baseline JSONs in `docs/cycles-*-4.5.10.json` and 5.2/5.3 binary/pass captures under `tests/data/`). All **421/421** CPU CTest cases pass on Windows (`LuisaCompute next@2beb5a275`).
2. **Pass-Level & Per-Path Differential Testing:** Full scenes are compared across all 15 linear EXR passes via [scene-benchmark.md](scene-benchmark.md); first-divergence path investigations use the 344-AOV per-path tracer in [cycles-path-trace.md](cycles-path-trace.md).
3. **Structural Determinism Within Hardware Precision:** RNG hashes, bounce counters, SVM cursor transitions, visibility masks, and object/primitive identities are checked structurally; floating-point continuous values allow documented fast-math tolerances without sacrificing GPU/CPU speed.
