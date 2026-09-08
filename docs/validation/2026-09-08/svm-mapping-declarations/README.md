# Native Mapping declarations

Psycles `61443f70` restores Mapping's four input sockets and its output to
the original Cycles POINT type. The contract's generic Blender VECTOR types
remain intact until native socket reconstruction. The compiler restores all
producer types before reconnecting consumers. No SVM ranking, node-ID patch,
device arithmetic, inlining policy or Luisa change is involved.

Original `scene/shader_nodes.cpp:1830-1847` declares Vector, Location,
Rotation, Scale and the output as POINT in every Mapping operation mode.
The old projection preserved VECTOR. Sharing a float source with a texture's
POINT input therefore created separate FLOAT->VECTOR and FLOAT->POINT
conversions, where original Cycles shares one FLOAT->POINT conversion.
Equivalent arithmetic is not equivalent conversion-node identity.

The smallest original graph has five authored nodes: Light Path Ray Depth
feeds Mapping.Vector and Gradient.Vector; Mapping feeds Diffuse.Color and
Gradient feeds Diffuse.Roughness. Psycles produces 47 words instead of the
original 44. A separate-source control passes. The wider matrix varies all
four Mapping inputs, Color/Fac source and shared/separate texture coordinates.
Five of eighteen original images fail before the repair; all eighteen pass
raw-exact afterwards. Expected words remain unchanged after the red run.

The complete original Barbershop dump still has 279 named images, 895,648
linked words, 33 float stack lanes and 12 closures. Its classification remains
115 raw-equal / 158 resource-field-only with unresolved bindings / six
different layouts. `extinguisher_copper` improves from 2,331 to 1,633 differing
words, and its first layout difference moves from word 1,062 to 5,393; this
does not resolve the material or the remaining six schedules.

| Complete Barbershop 2048x858 / 64 spp / seed 0 | Before | After |
| --- | ---: | ---: |
| Render wall seconds | 10.4395 | 10.5088 |
| Surface GPU seconds | 5.915631274 | 5.904181622 |
| Closest GPU seconds | 1.681565371 | 1.689932574 |
| Volume GPU seconds | 0.424453466 | 0.423224443 |
| Surface invocations | 332,307,894 | 332,307,894 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 |
| VGPR / SGPR / private bytes | 256 / 107 / 2,496 | 256 / 107 / 2,496 |

These sequential profiler runs have identical scene inputs and no overlapping
build, test or render. They show no established speedup. Stage-map, final LLVM
and ELF identify `kernel_9bbb1beb16d04044`; its full `.text` remains identical
to the previous checkpoint, SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`.
All 46 channels are finite and all 15 pass controls complete. Combined
relative RMSE versus before is 0.0000037841 and DiffInd 0.0000152451;
this is an intervention control, not original-Cycles image parity.

The 32-thread full rebuild completes 308 targets. Host passes 167/167
(1.32 s), HIP 182/182 (139.58 s), fallback 184/184 (89.49 s), followed by
strict native Vulkan 2/2 (22.03 s), 31 SPIR-V modules and no DXC/DXIL load.
Luisa `85e5300f1` is unchanged. The latest full-resolution 256-spp campaign
remains pinned to `cbb73185` in the [Math report](../svm-math-expansion/README.md);
no new multi-scene timing pairs are claimed here.

Evidence root `/var/tmp/psycles-mapping-declarations-2mxqRk` retains the
pre-fix formal analysis, v1/v2 original blends, HIP EXRs, exports, dumps,
red/build/green logs, complete original-scene dumps and profile `profile-ciK3u2`.
Generate graphs with `tools/create_cycles_mapping_declarations_probe.py`,
capture original HIP words with `PSYCLES_CYCLES_SVM_DUMP` and
`tools/render_cycles_golden.py`, then export with `tools/export_psycles_scene.py`.
`psycles.cycles_svm_mapping_declarations` reads the retained original fixture.
The sibling Math report supplies the fail-closed typed decoder.

Run this directory's `archive_results.py EVIDENCE results.json` to freeze
the raw/typed audit, source hashes, gates and correctly scoped profile in
[results.json](results.json). Original Cycles remains the only oracle;
no resource-ID normalization or CPU shader evaluator is added. Remaining
default-input phase differences are a separate investigation, not silently
folded into this declaration repair.
