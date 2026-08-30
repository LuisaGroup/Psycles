# Benchmark forward-emission path trace

This checkpoint isolates one apparent 25% forward-emission discrepancy in the
official Blender benchmark without changing renderer behavior. The reference
is Blender/Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`; both traces use HIP on the RX 9070
XT, full-film coordinate `(1150, 607)`, absolute Tabulated Sobol sample 508 of
1024, adaptive sampling off, and the fresh 5.2.1 scene export.

## Result

The production transport values agree:

| field | Cycles | Psycles |
|---|---:|---:|
| triangle selection PDF | 4.65398189e-5 | 4.65398225e-5 |
| total light PDF | 0.00196993630 | 0.00196991977 |
| previous BSDF PDF | 0.0898959637 | 0.0898968801 |
| forward MIS weight | 0.999520004 | 0.999520063 |
| emission RGB | (0.465249836, 0.542870045, 0.725488365) | identical |
| clamped contribution RGB | (0.593446195, 0.509159088, 0.668640137) | (0.593440711, 0.509150624, 0.668638527) |

Cycles and Psycles also build the same flat light measure: 178 emissive
triangles, total area 146.057266, and triangle distribution density
0.00342331477 after half of the distribution mass is assigned to triangles.
The two large bounce
objects each contribute about 72.9121 area and `LGT_meshlight_01` contributes
0.233093262.

The previously observed Cycles Combined value
`(0.445084631, 0.381869316, 0.501480103)` is not a path contribution. A device
probe read the full clamped value immediately before and after
`film_write_combined_pass`; the host pass accessor used one sample, exposure
one, and scale one; and `BlenderOutputDriver` received the full value. Blender
then stored 0.75 times that value for both 1x1 and 3x3 border crops. Rendering
the same absolute sample over the complete 2048x858 frame stored
`(0.593446195, 0.509159088, 0.668640137)` exactly. This formally rules out
surface SVM emission, light selection, forward MIS, sample clamp, and Cycles
HIP as causes of the apparent 25% discrepancy.

Schema version 3 appends four forward-emission records per event after every
version-2 index: evaluated emission, policy/selection measure, BSDF/light
measure and MIS weight, and post-clamp contribution. Unit tests lock the
append-only layout, multipart EXR decoding, comparison policy, and generated
Luisa header. The record is total over every ordinary surface event: a
non-emissive event has zero emission/PDF/contribution and the identity MIS
weight, while an emissive event overwrites that identity record. Cropped
ordinary passes are explicitly excluded from the
per-path oracle contract; only `PsyTraceNNN` fields are compared.
