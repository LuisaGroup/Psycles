"""Diagnostic scene copy, not an importer/renderer workaround.

Run with the original Lone Monk .blend loaded, passing a new .blend path
after --. Never overwrite the source scene. Only exact duplicate leaf faces
are removed; the checks intentionally reject any other scene revision.
"""
from pathlib import Path
import sys
import bpy
import bmesh

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
assert output != Path(bpy.data.filepath).resolve()
obj = bpy.data.objects["leaf.001"]
mesh = obj.data
assert len(mesh.vertices) == 20 and len(mesh.polygons) == 8
assert not obj.modifiers
for i in range(10):
    assert tuple(mesh.vertices[i].co) == tuple(mesh.vertices[i + 10].co)
for i in range(4):
    a, b = mesh.polygons[i], mesh.polygons[i + 4]
    assert tuple(a.vertices) == tuple(v - 10 for v in b.vertices)
    assert a.material_index == b.material_index and a.use_smooth == b.use_smooth
    for layer in mesh.uv_layers:
        for la, lb in zip(a.loop_indices, b.loop_indices):
            assert tuple(layer.data[la].uv) == tuple(layer.data[lb].uv)
bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=list(bm.faces)[4:], context="FACES")
bm.to_mesh(mesh)
bm.free()
mesh.update()
assert len(mesh.polygons) == 4
print("ABLATION_ONLY removed 4 coincident duplicate quads from leaf.001")
bpy.ops.wm.save_as_mainfile(filepath=str(output))
