"""Keep Blender socket availability distinct from its authored value."""

import importlib.util
from pathlib import Path
import sys

import bpy


path = Path(sys.argv[sys.argv.index("--") + 1])
spec = importlib.util.spec_from_file_location("socket_manifest", path)
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)

material = bpy.data.materials.new("Socket Availability")
material.use_nodes = True
tree = material.node_tree
tree.nodes.clear()
remap = tree.nodes.new("ShaderNodeMapRange")
remap.data_type = "FLOAT"
remap.interpolation_type = "LINEAR"
steps = next(s for s in remap.inputs if s.identifier == "Steps")
steps.default_value = 3.0
assert steps.is_unavailable
record = manifest._socket_manifest(steps)
assert record["available"] is False and record["default"] == 3.0

value = tree.nodes.new("ShaderNodeValue")
tree.links.new(value.outputs[0], steps)
assert manifest._node_tree_manifest(tree)["links"] == []

remap.interpolation_type = "STEPPED"
record = manifest._socket_manifest(steps)
assert record["available"] is True and record["default"] == 3.0
assert len(manifest._node_tree_manifest(tree)["links"]) == 1

emission = tree.nodes.new("ShaderNodeEmission")
vector = next(s for s in remap.outputs if s.identifier == "Vector")
tree.links.new(vector, emission.inputs[0])
assert vector.is_unavailable
assert manifest._socket_manifest(vector)["available"] is False
assert len(manifest._node_tree_manifest(tree)["links"]) == 1

# SOCK_HIDE_VALUE participates in ShaderNodesInliner::set_input_socket_value:
# it is not just presentation metadata. An unlinked hidden group input must
# remain distinguishable from a linked, explicitly authored zero vector.
group = bpy.data.node_groups.new("Hidden Input Contract", "ShaderNodeTree")
interface = group.interface.new_socket(
    name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
interface.hide_value = True
instance = tree.nodes.new("ShaderNodeGroup")
instance.node_tree = group
normal = instance.inputs["Normal"]
assert normal.hide_value and not normal.is_linked
record = manifest._socket_manifest(normal)
assert record["hide_value"] is True and record["linked"] is False
interface.hide_value = False
assert manifest._socket_manifest(normal)["hide_value"] is False
print("Unavailable socket defaults and links retain Cycles source semantics")
