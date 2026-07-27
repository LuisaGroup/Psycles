#!/usr/bin/env python3
"""Emit the shader-node contract exposed by the running Blender build.

Run through Blender, not the system Python:

    blender --background --factory-startup \
      --python tools/cycles_shader_node_inventory.py -- output.json

The resulting file is intentionally versioned. CI compares a fresh inventory
with it so a Blender upgrade cannot silently add or change shader nodes without
an explicit Psycles coverage decision.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import bpy


EEVEE_ONLY = {
    "ShaderNodeEeveeSpecular",
    "ShaderNodeShaderToRGB",
}

LINE_STYLE_ONLY = {
    "ShaderNodeOutputLineStyle",
    "ShaderNodeUVAlongStroke",
}

STRUCTURAL = {
    "ShaderNode",
    "ShaderNodeCustomGroup",
    "ShaderNodeGroup",
    "ShaderNodeTree",
}

OSL_ONLY = {
    "ShaderNodeScript",
}

# Blender keeps this compatibility alias even though the canonical RNA class
# was renamed to ShaderNodeBsdfAnisotropic.
ALIASES = {
    "ShaderNodeBsdfAnisotropic": ["ShaderNodeBsdfGlossy"],
}


def command_line_arguments() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_list"):
        return [json_value(item) for item in value.to_list()]
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    try:
        return [json_value(item) for item in value]
    except TypeError:
        return str(value)


def socket_contract(socket: Any) -> dict[str, Any]:
    return {
        "identifier": socket.identifier,
        "name": socket.name,
        "socket_type": socket.bl_idname,
        "default": (
            json_value(socket.default_value)
            if hasattr(socket, "default_value")
            else None
        ),
        "enabled": bool(socket.enabled),
        "hide_value": bool(socket.hide_value),
    }


def property_contract(node: Any) -> list[dict[str, Any]]:
    inherited = {
        prop.identifier for prop in bpy.types.Node.bl_rna.properties
    }
    result: list[dict[str, Any]] = []
    for prop in node.bl_rna.properties:
        if (
            prop.identifier in inherited
            or prop.identifier == "rna_type"
            or prop.is_readonly
            or prop.is_hidden
        ):
            continue
        item: dict[str, Any] = {
            "identifier": prop.identifier,
            "property_type": prop.type,
            "default": json_value(getattr(node, prop.identifier)),
        }
        if prop.type == "ENUM":
            item["enum_items"] = [
                enum_item.identifier for enum_item in prop.enum_items
            ]
        result.append(item)
    return sorted(result, key=lambda item: item["identifier"])


def applicability(bl_idname: str) -> str:
    if bl_idname in EEVEE_ONLY:
        return "eevee_only"
    if bl_idname in LINE_STYLE_ONLY:
        return "line_style_only"
    if bl_idname in STRUCTURAL:
        return "structural"
    if bl_idname in OSL_ONLY:
        return "cycles_osl_only"
    return "cycles"


def inventory() -> dict[str, Any]:
    node_tree = bpy.data.node_groups.new(
        "Psycles Shader Node Inventory", "ShaderNodeTree"
    )
    nodes: list[dict[str, Any]] = []
    for bl_idname in sorted(
        name for name in dir(bpy.types) if name.startswith("ShaderNode")
    ):
        node_class = getattr(bpy.types, bl_idname)
        if not hasattr(node_class, "bl_rna"):
            continue
        entry: dict[str, Any] = {
            "bl_idname": bl_idname,
            "display_name": node_class.bl_rna.name,
            "applicability": applicability(bl_idname),
            "aliases": ALIASES.get(bl_idname, []),
        }
        try:
            node = node_tree.nodes.new(bl_idname)
        except (RuntimeError, TypeError) as error:
            entry["constructible"] = False
            entry["construction_error"] = str(error)
        else:
            entry.update(
                {
                    "constructible": True,
                    "node_type": node.type,
                    "inputs": [
                        socket_contract(socket) for socket in node.inputs
                    ],
                    "outputs": [
                        socket_contract(socket) for socket in node.outputs
                    ],
                    "properties": property_contract(node),
                }
            )
            node_tree.nodes.remove(node)
        nodes.append(entry)

    return {
        "schema": "psycles.cycles-shader-node-inventory.v1",
        "blender_version": bpy.app.version_string,
        "blender_version_tuple": list(bpy.app.version),
        "node_count": len(nodes),
        "nodes": nodes,
    }


def main() -> None:
    arguments = command_line_arguments()
    if len(arguments) != 1:
        raise SystemExit(
            "expected one output path after '--': "
            "cycles_shader_node_inventory.py -- output.json"
        )
    output = pathlib.Path(arguments[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Cycles shader-node inventory: {output}")


if __name__ == "__main__":
    main()
