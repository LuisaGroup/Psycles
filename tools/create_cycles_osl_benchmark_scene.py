"""Create Cycles 5.3.0 OSL parity and benchmark scene."""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import bpy
from cycles import osl as cycles_osl
from mathutils import Euler


DEFAULT_BLEND = pathlib.Path(
    r"C:\Users\Mike\Projects\psycles-scenes\osl_benchmark\osl_benchmark.blend"
)


def _compile_osl_node(node: Any, text_name: str, source: str) -> None:
    text = bpy.data.texts.new(text_name)
    text.from_string(source.strip() + "\n")
    node.mode = "INTERNAL"
    node.script = text

    messages: list[tuple[set[str], str]] = []

    def report(level: set[str], msg: str) -> None:
        messages.append((set(level), str(msg)))

    ok = cycles_osl.update_script_node(node, report)
    if not ok or not str(getattr(node, "bytecode", "") or "").strip():
        raise RuntimeError(
            f"OSL compilation failed for {text_name}: ok={ok}, messages={messages}"
        )


def _look_at(obj: Any, target: tuple[float, float, float]) -> None:
    from mathutils import Vector

    direction = Vector(target) - obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def _setup_world(scene: Any) -> None:
    world = bpy.data.worlds.new("OSL_World")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Color"].default_value = (0.12, 0.14, 0.18, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    scene.world = world


def _build_ground_material() -> Any:
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.38, 0.39, 0.41, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65
    return mat


def _build_procedural_wood_material() -> Any:
    mat = bpy.data.materials.new("OSL_ProceduralWood")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.35
    script = nt.nodes.new("ShaderNodeScript")

    osl_src = """
shader osl_procedural_wood(
    point Vector = P,
    color ColorA = color(0.75, 0.32, 0.10),
    color ColorB = color(0.16, 0.08, 0.03),
    float Scale = 4.5,
    output color OutColor = color(0.5),
    output float OutRoughness = 0.3
) {
    point p_obj = transform("world", "object", Vector);
    float n0 = noise("perlin", p_obj * Scale);
    float n1 = noise("perlin", p_obj * (Scale * 2.3) + vector(3.1, 1.7, 5.2));
    float turb = n0 * 0.65 + n1 * 0.35;
    float r = length(vector(p_obj[0], p_obj[1], 0.0)) * Scale + turb * 2.2;
    float ring = 0.5 + 0.5 * sin(r * 6.2831853);
    float f = smoothstep(0.15, 0.85, ring);
    OutColor = mix(ColorA, ColorB, f);
    OutRoughness = 0.18 + 0.42 * f;
}
"""
    _compile_osl_node(script, "osl_procedural_wood.osl", osl_src)
    nt.links.new(script.outputs["OutColor"], principled.inputs["Base Color"])
    nt.links.new(script.outputs["OutRoughness"], principled.inputs["Roughness"])
    nt.links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_direct_closure_material() -> Any:
    mat = bpy.data.materials.new("OSL_DirectClosures")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    script = nt.nodes.new("ShaderNodeScript")

    osl_src = """
surface osl_direct_closures(
    color DiffColor = color(0.14, 0.42, 0.78),
    color SpecColor = color(0.9, 0.9, 0.9),
    float Roughness = 0.22,
    float OrenRough = 0.4,
    output closure color BSDF = 0
) {
    color dcol = backfacing() ? color(0.45, 0.12, 0.12) : DiffColor;
    closure color c_diff = 0;
    if (raytype("camera")) {
        c_diff = diffuse(N) * dcol;
    } else {
        c_diff = oren_nayar(N, OrenRough) * dcol;
    }
    closure color c_spec = microfacet("ggx", N, Roughness * Roughness, 0.0, 0) * SpecColor;
    BSDF = 0.72 * c_diff + 0.28 * c_spec;
}
"""
    _compile_osl_node(script, "osl_direct_closures.osl", osl_src)
    nt.links.new(script.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_emissive_stripes_material() -> Any:
    mat = bpy.data.materials.new("OSL_EmissiveStripes")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    script = nt.nodes.new("ShaderNodeScript")

    osl_src = """
surface osl_emissive_stripes(
    color ColorA = color(1.8, 0.45, 0.12),
    color ColorB = color(0.12, 1.1, 2.1),
    float Freq = 6.0,
    output closure color EmissionOut = 0
) {
    float phase = P[0] * Freq + P[2] * (Freq * 0.7) + u * 3.1415926;
    float band = smoothstep(0.25, 0.75, 0.5 + 0.5 * sin(phase));
    color ecol = mix(ColorA, ColorB, band);
    EmissionOut = ecol * emission() * 0.65 + diffuse(N) * color(0.08);
}
"""
    _compile_osl_node(script, "osl_emissive_stripes.osl", osl_src)
    nt.links.new(script.outputs["EmissionOut"], out.inputs["Surface"])
    return mat


def _build_custom_attr_material() -> Any:
    mat = bpy.data.materials.new("OSL_CustomAttr")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    script = nt.nodes.new("ShaderNodeScript")
    attr_node = nt.nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "my_osl_tint"
    uv_node = nt.nodes.new("ShaderNodeUVMap")

    osl_src = """
surface osl_custom_attr(
    vector AttrHint = vector(0.0),
    point UvHint = point(0.0),
    output closure color BSDF = 0
) {
    vector tint = vector(0.1, 0.1, 0.1);
    int has_tint = getattribute("my_osl_tint", tint);
    point uv_p = point(0.0, 0.0, 0.0);
    int has_uv = getattribute("geom:uv", uv_p);
    float hint_noop = 0.0 * (AttrHint[0] + UvHint[0]);
    float cx = floor(uv_p[0] * 6.0);
    float cy = floor(uv_p[1] * 6.0);
    float checker = mod(cx + cy + hint_noop, 2.0);
    color base = has_tint ? color(tint[0], tint[1], tint[2]) : color(0.8, 0.1, 0.8);
    color final_c = has_uv ? mix(base * 0.35, base, checker) : base;
    BSDF = diffuse(N) * final_c * 0.85 + microfacet("ggx", N, 0.12, 0.0, 0) * 0.15;
}
"""
    _compile_osl_node(script, "osl_custom_attr.osl", osl_src)
    nt.links.new(attr_node.outputs["Vector"], script.inputs["AttrHint"])
    nt.links.new(uv_node.outputs["UV"], script.inputs["UvHint"])
    nt.links.new(script.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_view_deriv_material() -> Any:
    mat = bpy.data.materials.new("OSL_ViewDeriv")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    script = nt.nodes.new("ShaderNodeScript")

    osl_src = """
surface osl_view_deriv(
    color CoreColor = color(0.12, 0.55, 0.32),
    color RimColor = color(0.95, 0.82, 0.25),
    output closure color BSDF = 0
) {
    float ndoti = clamp(dot(normalize(N), normalize(I)), 0.0, 1.0);
    float rim = pow(1.0 - ndoti, 2.5);
    float dmag = clamp((length(Dx(P)) + length(Dy(P))) * 4.0, 0.0, 0.35);
    float stripe = 0.5 + 0.5 * sin((P[1] + P[2]) * 10.0);
    color c = mix(CoreColor, RimColor, clamp(rim + dmag * 0.5, 0.0, 1.0));
    c *= 0.75 + 0.25 * stripe;
    BSDF = diffuse(N) * c * 0.8 + microfacet("ggx", N, 0.06, 0.0, 0) * 0.2;
}
"""
    _compile_osl_node(script, "osl_view_deriv.osl", osl_src)
    nt.links.new(script.outputs["BSDF"], out.inputs["Surface"])
    return mat


def main() -> None:
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
        out_path = pathlib.Path(args[0]).resolve() if args else DEFAULT_BLEND
    else:
        out_path = DEFAULT_BLEND
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.shading_system = True
    scene.cycles.samples = 64
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100

    _setup_world(scene)

    ground_mat = _build_ground_material()
    mat_wood = _build_procedural_wood_material()
    mat_closures = _build_direct_closure_material()
    mat_emissive = _build_emissive_stripes_material()
    mat_attr = _build_custom_attr_material()
    mat_view = _build_view_deriv_material()

    bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "Ground"
    ground.data.materials.append(ground_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=48, ring_count=24, radius=0.72, location=(-2.8, 0.2, 0.72)
    )
    obj_wood = bpy.context.object
    obj_wood.name = "Sphere_OSL_Wood"
    obj_wood.rotation_euler = Euler((0.2, -0.4, 0.6))
    bpy.ops.object.shade_smooth()
    obj_wood.data.materials.append(mat_wood)

    bpy.ops.mesh.primitive_monkey_add(
        size=1.35, location=(-1.4, -0.1, 0.65)
    )
    obj_closures = bpy.context.object
    obj_closures.name = "Suzanne_OSL_Closures"
    obj_closures.rotation_euler = Euler((0.0, 0.0, 0.25))
    bpy.ops.object.shade_smooth()
    obj_closures.data.materials.append(mat_closures)

    bpy.ops.mesh.primitive_torus_add(
        major_segments=48,
        minor_segments=24,
        major_radius=0.55,
        minor_radius=0.22,
        location=(0.0, 0.3, 0.65),
    )
    obj_emit = bpy.context.object
    obj_emit.name = "Torus_OSL_Emission"
    obj_emit.rotation_euler = Euler((0.75, 0.1, -0.2))
    bpy.ops.object.shade_smooth()
    obj_emit.data.materials.append(mat_emissive)

    bpy.ops.mesh.primitive_cube_add(size=1.15, location=(1.4, 0.0, 0.58))
    obj_attr = bpy.context.object
    obj_attr.name = "Cube_OSL_Attr"
    obj_attr.rotation_euler = Euler((0.0, 0.0, -0.45))
    mesh_attr = obj_attr.data
    attr = mesh_attr.attributes.new(
        name="my_osl_tint", type="FLOAT_VECTOR", domain="POINT"
    )
    for idx, vert in enumerate(mesh_attr.vertices):
        tx = 0.5 + 0.5 * math.sin(vert.co.x * 2.0 + 0.4)
        ty = 0.5 + 0.5 * math.cos(vert.co.y * 2.0 - 0.3)
        tz = 0.25 + 0.65 * (0.5 + 0.5 * vert.co.z)
        attr.data[idx].vector = (tx, ty, tz)
    mesh_attr.materials.append(mat_attr)

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=4, radius=0.72, location=(2.8, 0.15, 0.72)
    )
    obj_view = bpy.context.object
    obj_view.name = "Ico_OSL_ViewDeriv"
    bpy.ops.object.shade_smooth()
    obj_view.data.materials.append(mat_view)

    light_data = bpy.data.lights.new(name="KeyAreaLight", type="AREA")
    light_data.energy = 650.0
    light_data.size = 4.0
    light_obj = bpy.data.objects.new(name="KeyAreaLight", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (-2.5, -4.5, 5.5)
    _look_at(light_obj, (0.0, 0.0, 0.6))

    fill_data = bpy.data.lights.new(name="FillAreaLight", type="AREA")
    fill_data.energy = 220.0
    fill_data.size = 5.0
    fill_data.color = (0.75, 0.85, 1.0)
    fill_obj = bpy.data.objects.new(name="FillAreaLight", object_data=fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.location = (4.5, -3.5, 4.0)
    _look_at(fill_obj, (0.0, 0.0, 0.6))

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = 42.0
    cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -7.8, 2.55)
    _look_at(cam_obj, (0.0, 0.0, 0.68))
    scene.camera = cam_obj

    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"OSL_BENCHMARK_SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
