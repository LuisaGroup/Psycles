"""End-to-end tests for the Psycles Blender Extension running inside Blender.

Run inside Blender via:
    blender --background --factory-startup --python tests/test_blender_plugin.py -- \
        --suite all --out-dir /path/to/out --bmw27 /path/to/bmw27_gpu.blend
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import math
import pathlib
import sys
import time
from typing import Any

import addon_utils
import bpy
from mathutils import Euler, Matrix, Vector
import numpy as np


_EXT_MODULE = "bl_ext.user_default.psycles"


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "a", "b"), default="all")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bmw27", type=pathlib.Path, default=None)
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def _enable_extension() -> Any:
    addon_utils.enable(_EXT_MODULE, default_set=False, persistent=False)
    mod = importlib.import_module(_EXT_MODULE)
    rt_dir = mod.bridge.resolve_runtime_dir()
    expected_bin = pathlib.Path(mod.__file__).resolve().parent / "bin"
    assert rt_dir == expected_bin, f"Expected packaged bin {expected_bin}, got {rt_dir}"
    backends = mod.bridge.probe_backends()
    print(f"[PASS] Enabled {_EXT_MODULE}; runtime={rt_dir}; backends={backends}")
    assert "fallback" in backends, f"fallback backend missing: {backends}"
    return mod


def _find_panel(name: str) -> Any | None:
    stack = [bpy.types.Panel]
    seen: set[type] = set()
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child in seen:
                continue
            seen.add(child)
            if child.__name__ == name:
                return child
            stack.append(child)
    return None


def test_ui_and_pass_registration(mod: Any) -> None:
    scene = bpy.context.scene
    scene.render.engine = "PSYCLES"
    assert scene.render.engine == "PSYCLES"

    for name in (
        "PSYCLES_RENDER_PT_device",
        "PSYCLES_RENDER_PT_sampling",
        "PSYCLES_RENDER_PT_light_paths",
        "PSYCLES_RENDER_PT_film",
    ):
        cls = _find_panel(name)
        assert cls is not None, f"Missing custom panel {name}"
        assert cls.poll(bpy.context), f"Panel {name}.poll failed for PSYCLES"

    for name in (
        "CYCLES_MATERIAL_PT_preview",
        "CYCLES_MATERIAL_PT_surface",
        "CYCLES_WORLD_PT_surface",
        "CYCLES_LIGHT_PT_light",
        "CYCLES_RENDER_PT_passes_light",
        "RENDER_PT_color_management",
    ):
        cls = _find_panel(name)
        assert cls is not None, f"Missing native panel {name}"
        assert "PSYCLES" in cls.COMPAT_ENGINES, f"PSYCLES missing from {name}.COMPAT_ENGINES"

    for name in (
        "CYCLES_RENDER_PT_sampling",
        "CYCLES_RENDER_PT_light_paths",
        "CYCLES_RENDER_PT_performance",
        "CYCLES_OBJECT_PT_motion_blur",
    ):
        cls = _find_panel(name)
        if cls is not None:
            assert not cls.poll(bpy.context), f"Excluded panel {name} polled True for PSYCLES"

    vl = bpy.context.view_layer
    vl.use_pass_normal = True
    vl.use_pass_diffuse_color = True
    vl.use_pass_glossy_color = True
    vl.use_pass_transmission_color = True
    vl.use_pass_diffuse_direct = True
    vl.use_pass_diffuse_indirect = True
    vl.use_pass_glossy_direct = True
    vl.use_pass_glossy_indirect = True
    vl.use_pass_transmission_direct = True
    vl.use_pass_transmission_indirect = True
    vl.use_pass_emit = True
    vl.use_pass_environment = True
    vl.cycles.use_pass_volume_direct = True
    vl.cycles.use_pass_volume_indirect = True
    vl.cycles.denoising_store_passes = True
    vl.cycles.pass_debug_sample_count = True
    passes = mod.engine._collect_passes(vl, is_preview=False)
    names = [item[2] for item in passes]
    for expected in (
        "Combined",
        "Normal",
        "Diffuse Color",
        "Glossy Color",
        "Diffuse Direct",
        "Glossy Direct",
        "Emission",
        "Environment",
        "Volume Direct",
        "Volume Indirect",
        "Denoising Normal",
        "Denoising Albedo",
        "Debug Sample Count",
    ):
        assert expected in names, f"Pass {expected} missing in _collect_passes: {names}"
    print(f"[PASS] UI compatibility & {len(names)} pass mappings verified")


def _gl_persp_matrix(width: int, height: int, fov_y: float, shift_x: float, shift_y: float, near: float, far: float) -> Matrix:
    aspect = float(width) / float(height)
    top = near * math.tan(0.5 * fov_y)
    bottom = -top
    left = -aspect * top
    right = aspect * top
    dx = shift_x * (right - left)
    dy = shift_y * (top - bottom)
    left += dx
    right += dx
    bottom += dy
    top += dy
    P = Matrix.Identity(4)
    P.zero()
    P[0][0] = 2.0 * near / (right - left)
    P[1][1] = 2.0 * near / (top - bottom)
    P[0][2] = (right + left) / (right - left)
    P[1][2] = (top + bottom) / (top - bottom)
    P[2][2] = -(far + near) / (far - near)
    P[2][3] = -2.0 * far * near / (far - near)
    P[3][2] = -1.0
    return P


def _gl_ortho_matrix(ortho_scale: float, aspect: float, shift_x: float, shift_y: float, z_near: float, z_far: float) -> Matrix:
    # Matches OpenGL glOrtho(left, right, bottom, top, -z_far, -z_near) where
    # z_near / z_far are eye-space Z bounds (z_near may be > 0 in viewport ORTHO).
    half_x = ortho_scale * 0.5
    half_y = half_x / aspect
    dx = shift_x * 2.0 * half_x
    dy = shift_y * 2.0 * half_x
    left, right = -half_x + dx, half_x + dx
    bottom, top = -half_y + dy, half_y + dy
    near_dist = -z_near
    far_dist = -z_far
    P = Matrix.Identity(4)
    P[0][0] = 2.0 / (right - left)
    P[1][1] = 2.0 / (top - bottom)
    P[2][2] = -2.0 / (far_dist - near_dist)
    P[0][3] = -(right + left) / (right - left)
    P[1][3] = -(top + bottom) / (top - bottom)
    P[2][3] = -(far_dist + near_dist) / (far_dist - near_dist)
    return P


def _cycles_world_to_ndc(capi_cam: Any, width: int, height: int) -> Matrix:
    """Exact Python mirror of src/compiler/cycles_camera.cpp world_to_ndc."""
    xratio = float(width) * float(capi_cam.pixel_aspect_x)
    yratio = float(height) * float(capi_cam.pixel_aspect_y)
    # extract_viewport_camera uses HORIZONTAL sensor_fit
    ratio = xratio / yratio
    x = ratio
    y = 1.0
    fov = 2.0 * math.atan(math.tan(0.5 * float(capi_cam.horizontal_field_of_view)) / ratio)
    is_ortho = int(capi_cam.projection) == 1
    if is_ortho:
        factor = float(capi_cam.orthographic_scale) / (ratio * 2.0)
        x *= factor
        y *= factor
        ratio = float(capi_cam.orthographic_scale) * 0.5
    sx = 2.0 * ratio * float(capi_cam.lens_shift_x)
    sy = 2.0 * ratio * float(capi_cam.lens_shift_y)
    left, right = -x + sx, x + sx
    bottom, top = -y + sy, y + sy
    screen_to_ndc = Matrix.Diagonal((1.0 / (right - left), 1.0 / (top - bottom), 1.0, 1.0)) @ Matrix.Translation((-left, -bottom, 0.0))

    n = float(capi_cam.near_clip)
    f = float(capi_cam.far_clip)
    if is_ortho:
        proj = Matrix.Diagonal((1.0, 1.0, 1.0 / (f - n), 1.0))
    else:
        inv_tan = 1.0 / math.tan(0.5 * fov)
        proj = Matrix((
            (inv_tan, 0.0, 0.0, 0.0),
            (0.0, inv_tan, 0.0, 0.0),
            (0.0, 0.0, f / (f - n), -f * n / (f - n)),
            (0.0, 0.0, 1.0, 0.0),
        ))
    # Unpack column-major transform; flip local +Z column for Cycles world_to_camera
    raw = list(capi_cam.transform)
    c2w = Matrix([[raw[col * 4 + row] for col in range(4)] for row in range(4)])
    for row in range(3):
        c2w[row][2] *= -1.0
    return screen_to_ndc @ proj @ c2w.inverted()


def test_viewport_camera_projection(mod: Any) -> None:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    class _Reg:
        def __init__(self, w: int, h: int) -> None:
            self.width, self.height = w, h

    class _V3D:
        clip_start = 0.05
        clip_end = 500.0

    class _RV3D:
        def __init__(self, persp: bool, view_persp: str, vm: Matrix, wm: Matrix) -> None:
            self.is_perspective = persp
            self.view_perspective = view_persp
            self.view_matrix = vm
            self.window_matrix = wm

    cam_world = Matrix.Translation((2.5, -6.0, 3.0)) @ Euler((1.1, 0.05, 0.35), "XYZ").to_matrix().to_4x4()
    view_mat = cam_world.inverted()
    world_pts = [
        cam_world @ Vector((0.0, 0.0, -5.0, 1.0)),
        cam_world @ Vector((1.2, -0.7, -5.0, 1.0)),
        cam_world @ Vector((-0.8, 0.9, -12.0, 1.0)),
        cam_world @ Vector((0.3, 0.2, -3.5, 1.0)),
    ]

    cases = [
        ("PERSP_landscape", _Reg(640, 360), True, "PERSP", _gl_persp_matrix(640, 360, 0.72, 0.11, -0.07, 0.05, 500.0)),
        ("PERSP_portrait", _Reg(300, 520), True, "PERSP", _gl_persp_matrix(300, 520, 0.95, -0.05, 0.14, 0.05, 500.0)),
        ("ORTHO_pivot_behind", _Reg(500, 400), False, "ORTHO", _gl_ortho_matrix(8.5, 1.25, 0.15, -0.08, z_near=40.0, z_far=-200.0)),
        ("CAMERA_view", _Reg(480, 270), True, "CAMERA", _gl_persp_matrix(480, 270, 0.65, -0.09, 0.06, 0.1, 100.0)),
    ]
    for label, reg, is_persp, view_persp, win_mat in cases:
        rv3d = _RV3D(is_persp, view_persp, view_mat, win_mat)
        _jcam, capi_cam, _sig = mod.scene_sync.extract_viewport_camera(
            reg, _V3D(), rv3d, scene, depsgraph
        )
        w2ndc = _cycles_world_to_ndc(capi_cam, reg.width, reg.height)
        max_err = 0.0
        for p in world_pts:
            gl_clip = win_mat @ (view_mat @ p)
            gl_ndc = (gl_clip.x / gl_clip.w, gl_clip.y / gl_clip.w)
            cy_clip = w2ndc @ p
            cy_ndc = (2.0 * (cy_clip.x / cy_clip.w) - 1.0, 2.0 * (cy_clip.y / cy_clip.w) - 1.0)
            err = max(abs(gl_ndc[0] - cy_ndc[0]), abs(gl_ndc[1] - cy_ndc[1]))
            max_err = max(max_err, err)
        assert max_err < 1.0e-4, f"{label} projection mismatch: max_err={max_err}"
        print(f"[PASS] Viewport projection {label}: max_ndc_err={max_err:.2e}")


def _clear_scene_datablocks() -> None:
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for col in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
        bpy.data.images,
    ):
        for item in list(col):
            if getattr(item, "users", 0) == 0:
                col.remove(item)
    scene.world = None


def _build_synthetic_scene() -> None:
    _clear_scene_datablocks()
    scene = bpy.context.scene
    scene.render.engine = "PSYCLES"
    scene.render.resolution_x = 64
    scene.render.resolution_y = 48
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 8
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False

    # World: uniform subtle blue sky
    world = bpy.data.worlds.new("TestWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.04, 0.07, 0.14, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    scene.world = world

    # Camera at (0, -6, 0) looking +Y: image top = +Z, bottom = -Z, right = +X.
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    cam_obj.location = (0.0, -6.0, 0.0)
    cam_obj.rotation_euler = (math.pi * 0.5, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Generated checkerboard texture datablock (tests in-memory PNG encoding without disk write)
    gen_img = bpy.data.images.new("SynthChecker", width=16, height=16, alpha=True)
    px = np.zeros((16, 16, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:16, 0:16]
    checker = ((xx // 4 + yy // 4) % 2).astype(np.float32)
    px[:, :, 0] = 0.2 + 0.6 * checker
    px[:, :, 1] = 0.8 - 0.5 * checker
    px[:, :, 2] = 0.2
    px[:, :, 3] = 1.0
    gen_img.pixels.foreach_set(px.ravel())

    def _add_cube(name: str, loc: tuple[float, float, float], scale: float, emit_rgb: tuple[float, float, float], base_rgb: tuple[float, float, float]) -> Any:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (scale, scale, scale)
        mat = bpy.data.materials.new(name + "_Mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.25
        bsdf.inputs["Emission Color"].default_value = (*emit_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 4.0 if max(emit_rgb) > 0.0 else 0.0
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        obj.data.materials.append(mat)
        return obj

    # Top Red emissive + diffuse cube (+Z)
    _add_cube("TopRed", (0.0, 0.0, 1.1), 0.9, (1.0, 0.04, 0.04), (0.85, 0.08, 0.08))
    # Bottom Blue emissive + diffuse cube (-Z)
    _add_cube("BotBlue", (0.0, 0.0, -1.1), 0.9, (0.04, 0.04, 1.0), (0.08, 0.08, 0.85))

    # Center Glossy sphere using Generated texture
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55, location=(0.0, 0.0, 0.0))
    sph = bpy.context.active_object
    sph.name = "CenterSphere"
    smat = bpy.data.materials.new("SphereMat")
    smat.use_nodes = True
    sn = smat.node_tree.nodes
    sl = smat.node_tree.links
    sn.clear()
    sout = sn.new("ShaderNodeOutputMaterial")
    sbsdf = sn.new("ShaderNodeBsdfPrincipled")
    sbsdf.inputs["Metallic"].default_value = 0.8
    sbsdf.inputs["Roughness"].default_value = 0.15
    stex = sn.new("ShaderNodeTexImage")
    stex.image = gen_img
    sl.new(stex.outputs["Color"], sbsdf.inputs["Base Color"])
    sl.new(sbsdf.outputs["BSDF"], sout.inputs["Surface"])
    sph.data.materials.append(smat)

    # Area light
    ldata = bpy.data.lights.new("AreaKey", type="AREA")
    ldata.energy = 80.0
    ldata.size = 1.5
    lobj = bpy.data.objects.new("AreaKey", ldata)
    lobj.location = (2.5, -3.5, 2.0)
    lobj.rotation_euler = (math.radians(65), 0.0, math.radians(35))
    scene.collection.objects.link(lobj)


def test_fast_update_camera(mod: Any) -> None:
    _build_synthetic_scene()
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    t0 = time.perf_counter()
    json_b, geom_b, base_dir = mod.scene_sync.export_depsgraph_memory(
        depsgraph, scene=scene, view_layer=bpy.context.view_layer
    )
    engine = mod.bridge.get_or_create_engine(backend="fallback", scheduler="AUTO")
    compiled, _info = engine.compile_scene_memory(json_b, geom_b, base_dir)
    params = mod.engine._make_render_params(full_width=64, full_height=48, scene=scene)
    session = compiled.create_session(params, [(mod.bridge.PASS_KIND_COMBINED, 4, "Combined", "")])
    compile_s = time.perf_counter() - t0

    assert session.render_samples(0, 4, 4, flip_y=True)
    img0 = session.get_pass_array(0).copy()
    assert np.isfinite(img0).all() and img0[:, :, :3].mean() > 0.05

    # Pan camera far away -> black sky/background (zero emissive cube pixels)
    moved_xform = Matrix.Translation((50.0, -6.0, 0.0)) @ Euler((math.pi * 0.5, 0.0, 0.0), "XYZ").to_matrix().to_4x4()
    xform_col = mod._export.geometry.column_major(moved_xform)
    cam_desc = mod.bridge.PsyclesCameraDesc(
        projection=mod.bridge.PSYCLES_PROJECTION_PERSPECTIVE,
        sensor_fit=mod.bridge.PSYCLES_SENSOR_FIT_HORIZONTAL,
        transform=(ctypes.c_float * 16)(*xform_col),
        field_of_view=float(scene.camera.data.angle_y),
        horizontal_field_of_view=float(scene.camera.data.angle_x),
        orthographic_scale=1.0,
        lens_shift_x=0.0,
        lens_shift_y=0.0,
        near_clip=0.1,
        far_clip=100.0,
        aperture_radius=0.0,
        focal_distance=10.0,
        aperture_blades=0,
        aperture_rotation=0.0,
        aperture_ratio=1.0,
        has_sensor=0,
        reserved=(ctypes.c_uint8 * 3)(0, 0, 0),
        sensor_width_mm=0.0,
        sensor_height_mm=0.0,
        sensor_lens_mm=0.0,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )
    t_upd = time.perf_counter()
    session.update_camera(cam_desc, params)
    upd_s = time.perf_counter() - t_upd
    assert session.render_samples(0, 4, 4, flip_y=True)
    img1 = session.get_pass_array(0).copy()
    assert not np.allclose(img0, img1, atol=1e-2)
    # Cancel check
    session.cancel()
    assert session.render_samples(0, 1, 1, flip_y=True) is False
    session.destroy()
    compiled.destroy()
    print(f"[PASS] Fast update_camera: compile+init={compile_s:.3f}s vs update_camera={upd_s:.4f}s")


def test_synthetic_multipass_and_border(mod: Any, out_dir: pathlib.Path) -> None:
    backends = mod.bridge.probe_backends()
    test_backends = [b for b in ("vk", "fallback") if b in backends]
    assert test_backends, f"Neither vk nor fallback available in {backends}"

    for backend_name in test_backends:
        _build_synthetic_scene()
        scene = bpy.context.scene
        scene.psycles.backend = backend_name
        scene.psycles.scheduler = "AUTO"

        vl = bpy.context.view_layer
        vl.use_pass_normal = True
        vl.use_pass_diffuse_color = True
        vl.use_pass_glossy_color = True
        vl.use_pass_diffuse_direct = True
        vl.use_pass_diffuse_indirect = True
        vl.use_pass_glossy_direct = True
        vl.use_pass_glossy_indirect = True
        vl.use_pass_emit = True
        vl.use_pass_environment = True

        # 1) Full multilayer EXR render
        full_exr = out_dir / f"synth_{backend_name}_full.exr"
        scene.render.use_border = False
        img_set = scene.render.image_settings
        if hasattr(img_set, "media_type"):
            img_set.media_type = "MULTI_LAYER_IMAGE"
        img_set.file_format = "OPEN_EXR_MULTILAYER"
        img_set.color_depth = "32"
        scene.render.filepath = str(full_exr)
        bpy.ops.render.render(write_still=True)

        # Also save a PNG to check right-side-up top/bottom RGB orientation cleanly
        full_png = out_dir / f"synth_{backend_name}_full.png"
        if hasattr(img_set, "media_type"):
            img_set.media_type = "IMAGE"
        img_set.file_format = "PNG"
        img_set.color_mode = "RGBA"
        bpy.data.images["Render Result"].save_render(filepath=str(full_png), scene=scene)

        img = bpy.data.images.load(str(full_png))
        px = np.empty(64 * 48 * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(48, 64, 4)  # bottom-up: row 0 = bottom, row 47 = top
        bot = px[4:16, 28:36, :]
        top = px[32:44, 28:36, :]
        bpy.data.images.remove(img)
        assert top[:, :, 0].mean() > top[:, :, 2].mean() * 1.5, (
            f"{backend_name} top-red orientation check failed: R={top[:,:,0].mean():.3f}, B={top[:,:,2].mean():.3f}"
        )
        assert bot[:, :, 2].mean() > bot[:, :, 0].mean() * 1.5, (
            f"{backend_name} bottom-blue orientation check failed: R={bot[:,:,0].mean():.3f}, B={bot[:,:,2].mean():.3f}"
        )

        # 2) Border render (uncropped & cropped)
        scene.render.use_border = True
        scene.render.border_min_x = 0.25
        scene.render.border_max_x = 0.75
        scene.render.border_min_y = 0.25
        scene.render.border_max_y = 0.75

        scene.render.use_crop_to_border = False
        if hasattr(img_set, "media_type"):
            img_set.media_type = "IMAGE"
        img_set.file_format = "OPEN_EXR"
        img_set.color_depth = "32"
        uncrop_exr = out_dir / f"synth_{backend_name}_border_uncrop.exr"
        scene.render.filepath = str(uncrop_exr)
        bpy.ops.render.render(write_still=True)

        scene.render.use_crop_to_border = True
        crop_exr = out_dir / f"synth_{backend_name}_border_crop.exr"
        scene.render.filepath = str(crop_exr)
        bpy.ops.render.render(write_still=True)
        print(f"[PASS] Synthetic multipass + orientation + border renders on {backend_name}")


def test_unregister(mod: Any) -> None:
    addon_utils.disable(_EXT_MODULE, default_set=False)
    cls = _find_panel("CYCLES_MATERIAL_PT_surface")
    assert cls is not None and "PSYCLES" not in cls.COMPAT_ENGINES
    assert not hasattr(bpy.types.Scene, "psycles")
    print(f"[PASS] Cleanly unregistered {_EXT_MODULE} without leaking COMPAT_ENGINES")


def _configure_cycles_hip(scene: Any) -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "HIP"
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == "HIP"]
        # Prefer discrete RX 9070 / non-APU if multiple HIP devices exist
        discrete = [d for d in gpus if "9070" in d.name or "gfx120" in d.name.lower()]
        chosen = discrete or gpus[:1]
        chosen_ids = {d.id for d in chosen}
        for d in prefs.devices:
            d.use = d.id in chosen_ids
        if chosen:
            scene.cycles.device = "GPU"
            return f"HIP ({chosen[0].name})"
    except Exception:
        pass
    scene.cycles.device = "CPU"
    return "CPU"


def test_bmw27_render(mod: Any, bmw27_path: pathlib.Path, out_dir: pathlib.Path, samples: int) -> None:
    assert bmw27_path.is_file(), f"Missing bmw27 scene: {bmw27_path}"

    # 1) Render with Psycles plugin
    bpy.ops.wm.open_mainfile(filepath=str(bmw27_path))
    scene = bpy.context.scene
    scene.render.engine = "PSYCLES"
    scene.psycles.backend = "vk"
    scene.psycles.scheduler = "AUTO"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False

    psy_png = out_dir / "bmw27_psycles_plugin.png"
    psy_exr = out_dir / "bmw27_psycles_plugin.exr"

    t0 = time.perf_counter()
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(psy_exr)
    bpy.ops.render.render(write_still=True)
    psy_s = time.perf_counter() - t0

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.data.images["Render Result"].save_render(filepath=str(psy_png), scene=scene)
    print(f"[PASS] Rendered bmw27 with Psycles plugin ({samples} spp, vk) in {psy_s:.2f}s -> {psy_png}")

    # 2) Render matching reference with Cycles 5.3.0
    bpy.ops.wm.open_mainfile(filepath=str(bmw27_path))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    dev_label = _configure_cycles_hip(scene)
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    cyc_exr = out_dir / "bmw27_cycles_ref.exr"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(cyc_exr)
    t1 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    cyc_s = time.perf_counter() - t1
    print(f"[PASS] Rendered bmw27 reference with Cycles ({samples} spp, {dev_label}) in {cyc_s:.2f}s -> {cyc_exr}")


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mod = _enable_extension()
    try:
        if args.suite in ("all", "a"):
            test_ui_and_pass_registration(mod)
            test_viewport_camera_projection(mod)
            test_fast_update_camera(mod)
            test_synthetic_multipass_and_border(mod, args.out_dir)
        if args.suite in ("all", "b"):
            assert args.bmw27 is not None, "--bmw27 required for suite b"
            test_bmw27_render(mod, args.bmw27, args.out_dir, args.samples)
        if args.suite in ("all", "a"):
            test_unregister(mod)
    finally:
        mod.bridge.shutdown_all()
    print("[ALL BLENDER PLUGIN TESTS PASSED]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
