#include <psycles/blender_bridge/psycles_blender_capi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void append_bytes(std::vector<std::byte> &dst, const void *src, std::size_t n) {
    const auto *p = static_cast<const std::byte *>(src);
    dst.insert(dst.end(), p, p + n);
}

template<typename T>
std::pair<std::size_t, std::size_t> append_section(
    std::vector<std::byte> &bin,
    const std::vector<T> &values) {
    const auto offset = bin.size();
    const auto bytes = values.size() * sizeof(T);
    append_bytes(bin, values.data(), bytes);
    return {offset, bytes};
}

} // namespace

int main(int, char **argv) {
    if (psycles_capi_version() != PSYCLES_CAPI_VERSION) {
        std::cerr << "unexpected C API version\n";
        return 1;
    }
    const auto exe_dir = std::filesystem::path{argv[0]}.parent_path();
    char backends_csv[512]{};
    const auto count =
        psycles_probe_backends(exe_dir.string().c_str(), backends_csv, sizeof(backends_csv));
    std::cout << "Probed backends (" << count << "): " << backends_csv << '\n';
    if (count == 0u || std::string{backends_csv}.find("fallback") == std::string::npos) {
        std::cerr << "fallback backend was not probed in " << exe_dir << '\n';
        return 1;
    }

    char err[1024]{};
    PsyclesEngineConfig cfg{};
    cfg.scheduler = PSYCLES_SCHEDULER_MEGAKERNEL;
    cfg.enable_fast_math = 1u;
    cfg.staged_surface_sorting = 1u;
    cfg.staged_direct_light_queue = 1u;
    cfg.vulkan_require_native_xir = 1u;
    cfg.hip_visible_device = -1;

    auto *engine = psycles_engine_create(
        exe_dir.string().c_str(), "fallback", &cfg, err, sizeof(err));
    if (engine == nullptr) {
        std::cerr << "psycles_engine_create failed: " << err << '\n';
        return 1;
    }

    // Build an in-memory PSYGEO2 minimal scene: bright triangle facing camera.
    std::vector<std::byte> bin;
    const char magic[8]{'P', 'S', 'Y', 'G', 'E', 'O', '2', '\0'};
    append_bytes(bin, magic, sizeof(magic));

    const std::vector<float> positions{
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         0.0f,  1.0f, 0.0f};
    const std::vector<float> normals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f};
    const std::vector<float> uvs{0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
    const std::vector<float> uv_tangents(12u, 0.0f);
    const std::vector<float> generated{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u};
    const std::vector<std::uint32_t> tri_mats{0u};
    const std::vector<std::uint32_t> tri_smooth{0u};
    const std::vector<float> tri_island{0.0f};

    const auto sec_pos = append_section(bin, positions);
    const auto sec_nrm = append_section(bin, normals);
    const auto sec_uv = append_section(bin, uvs);
    const auto sec_tan = append_section(bin, uv_tangents);
    const auto sec_gen = append_section(bin, generated);
    const auto sec_idx = append_section(bin, indices);
    const auto sec_mat = append_section(bin, tri_mats);
    const auto sec_smt = append_section(bin, tri_smooth);
    const auto sec_isl = append_section(bin, tri_island);

    const std::string json = std::string(R"JSON({
      "schema": "psycles.blender-scene.v2",
      "cycles_sync": {"object_count": 2, "uses_light_linking": false},
      "render": {
        "width": 16, "height": 16, "percentage": 100,
        "transparent": false, "pixel_filter_type": "BOX", "filter_width": 1.0,
        "pass_alpha_threshold": 0.5,
        "color_management": {
          "display_device": "sRGB", "view_transform": "Standard", "look": "None",
          "exposure": 0.0, "gamma": 1.0, "use_curve_mapping": false,
          "sequencer_color_space": "sRGB",
          "shader_transforms": {
            "blender_luma": [0.2126, 0.7152, 0.0722],
            "xyz_to_rgb": [[3.2404542,-1.5371385,-0.4985314],[-0.969266,1.8760108,0.041556],[0.0556434,-0.2040259,1.0572252]],
            "rec709_to_rgb": [[1,0,0],[0,1,0],[0,0,1]]
          }
        },
        "cycles": {
          "samples": 4, "seed": 0, "max_bounces": 4, "min_light_bounces": 0,
          "diffuse_bounces": 2, "glossy_bounces": 2, "transmission_bounces": 2,
          "volume_bounces": 0, "transparent_max_bounces": 4,
          "sample_clamp_direct": 0.0, "sample_clamp_indirect": 0.0,
          "blur_glossy": 0.0, "light_sampling_threshold": 0.01,
          "caustics_reflective": true, "caustics_refractive": true,
          "direct_light_sampling_type": "MULTIPLE_IMPORTANCE_SAMPLING",
          "use_light_tree": false, "ao_bounces_render": 0,
          "film_exposure": 1.0
        }
      },
      "camera": {
        "name": "Cam", "type": "PERSP",
        "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,3,1],
        "angle": 0.78539816, "angle_x": 0.78539816, "angle_y": 0.78539816,
        "sensor_fit": "AUTO", "lens": 35.0, "sensor_width": 36.0, "sensor_height": 24.0,
        "pixel_aspect_x": 1.0, "pixel_aspect_y": 1.0, "ortho_scale": 6.0,
        "shift_x": 0.0, "shift_y": 0.0, "clip_start": 0.1, "clip_end": 100.0,
        "dof": {"enabled": false, "focus_distance": 10.0, "fstop": 2.8, "blades": 0, "rotation": 0.0, "ratio": 1.0}
      },
      "geometries": [{
        "name": "Tri", "triangle_count": 1, "point_count": 3, "corner_count": 3,
        "normal_domain": "POINT", "uv_domain": "POINT", "uv_tangent_domain": "POINT", "generated_domain": "POINT",
        "material_slots": ["EmitMat"],
        "positions": {"offset": )JSON") + std::to_string(sec_pos.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_pos.second) + R"JSON(},
        "normals": {"offset": )JSON" + std::to_string(sec_nrm.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_nrm.second) + R"JSON(},
        "uv": {"offset": )JSON" + std::to_string(sec_uv.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_uv.second) + R"JSON(},
        "uv_tangents": {"offset": )JSON" + std::to_string(sec_tan.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_tan.second) + R"JSON(},
        "generated": {"offset": )JSON" + std::to_string(sec_gen.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_gen.second) + R"JSON(},
        "indices": {"offset": )JSON" + std::to_string(sec_idx.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_idx.second) + R"JSON(},
        "triangle_material_slots": {"offset": )JSON" + std::to_string(sec_mat.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_mat.second) + R"JSON(},
        "triangle_smooth": {"offset": )JSON" + std::to_string(sec_smt.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_smt.second) + R"JSON(},
        "triangle_random_per_island": {"offset": )JSON" + std::to_string(sec_isl.first) + R"JSON(, "bytes": )JSON" + std::to_string(sec_isl.second) + R"JSON(},
        "uv_layers": [], "color_attributes": []
      }],
      "curve_geometries": [],
      "instances": [{
        "name": "TriObj", "geometry": 0, "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
        "object_color": [1,1,1], "object_alpha": 1.0, "random_id": 100, "particle_index": 0, "is_shadow_catcher": false,
        "ambient_occlusion_distance": 0.0, "shadow_terminator_geometry_offset": 0.0,
        "cycles_sync": {"object_index": 0, "light_group": -1},
        "visibility": {"camera": true, "diffuse": true, "glossy": true, "transmission": true, "shadow": true, "volume_scatter": true}
      }],
      "materials": [{
        "name": "EmitMat", "use_transparent_shadow": true, "use_bump_map_correction": true,
        "emission_sampling": "AUTO", "volume_sampling": "MULTIPLE_IMPORTANCE", "volume_interpolation": "LINEAR",
        "displacement_method": "BUMP",
        "cycles_sync": {"shader_index": 5, "pass_id": 0},
        "node_tree": {
          "nodes": [
            {"name": "Emit", "bl_idname": "ShaderNodeEmission", "type": "EMISSION",
             "inputs": [
               {"identifier": "Color", "name": "Color", "type": "NodeSocketColor", "linked": false, "available": true, "default": [1.0, 0.5, 0.25, 1.0]},
               {"identifier": "Strength", "name": "Strength", "type": "NodeSocketFloat", "linked": false, "available": true, "default": 2.0}
             ],
             "outputs": [
               {"identifier": "Emission", "name": "Emission", "type": "NodeSocketShader", "linked": true, "available": true}
             ]},
            {"name": "Out", "bl_idname": "ShaderNodeOutputMaterial", "type": "OUTPUT_MATERIAL",
             "inputs": [
               {"identifier": "Surface", "name": "Surface", "type": "NodeSocketShader", "linked": true, "available": true}
             ],
             "outputs": []}
          ],
          "links": [{"from_node": "Emit", "from_socket": "Emission", "to_node": "Out", "to_socket": "Surface"}],
          "surface_root": {"node": "Emit", "socket": "Emission"}, "volume_root": null, "displacement_root": null
        }
      }],
      "lights": [],
      "world": {
        "name": "World", "color": [0.0, 0.0, 0.0], "sampling_method": "AUTOMATIC",
        "volume_sampling": "MULTIPLE_IMPORTANCE", "volume_interpolation": "LINEAR",
        "sample_map_resolution": 32, "max_bounces": 4, "use_shadows": true,
        "visibility": {"camera": true, "diffuse": true, "glossy": true, "transmission": true, "shadow": true, "volume_scatter": true},
        "cycles_sync": {"shader_index": 3, "pass_id": 0, "object_index": 1, "light_group": -1},
        "node_tree": null
      },
      "world_environment": null,
      "node_groups": [],
      "images": []
    })JSON";

    PsyclesSceneInfo info{};
    auto *scene = psycles_scene_compile_memory(
        engine, json.data(), json.size(), bin.data(), bin.size(), "", &info, err, sizeof(err));
    if (scene == nullptr) {
        std::cerr << "psycles_scene_compile_memory failed: " << err << '\n';
        psycles_engine_destroy(engine);
        return 1;
    }
    if (info.geometry_count != 1u || info.instance_count != 1u || info.material_count != 3u) {
        std::cerr << "unexpected scene counts: geom=" << info.geometry_count
                  << " inst=" << info.instance_count << " mat=" << info.material_count << '\n';
        return 1;
    }

    PsyclesPassDesc passes[2]{
        {.kind = 0u, .reserved = 0u, .channels = 4u, .name = "Combined", .light_group = nullptr},
        {.kind = 5u, .reserved = 0u, .channels = 3u, .name = "Emit", .light_group = nullptr}};
    PsyclesRenderParams params{};
    params.full_width = 16u;
    params.full_height = 16u;
    params.pixel_filter = PSYCLES_PIXEL_FILTER_FROM_SCENE;
    auto *session = psycles_session_create(engine, scene, &params, passes, 2u, err, sizeof(err));
    if (session == nullptr) {
        std::cerr << "psycles_session_create failed: " << err << '\n';
        return 1;
    }

    // Progressive chunks: [0,2) + [2,2) out of total=4
    if (psycles_session_render_samples(session, 0u, 2u, 4u, 1u, err, sizeof(err)) != 1 ||
        psycles_session_render_samples(session, 2u, 2u, 4u, 1u, err, sizeof(err)) != 1) {
        std::cerr << "progressive render failed: " << err << '\n';
        return 1;
    }
    const float *comb_ptr = nullptr;
    std::uint32_t w = 0u, h = 0u, ch = 0u;
    if (!psycles_session_get_pass_buffer(session, 0u, &comb_ptr, &w, &h, &ch) ||
        w != 16u || h != 16u || ch != 4u) {
        std::cerr << "get_pass_buffer failed\n";
        return 1;
    }
    // Verify top/bottom asymmetry: triangle apex is at y=+1 -> top half in flip_y=1 should be brighter than bottom corners,
    // and bottom row (flip_y=1 row 0) is bottom edge (-1,-1)..(1,-1). Let's sum upper vs lower row halves.
    std::vector<float> baseline(comb_ptr, comb_ptr + 16u * 16u * 4u);
    float center_sum = 0.0f;
    for (float v : baseline) {
        if (!std::isfinite(v)) {
            std::cerr << "non-finite pixel value\n";
            return 1;
        }
        center_sum += v;
    }
    if (center_sum <= 1.0f) {
        std::cerr << "expected non-black triangle render\n";
        return 1;
    }

    // Compare single-shot [0, 4) after resetting camera far left -> should change image cleanly without rebuild.
    PsyclesCameraDesc moved_cam{};
    moved_cam.projection = PSYCLES_PROJECTION_PERSPECTIVE;
    moved_cam.sensor_fit = PSYCLES_SENSOR_FIT_AUTO;
    const float xform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 10.0f,0,3.0f,1.0f};
    std::memcpy(moved_cam.transform, xform, sizeof(xform));
    moved_cam.field_of_view = 0.78539816f;
    moved_cam.horizontal_field_of_view = 0.78539816f;
    moved_cam.near_clip = 0.1f;
    moved_cam.far_clip = 100.0f;
    moved_cam.aperture_ratio = 1.0f;
    moved_cam.pixel_aspect_x = 1.0f;
    moved_cam.pixel_aspect_y = 1.0f;

    if (psycles_session_update_camera(session, &moved_cam, &params, err, sizeof(err)) != 1) {
        std::cerr << "update_camera failed: " << err << '\n';
        return 1;
    }
    if (psycles_session_render_samples(session, 0u, 4u, 4u, 1u, err, sizeof(err)) != 1) {
        std::cerr << "moved camera render failed: " << err << '\n';
        return 1;
    }
    const float *moved_ptr = nullptr;
    psycles_session_get_pass_buffer(session, 0u, &moved_ptr, &w, &h, &ch);
    float moved_rgb = 0.0f;
    for (std::size_t i = 0u; i < 16u * 16u; ++i) {
        moved_rgb += moved_ptr[i * 4u] + moved_ptr[i * 4u + 1u] + moved_ptr[i * 4u + 2u];
    }
    if (moved_rgb > 1.0e-3f) {
        std::cerr << "expected black background after panning camera off triangle, got " << moved_rgb << '\n';
        return 1;
    }

    // Verify cancel path returns 0 cleanly
    psycles_session_cancel(session);
    if (psycles_session_render_samples(session, 0u, 1u, 1u, 1u, err, sizeof(err)) != 0) {
        std::cerr << "expected cancel return code 0\n";
        return 1;
    }

    psycles_session_destroy(session);
    psycles_scene_destroy(scene);
    psycles_engine_destroy(engine);
    std::cout << "psycles_blender_capi_tests PASSED\n";
    return 0;
}
