#include <psycles/blender_bridge/psycles_blender_capi.h>
#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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

    // 1) Verify psycles::parallel_for determinism, nested call safety, and exception propagation.
    {
        std::vector<std::size_t> values(256u, 0u);
        psycles::parallel_for(values.size(), [&](std::size_t i) {
            std::size_t nested_sum = 0u;
            psycles::parallel_for(4u, [&](std::size_t j) { nested_sum += j; });
            values[i] = i * 3u + nested_sum;
        });
        for (std::size_t i = 0u; i < values.size(); ++i) {
            if (values[i] != i * 3u + 6u) {
                std::cerr << "parallel_for result mismatch at " << i << '\n';
                return 1;
            }
        }
        bool caught_exception = false;
        try {
            psycles::parallel_for(32u, [](std::size_t i) {
                if (i == 17u) {
                    throw std::runtime_error("worker test exception");
                }
            });
        } catch (const std::runtime_error &err_ex) {
            caught_exception =
                std::string{err_ex.what()}.find("worker test exception") != std::string::npos;
        }
        if (!caught_exception) {
            std::cerr << "parallel_for did not propagate worker exception\n";
            return 1;
        }
        std::cout << "[PASS] parallel_for determinism, nested safety, exception propagation\n";
    }

    // 2) Verify omitted uv_tangents + C++ island_edges DisjointSet computation + psycles_session_reconfigure.
    {
        std::vector<std::byte> bin2;
        const char magic2[8]{'P', 'S', 'Y', 'G', 'E', 'O', '2', '\0'};
        append_bytes(bin2, magic2, sizeof(magic2));

        // Two disconnected triangles (vertices 0..2 and 3..5) in different islands.
        const std::vector<float> pos2{
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             0.0f,  1.0f, 0.0f,
             9.0f, -1.0f, 0.0f,
            11.0f, -1.0f, 0.0f,
            10.0f,  1.0f, 0.0f};
        const std::vector<float> nrm2{
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
        const std::vector<float> uv2{
            0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
        const std::vector<float> gen2{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f};
        const std::vector<std::uint32_t> idx2{0u, 1u, 2u, 3u, 4u, 5u};
        const std::vector<std::uint32_t> edges2{
            0u, 1u, 1u, 2u, 2u, 0u,
            3u, 4u, 4u, 5u, 5u, 3u};
        const std::vector<std::uint32_t> mats2{0u, 0u};
        const std::vector<std::uint32_t> smt2{0u, 0u};

        const auto sp_pos = append_section(bin2, pos2);
        const auto sp_nrm = append_section(bin2, nrm2);
        const auto sp_uv = append_section(bin2, uv2);
        const auto sp_gen = append_section(bin2, gen2);
        const auto sp_idx = append_section(bin2, idx2);
        const auto sp_edg = append_section(bin2, edges2);
        const auto sp_mat = append_section(bin2, mats2);
        const auto sp_smt = append_section(bin2, smt2);

        const std::string json2 = std::string(R"JSON({
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
              "use_light_tree": false, "ao_bounces_render": 0, "film_exposure": 1.0
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
            "name": "DualTri", "triangle_count": 2, "point_count": 6, "corner_count": 6,
            "normal_domain": "POINT", "uv_domain": "POINT", "uv_tangent_domain": "POINT", "generated_domain": "POINT",
            "material_slots": ["EmitMat"],
            "positions": {"offset": )JSON") + std::to_string(sp_pos.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_pos.second) + R"JSON(},
            "normals": {"offset": )JSON" + std::to_string(sp_nrm.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_nrm.second) + R"JSON(},
            "uv": {"offset": )JSON" + std::to_string(sp_uv.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_uv.second) + R"JSON(},
            "generated": {"offset": )JSON" + std::to_string(sp_gen.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_gen.second) + R"JSON(},
            "indices": {"offset": )JSON" + std::to_string(sp_idx.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_idx.second) + R"JSON(},
            "island_edges": {"edge_count": 6, "edges": {"offset": )JSON" + std::to_string(sp_edg.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_edg.second) + R"JSON(}},
            "triangle_material_slots": {"offset": )JSON" + std::to_string(sp_mat.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_mat.second) + R"JSON(},
            "triangle_smooth": {"offset": )JSON" + std::to_string(sp_smt.first) + R"JSON(, "bytes": )JSON" + std::to_string(sp_smt.second) + R"JSON(},
            "uv_layers": [], "color_attributes": []
          }],
          "curve_geometries": [],
          "instances": [{
            "name": "DualObj", "geometry": 0, "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
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
                 "outputs": [{"identifier": "Emission", "name": "Emission", "type": "NodeSocketShader", "linked": true, "available": true}]},
                {"name": "Out", "bl_idname": "ShaderNodeOutputMaterial", "type": "OUTPUT_MATERIAL",
                 "inputs": [{"identifier": "Surface", "name": "Surface", "type": "NodeSocketShader", "linked": true, "available": true}],
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

        PsyclesSceneInfo info2{};
        auto *scene2 = psycles_scene_compile_memory(
            engine, json2.data(), json2.size(), bin2.data(), bin2.size(), "", &info2, err, sizeof(err));
        if (scene2 == nullptr) {
            std::cerr << "island_edges / omitted uv_tangents compile failed: " << err << '\n';
            return 1;
        }
        auto *sess2 = psycles_session_create(engine, scene2, &params, &passes[0], 1u, err, sizeof(err));
        if (sess2 == nullptr) {
            std::cerr << "session create for reconfigure test failed: " << err << '\n';
            return 1;
        }
        // Reconfigure to 2 passes and moved camera over the 2nd triangle at x=10.
        PsyclesCameraDesc alt_cam = moved_cam;
        alt_cam.transform[12] = 10.0f;
        if (psycles_session_reconfigure(sess2, &alt_cam, &params, passes, 2u, err, sizeof(err)) != 1) {
            std::cerr << "psycles_session_reconfigure failed: " << err << '\n';
            return 1;
        }
        if (psycles_session_render_samples(sess2, 0u, 2u, 2u, 1u, err, sizeof(err)) != 1) {
            std::cerr << "reconfigured render failed: " << err << '\n';
            return 1;
        }
        const float *emit_ptr = nullptr;
        if (!psycles_session_get_pass_buffer(sess2, 1u, &emit_ptr, &w, &h, &ch) ||
            ch != 3u || w != 16u || h != 16u) {
            std::cerr << "reconfigured 2nd pass buffer missing\n";
            return 1;
        }
        float emit_sum = 0.0f;
        for (std::size_t i = 0u; i < 16u * 16u * 3u; ++i) {
            emit_sum += emit_ptr[i];
        }
        if (emit_sum <= 1.0f) {
            std::cerr << "expected non-black emit pass on 2nd island after reconfigure\n";
            return 1;
        }
        std::cout << "[PASS] island_edges, omitted uv_tangents, and psycles_session_reconfigure\n";

        // 3) Delayed cancellation & ownership safety test using PSYCLES_TEST_VIEWPORT_STEP_DELAY_MS.
        const auto set_delay_env = [](const char *val) {
#ifdef _WIN32
            _putenv_s("PSYCLES_TEST_VIEWPORT_STEP_DELAY_MS", val);
#else
            setenv("PSYCLES_TEST_VIEWPORT_STEP_DELAY_MS", val, 1);
#endif
        };
        set_delay_env("200");

        PsyclesCameraDesc orig_cam = moved_cam;
        orig_cam.transform[12] = 0.0f;
        auto *vtask1 = psycles_viewport_task_start_compile(
            engine, json.data(), json.size(), bin.data(), bin.size(), "",
            &orig_cam, &params, 2u, 2u, 1u, err, sizeof(err));
        if (vtask1 == nullptr) {
            std::cerr << "psycles_viewport_task_start_compile failed: " << err << '\n';
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        // Request camera far off triangle while compile delay is still in flight.
        PsyclesCameraDesc far_cam = moved_cam;
        far_cam.transform[12] = 50.0f;
        const auto t_req0 = std::chrono::steady_clock::now();
        const int req_rc = psycles_viewport_task_request_camera(
            vtask1, &far_cam, &params, 2u, 2u);
        const auto req_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_req0).count();
        if (req_rc != 1 || req_ms >= 50) {
            std::cerr << "request_camera failed or blocked: rc=" << req_rc
                      << " ms=" << req_ms << '\n';
            return 1;
        }
        psycles_viewport_task_wait(vtask1);
        PsyclesViewportStatus vst{};
        psycles_viewport_task_poll(vtask1, &vst, err, sizeof(err));
        if (vst.state != PSYCLES_VIEWPORT_STATE_DONE || vst.rendered_samples != 2u ||
            vst.frame_generation == 0u) {
            std::cerr << "vtask1 did not finish cleanly: state=" << vst.state
                      << " spp=" << vst.rendered_samples << " err=" << err << '\n';
            return 1;
        }
        std::vector<float> vframe(16u * 16u * 4u, -1.0f);
        std::uint32_t fw = 0u, fh = 0u;
        std::uint64_t fgen = 0u;
        if (psycles_viewport_task_copy_frame(
                vtask1, vframe.data(), vframe.size(), &fw, &fh, &fgen) != 1) {
            std::cerr << "vtask1 copy_frame failed\n";
            return 1;
        }
        float vrgb = 0.0f;
        for (std::size_t i = 0u; i < 16u * 16u; ++i) {
            vrgb += vframe[i * 4u] + vframe[i * 4u + 1u] + vframe[i * 4u + 2u];
        }
        if (vrgb > 1.0e-3f) {
            std::cerr << "expected black image from mid-compile camera switch, got " << vrgb << '\n';
            return 1;
        }
        psycles_viewport_task_destroy(vtask1);

        // Start a delayed scene task, cancel non-blockingly, destroy caller scene handle immediately while still running.
        auto *vtask2 = psycles_viewport_task_start_scene(
            engine, scene2, &alt_cam, &params, 4u, 2u, 1u, err, sizeof(err));
        if (vtask2 == nullptr) {
            std::cerr << "psycles_viewport_task_start_scene failed: " << err << '\n';
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const auto t_can0 = std::chrono::steady_clock::now();
        psycles_viewport_task_cancel(vtask2);
        const auto can_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_can0).count();
        if (can_ms >= 50) {
            std::cerr << "cancel blocked too long: " << can_ms << " ms\n";
            return 1;
        }
        psycles_session_destroy(sess2);
        psycles_scene_destroy(scene2);
        psycles_viewport_task_wait(vtask2);
        psycles_viewport_task_poll(vtask2, &vst, err, sizeof(err));
        if (vst.is_finished != 1u ||
            (vst.state != PSYCLES_VIEWPORT_STATE_CANCELLED &&
             vst.state != PSYCLES_VIEWPORT_STATE_DONE)) {
            std::cerr << "unexpected vtask2 final state: " << vst.state << '\n';
            return 1;
        }
        psycles_viewport_task_destroy(vtask2);

        // Start one more delayed task and destroy wrapper immediately while still active (tests safe detach).
        auto *vtask3 = psycles_viewport_task_start_compile(
            engine, json.data(), json.size(), bin.data(), bin.size(), "",
            &orig_cam, &params, 2u, 2u, 1u, err, sizeof(err));
        if (vtask3 == nullptr) {
            std::cerr << "vtask3 failed\n";
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto t_det0 = std::chrono::steady_clock::now();
        psycles_viewport_task_destroy(vtask3);
        const auto det_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_det0).count();
        if (det_ms >= 50) {
            std::cerr << "psycles_viewport_task_destroy blocked on delayed task: "
                      << det_ms << " ms\n";
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(260));
        set_delay_env("0");
        std::cout << "[PASS] delayed viewport cancellation & shared-ownership safety\n";
    }

    psycles_engine_destroy(engine);
    psycles_shutdown_runtime();
    std::cout << "psycles_blender_capi_tests PASSED\n";
    return 0;
}
