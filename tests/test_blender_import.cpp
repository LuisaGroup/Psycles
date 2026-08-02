#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using psycles::contract::DirectLightSampling;
using psycles::contract::EmissionSampling;
using psycles::contract::PixelFilter;
using psycles::contract::WorldSampling;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

void expect_near(float actual, float expected, const std::string &message) {
    expect(
        std::abs(actual - expected) <= 1.0e-6f,
        message + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
}

class TemporaryDirectory {

private:
    std::filesystem::path _path;

public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        _path = std::filesystem::temp_directory_path() /
                ("psycles-integrator-import-" +
                 std::to_string(nonce));
        std::filesystem::create_directories(_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
    }

    [[nodiscard]] const std::filesystem::path &path() const noexcept {
        return _path;
    }
};

void test_integrator_settings_round_trip() {
    TemporaryDirectory temporary;
    {
        std::ofstream geometry{
            temporary.path() / "geometry.bin",
            std::ios::binary};
        geometry.write("PSYGEO1\0", 8);
    }
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << R"JSON({
  "schema": "psycles.blender-scene.v1",
  "images": [],
  "node_groups": [],
  "materials": [
    {
      "name": "Raw Emissive",
      "emission_sampling": "BACK",
      "volume_sampling": "EQUIANGULAR",
      "cycles_sync": {
        "shader_index": 6
      },
      "node_tree": {
        "name": "Raw Volume Material",
        "surface_root": null,
        "volume_root": {
          "node": "Mix Volume",
          "socket": "Shader"
        },
        "displacement_root": null,
        "links": [
          {
            "from_node": "Absorption",
            "from_socket": "Volume",
            "to_node": "Mix Volume",
            "to_socket": "Shader"
          },
          {
            "from_node": "Scatter",
            "from_socket": "Volume",
            "to_node": "Mix Volume",
            "to_socket": "Shader_001"
          }
        ],
        "nodes": [
          {
            "name": "Absorption",
            "label": "",
            "type": "VOLUME_ABSORPTION",
            "bl_idname": "ShaderNodeVolumeAbsorption",
            "inputs": [
              {
                "identifier": "Color",
                "name": "Color",
                "type": "NodeSocketColor",
                "linked": false,
                "default": [0.2, 0.4, 0.6, 1.0]
              },
              {
                "identifier": "Density",
                "name": "Density",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.75
              }
            ],
            "outputs": [
              {
                "identifier": "Volume",
                "name": "Volume",
                "type": "NodeSocketShader",
                "linked": true
              }
            ],
            "properties": {},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Scatter",
            "label": "",
            "type": "VOLUME_SCATTER",
            "bl_idname": "ShaderNodeVolumeScatter",
            "inputs": [
              {
                "identifier": "Color",
                "name": "Color",
                "type": "NodeSocketColor",
                "linked": false,
                "default": [0.8, 0.7, 0.6, 1.0]
              },
              {
                "identifier": "Density",
                "name": "Density",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.5
              },
              {
                "identifier": "Anisotropy",
                "name": "Anisotropy",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.25
              },
              {
                "identifier": "IOR",
                "name": "IOR",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 1.33
              },
              {
                "identifier": "Backscatter",
                "name": "Backscatter",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.1
              },
              {
                "identifier": "Alpha",
                "name": "Alpha",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.5
              },
              {
                "identifier": "Diameter",
                "name": "Diameter",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 20.0
              }
            ],
            "outputs": [
              {
                "identifier": "Volume",
                "name": "Volume",
                "type": "NodeSocketShader",
                "linked": true
              }
            ],
            "properties": {
              "phase": "DRAINE"
            },
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Mix Volume",
            "label": "",
            "type": "MIX_SHADER",
            "bl_idname": "ShaderNodeMixShader",
            "inputs": [
              {
                "identifier": "Fac",
                "name": "Fac",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.375
              },
              {
                "identifier": "Shader",
                "name": "Shader",
                "type": "NodeSocketShader",
                "linked": true
              },
              {
                "identifier": "Shader_001",
                "name": "Shader",
                "type": "NodeSocketShader",
                "linked": true
              }
            ],
            "outputs": [
              {
                "identifier": "Shader",
                "name": "Shader",
                "type": "NodeSocketShader",
                "linked": true
              }
            ],
            "properties": {},
            "special": {},
            "image": null,
            "node_tree": null
          }
        ]
      }
    }
  ],
  "geometries": [],
  "instances": [],
  "lights": [
    {
      "name": "Key",
      "type": "POINT",
      "transform": [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ],
      "color": [1, 1, 1],
      "energy": 10,
      "use_multiple_importance_sampling": false,
      "cast_shadow": false,
      "visibility": {
        "camera": false,
        "diffuse": true,
        "glossy": false,
        "transmission": true,
        "shadow": true,
        "volume_scatter": false
      },
      "is_shadow_catcher": true,
      "cycles_sync": {
        "shader_index": 5,
        "object_index": 9,
        "light_group": 2
      },
      "node_tree": null
    }
  ],
  "world": {
    "name": "Nishita World",
    "color": [0.05, 0.05, 0.05],
    "sampling_method": "MANUAL",
    "sample_map_resolution": 2048,
    "visibility": {
      "camera": true,
      "diffuse": false,
      "glossy": true,
      "transmission": false,
      "shadow": true,
      "volume_scatter": false
    },
    "cycles_sync": {
      "shader_index": 3,
      "object_index": 12
    },
    "node_tree": {
      "name": "World Nodes",
      "surface_root": {
        "node": "Background",
        "socket": "Background"
      },
      "volume_root": null,
      "displacement_root": null,
      "links": [
        {
          "from_node": "Sky",
          "from_socket": "Color",
          "to_node": "Background",
          "to_socket": "Color"
        },
        {
          "from_node": "Background",
          "from_socket": "Background",
          "to_node": "Output",
          "to_socket": "Surface"
        }
      ],
      "nodes": [
        {
          "name": "Sky",
          "label": "",
          "type": "TEX_SKY",
          "bl_idname": "ShaderNodeTexSky",
          "inputs": [
            {
              "identifier": "Vector",
              "name": "Vector",
              "type": "NodeSocketVector",
              "linked": false,
              "default": [0.0, 0.0, 0.0]
            }
          ],
          "outputs": [
            {
              "identifier": "Color",
              "name": "Color",
              "type": "NodeSocketColor",
              "linked": true,
              "default": [0.0, 0.0, 0.0, 0.0]
            }
          ],
          "properties": {
            "sky_type": "SINGLE_SCATTERING",
            "sun_elevation": 0.9250245094299316,
            "sun_rotation": 2.6179938316345215,
            "sun_size": 0.01745329238474369,
            "sun_intensity": 1.25,
            "sun_disc": true,
            "altitude": 123.0,
            "air_density": 0.9,
            "aerosol_density": 1.1,
            "ozone_density": 1.2
          },
          "special": {},
          "image": null,
          "node_tree": null
        },
        {
          "name": "Background",
          "label": "",
          "type": "BACKGROUND",
          "bl_idname": "ShaderNodeBackground",
          "inputs": [
            {
              "identifier": "Color",
              "name": "Color",
              "type": "NodeSocketColor",
              "linked": true,
              "default": [0.05, 0.05, 0.05, 1.0]
            },
            {
              "identifier": "Strength",
              "name": "Strength",
              "type": "NodeSocketFloat",
              "linked": false,
              "default": 2.0
            }
          ],
          "outputs": [
            {
              "identifier": "Background",
              "name": "Background",
              "type": "NodeSocketShader",
              "linked": true
            }
          ],
          "properties": {},
          "special": {},
          "image": null,
          "node_tree": null
        },
        {
          "name": "Output",
          "label": "",
          "type": "OUTPUT_WORLD",
          "bl_idname": "ShaderNodeOutputWorld",
          "inputs": [
            {
              "identifier": "Surface",
              "name": "Surface",
              "type": "NodeSocketShader",
              "linked": true
            },
            {
              "identifier": "Volume",
              "name": "Volume",
              "type": "NodeSocketShader",
              "linked": false
            }
          ],
          "outputs": [],
          "properties": {
            "is_active_output": true,
            "target": "ALL"
          },
          "special": {},
          "image": null,
          "node_tree": null
        }
      ]
    }
  },
  "world_environment": null,
  "render": {
    "width": 480,
    "height": 270,
    "percentage": 50,
    "transparent": true,
    "pixel_filter_type": "GAUSSIAN",
    "filter_width": 2.25,
    "pass_alpha_threshold": 0.375,
    "color_management": {
      "display_device": "Display P3",
      "view_transform": "Filmic",
      "look": "Medium Contrast",
      "exposure": -2.0,
      "gamma": 1.1,
      "use_curve_mapping": false,
      "sequencer_color_space": "Linear Rec.709"
    },
    "cycles": {
      "samples": 37,
      "seed": 305419896,
      "use_adaptive_sampling": false,
      "use_denoising": false,
      "max_bounces": 11,
      "min_light_bounces": 2,
      "diffuse_bounces": 3,
      "glossy_bounces": 4,
      "transmission_bounces": 5,
      "volume_bounces": 6,
      "min_transparent_bounces": 1,
      "transparent_max_bounces": 9,
      "sample_clamp_direct": 1.25,
      "sample_clamp_indirect": 2.5,
      "blur_glossy": 0.75,
      "film_exposure": 0.75,
      "light_sampling_threshold": 0.125,
      "caustics_reflective": false,
      "caustics_refractive": true,
      "use_light_tree": false,
      "direct_light_sampling_type": "NEXT_EVENT_ESTIMATION"
    }
  },
  "camera": {
    "name": "Camera",
    "type": "PERSP",
    "transform": [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ],
    "angle": 0.75,
    "angle_x": 0.9,
    "angle_y": 0.6,
    "sensor_fit": "AUTO",
    "ortho_scale": 1.0,
    "shift_x": 0.0,
    "shift_y": 0.0,
    "clip_start": 0.01,
    "clip_end": 100.0,
    "lens": 50.0,
    "dof": {
      "enabled": false,
      "focus_distance": 10.0,
      "fstop": 2.8,
      "blades": 0,
      "rotation": 0.0,
      "ratio": 1.0
    }
  }
})JSON";
    }

    const auto imported =
        load_blender_scene_bundle(temporary.path());
    expect(imported.ok(), "minimal Blender bundle did not import");
    expect(imported.width == 240u, "render width did not round-trip");
    expect(imported.height == 135u, "render height did not round-trip");
    expect(imported.samples == 37u, "sample count did not round-trip");
    expect(
        imported.seed == 305419896u,
        "sampling seed did not round-trip");
    expect(
        imported.transparent_background,
        "transparent-film setting did not round-trip");
    expect_near(
        imported.pass_alpha_threshold,
        0.375f,
        "pass alpha threshold did not round-trip");
    expect(
        imported.pixel_filter == PixelFilter::gaussian,
        "Gaussian pixel filter did not round-trip");
    expect_near(
        imported.filter_width,
        2.25f,
        "Gaussian filter width did not round-trip");
    expect(
        imported.color_management.display_device == "Display P3",
        "display device did not round-trip");
    expect(
        imported.color_management.view_transform == "Filmic",
        "view transform did not round-trip");
    expect(
        imported.color_management.look == "Medium Contrast",
        "view look did not round-trip");
    expect_near(
        imported.color_management.exposure,
        -2.0f,
        "view exposure did not round-trip");
    expect_near(
        imported.color_management.gamma,
        1.1f,
        "view gamma did not round-trip");
    expect(
        !imported.color_management.use_curve_mapping,
        "curve-mapping setting did not round-trip");
    expect(
        imported.color_management.sequencer_color_space ==
            "Linear Rec.709",
        "sequencer color space did not round-trip");
    expect(
        imported.scene->environment.has_value(),
        "Nishita world did not produce an environment");
    expect(
        imported.scene->world_sampling ==
            WorldSampling::manual,
        "world sampling method did not round-trip");
    expect(
        imported.scene->world_sample_map_resolution == 2048u,
        "world sample-map resolution did not round-trip");
    expect(
        imported.scene->world_visibility_mask ==
            (psycles::contract::visibility_bit(
                 psycles::contract::RayVisibility::camera) |
             psycles::contract::visibility_bit(
                 psycles::contract::RayVisibility::glossy) |
             psycles::contract::visibility_bit(
                 psycles::contract::RayVisibility::shadow)),
        "world ray visibility did not round-trip");
    const auto imported_material =
        imported.scene->materials.find(
            psycles::contract::MaterialId{2u});
    const auto default_material =
        imported.scene->materials.find(
            psycles::contract::MaterialId{1u});
    expect(
        default_material != imported.scene->materials.end() &&
            default_material->second.name ==
                "__cycles_default_surface__",
        "empty Blender material slots did not retain the Cycles default "
        "surface");
    bool default_is_principled = false;
    if (default_material != imported.scene->materials.end()) {
        for (const auto &node :
             default_material->second.shader.nodes()) {
            default_is_principled |=
                node.type ==
                psycles::compiler::node_type::principled_bsdf;
        }
    }
    expect(
        default_is_principled,
        "Cycles default surface was replaced by a diagnostic material");
    expect(
        imported_material !=
            imported.scene->materials.end() &&
            imported_material->second.emission_sampling ==
                EmissionSampling::back &&
            imported_material->second.volume_sampling ==
                psycles::contract::VolumeSampling::
                    equiangular &&
            imported_material->second.cycles_shader_index ==
                std::optional<std::uint32_t>{6u},
        "material sampling policies did not round-trip");
    expect(
        imported_material->second.shader.root(
            psycles::contract::ShaderDomain::volume)
            .has_value(),
        "raw Blender Volume root was not retained");
    bool has_absorption = false;
    bool has_scatter = false;
    bool has_volume_mix = false;
    bool has_transparent_boundary = false;
    for (const auto &node :
         imported_material->second.shader.nodes()) {
        has_absorption |=
            node.type ==
            psycles::compiler::node_type::
                volume_absorption;
        has_scatter |=
            node.type ==
            psycles::compiler::node_type::volume_scatter;
        has_volume_mix |=
            node.type ==
            psycles::compiler::node_type::mix_volume;
        has_transparent_boundary |=
            node.type ==
            psycles::compiler::node_type::transparent_bsdf;
    }
    expect(
        has_absorption && has_scatter && has_volume_mix,
        "Blender Volume closure tree was flattened or typed as a "
        "surface closure");
    expect(
        has_transparent_boundary,
        "volume-only Blender material did not receive a transparent "
        "surface boundary");
    const auto imported_light =
        imported.scene->lights.find(
            psycles::contract::LightId{1u});
    expect(
        imported_light != imported.scene->lights.end() &&
            !imported_light->second.use_mis &&
            !imported_light->second.cast_shadow &&
            imported_light->second.is_shadow_catcher &&
            imported_light->second.cycles_shader_index ==
                std::optional<std::uint32_t>{5u} &&
            imported_light->second.cycles_object_index ==
                std::optional<std::uint32_t>{9u} &&
            imported_light->second.cycles_light_group == 2 &&
            imported_light->second.visibility_mask ==
                (psycles::contract::visibility_bit(
                     psycles::contract::RayVisibility::diffuse) |
                 psycles::contract::visibility_bit(
                     psycles::contract::RayVisibility::transmission) |
                 psycles::contract::visibility_bit(
                     psycles::contract::RayVisibility::shadow)),
        "light Cycles identity or shader policy did not round-trip");
    const auto imported_world =
        imported.scene->materials.find(
            psycles::contract::MaterialId{3u});
    expect(
        imported_world != imported.scene->materials.end() &&
            imported_world->second.cycles_shader_index ==
                std::optional<std::uint32_t>{3u} &&
            imported.scene
                    ->cycles_background_object_index ==
                std::optional<std::uint32_t>{12u},
        "world object/shader identity did not round-trip");
    expect(
        imported.scene->environment->nishita.has_value(),
        "Nishita world was not kept procedural");
    expect(
        imported.scene->environment->pixels.empty(),
        "Nishita world was unexpectedly baked to pixels");
    const auto &sky =
        *imported.scene->environment->nishita;
    expect_near(
        sky.sun_elevation,
        0.9250245094299316f,
        "Nishita elevation mismatch");
    expect_near(
        sky.sun_rotation,
        3.6651914755450647f,
        "Nishita Cycles rotation mismatch");
    expect_near(
        sky.angular_diameter,
        0.01745329238474369f,
        "Nishita angular diameter mismatch");
    expect_near(
        sky.sun_intensity,
        1.25f,
        "Nishita sun intensity mismatch");
    expect_near(
        sky.altitude,
        123.0f,
        "Nishita altitude mismatch");
    expect_near(
        sky.air_density,
        0.9f,
        "Nishita air density mismatch");
    expect_near(
        sky.dust_density,
        1.1f,
        "Nishita dust density mismatch");
    expect_near(
        sky.ozone_density,
        1.2f,
        "Nishita ozone density mismatch");
    expect_near(
        sky.background_strength,
        2.0f,
        "Nishita background strength mismatch");

    const auto &integrator = imported.integrator;
    expect(integrator.max_bounces == 11u, "max bounce mismatch");
    expect(integrator.min_bounces == 2u, "min bounce mismatch");
    expect(integrator.diffuse_bounces == 3u, "diffuse bounce mismatch");
    expect(integrator.glossy_bounces == 4u, "glossy bounce mismatch");
    expect(
        integrator.transmission_bounces == 5u,
        "transmission bounce mismatch");
    expect(integrator.volume_bounces == 6u, "volume bounce mismatch");
    expect(
        integrator.transparent_min_bounces == 1u,
        "transparent minimum mismatch");
    expect(
        integrator.transparent_max_bounces == 9u,
        "transparent maximum mismatch");
    expect_near(
        integrator.sample_clamp_direct,
        1.25f,
        "direct clamp mismatch");
    expect_near(
        integrator.sample_clamp_indirect,
        2.5f,
        "indirect clamp mismatch");
    expect_near(
        integrator.filter_glossy,
        0.75f,
        "filter glossy mismatch");
    expect_near(
        integrator.film_exposure,
        0.75f,
        "film exposure mismatch");
    expect_near(
        integrator.light_sampling_threshold,
        0.125f,
        "light sampling threshold mismatch");
    expect(
        !integrator.reflective_caustics &&
            integrator.refractive_caustics,
        "caustics settings mismatch");
    expect(
        !integrator.use_light_tree,
        "light-tree setting mismatch");
    expect(
        integrator.direct_light_sampling ==
            DirectLightSampling::next_event_estimation,
        "direct-light sampling mode mismatch");

    std::ifstream source{temporary.path() / "scene.json"};
    std::string light_tree_scene{
        std::istreambuf_iterator<char>{source},
        std::istreambuf_iterator<char>{}};

    auto bump_scene = light_tree_scene;
    const auto replace_once = [](
                                  std::string &text,
                                  std::string_view before,
                                  std::string_view after) {
        const auto position = text.find(before);
        expect(
            position != std::string::npos,
            "bump regression fixture is missing its insertion point");
        text.replace(position, before.size(), after);
    };
    replace_once(
        bump_scene,
        "\"volume_sampling\": \"EQUIANGULAR\",",
        "\"volume_sampling\": \"EQUIANGULAR\",\n"
        "      \"displacement_method\": \"BUMP\",");
    replace_once(
        bump_scene,
        "\"surface_root\": null,",
        "\"surface_root\": {\n"
        "          \"node\": \"Diffuse\",\n"
        "          \"socket\": \"BSDF\"\n"
        "        },");
    replace_once(
        bump_scene,
        "\"displacement_root\": null,",
        "\"displacement_root\": {\n"
        "          \"node\": \"Displacement\",\n"
        "          \"socket\": \"Displacement\"\n"
        "        },");
    replace_once(
        bump_scene,
        "\"links\": [",
        "\"links\": [\n"
        "          {\n"
        "            \"from_node\": \"Texture Coordinate\",\n"
        "            \"from_socket\": \"Reflection\",\n"
        "            \"to_node\": \"Diffuse\",\n"
        "            \"to_socket\": \"Color\"\n"
        "          },");
    replace_once(
        bump_scene,
        R"JSON(        "nodes": [
          {
            "name": "Absorption",)JSON",
        R"JSON(        "nodes": [
          {
            "name": "Texture Coordinate",
            "label": "",
            "type": "TEX_COORD",
            "bl_idname": "ShaderNodeTexCoord",
            "inputs": [],
            "outputs": [
              {
                "identifier": "Reflection",
                "name": "Reflection",
                "type": "NodeSocketVector",
                "linked": true,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "properties": {"from_instancer": false},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Diffuse",
            "label": "",
            "type": "BSDF_DIFFUSE",
            "bl_idname": "ShaderNodeBsdfDiffuse",
            "inputs": [
              {
                "identifier": "Color",
                "name": "Color",
                "type": "NodeSocketColor",
                "linked": false,
                "default": [0.8, 0.8, 0.8, 1.0]
              },
              {
                "identifier": "Roughness",
                "name": "Roughness",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.0
              },
              {
                "identifier": "Normal",
                "name": "Normal",
                "type": "NodeSocketVector",
                "linked": false,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "outputs": [
              {
                "identifier": "BSDF",
                "name": "BSDF",
                "type": "NodeSocketShader",
                "linked": true
              }
            ],
            "properties": {},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Displacement",
            "label": "",
            "type": "DISPLACEMENT",
            "bl_idname": "ShaderNodeDisplacement",
            "inputs": [
              {
                "identifier": "Height",
                "name": "Height",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.75
              },
              {
                "identifier": "Midlevel",
                "name": "Midlevel",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.5
              },
              {
                "identifier": "Scale",
                "name": "Scale",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.1
              },
              {
                "identifier": "Normal",
                "name": "Normal",
                "type": "NodeSocketVector",
                "linked": false,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "outputs": [
              {
                "identifier": "Displacement",
                "name": "Displacement",
                "type": "NodeSocketVector",
                "linked": true,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "properties": {"space": "OBJECT"},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Absorption",)JSON");
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << bump_scene;
    }
    const auto bump_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        bump_imported.ok(),
        "bump-only displacement material did not import");
    const auto bump_material =
        bump_imported.scene->materials.find(
            psycles::contract::MaterialId{2u});
    expect(
        bump_material != bump_imported.scene->materials.end() &&
            bump_material->second.shader.root(
                psycles::contract::ShaderDomain::displacement)
                .has_value(),
        "automatic bump was not retained as a displacement root");
    bool has_automatic_bump = false;
    for (const auto &node : bump_material->second.shader.nodes()) {
        has_automatic_bump |=
            node.type == psycles::compiler::node_type::bump;
    }
    expect(
        has_automatic_bump,
        "Blender bump-only displacement did not lower to a bump node");

    psycles::compiler::ShaderCompiler bump_compiler{
        psycles::compiler::make_core_node_registry()};
    const auto bump_shader = bump_compiler.compile(
        bump_material->second.shader);
    expect(
        bump_shader.ok(),
        "automatic displacement bump graph did not validate");
    const auto bump_surface =
        psycles::compiler::compile_surface_program(
            *bump_shader.program);
    expect(
        bump_surface.ok(),
        "automatic displacement bump did not lower to a surface program");
    bool has_bump_instruction = false;
    for (const auto &instruction :
         bump_surface.program->value_instructions()) {
        has_bump_instruction |=
            instruction.operation ==
            psycles::compiler::ValueOperation::bump;
    }
    expect(
        has_bump_instruction,
        "automatic displacement bump emitted no value instruction");

    auto combined_displacement_scene = bump_scene;
    replace_once(
        combined_displacement_scene,
        "\"displacement_method\": \"BUMP\"",
        "\"displacement_method\": \"BOTH\"");
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << combined_displacement_scene;
    }
    const auto combined_displacement_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        combined_displacement_imported.ok(),
        "combined displacement was not accepted as a bump approximation");
    bool named_combined_displacement_warning = false;
    for (const auto &diagnostic :
         combined_displacement_imported.diagnostics) {
        named_combined_displacement_warning |=
            diagnostic.severity ==
                psycles::adapter::
                    BlenderSceneDiagnosticSeverity::warning &&
            diagnostic.message.find("using a bump approximation") !=
                std::string::npos;
    }
    expect(
        named_combined_displacement_warning,
        "combined displacement approximation has no named warning");

    auto true_displacement_scene = bump_scene;
    replace_once(
        true_displacement_scene,
        "\"displacement_method\": \"BUMP\"",
        "\"displacement_method\": \"DISPLACEMENT\"");
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << true_displacement_scene;
    }
    const auto true_displacement_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        true_displacement_imported.ok(),
        "true geometry displacement was not accepted as a bump approximation");
    bool named_displacement_warning = false;
    for (const auto &diagnostic :
         true_displacement_imported.diagnostics) {
        named_displacement_warning |=
            diagnostic.severity ==
                psycles::adapter::
                    BlenderSceneDiagnosticSeverity::warning &&
            diagnostic.message.find("using a bump approximation") !=
                std::string::npos;
    }
    expect(
        named_displacement_warning,
        "true displacement approximation has no named warning");

    auto box_filter_scene = light_tree_scene;
    constexpr std::string_view gaussian_filter =
        "\"pixel_filter_type\": \"GAUSSIAN\"";
    constexpr std::string_view box_filter =
        "\"pixel_filter_type\": \"BOX\"";
    const auto filter_position =
        box_filter_scene.find(gaussian_filter);
    expect(
        filter_position != std::string::npos,
        "test scene has no Gaussian pixel filter");
    box_filter_scene.replace(
        filter_position,
        gaussian_filter.size(),
        box_filter);
    constexpr std::string_view gaussian_width =
        "\"filter_width\": 2.25";
    constexpr std::string_view dormant_box_width =
        "\"filter_width\": 0.01";
    const auto filter_width_position =
        box_filter_scene.find(gaussian_width);
    expect(
        filter_width_position != std::string::npos,
        "test scene has no Gaussian filter width");
    box_filter_scene.replace(
        filter_width_position,
        gaussian_width.size(),
        dormant_box_width);
    {
        std::ofstream scene{
            temporary.path() / "scene.json"};
        scene << box_filter_scene;
    }
    const auto box_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        box_imported.ok() &&
            box_imported.pixel_filter == PixelFilter::box,
        "BOX pixel filter did not import");
    expect_near(
        box_imported.filter_width,
        1.0f,
        "BOX filter did not normalize to one pixel");

    auto legacy_nishita_scene = light_tree_scene;
    constexpr std::string_view current_sky_type =
        "\"sky_type\": \"SINGLE_SCATTERING\"";
    constexpr std::string_view legacy_sky_type =
        "\"sky_type\": \"NISHITA\"";
    const auto sky_type_position =
        legacy_nishita_scene.find(current_sky_type);
    expect(
        sky_type_position != std::string::npos,
        "test scene has no current single-scattering sky type");
    legacy_nishita_scene.replace(
        sky_type_position,
        current_sky_type.size(),
        legacy_sky_type);
    constexpr std::string_view current_aerosol =
        "\"aerosol_density\": 1.1";
    constexpr std::string_view legacy_dust =
        "\"dust_density\": 1.1";
    const auto aerosol_position =
        legacy_nishita_scene.find(current_aerosol);
    expect(
        aerosol_position != std::string::npos,
        "test scene has no current aerosol-density property");
    legacy_nishita_scene.replace(
        aerosol_position,
        current_aerosol.size(),
        legacy_dust);
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << legacy_nishita_scene;
    }
    const auto legacy_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        legacy_imported.ok() &&
            legacy_imported.scene->environment.has_value() &&
            legacy_imported.scene->environment->nishita.has_value(),
        "legacy NISHITA world was not kept procedural");
    expect_near(
        legacy_imported.scene->environment->nishita
            ->dust_density,
        1.1f,
        "legacy Nishita dust density mismatch");

    auto hosek_scene = light_tree_scene;
    replace_once(
        hosek_scene,
        "\"sky_type\": \"SINGLE_SCATTERING\"",
        "\"sky_type\": \"HOSEK_WILKIE\"");
    replace_once(
        hosek_scene,
        "\"sun_elevation\": 0.9250245094299316,",
        "\"sun_direction\": [-0.94615382, 0.06153846, "
        "0.31781429],\n"
        "            \"turbidity\": 2.9,\n"
        "            \"ground_albedo\": 0.3,\n"
        "            \"sun_elevation\": 0.9250245094299316,");
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << hosek_scene;
    }
    const auto hosek_imported =
        load_blender_scene_bundle(temporary.path());
    expect(
        hosek_imported.ok(),
        "legacy Hosek-Wilkie world did not import");
    const auto hosek_world =
        hosek_imported.scene->materials.find(
            psycles::contract::MaterialId{3u});
    expect(
        hosek_world != hosek_imported.scene->materials.end(),
        "legacy Hosek-Wilkie world material is missing");
    bool has_hosek_node = false;
    for (const auto &node : hosek_world->second.shader.nodes()) {
        has_hosek_node |=
            node.type ==
            psycles::compiler::node_type::hosek_wilkie_sky;
    }
    expect(
        has_hosek_node,
        "legacy Hosek-Wilkie sky was rewritten to another model");
    psycles::compiler::ShaderCompiler hosek_compiler{
        psycles::compiler::make_core_node_registry()};
    const auto hosek_shader = hosek_compiler.compile(
        hosek_world->second.shader);
    expect(
        hosek_shader.ok(),
        "legacy Hosek-Wilkie graph did not validate");
    const auto hosek_surface =
        psycles::compiler::compile_surface_program(
            *hosek_shader.program);
    expect(
        hosek_surface.ok(),
        "legacy Hosek-Wilkie graph did not lower");
    bool has_hosek_instruction = false;
    for (const auto &instruction :
         hosek_surface.program->value_instructions()) {
        has_hosek_instruction |=
            instruction.operation ==
                psycles::compiler::ValueOperation::
                    hosek_wilkie_sky &&
            instruction.static_table.size() == 33u;
    }
    expect(
        has_hosek_instruction,
        "legacy Hosek-Wilkie sky emitted no cooked model");

    const auto setting =
        light_tree_scene.find("\"use_light_tree\": false");
    expect(
        setting != std::string::npos,
        "test scene has no light-tree setting");
    light_tree_scene.replace(
        setting,
        std::string{"\"use_light_tree\": false"}.size(),
        "\"use_light_tree\": true");
    {
        std::ofstream scene{temporary.path() / "scene.json"};
        scene << light_tree_scene;
    }
    const auto rejected =
        load_blender_scene_bundle(temporary.path());
    expect(
        !rejected.ok(),
        "unsupported Cycles light tree was silently accepted");
    bool named_diagnostic = false;
    for (const auto &diagnostic : rejected.diagnostics) {
        named_diagnostic |=
            diagnostic.message.find("light-tree") !=
            std::string::npos;
    }
    expect(
        named_diagnostic,
        "light-tree rejection has no named diagnostic");
}

}// namespace

int main() {
    try {
        test_integrator_settings_round_trip();
    } catch (const std::exception &exception) {
        std::cerr << "test failure: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
