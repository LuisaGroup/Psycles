#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

#include "blender_texture_coordinate_import_expectations.h"

#include <array>
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
#include <utility>
#include <vector>
void test_blender_legacy_mix_import();
void test_blender_light_path_portal_depth_import();
void test_blender_vector_import();
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
  expect(std::abs(actual - expected) <= 1.0e-6f,
         message + ": got " + std::to_string(actual) + ", expected " +
             std::to_string(expected));
}

class TemporaryDirectory {

private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-integrator-import-" + std::to_string(nonce));
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
    std::ofstream geometry{temporary.path() / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema": "psycles.blender-scene.v1",
  "cycles_sync": {"object_count": 13, "uses_light_linking": true},
  "images": [],
  "node_groups": [],
  "materials": [
    {
      "name": "Raw Emissive", "use_transparent_shadow": false,
      "use_bump_map_correction": false,
      "emission_sampling": "BACK",
      "volume_sampling": "EQUIANGULAR",
      "volume_interpolation": "CUBIC",
      "cycles_sync": {
        "shader_index": 6,
        "pass_id": 27
      },
      "node_tree": {
        "name": "Raw Volume Material",
        "surface_root": null,
        "volume_root": {
          "node": "Add Volume",
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
          },
          {
            "from_node": "Mix Volume",
            "from_socket": "Shader",
            "to_node": "Add Volume",
            "to_socket": "Shader"
          },
          {
            "from_node": "Geometry",
            "from_socket": "Position",
            "to_node": "Emission",
            "to_socket": "Color"
          },
          {
            "from_node": "Hair Info",
            "from_socket": "Intercept",
            "to_node": "Emission",
            "to_socket": "Strength"
          },
          {
            "from_node": "Emission",
            "from_socket": "Emission",
            "to_node": "Add Volume",
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
          },
          {
            "name": "Geometry",
            "label": "",
            "type": "NEW_GEOMETRY",
            "bl_idname": "ShaderNodeNewGeometry",
            "inputs": [],
            "outputs": [
              {
                "identifier": "Position",
                "name": "Position",
                "type": "NodeSocketVector",
                "linked": true,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "properties": {},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Hair Info",
            "label": "",
            "type": "HAIR_INFO",
            "bl_idname": "ShaderNodeHairInfo",
            "inputs": [],
            "outputs": [
              {
                "identifier": "Intercept",
                "name": "Intercept",
                "type": "NodeSocketFloat",
                "linked": true,
                "default": 0.0
              }
            ],
            "properties": {},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Emission",
            "label": "",
            "type": "EMISSION",
            "bl_idname": "ShaderNodeEmission",
            "inputs": [
              {
                "identifier": "Color",
                "name": "Color",
                "type": "NodeSocketColor",
                "linked": true,
                "default": [0.25, 0.5, 0.75, 1.0]
              },
              {
                "identifier": "Strength",
                "name": "Strength",
                "type": "NodeSocketFloat",
                "linked": true,
                "default": 1.2
              }
            ],
            "outputs": [
              {
                "identifier": "Emission",
                "name": "Emission",
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
            "name": "Add Volume",
            "label": "",
            "type": "ADD_SHADER",
            "bl_idname": "ShaderNodeAddShader",
            "inputs": [
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
      "type": "AREA",
      "transform": [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ],
      "color": [1, 1, 1],
      "energy": 10,
      "shape": "DISK",
      "size": 1.3,
      "size_y": 0.25,
      "use_multiple_importance_sampling": false,
      "is_portal": true,
      "max_bounces": 13,
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
      "asset_name": "Lighting Collection",
      "use_holdout": true,
      "is_caustics_caster": true,
      "is_caustics_receiver": false,
      "object_color": [0.125, 0.25, 0.5],
      "object_alpha": 0.75,
      "object_pass_id": 19,
      "random_id": 2147483648,
      "particle_index": 7,
      "dupli_generated": [-0.25, 0.125, 0.75],
      "dupli_uv": [0.2, 0.8],
      "shadow_terminator_shading_offset": 0.375,
      "shadow_terminator_geometry_offset": 0.625,
      "cycles_particle_source": {
        "system": 3,
        "source_index": 5,
        "age": 2.5,
        "lifetime": 12.0,
        "location": [1, 2, 3],
        "rotation": [0.1, 0.2, 0.3, 0.4],
        "size": 0.75,
        "velocity": [4, 5, 6],
        "angular_velocity": [7, 8, 9]
      },
      "cycles_sync": {
        "shader_index": 5,
        "pass_id": 0,
        "object_index": 9,
        "light_group": 2
      },
      "node_tree": null
    }
  ],
  "world": {
    "name": "Nishita World",
    "color": [0.05, 0.05, 0.05],
    "volume_sampling": "DISTANCE",
    "volume_interpolation": "CUBIC",
    "sampling_method": "MANUAL",
    "sample_map_resolution": 2048,
    "max_bounces": 23,
    "use_shadows": false,
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
      "pass_id": 0,
      "object_index": 12,
      "light_group": 4
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
      "effective_seed": 2271560481,
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
      "use_fast_gi": true, "fast_gi_method": "REPLACE",
      "ao_bounces": 3, "ao_bounces_render": 7, "ao_factor": 0.625,
      "blur_glossy": 0.75,
      "ao_distance": 4.25,
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

  const auto imported = load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "minimal Blender bundle did not import");
  expect(imported.width == 240u, "render width did not round-trip");
  expect(imported.height == 135u, "render height did not round-trip");
  expect(imported.samples == 37u, "sample count did not round-trip");
  expect(imported.seed == 2271560481u,
         "effective Cycles sampling seed did not round-trip");
  expect(imported.transparent_background,
         "transparent-film setting did not round-trip");
  expect_near(imported.pass_alpha_threshold, 0.375f,
              "pass alpha threshold did not round-trip");
  expect(imported.pixel_filter == PixelFilter::gaussian,
         "Gaussian pixel filter did not round-trip");
  expect_near(imported.filter_width, 2.25f,
              "Gaussian filter width did not round-trip");
  expect(imported.color_management.display_device == "Display P3",
         "display device did not round-trip");
  expect(imported.color_management.view_transform == "Filmic",
         "view transform did not round-trip");
  expect(imported.color_management.look == "Medium Contrast",
         "view look did not round-trip");
  expect_near(imported.color_management.exposure, -2.0f,
              "view exposure did not round-trip");
  expect_near(imported.color_management.gamma, 1.1f,
              "view gamma did not round-trip");
  expect(!imported.color_management.use_curve_mapping,
         "curve-mapping setting did not round-trip");
  expect(imported.color_management.sequencer_color_space == "Linear Rec.709",
         "sequencer color space did not round-trip");
  expect(imported.scene->environment.has_value(),
         "Nishita world did not produce an environment");
  expect(imported.scene->world_sampling == WorldSampling::manual,
         "world sampling method did not round-trip");
  expect(imported.scene->world_sample_map_resolution == 2048u,
         "world sample-map resolution did not round-trip");
  expect(imported.scene->world_max_bounces == 23u,
         "world max-bounces policy did not round-trip");
  expect(!imported.scene->world_cast_shadow &&
             imported.scene->cycles_background_light_group == 4,
         "world shadow/light-group policy did not round-trip");
  expect(imported.scene->cycles_background_asset_name == "Nishita World",
         "world asset identity did not round-trip");
  expect(imported.scene->cycles_uses_light_linking,
         "light-linking capability did not round-trip");
  expect(imported.scene->world_visibility_mask ==
             (psycles::contract::visibility_bit(
                  psycles::contract::RayVisibility::camera) |
              psycles::contract::visibility_bit(
                  psycles::contract::RayVisibility::glossy) |
              psycles::contract::visibility_bit(
                  psycles::contract::RayVisibility::shadow)),
         "world ray visibility did not round-trip");
  const auto imported_material =
      imported.scene->materials.find(psycles::contract::MaterialId{2u});
  const auto default_material =
      imported.scene->materials.find(psycles::contract::MaterialId{1u});
  expect(default_material != imported.scene->materials.end() &&
             default_material->second.name == "__cycles_default_surface__",
         "empty Blender material slots did not retain the Cycles default "
         "surface");
  bool default_is_principled = false;
  if (default_material != imported.scene->materials.end()) {
    for (const auto &node : default_material->second.shader.nodes()) {
      default_is_principled |=
          node.type == psycles::compiler::node_type::principled_bsdf;
    }
  }
  expect(default_is_principled,
         "Cycles default surface was replaced by a diagnostic material");
  expect(imported_material != imported.scene->materials.end() &&
             !imported_material->second.use_transparent_shadow &&
             !imported_material->second.use_bump_map_correction &&
             imported_material->second.emission_sampling ==
                 EmissionSampling::back &&
             imported_material->second.volume_sampling ==
                 psycles::contract::VolumeSampling::equiangular &&
             imported_material->second.volume_interpolation ==
                 psycles::contract::VolumeInterpolation::cubic &&
             imported_material->second.cycles_shader_index ==
                 std::optional<std::uint32_t>{6u} &&
             imported_material->second.cycles_pass_id == 27,
         "material sampling policies did not round-trip");
  expect(imported_material->second.shader
             .root(psycles::contract::ShaderDomain::volume)
             .has_value(),
         "raw Blender Volume root was not retained");
  bool has_absorption = false;
  bool has_scatter = false;
  bool has_volume_mix = false;
  bool has_volume_emission = false;
  bool has_hair_info = false;
  bool has_point_to_vector = false;
  bool has_vector_to_color = false;
  bool has_transparent_boundary = false;
  bool has_null_surface = false;
  for (const auto &node : imported_material->second.shader.nodes()) {
    has_absorption |=
        node.type == psycles::compiler::node_type::volume_absorption;
    has_scatter |= node.type == psycles::compiler::node_type::volume_scatter;
    has_volume_mix |= node.type == psycles::compiler::node_type::mix_volume;
    has_volume_emission |=
        node.type == psycles::compiler::node_type::volume_emission;
    has_hair_info |= node.type == psycles::compiler::node_type::hair_info;
    has_point_to_vector |=
        node.type == psycles::compiler::node_type::point_to_vector;
    has_vector_to_color |=
        node.type == psycles::compiler::node_type::vector_to_color;
    has_transparent_boundary |=
        node.type == psycles::compiler::node_type::transparent_bsdf;
    has_null_surface |=
        node.type == psycles::compiler::node_type::null_closure;
  }
  expect(has_absorption && has_scatter && has_volume_mix &&
             has_volume_emission && has_hair_info,
         "Blender Volume closure tree was flattened or typed as a "
         "surface closure, or Hair Info was not retained");
  expect(has_point_to_vector && has_vector_to_color,
         "Geometry Position to Color did not follow the component-preserving "
         "float3 conversion path");
  for (const auto &diagnostic : imported.diagnostics) {
    expect(diagnostic.message.find("unsupported implicit socket conversion") ==
               std::string::npos,
           "supported float3-family conversion emitted a warning");
    expect(diagnostic.message.find("Hair Info") == std::string::npos,
           "supported Hair Info emitted an importer warning");
  }
  expect(has_null_surface && !has_transparent_boundary,
         "volume-only Blender material acquired a synthetic surface BSDF");
  const auto imported_light =
      imported.scene->lights.find(psycles::contract::LightId{1u});
  expect(imported_light != imported.scene->lights.end() &&
             imported_light->second.type ==
                 psycles::contract::LightType::area &&
             imported_light->second.ellipse &&
             imported_light->second.is_portal &&
             std::abs(imported_light->second.size - 1.3f) <= 1.0e-6f &&
             std::abs(imported_light->second.size_y - 1.3f) <= 1.0e-6f &&
             !imported_light->second.use_mis &&
             !imported_light->second.cast_shadow &&
             imported_light->second.is_shadow_catcher &&
             imported_light->second.cycles_asset_name ==
                 "Lighting Collection" &&
             imported_light->second.use_holdout &&
             imported_light->second.is_caustics_caster &&
             !imported_light->second.is_caustics_receiver &&
             imported_light->second.cycles_shader_index ==
                 std::optional<std::uint32_t>{5u} &&
             imported_light->second.cycles_object_index ==
                 std::optional<std::uint32_t>{9u} &&
             imported_light->second.object_color ==
                 psycles::Vec3f{0.125f, 0.25f, 0.5f} &&
             std::abs(imported_light->second.object_alpha - 0.75f) <=
                 1.0e-6f &&
             imported_light->second.object_pass_id == 19 &&
             std::abs(imported_light->second.object_random -
                      (2147483648.0f / 4294967295.0f)) <= 1.0e-6f &&
             imported_light->second.cycles_random_id ==
                 std::optional<std::uint32_t>{2147483648u} &&
             imported_light->second.particle_index == 7u &&
             imported_light->second.dupli_generated ==
                 psycles::Vec3f{-0.25f, 0.125f, 0.75f} &&
             imported_light->second.dupli_uv ==
                 psycles::Vec2f{0.2f, 0.8f} &&
             std::abs(imported_light->second
                          .shadow_terminator_shading_offset -
                      0.375f) <= 1.0e-6f &&
             std::abs(imported_light->second
                          .shadow_terminator_geometry_offset -
                      0.625f) <= 1.0e-6f &&
             imported_light->second.cycles_particle_source &&
             imported_light->second.cycles_particle_source->system == 3u &&
             imported_light->second.cycles_particle_source->source_index ==
                 5u &&
             imported_light->second.cycles_particle_source->location ==
                 psycles::Vec3f{1.0f, 2.0f, 3.0f} &&
             imported_light->second.cycles_particle_source->rotation ==
                 psycles::Vec4f{0.1f, 0.2f, 0.3f, 0.4f} &&
             imported_light->second.cycles_light_group == 2 &&
             imported_light->second.max_bounces == 13u &&
             imported_light->second.visibility_mask ==
                 (psycles::contract::visibility_bit(
                      psycles::contract::RayVisibility::diffuse) |
                  psycles::contract::visibility_bit(
                      psycles::contract::RayVisibility::transmission) |
                  psycles::contract::visibility_bit(
                      psycles::contract::RayVisibility::shadow)),
         "light shape, Cycles identity, or shader policy did not round-trip");
  const auto imported_world =
      imported.scene->materials.find(psycles::contract::MaterialId{3u});
  expect(imported_world != imported.scene->materials.end() &&
             imported_world->second.cycles_shader_index ==
                 std::optional<std::uint32_t>{3u} &&
             imported_world->second.volume_sampling ==
                 psycles::contract::VolumeSampling::distance &&
             imported_world->second.volume_interpolation ==
                 psycles::contract::VolumeInterpolation::cubic &&
             imported.scene->cycles_background_object_index ==
                 std::optional<std::uint32_t>{12u} &&
             imported.scene->cycles_object_count ==
                 std::optional<std::uint32_t>{13u},
         "world object/shader identity did not round-trip");
  expect(imported.scene->environment->nishita.has_value(),
         "Nishita world was not kept procedural");
  expect(imported.scene->environment->pixels.empty(),
         "Nishita world was unexpectedly baked to pixels");
  const auto &sky = *imported.scene->environment->nishita;
  expect_near(sky.sun_elevation, 0.9250245094299316f,
              "Nishita elevation mismatch");
  expect_near(sky.sun_rotation, 3.6651914755450647f,
              "Nishita Cycles rotation mismatch");
  expect_near(sky.angular_diameter, 0.01745329238474369f,
              "Nishita angular diameter mismatch");
  expect_near(sky.sun_intensity, 1.25f, "Nishita sun intensity mismatch");
  expect_near(sky.altitude, 123.0f, "Nishita altitude mismatch");
  expect_near(sky.air_density, 0.9f, "Nishita air density mismatch");
  expect_near(sky.dust_density, 1.1f, "Nishita dust density mismatch");
  expect_near(sky.ozone_density, 1.2f, "Nishita ozone density mismatch");
  expect_near(sky.background_strength, 2.0f,
              "Nishita background strength mismatch");

  const auto &integrator = imported.integrator;
  expect(integrator.max_bounces == 11u, "max bounce mismatch");
  expect(integrator.min_bounces == 2u, "min bounce mismatch");
  expect(integrator.diffuse_bounces == 3u, "diffuse bounce mismatch");
  expect(integrator.glossy_bounces == 4u, "glossy bounce mismatch");
  expect(integrator.transmission_bounces == 5u, "transmission bounce mismatch");
  expect(integrator.volume_bounces == 6u, "volume bounce mismatch");
  expect(integrator.transparent_min_bounces == 1u,
         "transparent minimum mismatch");
  expect(integrator.transparent_max_bounces == 9u,
         "transparent maximum mismatch");
  expect_near(integrator.sample_clamp_direct, 1.25f, "direct clamp mismatch");
  expect_near(integrator.sample_clamp_indirect, 2.5f,
              "indirect clamp mismatch");
  expect(integrator.ambient_occlusion_bounces == 7u, "Fast GI bounce mismatch");
  expect_near(integrator.ambient_occlusion_factor, 0.625f, "Fast GI factor mismatch");
  expect_near(integrator.ambient_occlusion_additive_factor, 0.0f,
              "Fast GI additive mismatch");
  expect_near(integrator.filter_glossy, 0.75f, "filter glossy mismatch");
  expect_near(imported.scene->ambient_occlusion_distance, 4.25f, "world AO distance mismatch");
  expect_near(integrator.ambient_occlusion_distance, 4.25f, "Fast GI distance mismatch");
  expect_near(integrator.film_exposure, 0.75f, "film exposure mismatch");
  expect_near(integrator.light_sampling_threshold, 0.125f,
              "light sampling threshold mismatch");
  expect(!integrator.reflective_caustics && integrator.refractive_caustics,
         "caustics settings mismatch");
  expect(!integrator.use_light_tree, "light-tree setting mismatch");
  expect(integrator.direct_light_sampling ==
             DirectLightSampling::next_event_estimation,
         "direct-light sampling mode mismatch");

  std::ifstream source{temporary.path() / "scene.json"};
  std::string light_tree_scene{std::istreambuf_iterator<char>{source},
                               std::istreambuf_iterator<char>{}};

  auto bump_scene = light_tree_scene;
  const auto replace_once = [](std::string &text, std::string_view before,
                               std::string_view after) {
    const auto position = text.find(before);
    expect(position != std::string::npos,
           "bump regression fixture is missing its insertion point");
    text.replace(position, before.size(), after);
  };
  auto legacy_seed_scene = light_tree_scene;
  replace_once(legacy_seed_scene,
               "      \"effective_seed\": 2271560481,\n", "");
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << legacy_seed_scene;
  }
  const auto legacy_seed_imported =
      load_blender_scene_bundle(temporary.path());
  expect(legacy_seed_imported.ok() &&
             legacy_seed_imported.seed == 305419896u,
         "legacy authored sampling seed fallback did not round-trip");
  replace_once(bump_scene, "\"volume_sampling\": \"EQUIANGULAR\",",
               "\"volume_sampling\": \"EQUIANGULAR\",\n"
               "      \"displacement_method\": \"BUMP\",");
  replace_once(bump_scene, "\"surface_root\": null,",
               "\"surface_root\": {\n"
               "          \"node\": \"Diffuse\",\n"
               "          \"socket\": \"BSDF\"\n"
               "        },");
  replace_once(bump_scene, "\"displacement_root\": null,",
               "\"displacement_root\": {\n"
               "          \"node\": \"Displacement\",\n"
               "          \"socket\": \"Displacement\"\n"
               "        },");
  replace_once(bump_scene, "\"links\": [",
               "\"links\": [\n"
               "          {\n"
               "            \"from_node\": \"Texture Coordinate\",\n"
               "            \"from_socket\": \"Reflection\",\n"
               "            \"to_node\": \"Diffuse\",\n"
               "            \"to_socket\": \"Normal\"\n"
               "          },\n"
               "          {\n"
               "            \"from_node\": \"Projector Coordinates\",\n"
               "            \"from_socket\": \"Object\",\n"
               "            \"to_node\": \"Legacy Mapped Image\",\n"
               "            \"to_socket\": \"Vector\"\n"
               "          },\n"
               "          {\n"
               "            \"from_node\": \"Legacy Mapped Image\",\n"
               "            \"from_socket\": \"Color\",\n"
               "            \"to_node\": \"Diffuse\",\n"
               "            \"to_socket\": \"Color\"\n"
               "          },");
  replace_once(bump_scene,
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
            "properties": {"from_instancer": true},
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Projector Coordinates",
            "label": "",
            "type": "TEX_COORD",
            "bl_idname": "ShaderNodeTexCoord",
            "inputs": [],
            "outputs": [
              {
                "identifier": "Object",
                "name": "Object",
                "type": "NodeSocketVector",
                "linked": true,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "properties": {"from_instancer": false},
            "special": {
              "object_coordinates": {
                "object": "Coordinate Projector",
                "object_to_world": [
                  0.46153846, 0.15384615, -0.02884615, 0.0,
                  -0.07692308, 0.30769231, 0.00480769, 0.0,
                  0.0, 0.0, 0.25, 0.0,
                  -0.61538462, 0.46153846, -0.08653846, 1.0
                ]
              }
            },
            "image": null,
            "node_tree": null
          },
          {
            "name": "Legacy Mapped Image",
            "label": "",
            "type": "TEX_IMAGE",
            "bl_idname": "ShaderNodeTexImage",
            "inputs": [
              {
                "identifier": "Vector",
                "name": "Vector",
                "type": "NodeSocketVector",
                "linked": true,
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
              "extension": "REPEAT",
              "interpolation": "Linear",
              "projection": "FLAT",
              "projection_blend": 0.0
            },
            "special": {
              "texture_mapping": {
                "vector_type": "TEXTURE",
                "translation": [0.25, -0.5, 1.25],
                "rotation": [0.125, -0.25, 0.375],
                "scale": [0.0, -2.0, 0.5],
                "mapping_x": "Z",
                "mapping_y": "NONE",
                "mapping_z": "X"
              }
            },
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
  const auto bump_imported = load_blender_scene_bundle(temporary.path());
  expect(bump_imported.ok(), "bump-only displacement material did not import");
  const auto bump_material =
      bump_imported.scene->materials.find(psycles::contract::MaterialId{2u});
  expect(bump_material != bump_imported.scene->materials.end() &&
             bump_material->second.shader
                 .root(psycles::contract::ShaderDomain::surface_normal)
                 .has_value() &&
             !bump_material->second.shader
                  .root(psycles::contract::ShaderDomain::displacement)
                  .has_value() &&
             bump_material->second.displacement_method ==
                 psycles::contract::DisplacementMethod::bump,
         "automatic bump was not retained as a surface-normal root");
  bool has_automatic_bump = false;
  bool has_texture_mapping_node = false;
  for (const auto &node : bump_material->second.shader.nodes()) {
    has_automatic_bump |= node.type == psycles::compiler::node_type::bump;
    if (node.type == psycles::compiler::node_type::mapping &&
        node.label == "Legacy Mapped Image / Texture Mapping") {
      const auto scale = node.inputs.find("Scale");
      const auto x_mapping = node.properties.find("XMapping");
      const auto y_mapping = node.properties.find("YMapping");
      const auto z_mapping = node.properties.find("ZMapping");
      has_texture_mapping_node =
          scale != node.inputs.end() && scale->second.value.has_value() &&
          std::get<psycles::Vec3f>(scale->second.value->value) ==
              psycles::Vec3f{1.0e-5f, -2.0f, 0.5f} &&
          x_mapping != node.properties.end() &&
          std::get<std::string>(x_mapping->second.value) == "Z" &&
          y_mapping != node.properties.end() &&
          std::get<std::string>(y_mapping->second.value) == "NONE" &&
          z_mapping != node.properties.end() &&
          std::get<std::string>(z_mapping->second.value) == "X";
    }
  }
  expect(has_automatic_bump,
         "Blender bump-only displacement did not lower to a bump node");
  expect(has_texture_mapping_node,
         "legacy TextureNode mapping inputs were not normalized");

  psycles::compiler::ShaderCompiler bump_compiler{
      psycles::compiler::make_core_node_registry()};
  const auto bump_shader = bump_compiler.compile(bump_material->second.shader);
  expect(bump_shader.ok(),
         "automatic displacement bump graph did not validate");
  const auto bump_surface =
      psycles::compiler::compile_surface_program(*bump_shader.program);
  expect(bump_surface.ok(),
         "automatic displacement bump did not lower to a surface program");
  bool has_bump_instruction = false;
  bool bump_uses_object_space = false;
  bool has_texture_mapping = false;
  for (const auto &instruction : bump_surface.program->value_instructions()) {
    has_bump_instruction |=
        instruction.operation == psycles::compiler::ValueOperation::bump;
    bump_uses_object_space |=
        instruction.operation == psycles::compiler::ValueOperation::bump &&
        (instruction.static_u0 & 4u) != 0u;
    has_texture_mapping |=
        instruction.operation == psycles::compiler::ValueOperation::mapping &&
        instruction.static_u0 == 1u && instruction.static_u1 == 19u;
  }
  expect(has_bump_instruction,
         "automatic displacement bump emitted no value instruction");
  expect(!bump_uses_object_space,
         "BUMP-only displacement incorrectly selected object-space bump");
  psycles::tests::expect_texture_coordinate_import(
      bump_material->second, *bump_surface.program);
  expect(has_texture_mapping,
         "legacy TextureNode mapping type or axis map was not lowered");

  auto normal_map_scene = bump_scene;
  replace_once(
      normal_map_scene,
      "            \"from_node\": \"Texture Coordinate\",\n"
      "            \"from_socket\": \"Reflection\",\n"
      "            \"to_node\": \"Diffuse\",\n"
      "            \"to_socket\": \"Normal\"",
      "            \"from_node\": \"Normal Map\",\n"
      "            \"from_socket\": \"Normal\",\n"
      "            \"to_node\": \"Diffuse\",\n"
      "            \"to_socket\": \"Normal\"");
  replace_once(
      normal_map_scene,
      R"JSON(          {
            "name": "Diffuse",)JSON",
      R"JSON(          {
            "name": "Normal Map",
            "label": "",
            "type": "NORMAL_MAP",
            "bl_idname": "ShaderNodeNormalMap",
            "inputs": [
              {
                "identifier": "Strength",
                "name": "Strength",
                "type": "NodeSocketFloat",
                "linked": false,
                "default": 0.75
              },
              {
                "identifier": "Color",
                "name": "Color",
                "type": "NodeSocketColor",
                "linked": false,
                "default": [0.8, 0.3, 0.9, 1.0]
              }
            ],
            "outputs": [
              {
                "identifier": "Normal",
                "name": "Normal",
                "type": "NodeSocketVector",
                "linked": true,
                "default": [0.0, 0.0, 0.0]
              }
            ],
            "properties": {
              "space": "TANGENT",
              "uv_map": "MappedUV",
              "convention": "DIRECTX",
              "base": "ORIGINAL"
            },
            "special": {},
            "image": null,
            "node_tree": null
          },
          {
            "name": "Diffuse",)JSON");
  struct LoweredNormalMap {
    psycles::compiler::ValueInstruction instruction;
    std::uint64_t uv_map_id{};
  };
  const auto lower_imported_normal_map =
      [&](const std::string &scene_text) {
        {
          std::ofstream scene{temporary.path() / "scene.json"};
          scene << scene_text;
        }
        const auto loaded = load_blender_scene_bundle(temporary.path());
        expect(loaded.ok(), "Normal Map semantic scene did not import");
        const auto material = loaded.scene->materials.find(
            psycles::contract::MaterialId{2u});
        expect(material != loaded.scene->materials.end(),
               "Normal Map semantic material is missing");
        const auto shader = bump_compiler.compile(material->second.shader);
        expect(shader.ok(), "Normal Map semantic graph did not validate");
        const auto surface =
            psycles::compiler::compile_surface_program(*shader.program);
        expect(surface.ok(), "Normal Map semantic graph did not lower");
        const auto &values = surface.program->value_instructions();
        const auto &parameters = surface.program->parameters();
        for (const auto &instruction : values) {
          if (instruction.operation ==
              psycles::compiler::ValueOperation::normal_map) {
            const auto uv_map = instruction.operand(
                psycles::compiler::value_operand::normal_map::uv_map);
            expect(uv_map.valid() && uv_map.value < values.size(),
                   "Normal Map UV operand is invalid");
            const auto &parameter_value = values[uv_map.value];
            expect(parameter_value.operation ==
                           psycles::compiler::ValueOperation::parameter &&
                       parameter_value.parameter.valid() &&
                       parameter_value.parameter.value < parameters.size(),
                   "Normal Map UV identity is not a typed parameter");
            const auto &parameter =
                parameters[parameter_value.parameter.value];
            expect(parameter.source ==
                           psycles::compiler::ParameterSource::property &&
                       parameter.socket == "UvMapId" &&
                       parameter.type ==
                           psycles::contract::SocketType::unsigned_integer,
                   "Normal Map UV identity lost its property contract");
            return LoweredNormalMap{
                .instruction = instruction,
                .uv_map_id = std::get<std::uint64_t>(
                    parameter.default_value.value)};
          }
        }
        throw std::runtime_error{
            "Normal Map semantic graph emitted no instruction"};
      };
  const auto original_normal_map =
      lower_imported_normal_map(normal_map_scene);
  expect(
      original_normal_map.instruction.static_u0 ==
              psycles::compiler::encode_normal_map_configuration(
                  psycles::compiler::NormalMapSpace::tangent,
                  true,
                  psycles::compiler::NormalMapBase::original,
                  psycles::compiler::NormalMapConvention::direct_x) &&
          original_normal_map.uv_map_id ==
              psycles::contract::uv_undisplaced_tangent_attribute_id(
                  "MappedUV"),
      "Blender ORIGINAL/DirectX Normal Map lost its Cycles attributes");

  auto displaced_normal_map_scene = normal_map_scene;
  replace_once(displaced_normal_map_scene,
               "\"base\": \"ORIGINAL\"",
               "\"base\": \"DISPLACED\"");
  const auto displaced_normal_map =
      lower_imported_normal_map(displaced_normal_map_scene);
  expect(
      displaced_normal_map.instruction.static_u0 ==
              psycles::compiler::encode_normal_map_configuration(
                  psycles::compiler::NormalMapSpace::tangent,
                  true,
                  psycles::compiler::NormalMapBase::displaced,
                  psycles::compiler::NormalMapConvention::direct_x) &&
          displaced_normal_map.uv_map_id ==
              psycles::contract::uv_tangent_attribute_id("MappedUV"),
      "Blender DISPLACED Normal Map did not select the current Mikk frame");

  auto combined_displacement_scene = bump_scene;
  replace_once(combined_displacement_scene, "\"displacement_method\": \"BUMP\"",
               "\"displacement_method\": \"BOTH\"");
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << combined_displacement_scene;
  }
  const auto combined_displacement_imported =
      load_blender_scene_bundle(temporary.path());
  expect(combined_displacement_imported.ok(),
         "combined displacement did not import");
  const auto &combined_material =
      combined_displacement_imported.scene->materials.at(
          psycles::contract::MaterialId{2u});
  expect(
      combined_material.displacement_method ==
              psycles::contract::DisplacementMethod::both &&
          combined_material.shader
              .root(psycles::contract::ShaderDomain::surface_normal)
              .has_value() &&
          combined_material.shader
              .root(psycles::contract::ShaderDomain::displacement)
              .has_value(),
      "combined displacement did not retain both semantic roots");
  const auto combined_shader =
      bump_compiler.compile(combined_material.shader);
  expect(combined_shader.ok(),
         "combined displacement graph did not validate");
  const auto combined_surface =
      psycles::compiler::compile_surface_program(
          *combined_shader.program);
  expect(combined_surface.ok(),
         "combined displacement graph did not lower");
  bool combined_bump_uses_object_space = false;
  for (const auto &instruction :
       combined_surface.program->value_instructions()) {
    combined_bump_uses_object_space |=
        instruction.operation == psycles::compiler::ValueOperation::bump &&
        (instruction.static_u0 & 4u) != 0u;
  }
  expect(combined_bump_uses_object_space,
         "BOTH automatic bump lost Cycles object-space semantics");

  auto true_displacement_scene = bump_scene;
  replace_once(true_displacement_scene, "\"displacement_method\": \"BUMP\"",
               "\"displacement_method\": \"DISPLACEMENT\"");
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << true_displacement_scene;
  }
  const auto true_displacement_imported =
      load_blender_scene_bundle(temporary.path());
  expect(true_displacement_imported.ok(),
         "true geometry displacement did not import");
  const auto &true_material =
      true_displacement_imported.scene->materials.at(
          psycles::contract::MaterialId{2u});
  expect(
      true_material.displacement_method ==
              psycles::contract::DisplacementMethod::displacement &&
          true_material.shader
              .root(psycles::contract::ShaderDomain::displacement)
              .has_value() &&
          !true_material.shader
               .root(psycles::contract::ShaderDomain::surface_normal)
               .has_value(),
      "true displacement was conflated with automatic bump");

  auto box_filter_scene = light_tree_scene;
  constexpr std::string_view gaussian_filter =
      "\"pixel_filter_type\": \"GAUSSIAN\"";
  constexpr std::string_view box_filter = "\"pixel_filter_type\": \"BOX\"";
  const auto filter_position = box_filter_scene.find(gaussian_filter);
  expect(filter_position != std::string::npos,
         "test scene has no Gaussian pixel filter");
  box_filter_scene.replace(filter_position, gaussian_filter.size(), box_filter);
  constexpr std::string_view gaussian_width = "\"filter_width\": 2.25";
  constexpr std::string_view dormant_box_width = "\"filter_width\": 0.01";
  const auto filter_width_position = box_filter_scene.find(gaussian_width);
  expect(filter_width_position != std::string::npos,
         "test scene has no Gaussian filter width");
  box_filter_scene.replace(filter_width_position, gaussian_width.size(),
                           dormant_box_width);
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << box_filter_scene;
  }
  const auto box_imported = load_blender_scene_bundle(temporary.path());
  expect(box_imported.ok() && box_imported.pixel_filter == PixelFilter::box,
         "BOX pixel filter did not import");
  expect_near(box_imported.filter_width, 1.0f,
              "BOX filter did not normalize to one pixel");

  auto legacy_nishita_scene = light_tree_scene;
  constexpr std::string_view current_sky_type =
      "\"sky_type\": \"SINGLE_SCATTERING\"";
  constexpr std::string_view legacy_sky_type = "\"sky_type\": \"NISHITA\"";
  const auto sky_type_position = legacy_nishita_scene.find(current_sky_type);
  expect(sky_type_position != std::string::npos,
         "test scene has no current single-scattering sky type");
  legacy_nishita_scene.replace(sky_type_position, current_sky_type.size(),
                               legacy_sky_type);
  constexpr std::string_view current_aerosol = "\"aerosol_density\": 1.1";
  constexpr std::string_view legacy_dust = "\"dust_density\": 1.1";
  const auto aerosol_position = legacy_nishita_scene.find(current_aerosol);
  expect(aerosol_position != std::string::npos,
         "test scene has no current aerosol-density property");
  legacy_nishita_scene.replace(aerosol_position, current_aerosol.size(),
                               legacy_dust);
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << legacy_nishita_scene;
  }
  const auto legacy_imported = load_blender_scene_bundle(temporary.path());
  expect(legacy_imported.ok() &&
             legacy_imported.scene->environment.has_value() &&
             legacy_imported.scene->environment->nishita.has_value(),
         "legacy NISHITA world was not kept procedural");
  expect_near(legacy_imported.scene->environment->nishita->dust_density, 1.1f,
              "legacy Nishita dust density mismatch");

  auto hosek_scene = light_tree_scene;
  replace_once(hosek_scene, "\"sky_type\": \"SINGLE_SCATTERING\"",
               "\"sky_type\": \"HOSEK_WILKIE\"");
  replace_once(hosek_scene, "\"sun_elevation\": 0.9250245094299316,",
               "\"sun_direction\": [-0.94615382, 0.06153846, "
               "0.31781429],\n"
               "            \"turbidity\": 2.9,\n"
               "            \"ground_albedo\": 0.3,\n"
               "            \"sun_elevation\": 0.9250245094299316,");
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << hosek_scene;
  }
  const auto hosek_imported = load_blender_scene_bundle(temporary.path());
  expect(hosek_imported.ok(), "legacy Hosek-Wilkie world did not import");
  const auto hosek_world =
      hosek_imported.scene->materials.find(psycles::contract::MaterialId{3u});
  expect(hosek_world != hosek_imported.scene->materials.end(),
         "legacy Hosek-Wilkie world material is missing");
  bool has_hosek_node = false;
  for (const auto &node : hosek_world->second.shader.nodes()) {
    has_hosek_node |=
        node.type == psycles::compiler::node_type::hosek_wilkie_sky;
  }
  expect(has_hosek_node,
         "legacy Hosek-Wilkie sky was rewritten to another model");
  psycles::compiler::ShaderCompiler hosek_compiler{
      psycles::compiler::make_core_node_registry()};
  const auto hosek_shader = hosek_compiler.compile(hosek_world->second.shader);
  if (!hosek_shader.ok()) {
    for (const auto &diagnostic : hosek_shader.diagnostics) {
      std::cerr << "Hosek graph diagnostic: " << diagnostic.message << '\n';
    }
  }
  expect(hosek_shader.ok(), "legacy Hosek-Wilkie graph did not validate");
  const auto hosek_surface =
      psycles::compiler::compile_surface_program(*hosek_shader.program);
  expect(hosek_surface.ok(), "legacy Hosek-Wilkie graph did not lower");
  bool has_hosek_instruction = false;
  for (const auto &instruction : hosek_surface.program->value_instructions()) {
    has_hosek_instruction |=
        instruction.operation ==
            psycles::compiler::ValueOperation::hosek_wilkie_sky &&
        instruction.static_table.size() == 33u;
  }
  expect(has_hosek_instruction,
         "legacy Hosek-Wilkie sky emitted no cooked model");

  const auto setting = light_tree_scene.find("\"use_light_tree\": false");
  expect(setting != std::string::npos, "test scene has no light-tree setting");
  light_tree_scene.replace(setting,
                           std::string{"\"use_light_tree\": false"}.size(),
                           "\"use_light_tree\": true");
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << light_tree_scene;
  }
  const auto light_tree_imported = load_blender_scene_bundle(temporary.path());
  expect(light_tree_imported.ok(), "supported Cycles light tree was rejected");
  expect(light_tree_imported.integrator.use_light_tree,
         "Cycles light-tree setting was not preserved");
}

void test_cycles_float_to_boolean_conversion() {
  TemporaryDirectory temporary;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin",
                           std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[],
  "materials":[{
    "name":"Linked Thin Wall",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Linked Thin Wall",
      "surface_root":{"node":"Principled","socket":"BSDF"},
      "volume_root":null,
      "displacement_root":null,
      "links":[
        {"from_node":"Value","from_socket":"Value",
         "to_node":"Principled","to_socket":"Thin Wall"}
      ],
      "nodes":[
        {
          "name":"Value","type":"VALUE","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Value","name":"Value",
            "type":"NodeSocketFloat","linked":true,"default":1.25}],
          "properties":{},"special":{}
        },
        {
          "name":"Principled","type":"BSDF_PRINCIPLED","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Thin Wall","name":"Thin Wall",
             "type":"NodeSocketBool","linked":true,"default":false}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{"distribution":"GGX",
                        "subsurface_method":"RANDOM_WALK"},
          "special":{}
        }
      ]
    }
  }],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
  }

  const auto imported = load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "linked float Thin Wall scene did not import");
  expect(!imported.scene->cycles_uses_light_linking,
         "absent light-linking capability did not default to false");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Linked Thin Wall") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "linked float Thin Wall material is missing");

  bool has_float_to_boolean = false;
  for (const auto &node : material->shader.nodes()) {
    has_float_to_boolean |=
        node.type ==
        psycles::compiler::node_type::scalar_to_boolean;
  }
  expect(has_float_to_boolean,
         "Cycles FLOAT-to-INT Boolean conversion was not materialized");
  for (const auto &diagnostic : imported.diagnostics) {
    expect(diagnostic.message.find("unsupported implicit socket conversion") ==
               std::string::npos,
           "supported FLOAT-to-Boolean conversion emitted a warning");
  }

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  expect(shader.ok(), "linked float Thin Wall graph did not validate");
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  expect(surface.ok(), "linked float Thin Wall graph did not lower");
  bool has_float_to_boolean_instruction = false;
  for (const auto &instruction : surface.program->value_instructions()) {
    has_float_to_boolean_instruction |=
        instruction.operation ==
        psycles::compiler::ValueOperation::scalar_to_boolean;
  }
  expect(has_float_to_boolean_instruction,
         "linked float Thin Wall emitted no conversion instruction");
}

void test_cycles_default_microfacet_tangent_import() {
  TemporaryDirectory temporary;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin",
                           std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[],
  "materials":[
    {
      "name":"Default Principled Tangent",
      "cycles_sync":{"shader_index":8},
      "node_tree":{
        "name":"Default Principled Tangent",
        "surface_root":{"node":"Principled","socket":"BSDF"},
        "volume_root":null,
        "displacement_root":null,
        "links":[],
        "nodes":[{
          "name":"Principled","type":"BSDF_PRINCIPLED","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Metallic","name":"Metallic",
             "type":"NodeSocketFloat","linked":false,"default":1.0},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.5},
            {"identifier":"Anisotropic","name":"Anisotropic",
             "type":"NodeSocketFloat","linked":false,"default":0.6},
            {"identifier":"Anisotropic Rotation",
             "name":"Anisotropic Rotation","type":"NodeSocketFloat",
             "linked":false,"default":0.25},
            {"identifier":"Tangent","name":"Tangent",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{"distribution":"GGX",
                        "subsurface_method":"RANDOM_WALK"},
          "special":{}
        }]
      }
    },
    {
      "name":"Default Glossy Tangent",
      "cycles_sync":{"shader_index":9},
      "node_tree":{
        "name":"Default Glossy Tangent",
        "surface_root":{"node":"Glossy","socket":"BSDF"},
        "volume_root":null,
        "displacement_root":null,
        "links":[],
        "nodes":[{
          "name":"Glossy","type":"BSDF_GLOSSY","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.5},
            {"identifier":"Anisotropy","name":"Anisotropy",
             "type":"NodeSocketFloat","linked":false,"default":0.6},
            {"identifier":"Rotation","name":"Rotation",
             "type":"NodeSocketFloat","linked":false,"default":0.25},
            {"identifier":"Tangent","name":"Tangent",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{"distribution":"GGX"},
          "special":{}
        }]
      }
    },
    {
      "name":"Default Metallic Tangent",
      "cycles_sync":{"shader_index":10},
      "node_tree":{
        "name":"Default Metallic Tangent",
        "surface_root":{"node":"Metallic","socket":"BSDF"},
        "volume_root":null,
        "displacement_root":null,
        "links":[],
        "nodes":[{
          "name":"Metallic","type":"BSDF_METALLIC","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Base Color","name":"Base Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.17,0.31,0.73,1.0]},
            {"identifier":"Edge Tint","name":"Edge Tint",
             "type":"NodeSocketColor","linked":false,
             "default":[0.91,0.67,0.43,1.0]},
            {"identifier":"IOR","name":"IOR",
             "type":"NodeSocketVector","linked":false,
             "default":[1.9,2.7,3.4]},
            {"identifier":"Extinction","name":"Extinction",
             "type":"NodeSocketVector","linked":false,
             "default":[4.6,3.2,2.1]},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.38},
            {"identifier":"Anisotropy","name":"Anisotropy",
             "type":"NodeSocketFloat","linked":false,"default":0.63},
            {"identifier":"Rotation","name":"Rotation",
             "type":"NodeSocketFloat","linked":false,"default":0.2},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Tangent","name":"Tangent",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Weight","name":"Weight",
             "type":"NodeSocketFloat","linked":false,"default":0.0},
            {"identifier":"Thin Film Thickness",
             "name":"Thin Film Thickness","type":"NodeSocketFloat",
             "linked":false,"default":420.0},
            {"identifier":"Thin Film IOR","name":"Thin Film IOR",
             "type":"NodeSocketFloat","linked":false,"default":1.37}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{"distribution":"BECKMANN",
                        "fresnel_type":"PHYSICAL_CONDUCTOR"},
          "special":{}
        }]
      }
    },
    {
      "name":"Standalone Ashikhmin Sheen",
      "cycles_sync":{"shader_index":11},
      "node_tree":{
        "name":"Standalone Ashikhmin Sheen",
        "surface_root":{"node":"Sheen","socket":"BSDF"},
        "volume_root":null,
        "displacement_root":null,
        "links":[],
        "nodes":[{
          "name":"Sheen","type":"BSDF_SHEEN","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.6,0.3,0.15,1.0]},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.37},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Weight","name":"Weight",
             "type":"NodeSocketFloat","linked":false,"default":0.0}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{"distribution":"ASHIKHMIN"},
          "special":{}
        }]
      }
    }
  ],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
  }

  const auto imported = load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "unlinked microfacet tangent scene did not import");
  for (const auto &[material_name, expected_type] :
       std::array{
           std::pair{"Default Principled Tangent",
                     std::string_view{
                         psycles::compiler::node_type::principled_bsdf}},
           std::pair{"Default Glossy Tangent",
                     std::string_view{
                         psycles::compiler::node_type::glossy_bsdf}},
           std::pair{"Default Metallic Tangent",
                     std::string_view{
                         psycles::compiler::node_type::metallic_bsdf}},
           std::pair{"Standalone Ashikhmin Sheen",
                     std::string_view{
                         psycles::compiler::node_type::sheen_bsdf}}}) {
    const psycles::contract::MaterialDesc *material = nullptr;
    for (const auto &[id, candidate] : imported.scene->materials) {
      static_cast<void>(id);
      if (candidate.name == material_name) {
        material = &candidate;
        break;
      }
    }
    expect(material != nullptr,
           std::string{material_name} + " material is missing");
    const psycles::contract::ShaderNode *closure_node = nullptr;
    for (const auto &node : material->shader.nodes()) {
      if (node.type == expected_type) {
        closure_node = &node;
        break;
      }
    }
    expect(closure_node != nullptr,
           std::string{material_name} + " closure is missing");
    if (expected_type == psycles::compiler::node_type::metallic_bsdf) {
      expect(
          closure_node->properties.contains("FresnelType") &&
              closure_node->properties.at("FresnelType") ==
                  psycles::contract::SocketValue::string(
                      "PHYSICAL_CONDUCTOR") &&
              closure_node->properties.contains("Distribution") &&
              closure_node->properties.at("Distribution") ==
                  psycles::contract::SocketValue::string("BECKMANN") &&
              closure_node->inputs.contains("BaseColor") &&
              closure_node->inputs.contains("EdgeTint") &&
              closure_node->inputs.contains("IOR") &&
              closure_node->inputs.contains("Extinction") &&
              closure_node->inputs.contains("ThinFilmThickness") &&
              closure_node->inputs.contains("ThinFilmIOR") &&
              !closure_node->inputs.contains("Weight"),
          "Blender Metallic import did not preserve the original closure "
          "sockets and static model tags");
    }
    if (expected_type == psycles::compiler::node_type::sheen_bsdf) {
      expect(
          closure_node->properties.contains("Distribution") &&
              closure_node->properties.at("Distribution") ==
                  psycles::contract::SocketValue::string("ASHIKHMIN") &&
              closure_node->inputs.contains("Color") &&
              closure_node->inputs.contains("Roughness") &&
              closure_node->inputs.contains("Normal") &&
              !closure_node->inputs.contains("Weight"),
          "Blender Sheen import did not preserve the raw closure sockets "
          "and static distribution tag");
    } else {
      const auto tangent = closure_node->inputs.find("Tangent");
      expect(tangent != closure_node->inputs.end() &&
                 tangent->second.source.has_value(),
             std::string{material_name} +
                 " did not materialize Cycles' default tangent edge");
      const auto *geometry_node = material->shader.find(
          tangent->second.source->node);
      expect(geometry_node != nullptr &&
                 geometry_node->type ==
                     psycles::compiler::node_type::geometry &&
                 tangent->second.source->socket == "Tangent",
             std::string{material_name} +
                 " default tangent is not Geometry.Tangent");
    }

    psycles::compiler::ShaderCompiler compiler{
        psycles::compiler::make_core_node_registry()};
    const auto shader = compiler.compile(material->shader);
    expect(shader.ok(),
           std::string{material_name} + " graph did not validate");
    const auto surface =
        psycles::compiler::compile_surface_program(*shader.program);
    expect(surface.ok() &&
               surface.program->closure_instructions().size() == 1u,
           std::string{material_name} + " graph did not lower");
    const auto &closure = surface.program->closure_instructions().front();
    if (expected_type != psycles::compiler::node_type::sheen_bsdf) {
      expect(closure.microfacet_anisotropy.valid() &&
                 closure.microfacet_rotation.valid() &&
                 closure.tangent.valid(),
             std::string{material_name} +
                 " lost anisotropy, rotation, or tangent during lowering");
    }
    if (expected_type == psycles::compiler::node_type::metallic_bsdf) {
      expect(
          closure.operation ==
                  psycles::compiler::ClosureOperation::metallic_conductor &&
              closure.metallic_base_ior.valid() &&
              closure.metallic_edge_tint_k.valid() &&
              !closure.color.valid() && closure.beckmann &&
              !closure.preserve_ggx_energy,
          "Blender Metallic physical-conductor tag was not lowered "
          "without pre-baking");
    }
    if (expected_type == psycles::compiler::node_type::sheen_bsdf) {
      expect(
          closure.operation ==
                  psycles::compiler::ClosureOperation::sheen_ashikhmin &&
              closure.color.valid() && closure.normal.valid() &&
              closure.roughness.valid(),
          "Blender Ashikhmin Sheen did not lower as a raw closure");
    }
  }
}

} // namespace
int main() {
  try {
    test_integrator_settings_round_trip();
    test_cycles_float_to_boolean_conversion();
    test_cycles_default_microfacet_tangent_import();
    test_blender_light_path_portal_depth_import();
    test_blender_legacy_mix_import();
    test_blender_vector_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
