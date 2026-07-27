#include <psycles/adapter/blender_scene.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using psycles::contract::DirectLightSampling;

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
  "materials": [],
  "geometries": [],
  "instances": [],
  "lights": [],
  "world": {
    "name": "Nishita World",
    "color": [0.05, 0.05, 0.05],
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
            "sky_type": "NISHITA",
            "sun_elevation": 0.9250245094299316,
            "sun_rotation": 2.6179938316345215,
            "sun_size": 0.01745329238474369,
            "sun_intensity": 1.25,
            "sun_disc": true,
            "altitude": 123.0,
            "air_density": 0.9,
            "dust_density": 1.1,
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
        imported.scene->environment.has_value(),
        "Nishita world did not produce an environment");
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
