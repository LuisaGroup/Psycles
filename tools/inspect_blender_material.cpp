#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void print_value(const psycles::contract::SocketValue &value) {
    using psycles::contract::SocketType;
    switch (value.type) {
        case SocketType::floating:
            std::cout << std::get<float>(value.value);
            break;
        case SocketType::unsigned_integer:
            std::cout << std::get<std::uint64_t>(value.value);
            break;
        case SocketType::boolean:
            std::cout << (std::get<bool>(value.value) ? "true" : "false");
            break;
        case SocketType::color:
        case SocketType::float3:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal:
        case SocketType::spectrum: {
            const auto vector =
                std::get<psycles::Vec3f>(value.value);
            std::cout << '(' << vector.x << ", " << vector.y << ", "
                      << vector.z << ')';
            break;
        }
        default:
            std::cout << "<type " << static_cast<unsigned>(value.type)
                      << '>';
            break;
    }
}

[[nodiscard]] psycles::Vec3f transform_point(
    const psycles::Mat4f &transform,
    psycles::Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        e[0u] * point.x + e[4u] * point.y +
            e[8u] * point.z + e[12u],
        e[1u] * point.x + e[5u] * point.y +
            e[9u] * point.z + e[13u],
        e[2u] * point.x + e[6u] * point.y +
            e[10u] * point.z + e[14u]};
}

[[nodiscard]] bool near(
    float actual,
    float expected) noexcept {
    return std::abs(actual - expected) <=
           1.0e-6f *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

}// namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "usage: psycles_inspect_blender_material "
               "<export-directory> <material-name> "
               "[--require-generated-transform]\n";
        return EXIT_FAILURE;
    }
    const auto require_generated_transform =
        argc == 4 &&
        std::string_view{argv[3]} ==
            "--require-generated-transform";
    if (argc == 4 && !require_generated_transform) {
        std::cerr
            << "unknown option: " << argv[3] << '\n';
        return EXIT_FAILURE;
    }
    auto imported =
        psycles::adapter::load_blender_scene_bundle(argv[1]);
    for (const auto &diagnostic : imported.diagnostics) {
        std::cerr
            << (diagnostic.severity ==
                        psycles::adapter::
                            BlenderSceneDiagnosticSeverity::error
                    ? "error: "
                    : "warning: ")
            << diagnostic.message << '\n';
    }
    if (!imported.ok()) {
        return EXIT_FAILURE;
    }
    if (require_generated_transform) {
        for (const auto &[geometry_id, geometry] :
             imported.scene->geometries) {
            static_cast<void>(geometry_id);
            if (!geometry.generated_transform) {
                std::cerr
                    << "geometry '" << geometry.name
                    << "' has no volume Generated transform\n";
                return EXIT_FAILURE;
            }
            if (geometry.generated.domain !=
                    psycles::contract::
                        MeshAttributeDomain::point ||
                geometry.generated.values.size() !=
                    geometry.positions.size()) {
                std::cerr
                    << "geometry '" << geometry.name
                    << "' cannot validate its point-domain "
                       "Generated transform\n";
                return EXIT_FAILURE;
            }
            for (std::size_t index = 0u;
                 index < geometry.positions.size();
                 ++index) {
                const auto expected = transform_point(
                    *geometry.generated_transform,
                    geometry.positions[index]);
                const auto actual =
                    geometry.generated.values[index];
                if (!near(actual.x, expected.x) ||
                    !near(actual.y, expected.y) ||
                    !near(actual.z, expected.z)) {
                    std::cerr
                        << "geometry '" << geometry.name
                        << "' has inconsistent surface and "
                           "volume Generated coordinates at point "
                        << index << '\n';
                    return EXIT_FAILURE;
                }
            }
        }
    }
    psycles::compiler::ShaderCompiler shader_compiler{
        psycles::compiler::make_core_node_registry()};
    psycles::compiler::MaterialLibrary materials;
    const auto update =
        materials.update(*imported.scene, shader_compiler);
    if (!update.committed) {
        for (const auto &diagnostic : update.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }

    const auto requested = std::string_view{argv[2]};
    const auto inspect_all = requested == "*";
    for (const auto &[id, material] : materials.materials()) {
        if (!inspect_all && material.name() != requested) {
            continue;
        }
        const auto &program = *material.surface_program();
        std::cout << "material " << id.value << " '" << material.name()
                  << "' signature " << program.structure_signature()
                  << "\nparameters " << program.parameters().size()
                  << "\n";
        if (inspect_all) {
            std::cout << "values "
                      << program.value_instructions().size()
                      << "\nclosures "
                      << program.closure_instructions().size()
                      << "\n";
            continue;
        }
        for (const auto &parameter : program.parameters()) {
            std::cout << "  [" << parameter.id.value << "] node "
                      << parameter.node.value << ' ' << parameter.socket
                      << " = ";
            const auto *bound =
                material.parameters().find(parameter.id);
            print_value(
                bound == nullptr ? parameter.default_value : *bound);
            std::cout << '\n';
        }
        std::cout << "values "
                  << program.value_instructions().size() << "\n";
        for (std::size_t i = 0u;
             i < program.value_instructions().size();
             ++i) {
            const auto &instruction =
                program.value_instructions()[i];
            std::cout << "  [" << i << "] op "
                      << static_cast<unsigned>(instruction.operation)
                      << " node " << instruction.source_node.value
                      << " a=" << instruction.a.value
                      << " b=" << instruction.b.value
                      << " u0=" << instruction.static_u0
                      << " u1=" << instruction.static_u1 << '\n';
        }
        std::cout << "closures "
                  << program.closure_instructions().size() << "\n";
        for (std::size_t i = 0u;
             i < program.closure_instructions().size();
             ++i) {
            const auto &closure =
                program.closure_instructions()[i];
            std::cout << "  [" << i << "] op "
                      << static_cast<unsigned>(closure.operation)
                      << " color=" << closure.color.value
                      << " strength=" << closure.strength.value << '\n';
        }
        return EXIT_SUCCESS;
    }
    if (inspect_all) {
        return EXIT_SUCCESS;
    }
    std::cerr << "material not found: " << requested << '\n';
    return EXIT_FAILURE;
}
