#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>

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

}// namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr
            << "usage: psycles_inspect_blender_material "
               "<export-directory> <material-name>\n";
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
    for (const auto &[id, material] : materials.materials()) {
        if (material.name() != requested) {
            continue;
        }
        const auto &program = *material.surface_program();
        std::cout << "material " << id.value << " '" << material.name()
                  << "'\nparameters " << program.parameters().size()
                  << "\n";
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
    std::cerr << "material not found: " << requested << '\n';
    return EXIT_FAILURE;
}
