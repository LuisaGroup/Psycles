#include <psycles/compiler/material_library.h>

#include <utility>

namespace psycles::compiler {

CompiledMaterial::CompiledMaterial(
    contract::MaterialId id,
    std::string name,
    std::shared_ptr<const ShaderProgram> shader,
    std::shared_ptr<const SurfaceProgram> surface,
    SurfaceParameterBlock parameters) noexcept
    : _id{id},
      _name{std::move(name)},
      _shader{std::move(shader)},
      _surface{std::move(surface)},
      _parameters{std::move(parameters)} {}

const CompiledMaterial *MaterialLibrary::find(
    contract::MaterialId id) const noexcept {
    auto iter = _materials.find(id);
    return iter == _materials.end() ? nullptr : &iter->second;
}

MaterialLibraryUpdate MaterialLibrary::update(
    const contract::SceneSnapshot &scene,
    const ShaderCompiler &shader_compiler) {
    MaterialLibraryUpdate result{
        .committed = false,
        .source_revision = _source_revision,
        .updates = {},
        .diagnostics = {}};
    auto candidate = _materials;
    std::vector<MaterialUpdate> updates;

    auto diagnose = [&](contract::MaterialId material,
                        std::string message) {
        result.diagnostics.emplace_back(
            MaterialCompilationDiagnostic{
                .material = material,
                .message = std::move(message)});
    };

    for (const auto &[id, material] : scene.materials) {
        auto shader = shader_compiler.compile(material.shader);
        if (!shader.ok()) {
            for (const auto &diagnostic : shader.diagnostics) {
                diagnose(id, diagnostic.message);
            }
            continue;
        }

        auto previous = _materials.find(id);
        if (previous != _materials.end() &&
            previous->second.surface_program()
                    ->structure_signature() ==
                shader.program->analysis().structure_signature) {
            auto parameters = bind_surface_parameters(
                *previous->second.surface_program(),
                *shader.program);
            if (!parameters.ok()) {
                for (const auto &diagnostic :
                     parameters.diagnostics) {
                    diagnose(id, diagnostic.message);
                }
                continue;
            }

            auto kind = MaterialUpdateKind::unchanged;
            if (previous->second.shader()
                    .analysis()
                    .parameter_signature !=
                shader.program->analysis().parameter_signature) {
                kind = MaterialUpdateKind::parameters_rebound;
            } else if (previous->second.name() != material.name) {
                kind = MaterialUpdateKind::metadata_updated;
            }
            candidate.insert_or_assign(
                id,
                CompiledMaterial{
                    id,
                    material.name,
                    std::move(shader.program),
                    previous->second.surface_program(),
                    std::move(*parameters.parameters)});
            updates.emplace_back(MaterialUpdate{
                .id = id,
                .kind = kind});
            continue;
        }

        auto surface = compile_surface_program(*shader.program);
        if (!surface.ok()) {
            for (const auto &diagnostic : surface.diagnostics) {
                diagnose(id, diagnostic.message);
            }
            continue;
        }
        auto parameters = bind_surface_parameters(
            *surface.program, *shader.program);
        if (!parameters.ok()) {
            for (const auto &diagnostic : parameters.diagnostics) {
                diagnose(id, diagnostic.message);
            }
            continue;
        }

        const auto kind =
            previous == _materials.end()
                ? MaterialUpdateKind::added
                : MaterialUpdateKind::program_recompiled;
        candidate.insert_or_assign(
            id,
            CompiledMaterial{
                id,
                material.name,
                std::move(shader.program),
                std::move(surface.program),
                std::move(*parameters.parameters)});
        updates.emplace_back(MaterialUpdate{
            .id = id,
            .kind = kind});
    }

    for (auto iter = candidate.begin(); iter != candidate.end();) {
        if (!scene.materials.contains(iter->first)) {
            updates.emplace_back(MaterialUpdate{
                .id = iter->first,
                .kind = MaterialUpdateKind::removed});
            iter = candidate.erase(iter);
        } else {
            ++iter;
        }
    }

    if (!result.diagnostics.empty()) {
        return result;
    }

    _materials = std::move(candidate);
    _source_revision = scene.revision;
    result.committed = true;
    result.source_revision = _source_revision;
    result.updates = std::move(updates);
    return result;
}

}// namespace psycles::compiler
