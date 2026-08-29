#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

namespace psycles::compiler {

enum class MaterialUpdateKind : std::uint8_t {
    added,
    program_recompiled,
    parameters_rebound,
    metadata_updated,
    unchanged,
    removed
};

struct MaterialUpdate {
    contract::MaterialId id;
    MaterialUpdateKind kind{MaterialUpdateKind::unchanged};
};

struct MaterialCompilationDiagnostic {
    contract::MaterialId material;
    std::string message;
};

struct MaterialLibraryUpdate {
    bool committed{false};
    std::uint64_t source_revision{};
    std::vector<MaterialUpdate> updates;
    std::vector<MaterialCompilationDiagnostic> diagnostics;
};

class CompiledMaterial {

private:
    contract::MaterialId _id;
    std::string _name;
    std::shared_ptr<const ShaderProgram> _shader;
    std::shared_ptr<const SurfaceProgram> _surface;
    SurfaceParameterBlock _parameters;

public:
    CompiledMaterial(
        contract::MaterialId id,
        std::string name,
        std::shared_ptr<const ShaderProgram> shader,
        std::shared_ptr<const SurfaceProgram> surface,
        SurfaceParameterBlock parameters) noexcept;

    [[nodiscard]] contract::MaterialId id() const noexcept {
        return _id;
    }
    [[nodiscard]] const std::string &name() const noexcept {
        return _name;
    }
    [[nodiscard]] const ShaderProgram &shader() const noexcept {
        return *_shader;
    }
    [[nodiscard]] const std::shared_ptr<const SurfaceProgram> &
    surface_program() const noexcept {
        return _surface;
    }
    [[nodiscard]] const SurfaceParameterBlock &parameters() const noexcept {
        return _parameters;
    }
};

class MaterialLibrary {

private:
    std::uint64_t _source_revision{};
    std::map<contract::MaterialId, CompiledMaterial> _materials;

public:
    [[nodiscard]] std::uint64_t source_revision() const noexcept {
        return _source_revision;
    }
    [[nodiscard]] const CompiledMaterial *find(
        contract::MaterialId id) const noexcept;
    [[nodiscard]] const auto &materials() const noexcept {
        return _materials;
    }

    [[nodiscard]] MaterialLibraryUpdate update(
        const contract::SceneSnapshot &scene,
        const ShaderCompiler &shader_compiler);

    // Compile exactly the closed material domain supplied by the scene
    // consumer. The transaction removes previously compiled entries outside
    // the domain and rejects missing roots without mutating the library.
    // This is a semantic reachability boundary, not a heuristic shader cache:
    // callers must include every MaterialId that their runtime can
    // dereference.
    [[nodiscard]] MaterialLibraryUpdate update(
        const contract::SceneSnapshot &scene,
        const ShaderCompiler &shader_compiler,
        const std::set<contract::MaterialId> &material_domain);
};

}// namespace psycles::compiler
