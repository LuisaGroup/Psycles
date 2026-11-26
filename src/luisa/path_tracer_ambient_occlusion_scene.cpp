#include "path_tracer_ambient_occlusion_scene.h"

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <span>

namespace psycles::luisa_backend::detail {

namespace {

[[nodiscard]] bool block_has_local_ao(
    std::span<const std::uint32_t> words,
    const compiler::cycles_svm::SvmBlockIr &block) noexcept {
  for (const auto &op : block) {
    if (op.is_conditional) {
      if (block_has_local_ao(words, op.children)) {
        return true;
      }
      continue;
    }
    if (op.instruction.opcode == compiler::cycles_svm::NODE_AMBIENT_OCCLUSION) {
      const auto flags_word_index =
          static_cast<std::size_t>(op.instruction.payload_offset) + 4u;
      if (flags_word_index >= words.size()) {
        return true;
      }
      const auto flags = words[flags_word_index] & 0xffu;
      if ((flags & compiler::cycles_svm::NODE_AO_ONLY_LOCAL) != 0u) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

AmbientOcclusionSceneComponent::AmbientOcclusionSceneComponent(
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    std::uint32_t triangle_instance_count) noexcept {
    const auto &table = compilation.table;
    _plan.enabled =
        table.valid &&
        table.node_types_used[compiler::cycles_svm::NODE_AMBIENT_OCCLUSION];
    _plan.triangle_instance_count = triangle_instance_count;
    if (_plan.enabled) {
        if (!table.ir.valid) {
            _plan.local_requested = true;
        } else {
            const auto words = std::span<const std::uint32_t>{table.words};
            for (const auto &shader : table.ir.shaders) {
                if (block_has_local_ao(
                        words,
                        shader.routine_for(
                            compiler::cycles_svm::SHADER_TYPE_SURFACE))) {
                    _plan.local_requested = true;
                    break;
                }
            }
        }
    }
}

void AmbientOcclusionSceneComponent::initialize(
    const std::shared_ptr<LuisaSceneData> &scene) const {
    scene->has_ambient_occlusion = _plan.enabled;
    scene->has_local_ambient_occlusion = _plan.local_requested;
    if (_plan.enabled) {
        scene->ambient_occlusion_distance_buffer.emplace(
            scene->device.create_buffer<float>(1u));
    }
    if (_plan.local_enabled()) {
        LUISA_INFO(
            "Psycles Only Local AO domain: triangle_instances={}.",
            _plan.triangle_instance_count);
        scene->ambient_occlusion_local_accel.emplace(
            scene->device.create_accel(static_scene_accel_option()));
    }
}

void AmbientOcclusionSceneComponent::upload_distance(
    Stream &stream,
    const std::shared_ptr<LuisaSceneData> &scene,
    float distance) const {
    if (!_plan.enabled) {
        return;
    }
    LUISA_ASSERT(
        scene->ambient_occlusion_distance_buffer,
        "AO distance buffer was not initialized.");
    const std::array values{distance};
    stream << scene->ambient_occlusion_distance_buffer->copy_from(
        luisa::span{values});
}

void AmbientOcclusionSceneComponent::append_triangle_instance(
    const std::shared_ptr<LuisaSceneData> &scene,
    std::uint32_t primary_instance,
    std::uint32_t geometry_index,
    const Mat4f &transform) const {
    if (!_plan.local_enabled()) {
        return;
    }
    LUISA_ASSERT(
        scene->ambient_occlusion_local_accel &&
            geometry_index < scene->geometries.size(),
        "Invalid Only Local AO scene plan.");
    scene->ambient_occlusion_local_accel->emplace_back(
        scene->geometries[geometry_index].mesh,
        to_luisa(transform),
        std::uint8_t{0xffu},
        false,
        primary_instance);
}

}// namespace psycles::luisa_backend::detail
