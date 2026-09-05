#include "path_tracer_cycles_svm_scene.h"

#include "cycles_svm_scene_image.h"
#include "path_tracer_cycles_svm_geometry.h"
#include "path_tracer_cycles_svm_object.h"
#include "path_tracer_scene_geometry.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/cycles_svm.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] constexpr std::size_t
device_scene_extent(std::size_t semantic_extent) noexcept {
  // Luisa buffers cannot represent a zero allocation. The one-element device
  // sentinel is outside every validated Cycles address domain; the exact host
  // image deliberately remains empty.
  return std::max<std::size_t>(semantic_extent, 1u);
}

template<typename T>
void upload_device_scene_array(Stream &stream, Buffer<T> &destination,
                               const std::vector<T> &source) noexcept {
  if (!source.empty()) {
    stream << destination.copy_from(luisa::span{source});
    return;
  }
  // Upload commands may outlive this host function. Static storage makes the
  // unreachable sentinel's lifetime independent of stream synchronization.
  static const std::array<T, 1u> zero{};
  stream << destination.copy_from(luisa::span{zero});
}

[[nodiscard]] bool material_uses_particle(
    const CyclesSvmRuntime &runtime,
    contract::MaterialId material,
    bool &uses_particle,
    std::string &diagnostic) {
  const auto shader = runtime.material_shader_indices.find(material);
  if (shader == runtime.material_shader_indices.end() ||
      shader->second >= runtime.compilation.shader_attribute_ids_used.size()) {
    diagnostic = "Cycles particle demand references unavailable material " +
                 std::to_string(material.value);
    return false;
  }
  const auto &attributes =
      runtime.compilation.shader_attribute_ids_used[shader->second];
  uses_particle = std::ranges::binary_search(
      attributes,
      static_cast<std::uint64_t>(compiler::cycles_svm::ATTR_STD_PARTICLE));
  return true;
}

[[nodiscard]] bool geometry_uses_particle(
    const CyclesSvmRuntime &runtime,
    const contract::SceneSnapshot &snapshot,
    const contract::InstanceDesc &instance,
    bool &uses_particle,
    std::string &diagnostic) {
  uses_particle = false;
  const std::vector<contract::MaterialId> *materials = nullptr;
  if (const auto mesh = snapshot.geometries.find(instance.geometry);
      mesh != snapshot.geometries.end()) {
    materials = &mesh->second.material_slots;
  } else if (const auto curves =
                 snapshot.curve_geometries.find(instance.geometry);
             curves != snapshot.curve_geometries.end()) {
    materials = &curves->second.material_slots;
  } else {
    diagnostic = "Cycles particle demand references unavailable geometry " +
                 std::to_string(instance.geometry.value);
    return false;
  }

  const auto include = [&](contract::MaterialId material) {
    bool shader_uses_particle = false;
    if (!material_uses_particle(runtime, material, shader_uses_particle,
                                diagnostic)) {
      return false;
    }
    uses_particle |= shader_uses_particle;
    return true;
  };
  for (const auto material : *materials) {
    if (!include(material)) {
      return false;
    }
  }
  for (const auto material : instance.material_overrides) {
    if (!include(material)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool build_particle_table(
    CyclesSvmRuntime &runtime,
    const contract::SceneSnapshot &snapshot,
    std::string &diagnostic) {
  std::vector<compiler::cycles_svm::ParticleTableObject> objects;
  objects.reserve(snapshot.instances.size() + snapshot.lights.size() + 1u);
  for (const auto &[instance_id, instance] : snapshot.instances) {
    bool needs_particle = false;
    if (!geometry_uses_particle(runtime, snapshot, instance, needs_particle,
                                diagnostic)) {
      return false;
    }
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index = runtime.object_identities.instance_indices.at(
            instance_id),
        .needs_particle = needs_particle,
        .source = instance.cycles_particle_source});
  }
  for (const auto &[light_id, light] : snapshot.lights) {
    bool needs_particle = false;
    if (light.shader &&
        !material_uses_particle(runtime, *light.shader, needs_particle,
                                diagnostic)) {
      return false;
    }
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index =
            runtime.object_identities.light_indices.at(light_id),
        .needs_particle = needs_particle,
        .source = light.cycles_particle_source});
  }
  if (runtime.object_identities.background_index) {
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index = *runtime.object_identities.background_index,
        .needs_particle = false,
        .source = std::nullopt});
  }
  runtime.particles = compiler::cycles_svm::pack_particle_table(objects);
  if (!runtime.particles.valid) {
    diagnostic = runtime.particles.diagnostic;
    return false;
  }
  runtime.particle_records.reserve(runtime.particles.particles.size());
  for (const auto &particle : runtime.particles.particles) {
    runtime.particle_records.emplace_back(make_cycles_svm_particle(particle));
  }
  return true;
}

} // namespace

std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic) {
  diagnostic.clear();
  auto runtime = std::make_unique<CyclesSvmRuntime>();
  runtime->object_identities =
      compiler::cycles_svm::plan_object_identities(snapshot);
  if (!runtime->object_identities.valid) {
    diagnostic = runtime->object_identities.diagnostic;
    return nullptr;
  }

  const auto shader_materials =
      collect_cycles_svm_shader_materials(snapshot);
  // The native compiler consumes the current validated source graph, not a
  // SurfaceProgram or a retained legacy material cache. Requiring legacy
  // lowering here both narrows Cycles' node domain (for example NODE_CAMERA)
  // and lets a same-id cached graph override a newer scene snapshot.
  const compiler::ShaderCompiler shader_compiler{
      compiler::make_core_node_registry()};
  std::map<contract::MaterialId, std::shared_ptr<const compiler::ShaderProgram>>
      shader_programs;
  for (const auto material : shader_materials) {
    const auto source = snapshot.materials.find(material);
    if (source == snapshot.materials.end()) {
      diagnostic = "Cycles used-shader domain references unavailable material " +
                   std::to_string(material.value);
      return nullptr;
    }
    auto shader = shader_compiler.compile(source->second.shader);
    if (!shader.ok()) {
      diagnostic = "Cycles used-shader material " +
                   std::to_string(material.value) + ": " +
                   (shader.diagnostics.empty()
                        ? "source graph validation failed"
                        : shader.diagnostics.front().message);
      return nullptr;
    }
    shader_programs.emplace(material, std::move(shader.program));
  }

  std::set<std::uint32_t> occupied_indices;
  auto maximum_source_index = std::uint32_t{};
  auto has_source_index = false;
  for (const auto material_id : shader_materials) {
    const auto &source = snapshot.materials.at(material_id);
    if (source.cycles_shader_index) {
      occupied_indices.emplace(*source.cycles_shader_index);
      maximum_source_index =
          std::max(maximum_source_index, *source.cycles_shader_index);
      has_source_index = true;
    }
  }

  auto next_authored_index =
      has_source_index ? static_cast<std::uint64_t>(maximum_source_index) + 1u
                       : std::uint64_t{};
  for (const auto material_id : shader_materials) {
    const auto &source = snapshot.materials.at(material_id);
    auto shader_index = source.cycles_shader_index;
    if (!shader_index) {
      while (next_authored_index <= std::numeric_limits<std::uint32_t>::max() &&
             occupied_indices.contains(
                 static_cast<std::uint32_t>(next_authored_index))) {
        ++next_authored_index;
      }
      if (next_authored_index > std::numeric_limits<std::uint32_t>::max()) {
        diagnostic =
            "renderer-authored Cycles shader identity overflows uint32";
        return nullptr;
      }
      shader_index = static_cast<std::uint32_t>(next_authored_index++);
      occupied_indices.emplace(*shader_index);
    }
    runtime->material_shader_indices.emplace(material_id, *shader_index);
  }

  std::vector<std::pair<std::uint32_t, contract::MaterialId>>
      shader_compile_order;
  shader_compile_order.reserve(shader_materials.size());
  for (const auto material : shader_materials) {
    shader_compile_order.emplace_back(
        runtime->material_shader_indices.at(material), material);
  }
  std::ranges::sort(shader_compile_order);

  std::vector<compiler::cycles_svm::ShaderTableCompileUnit> units;
  units.reserve(shader_compile_order.size());
  for (const auto [shader_index, material_id] : shader_compile_order) {
    const auto &source = snapshot.materials.at(material_id);
    units.emplace_back(compiler::cycles_svm::ShaderTableCompileUnit{
        .shader_index = shader_index,
        .shader = shader_programs.at(material_id).get(),
        .context = {.background = snapshot.world_shader == material_id,
                    .displacement_method = source.displacement_method,
                    .color_space = snapshot.shader_color_space},
        .kernel = {.name = source.name,
                   .use_transparent_shadow =
                       source.use_transparent_shadow,
                   .use_bump_map_correction =
                       source.use_bump_map_correction,
                   .emission_sampling = source.emission_sampling,
                   .volume_sampling = source.volume_sampling,
                   .volume_interpolation = source.volume_interpolation,
                   .pass_id = source.cycles_pass_id}});
  }

  runtime->compilation = compiler::cycles_svm::compile_shader_table(units);
  if (!runtime->compilation.table.valid) {
    diagnostic = runtime->compilation.table.diagnostic;
    return nullptr;
  }
  runtime->kernel_features =
      runtime->compilation.kernel_features |
      compiler::cycles_svm::kernel_feature_path_tracing;
  if (!build_particle_table(*runtime, snapshot, diagnostic)) {
    return nullptr;
  }
  if (!runtime->compilation.table.words.empty()) {
    runtime->word_buffer.emplace(scene->device.create_buffer<luisa::uint>(
        runtime->compilation.table.words.size()));
  }
  if (!runtime->compilation.kernel_shaders.empty()) {
    runtime->kernel_shader_buffer.emplace(
        scene->device.create_buffer<compiler::cycles_svm::KernelShader>(
            runtime->compilation.kernel_shaders.size()));
  }
  if (!runtime->compilation.ies.empty()) {
    runtime->ies_buffer.emplace(
        scene->device.create_buffer<float>(runtime->compilation.ies.size()));
  }
  runtime->image_bindings.reserve(runtime->compilation.images.size());
  auto next_generated_texture_slot = std::uint64_t{1u};
  for (const auto &[image_id, image] : snapshot.images) {
    static_cast<void>(image);
    if (image_id.value > std::numeric_limits<std::uint32_t>::max()) {
      diagnostic = "scene image identity exceeds the 32-bit Luisa bindless "
                   "address space";
      return nullptr;
    }
    next_generated_texture_slot =
        std::max(next_generated_texture_slot, image_id.value + 1u);
  }
  for (const auto &binding : runtime->compilation.images) {
    if (binding.nishita) {
      if (next_generated_texture_slot >
          std::numeric_limits<std::uint32_t>::max()) {
        diagnostic = "Cycles generated sky image exhausts the 32-bit Luisa "
                     "bindless address space";
        return nullptr;
      }
      const auto texture_slot =
          static_cast<std::uint32_t>(next_generated_texture_slot++);
      runtime->nishita_images.emplace_back(
          CyclesSvmNishitaImageRuntime{.parameters = *binding.nishita,
                                      .texture_slot = texture_slot});
      runtime->image_bindings.emplace_back(make_cycles_svm_image_binding(
          texture_slot, binding.interpolation, binding.extension));
      continue;
    }
    if (binding.resource_id > std::numeric_limits<std::uint32_t>::max()) {
      diagnostic = "Cycles SVM image resource identity exceeds the 32-bit "
                   "Luisa bindless address space";
      return nullptr;
    }
    const contract::ImageId image_id{binding.resource_id};
    if (!snapshot.images.contains(image_id)) {
      diagnostic = "Cycles SVM image handle references unavailable scene "
                   "resource " +
                   std::to_string(binding.resource_id);
      return nullptr;
    }
    runtime->image_bindings.emplace_back(make_cycles_svm_image_binding(
        static_cast<std::uint32_t>(binding.resource_id),
        binding.interpolation, binding.extension));
  }
  if (!runtime->image_bindings.empty()) {
    runtime->image_binding_buffer.emplace(
        scene->device.create_buffer<CyclesSvmImageBindingGpu>(
            runtime->image_bindings.size()));
  }
  // Cycles always materializes dummy particle entry zero, even when no object
  // has particle data. The formal host packer therefore makes this non-empty.
  runtime->particle_buffer.emplace(
      scene->device.create_buffer<CyclesSvmParticleGpu>(
          runtime->particle_records.size()));
  return runtime;
}

void upload_cycles_svm_runtime(Stream &stream,
                               CyclesSvmRuntime &runtime) noexcept {
  if (runtime.word_buffer) {
    stream << runtime.word_buffer->copy_from(
        luisa::span{runtime.compilation.table.words});
  }
  if (runtime.kernel_shader_buffer) {
    stream << runtime.kernel_shader_buffer->copy_from(
        luisa::span{runtime.compilation.kernel_shaders});
  }
  if (runtime.ies_buffer) {
    stream << runtime.ies_buffer->copy_from(
        luisa::span{runtime.compilation.ies});
  }
  if (runtime.image_binding_buffer) {
    stream << runtime.image_binding_buffer->copy_from(
        luisa::span{runtime.image_bindings});
  }
  if (runtime.particle_buffer) {
    stream << runtime.particle_buffer->copy_from(
        luisa::span{runtime.particle_records});
  }
}

bool finalize_cycles_svm_scene_runtime(
    const std::shared_ptr<LuisaSceneData> &scene,
    const contract::SceneSnapshot &snapshot,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets,
    std::string &diagnostic) {
  diagnostic.clear();
  if (!scene || !scene->cycles_svm) {
    diagnostic = "Cycles geometry finalization requires an SVM runtime";
    return false;
  }
  auto &runtime = *scene->cycles_svm;
  if (runtime.geometry || runtime.objects) {
    diagnostic = "Cycles geometry/object runtime was finalized more than once";
    return false;
  }

  auto geometry_image = build_cycles_svm_geometry_scene_image(
      snapshot, runtime.compilation, runtime.material_shader_indices,
      runtime.object_identities, intersection_plans, mesh_uploads,
      resource_geometry_indices, triangle_primitive_offsets,
      curve_primitive_offsets);
  if (!geometry_image.valid) {
    diagnostic = std::move(geometry_image.diagnostic);
    return false;
  }
  auto object_image = build_cycles_svm_object_scene_image(
      snapshot, runtime.object_identities, runtime.particles, geometry_image,
      intersection_plans, mesh_uploads, resource_geometry_indices);
  if (!object_image.valid) {
    diagnostic = std::move(object_image.diagnostic);
    return false;
  }

  const auto &attributes = geometry_image.attributes;
  auto attribute_map_buffer =
      scene->device.create_buffer<compiler::cycles_svm::AttributeMap>(
          device_scene_extent(attributes.attribute_map.size()));
  auto attribute_float_buffer = scene->device.create_buffer<float>(
      device_scene_extent(attributes.attributes_float.size()));
  auto attribute_float2_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float2>(
          device_scene_extent(attributes.attributes_float2.size()));
  auto attribute_float3_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float3>(
          device_scene_extent(attributes.attributes_float3.size()));
  auto attribute_float4_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float4>(
          device_scene_extent(attributes.attributes_float4.size()));
  auto attribute_uchar4_buffer =
      scene->device.create_buffer<compiler::cycles_svm::uchar4>(
          device_scene_extent(attributes.attributes_uchar4.size()));
  auto attribute_normal_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_normal>(
          device_scene_extent(attributes.attributes_normal.size()));
  auto triangle_vertex_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float3>(
          device_scene_extent(attributes.tri_verts.size()));
  auto curve_key_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float4>(
          device_scene_extent(attributes.curve_keys.size()));
  auto point_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_float4>(
          device_scene_extent(attributes.points.size()));
  auto triangle_index_buffer =
      scene->device.create_buffer<compiler::cycles_svm::packed_uint3>(
          device_scene_extent(geometry_image.triangle_vertex_indices.size()));
  auto triangle_shader_buffer = scene->device.create_buffer<luisa::uint>(
      device_scene_extent(geometry_image.triangle_shaders.size()));
  auto curve_buffer =
      scene->device.create_buffer<compiler::cycles_svm::KernelCurve>(
          device_scene_extent(geometry_image.curves.size()));
  auto object_buffer =
      scene->device.create_buffer<compiler::cycles_svm::KernelObject>(
          device_scene_extent(object_image.objects.size()));
  auto object_flag_buffer = scene->device.create_buffer<luisa::uint>(
      device_scene_extent(object_image.object_flags.size()));

  auto geometry_runtime = std::make_unique<CyclesSvmGeometryRuntime>(
      CyclesSvmGeometryRuntime{
          .image = std::move(geometry_image),
          .attribute_map_buffer = std::move(attribute_map_buffer),
          .attribute_float_buffer = std::move(attribute_float_buffer),
          .attribute_float2_buffer = std::move(attribute_float2_buffer),
          .attribute_float3_buffer = std::move(attribute_float3_buffer),
          .attribute_float4_buffer = std::move(attribute_float4_buffer),
          .attribute_uchar4_buffer = std::move(attribute_uchar4_buffer),
          .attribute_normal_buffer = std::move(attribute_normal_buffer),
          .triangle_vertex_buffer = std::move(triangle_vertex_buffer),
          .curve_key_buffer = std::move(curve_key_buffer),
          .point_buffer = std::move(point_buffer),
          .triangle_index_buffer = std::move(triangle_index_buffer),
          .triangle_shader_buffer = std::move(triangle_shader_buffer),
          .curve_buffer = std::move(curve_buffer)});
  auto object_runtime = std::make_unique<CyclesSvmObjectRuntime>(
      CyclesSvmObjectRuntime{.image = std::move(object_image),
                             .object_buffer = std::move(object_buffer),
                             .object_flag_buffer =
                                 std::move(object_flag_buffer)});
  auto kernel_features = runtime.kernel_features;
  // Cycles Scene::update_kernel_features() derives hair shape features from
  // instanced Geometry, not merely from every datablock present in the scene.
  for (const auto &[instance_id, instance] : snapshot.instances) {
    static_cast<void>(instance_id);
    const auto curve = snapshot.curve_geometries.find(instance.geometry);
    if (curve == snapshot.curve_geometries.end()) {
      continue;
    }
    kernel_features |=
        curve->second.shape == contract::CurveShape::ribbon
            ? compiler::cycles_svm::kernel_feature_hair_ribbon
            : compiler::cycles_svm::kernel_feature_hair_thick;
  }
  runtime.kernel_features = kernel_features;
  runtime.geometry = std::move(geometry_runtime);
  runtime.objects = std::move(object_runtime);
  return true;
}

void upload_cycles_svm_scene_runtime(Stream &stream,
                                     CyclesSvmRuntime &runtime) noexcept {
  if (!runtime.geometry || !runtime.objects) {
    return;
  }
  auto &geometry = *runtime.geometry;
  const auto &attributes = geometry.image.attributes;
  upload_device_scene_array(stream, geometry.attribute_map_buffer,
                            attributes.attribute_map);
  upload_device_scene_array(stream, geometry.attribute_float_buffer,
                            attributes.attributes_float);
  upload_device_scene_array(stream, geometry.attribute_float2_buffer,
                            attributes.attributes_float2);
  upload_device_scene_array(stream, geometry.attribute_float3_buffer,
                            attributes.attributes_float3);
  upload_device_scene_array(stream, geometry.attribute_float4_buffer,
                            attributes.attributes_float4);
  upload_device_scene_array(stream, geometry.attribute_uchar4_buffer,
                            attributes.attributes_uchar4);
  upload_device_scene_array(stream, geometry.attribute_normal_buffer,
                            attributes.attributes_normal);
  upload_device_scene_array(stream, geometry.triangle_vertex_buffer,
                            attributes.tri_verts);
  upload_device_scene_array(stream, geometry.curve_key_buffer,
                            attributes.curve_keys);
  upload_device_scene_array(stream, geometry.point_buffer, attributes.points);
  upload_device_scene_array(stream, geometry.triangle_index_buffer,
                            geometry.image.triangle_vertex_indices);
  upload_device_scene_array(stream, geometry.triangle_shader_buffer,
                            geometry.image.triangle_shaders);
  upload_device_scene_array(stream, geometry.curve_buffer,
                            geometry.image.curves);
  auto &objects = *runtime.objects;
  upload_device_scene_array(stream, objects.object_buffer,
                            objects.image.objects);
  upload_device_scene_array(stream, objects.object_flag_buffer,
                            objects.image.object_flags);
}

} // namespace psycles::luisa_backend::detail
