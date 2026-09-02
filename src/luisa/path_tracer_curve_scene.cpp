#include "path_tracer_curve_scene.h"

#include "path_tracer_scene_geometry.h"

#include <array>
#include <cmath>
#include <limits>

namespace psycles::luisa_backend::detail {
namespace {

struct ScalarBounds {
  double lower{};
  double upper{};
};

[[nodiscard]] ScalarBounds
catmull_rom_bounds(const std::array<float, 4u> &control) noexcept {
  const auto p0 = static_cast<double>(control[0u]);
  const auto p1 = static_cast<double>(control[1u]);
  const auto p2 = static_cast<double>(control[2u]);
  const auto p3 = static_cast<double>(control[3u]);
  const auto cubic = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const auto quadratic = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const auto linear = 0.5 * (-p0 + p2);
  const auto evaluate = [&](double u) noexcept {
    return ((cubic * u + quadratic) * u + linear) * u + p1;
  };

  ScalarBounds bounds{.lower = std::min(p1, p2), .upper = std::max(p1, p2)};
  const auto include = [&](double u) noexcept {
    if (u > 0.0 && u < 1.0) {
      const auto value = evaluate(u);
      bounds.lower = std::min(bounds.lower, value);
      bounds.upper = std::max(bounds.upper, value);
    }
  };
  if (cubic != 0.0) {
    const auto discriminant = quadratic * quadratic - 3.0 * cubic * linear;
    if (discriminant >= 0.0) {
      const auto root = std::sqrt(discriminant);
      include((-quadratic - root) / (3.0 * cubic));
      include((-quadratic + root) / (3.0 * cubic));
    }
  } else if (quadratic != 0.0) {
    include(-linear / (2.0 * quadratic));
  }
  return bounds;
}

[[nodiscard]] luisa::compute::AABB
segment_bounds(const contract::CurveGeometryDesc &geometry,
               const CurveSegmentGpu &segment) noexcept {
  const auto &p0 = geometry.keys[segment.key_before];
  const auto &p1 = geometry.keys[segment.key_begin];
  const auto &p2 = geometry.keys[segment.key_end];
  const auto &p3 = geometry.keys[segment.key_after];
  const auto x = catmull_rom_bounds({p0.x, p1.x, p2.x, p3.x});
  const auto y = catmull_rom_bounds({p0.y, p1.y, p2.y, p3.y});
  const auto z = catmull_rom_bounds({p0.z, p1.z, p2.z, p3.z});
  const auto radius = catmull_rom_bounds({p0.w, p1.w, p2.w, p3.w});
  const auto extent = std::max(std::abs(radius.lower), std::abs(radius.upper));
  const auto outward_lower = [](double value) noexcept {
    return std::nextafter(static_cast<float>(value),
                          -std::numeric_limits<float>::infinity());
  };
  const auto outward_upper = [](double value) noexcept {
    return std::nextafter(static_cast<float>(value),
                          std::numeric_limits<float>::infinity());
  };
  return {.packed_min = {outward_lower(x.lower - extent),
                         outward_lower(y.lower - extent),
                         outward_lower(z.lower - extent)},
          .packed_max = {outward_upper(x.upper + extent),
                         outward_upper(y.upper + extent),
                         outward_upper(z.upper + extent)}};
}

} // namespace

CurveGeometryUpload
build_curve_geometry_upload(const contract::CurveGeometryDesc &geometry,
                            std::uint32_t cycles_curve_offset,
                            std::uint32_t cycles_segment_offset) {
  CurveGeometryUpload upload;
  upload.keys.reserve(geometry.keys.size());
  for (const auto key : geometry.keys) {
    upload.keys.emplace_back(luisa::make_float4(key.x, key.y, key.z, key.w));
  }
  upload.material_slots.reserve(geometry.curve_first_key.size());
  upload.default_uv_layer = geometry.default_uv_layer;
  for (const auto &[name, source] : geometry.uv_layers) {
    auto &destination = upload.uv_layers[name];
    destination.reserve(source.size());
    for (const auto value : source) {
      destination.emplace_back(luisa::make_float2(value.x, value.y));
    }
  }
  for (const auto &[name, source] : geometry.color_attributes) {
    auto &destination = upload.color_attributes[name];
    destination.reserve(source.size());
    for (const auto value : source) {
      destination.emplace_back(
          luisa::make_float4(value.x, value.y, value.z, value.w));
    }
  }
  upload.length.reserve(geometry.curve_first_key.size());
  upload.random.reserve(geometry.curve_first_key.size());
  upload.intercept.reserve(geometry.keys.size());
  for (std::size_t key = 0u; key < geometry.keys.size(); ++key) {
    upload.intercept.emplace_back(
        key < geometry.intercept.size() ? geometry.intercept[key] : 0.0f);
  }

  std::uint32_t segment_index = 0u;
  for (std::size_t curve = 0u; curve < geometry.curve_first_key.size();
       ++curve) {
    const auto first =
        static_cast<std::size_t>(geometry.curve_first_key[curve]);
    const auto end =
        curve + 1u < geometry.curve_first_key.size()
            ? static_cast<std::size_t>(geometry.curve_first_key[curve + 1u])
            : geometry.keys.size();
    upload.material_slots.emplace_back(
        curve < geometry.curve_material_slots.size()
            ? geometry.curve_material_slots[curve]
            : 0u);
    upload.length.emplace_back(
        curve < geometry.length.size() ? geometry.length[curve] : 0.0f);
    upload.random.emplace_back(
        curve < geometry.random.size() ? geometry.random[curve] : 0.0f);
    for (auto key = first; key + 1u < end; ++key) {
      const CurveSegmentGpu segment{
          .key_before =
              static_cast<std::uint32_t>(key > first ? key - 1u : first),
          .key_begin = static_cast<std::uint32_t>(key),
          .key_end = static_cast<std::uint32_t>(key + 1u),
          .key_after = static_cast<std::uint32_t>(std::min(key + 2u, end - 1u)),
          .curve_index = static_cast<std::uint32_t>(curve),
          .cycles_curve_index =
              cycles_curve_offset + static_cast<std::uint32_t>(curve),
          .cycles_segment_index = cycles_segment_offset + segment_index};
      upload.segments.emplace_back(segment);
      upload.bounds.emplace_back(segment_bounds(geometry, segment));
      ++segment_index;
    }
  }
  return upload;
}

CurveSceneUploadResult CurveSceneUploadComponent::upload(
    const std::shared_ptr<LuisaSceneData> &data, const SceneSnapshot &snapshot,
    Stream &stream,
    std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    luisa::vector<GeometryGpu> &geometry_gpu,
    luisa::vector<MaterialBindingGpu> &geometry_materials,
    luisa::vector<AttributeBindingGpu> &attribute_bindings,
    luisa::vector<AttributeRangeGpu> &attribute_ranges,
    const SceneAttributeResidencyPlan &attribute_residency,
    std::uint32_t &next_attribute_slot) const {
  CurveSceneUploadResult result;
  CyclesPrimitiveIntervalResolver curve_intervals;
  CyclesPrimitiveIntervalResolver segment_intervals;
  data->curve_geometries.reserve(snapshot.curve_geometries.size());

  for (const auto &[geometry_id, geometry] : snapshot.curve_geometries) {
    if (geometry.shape != contract::CurveShape::ribbon) {
      result.diagnostic = "Curve geometry '" + geometry.name +
                          "' does not use the currently implemented Cycles "
                          "RIBBON intersection model.";
      return result;
    }
    if (geometry.keys.size() > std::numeric_limits<std::uint32_t>::max() ||
        geometry.curve_first_key.size() >
            std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = "Curve geometry '" + geometry.name +
                          "' exceeds the 32-bit Cycles geometry address space.";
      return result;
    }
    std::size_t segment_count = 0u;
    for (std::size_t curve = 0u; curve < geometry.curve_first_key.size();
         ++curve) {
      const auto first =
          static_cast<std::size_t>(geometry.curve_first_key[curve]);
      const auto end =
          curve + 1u < geometry.curve_first_key.size()
              ? static_cast<std::size_t>(geometry.curve_first_key[curve + 1u])
              : geometry.keys.size();
      segment_count += end - first - 1u;
    }
    const auto curve_interval = curve_intervals.resolve(
        geometry.curve_first_key.size(), geometry.cycles_curve_offset);
    const auto segment_interval = segment_intervals.resolve(
        segment_count, geometry.cycles_segment_offset);
    if (!curve_interval.offset || !segment_interval.offset) {
      result.diagnostic = "Curve geometry '" + geometry.name +
                          "' has overlapping or out-of-range Cycles curve "
                          "or segment intervals.";
      return result;
    }

    auto upload = build_curve_geometry_upload(geometry, *curve_interval.offset,
                                              *segment_interval.offset);
    const auto geometry_index = static_cast<std::uint32_t>(geometry_gpu.size());
    const auto bindless_base = geometry_index * geometry_bindless_stride;
    const auto material_offset =
        static_cast<std::uint32_t>(geometry_materials.size());
    for (const auto material_id : geometry.material_slots) {
      const auto material = data->material_bindings.find(material_id);
      geometry_materials.emplace_back(
          material != data->material_bindings.end()
              ? to_luisa(material->second)
              : inert_material_binding());
    }
    if (geometry.material_slots.empty()) {
      geometry_materials.emplace_back(inert_material_binding());
    }

    const auto resource_index =
        static_cast<std::uint32_t>(data->curve_geometries.size());
    auto &resource = data->curve_geometries.emplace_back();
    resource.bounds =
        data->device.create_buffer<luisa::compute::AABB>(upload.bounds.size());
    resource.segments =
        data->device.create_buffer<CurveSegmentGpu>(upload.segments.size());
    resource.keys =
        data->device.create_buffer<luisa::float4>(upload.keys.size());
    resource.material_slots =
        data->device.create_buffer<luisa::uint>(upload.material_slots.size());
    resource.intercept =
        data->device.create_buffer<float>(upload.intercept.size());
    resource.length = data->device.create_buffer<float>(upload.length.size());
    resource.random = data->device.create_buffer<float>(upload.random.size());
    resource.primitive =
        data->device.create_procedural_primitive(resource.bounds);

    data->heap.emplace_on_update(bindless_base, resource.segments);
    data->heap.emplace_on_update(bindless_base + 1u, resource.keys);
    data->heap.emplace_on_update(bindless_base + 3u, resource.intercept);
    data->heap.emplace_on_update(bindless_base + 4u, resource.material_slots);
    data->heap.emplace_on_update(bindless_base + 5u, resource.length);
    data->heap.emplace_on_update(bindless_base + 6u, resource.random);
    const auto attribute_offset =
        static_cast<std::uint32_t>(attribute_bindings.size());
    const auto &resident = attribute_residency.geometry(geometry_id);
    bool default_uv_available = false;
    for (const auto &[name, values] : upload.uv_layers) {
      const auto is_default =
          upload.default_uv_layer && *upload.default_uv_layer == name;
      const auto id = contract::uv_attribute_id(name);
      const auto retain_named = resident.contains(id);
      if (!is_default && !retain_named) {
        continue;
      }
      auto &buffer = resource.uv_layers.emplace_back(
          data->device.create_buffer<luisa::float2>(values.size()));
      const auto slot = is_default ? bindless_base + 2u
                                   : next_attribute_slot++;
      data->heap.emplace_on_update(slot, buffer);
      stream << buffer.copy_from(luisa::span{values});
      default_uv_available |= is_default;
      if (retain_named) {
        attribute_bindings.emplace_back(AttributeBindingGpu{
            .id = id,
            .value_slot = slot,
            .domain = pack_attribute_layout(
                attribute_domain_curve, attribute_format_float2)});
      }
    }
    for (const auto &[name, values] : upload.color_attributes) {
      const auto id = contract::attribute_id(name);
      if (!resident.contains(id)) {
        continue;
      }
      auto &buffer = resource.attributes.emplace_back(
          data->device.create_buffer<luisa::float4>(values.size()));
      const auto slot = next_attribute_slot++;
      data->heap.emplace_on_update(slot, buffer);
      stream << buffer.copy_from(luisa::span{values});
      attribute_bindings.emplace_back(AttributeBindingGpu{
          .id = id,
          .value_slot = slot,
          .domain = pack_attribute_layout(
              attribute_domain_curve, attribute_format_float4)});
    }
    stream << resource.bounds.copy_from(luisa::span{upload.bounds})
           << resource.segments.copy_from(luisa::span{upload.segments})
           << resource.keys.copy_from(luisa::span{upload.keys})
           << resource.material_slots.copy_from(
                  luisa::span{upload.material_slots})
           << resource.intercept.copy_from(luisa::span{upload.intercept})
           << resource.length.copy_from(luisa::span{upload.length})
           << resource.random.copy_from(luisa::span{upload.random})
           << resource.primitive.build();

    geometry_indices.emplace(geometry_id, geometry_index);
    result.resource_indices.emplace(geometry_id, resource_index);
    result.cycles_curve_offsets.emplace(geometry_id, *curve_interval.offset);
    result.cycles_segment_offsets.emplace(geometry_id,
                                          *segment_interval.offset);
    attribute_ranges.emplace_back(
        AttributeRangeGpu{.offset = attribute_offset,
                          .count = static_cast<std::uint32_t>(
                              attribute_bindings.size() - attribute_offset),
                          .primitive_slot = bindless_base,
                          .padding = 0u});
    geometry_gpu.emplace_back(GeometryGpu{
        .bindless_base = bindless_base,
        .material_offset = material_offset,
        .material_count = static_cast<std::uint32_t>(
            std::max<std::size_t>(geometry.material_slots.size(), 1u)),
        .attribute_domains = default_uv_available
                                 ? geometry_curve_default_uv
                                 : 0u,
        .cycles_primitive_offset = *curve_interval.offset,
        .cycles_segment_offset = *segment_interval.offset,
        .primitive_kind = geometry_kind_curve,
        .curve_shape = static_cast<std::uint32_t>(geometry.shape),
        .curve_subdivision_level = geometry.subdivisions,
        .generated_transform = luisa::make_float4x4(1.0f)});
  }
  return result;
}

} // namespace psycles::luisa_backend::detail
