#include "path_tracer_attribute_residency.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::uint64_t float4_bytes(std::size_t count) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 16u
               ? maximum
               : static_cast<std::uint64_t>(count) * 16u;
}

[[nodiscard]] std::uint64_t float2_bytes(std::size_t count) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 8u
               ? maximum
               : static_cast<std::uint64_t>(count) * 8u;
}

[[nodiscard]] std::uint64_t float3_bytes(std::size_t count) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 12u
               ? maximum
               : static_cast<std::uint64_t>(count) * 12u;
}

[[nodiscard]] std::uint64_t float_bytes(std::size_t count) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 4u
               ? maximum
               : static_cast<std::uint64_t>(count) * 4u;
}

void add_bytes(std::uint64_t &total, std::uint64_t bytes) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    total = bytes > maximum - total ? maximum : total + bytes;
}

void close_tangent_dependencies(
    SurfaceAttributeDemand &demand,
    const contract::TriangleMeshDesc &geometry) {
    if (demand.all) {
        return;
    }
    for (const auto &[name, tangent] : geometry.uv_tangent_layers) {
        static_cast<void>(tangent);
        if (demand.contains(contract::uv_tangent_attribute_id(name)) ||
            demand.contains(
                contract::uv_undisplaced_tangent_attribute_id(name))) {
            // MikkTSpace construction consumes the corresponding UV layer at
            // scene-build time.
            demand.ids.emplace(contract::uv_attribute_id(name));
        }
    }
}

void measure_geometry(
    SceneAttributeResidencyPlan &plan,
    const contract::TriangleMeshDesc &geometry,
    const GeometryAttributeResidency &residency) {
    const auto measure = [&](std::uint64_t id, std::size_t count) {
        const auto bytes = float4_bytes(count);
        ++plan.source_binding_count;
        add_bytes(plan.source_device_bytes, bytes);
        if (residency.contains(id)) {
            ++plan.resident_binding_count;
            add_bytes(plan.resident_device_bytes, bytes);
        }
    };
    for (const auto &[name, attribute] : geometry.color_attributes) {
        measure(contract::attribute_id(name), attribute.values.size());
    }
    const auto measure_typed = [&](const auto &attributes, auto bytes_fn) {
        for (const auto &[name, attribute] : attributes) {
            const auto id = contract::attribute_id(name);
            const auto bytes = bytes_fn(attribute.values.size());
            ++plan.source_binding_count;
            add_bytes(plan.source_device_bytes, bytes);
            if (residency.contains(id)) {
                ++plan.resident_binding_count;
                add_bytes(plan.resident_device_bytes, bytes);
            }
        }
    };
    measure_typed(geometry.float_attributes, float_bytes);
    measure_typed(geometry.float2_attributes, float2_bytes);
    measure_typed(geometry.float3_attributes, float3_bytes);
    for (const auto &[name, attribute] : geometry.uv_layers) {
        measure(contract::uv_attribute_id(name), attribute.values.size());
    }
    const auto tangent_count = geometry.triangles.size() * 3u;
    for (const auto &[name, attribute] : geometry.uv_tangent_layers) {
        static_cast<void>(attribute);
        measure(contract::uv_tangent_attribute_id(name), tangent_count);
        measure(
            contract::uv_undisplaced_tangent_attribute_id(name),
            tangent_count);
    }
    if (geometry.pointiness_source) {
        measure(
            contract::cycles_pointiness_attribute_id,
            geometry.positions.size());
    }
}

void measure_geometry(
    SceneAttributeResidencyPlan &plan,
    const contract::CurveGeometryDesc &geometry,
    const GeometryAttributeResidency &residency) {
    for (const auto &[name, values] : geometry.color_attributes) {
        const auto id = contract::attribute_id(name);
        const auto bytes = float4_bytes(values.size());
        ++plan.source_binding_count;
        add_bytes(plan.source_device_bytes, bytes);
        if (residency.contains(id)) {
            ++plan.resident_binding_count;
            add_bytes(plan.resident_device_bytes, bytes);
        }
    }
    for (const auto &[name, values] : geometry.uv_layers) {
        const auto id = contract::uv_attribute_id(name);
        const auto bytes = float2_bytes(values.size());
        ++plan.source_binding_count;
        add_bytes(plan.source_device_bytes, bytes);
        if (residency.contains(id)) {
            ++plan.resident_binding_count;
            add_bytes(plan.resident_device_bytes, bytes);
        }
    }
}

}// namespace

bool SurfaceAttributeDemand::contains(std::uint64_t id) const noexcept {
    return all || ids.contains(id);
}

const GeometryAttributeResidency &SceneAttributeResidencyPlan::geometry(
    contract::GeometryId id) const noexcept {
    static const GeometryAttributeResidency conservative{
        .demand = SurfaceAttributeDemand{.all = true}};
    const auto iter = geometries.find(id);
    return iter == geometries.end() ? conservative : iter->second;
}

SceneAttributeResidencyPlan build_scene_attribute_residency_plan(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t> &material_shader_indices) {
    namespace abi = compiler::cycles_svm;
    const auto geometry_requests = [&](contract::GeometryId id, const auto &geometry) {
        std::set<std::uint64_t> requests;
        const auto append = [&](contract::MaterialId material) {
            const auto index = material_shader_indices.at(material);
            const auto &ids = compilation.shader_attribute_ids_used.at(index);
            requests.insert(ids.begin(), ids.end());
        };
        for (const auto material : geometry.material_slots) { append(material); }
        for (const auto &[instance_id, instance] : snapshot.instances) {
            static_cast<void>(instance_id);
            if (instance.geometry == id) {
                for (const auto material : instance.material_overrides) { append(material); }
            }
        }
        return requests;
    };
    const auto project_named = [&](SurfaceAttributeDemand &demand,
                                   const auto &geometry, const auto &requests) {
        for (const auto &[name, id] : compilation.named_attributes) {
            if (!requests.contains(id)) { continue; }
            if (geometry.uv_layers.contains(name)) {
                demand.ids.emplace(contract::uv_attribute_id(name));
                continue;
            }
            if constexpr (requires { geometry.cycles_byte_color_attributes; }) {
                if (geometry.cycles_byte_color_attributes.contains(name)) { continue; }
            }
            if (geometry.color_attributes.contains(name)) {
                demand.ids.emplace(contract::attribute_id(name));
            } else if constexpr (requires { geometry.float_attributes; }) {
                if (geometry.float_attributes.contains(name) ||
                    geometry.float2_attributes.contains(name) ||
                    geometry.float3_attributes.contains(name)) {
                    demand.ids.emplace(contract::attribute_id(name));
                } else {
                    constexpr std::array<std::pair<std::string_view, bool>, 4u> suffixes{{
                        {".undisplaced_tangent_sign", true}, {".undisplaced_tangent", true},
                        {".tangent_sign", false}, {".tangent", false}}};
                    for (const auto &[suffix, undisplaced] : suffixes) {
                        if (!name.ends_with(suffix)) { continue; }
                        const auto base = name.substr(0u, name.size() - suffix.size());
                        if (geometry.uv_layers.contains(base)) {
                            demand.ids.emplace(undisplaced
                                ? contract::uv_undisplaced_tangent_attribute_id(base)
                                : contract::uv_tangent_attribute_id(base));
                            demand.ids.emplace(contract::uv_attribute_id(base));
                        }
                        break;
                    }
                }
            }
        }
    };
    SceneAttributeResidencyPlan result;
    for (const auto &[id, geometry] : snapshot.geometries) {
        auto &residency = result.geometries[id];
        const auto requests = geometry_requests(id, geometry);
        if (requests.contains(abi::ATTR_STD_POINTINESS)) {
            residency.demand.ids.emplace(contract::cycles_pointiness_attribute_id);
        }
        project_named(residency.demand, geometry, requests);
        close_tangent_dependencies(residency.demand, geometry);
        measure_geometry(result, geometry, residency);
    }
    for (const auto &[id, geometry] : snapshot.curve_geometries) {
        auto &residency = result.geometries[id];
        const auto requests = geometry_requests(id, geometry);
        project_named(residency.demand, geometry, requests);
        measure_geometry(result, geometry, residency);
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
