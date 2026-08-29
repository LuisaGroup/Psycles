#include "path_tracer_tangent_space.h"

#include "path_tracer_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

// Blender's standalone headers normally inherit these branch-hint macros and
// the uint alias from Blender's platform layer. Keep the vendored files
// byte-identical and provide their tiny integration surface here instead.
#define LIKELY(value) (value)
#define UNLIKELY(value) (value)
using uint = unsigned int;
#include <mikktspace.hh>
#undef UNLIKELY
#undef LIKELY

namespace psycles::luisa_backend::detail {
namespace {

struct UvView {
    std::span<const luisa::float2> vectors;
    std::span<const luisa::float4> attributes;
    std::uint32_t domain{};
    bool available{};

    [[nodiscard]] luisa::float2 read(
        std::size_t corner,
        std::size_t vertex) const noexcept {
        const auto index =
            domain == attribute_domain_corner ? corner : vertex;
        if (!vectors.empty() && index < vectors.size()) {
            return vectors[index];
        }
        if (!attributes.empty() && index < attributes.size()) {
            return luisa::make_float2(
                attributes[index].x,
                attributes[index].y);
        }
        return luisa::make_float2(0.0f);
    }
};

[[nodiscard]] luisa::float3 subtract(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z);
}

[[nodiscard]] luisa::float3 cross_product(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return luisa::make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

[[nodiscard]] luisa::float3 normalize_or_zero(
    luisa::float3 value) noexcept {
    const auto magnitude_squared =
        value.x * value.x +
        value.y * value.y +
        value.z * value.z;
    if (!(magnitude_squared > 0.0f) ||
        !std::isfinite(magnitude_squared)) {
        return luisa::make_float3(0.0f);
    }
    const auto inverse_magnitude =
        1.0f / std::sqrt(magnitude_squared);
    return luisa::make_float3(
        value.x * inverse_magnitude,
        value.y * inverse_magnitude,
        value.z * inverse_magnitude);
}

[[nodiscard]] luisa::float2 map_to_sphere(
    luisa::float3 position) noexcept {
    constexpr auto inverse_two_pi =
        0.15915494309189533577f;
    constexpr auto inverse_pi =
        0.31830988618379067154f;
    const auto length_squared =
        position.x * position.x +
        position.y * position.y +
        position.z * position.z;
    if (!(length_squared > 0.0f)) {
        return luisa::make_float2(0.0f);
    }
    const auto u =
        position.x == 0.0f && position.y == 0.0f
            ? 0.0f
            : 0.5f -
                  std::atan2(position.x, position.y) *
                      inverse_two_pi;
    const auto cosine = std::clamp(
        position.z / std::sqrt(length_squared),
        -1.0f,
        1.0f);
    const auto v = 1.0f - std::acos(cosine) * inverse_pi;
    return luisa::make_float2(u, v);
}

class CyclesMikkMesh {

private:
    const GeometryUpload &_upload;
    UvView _uv;
    luisa::vector<luisa::float4> &_tangents;

    [[nodiscard]] std::size_t corner_index(
        int face,
        int vertex) const noexcept {
        return static_cast<std::size_t>(face) * 3u +
               static_cast<std::size_t>(vertex);
    }

    [[nodiscard]] std::uint32_t vertex_index(
        int face,
        int vertex) const noexcept {
        const auto &triangle =
            _upload.triangles[static_cast<std::size_t>(face)];
        switch (vertex) {
            case 0:
                return triangle.i0;
            case 1:
                return triangle.i1;
            default:
                return triangle.i2;
        }
    }

public:
    CyclesMikkMesh(
        const GeometryUpload &upload,
        UvView uv,
        luisa::vector<luisa::float4> &tangents) noexcept
        : _upload{upload},
          _uv{uv},
          _tangents{tangents} {
        _tangents.assign(
            _upload.triangles.size() * 3u,
            luisa::make_float4(0.0f));
    }

    [[nodiscard]] int GetNumFaces() const noexcept {
        return static_cast<int>(_upload.triangles.size());
    }

    [[nodiscard]] int GetNumVerticesOfFace(
        int /* face */) const noexcept {
        return 3;
    }

    [[nodiscard]] mikk::float3 GetPosition(
        int face,
        int vertex) const noexcept {
        const auto value =
            _upload.positions[vertex_index(face, vertex)];
        return {value.x, value.y, value.z};
    }

    [[nodiscard]] mikk::float3 GetTexCoord(
        int face,
        int vertex) const noexcept {
        const auto point = vertex_index(face, vertex);
        const auto value = _uv.available
                               ? _uv.read(
                                     corner_index(face, vertex),
                                     point)
                               : map_to_sphere(
                                     _upload.positions[point]);
        return {value.x, value.y, 1.0f};
    }

    [[nodiscard]] mikk::float3 GetNormal(
        int face,
        int vertex) const noexcept {
        const auto face_index =
            static_cast<std::size_t>(face);
        const auto smooth =
            face_index < _upload.triangle_smooth.size() &&
            _upload.triangle_smooth[face_index] != 0u;
        luisa::float3 value;
        if (smooth) {
            const auto index =
                (_upload.attribute_domains &
                 geometry_normal_corner) != 0u
                    ? corner_index(face, vertex)
                    : static_cast<std::size_t>(
                          vertex_index(face, vertex));
            value = index < _upload.normals.size()
                        ? _upload.normals[index]
                        : luisa::make_float3(0.0f);
        } else {
            const auto &triangle =
                _upload.triangles[face_index];
            value = normalize_or_zero(cross_product(
                subtract(
                    _upload.positions[triangle.i1],
                    _upload.positions[triangle.i0]),
                subtract(
                    _upload.positions[triangle.i2],
                    _upload.positions[triangle.i0])));
        }
        return {value.x, value.y, value.z};
    }

    void SetTangentSpace(
        int face,
        int vertex,
        mikk::float3 tangent,
        bool orientation) noexcept {
        _tangents[corner_index(face, vertex)] =
            luisa::make_float4(
                tangent.x,
                tangent.y,
                tangent.z,
                orientation ? 1.0f : -1.0f);
    }

    [[nodiscard]] bool has_uv() const noexcept {
        return _uv.available;
    }
};

void compute_tangents(
    const GeometryUpload &upload,
    UvView uv,
    luisa::vector<luisa::float4> &output) {
    CyclesMikkMesh mesh{upload, uv, output};
    mikk::Mikktspace{mesh}.genTangSpace();
}

}// namespace

void recompute_cycles_tangent_space(GeometryUpload &upload) {
    compute_tangents(
        upload,
        UvView{
            .vectors = {
                upload.uv.data(),
                upload.uv.size()},
            .domain =
                (upload.attribute_domains & geometry_uv_corner) != 0u
                    ? attribute_domain_corner
                    : attribute_domain_point,
            .available = upload.default_uv_available},
        upload.uv_tangents);
    upload.attribute_domains |= geometry_uv_tangent_corner;

    for (const auto &layer : upload.uv_tangent_layers) {
        if (layer.uv_attribute_index >= upload.attributes.size() ||
            !layer.tangent_attribute_index ||
            *layer.tangent_attribute_index >= upload.attributes.size()) {
            continue;
        }
        const auto &uv =
            upload.attributes[layer.uv_attribute_index];
        auto &tangent =
            upload.attributes[*layer.tangent_attribute_index];
        compute_tangents(
            upload,
            UvView{
                .attributes = {
                    uv.values.data(),
                    uv.values.size()},
                .domain = uv.domain,
                .available = true},
            tangent.values);
        tangent.domain = attribute_domain_corner;
    }
}

void initialize_cycles_undisplaced_tangent_space(
    GeometryUpload &upload) {
    for (const auto &layer : upload.uv_tangent_layers) {
        if (!layer.undisplaced_tangent_attribute_index ||
            layer.uv_attribute_index >= upload.attributes.size() ||
            *layer.undisplaced_tangent_attribute_index >=
                upload.attributes.size()) {
            continue;
        }
        auto &undisplaced = upload.attributes[
            *layer.undisplaced_tangent_attribute_index];
        if (layer.tangent_attribute_index &&
            *layer.tangent_attribute_index < upload.attributes.size()) {
            const auto &current =
                upload.attributes[*layer.tangent_attribute_index];
            undisplaced.domain = current.domain;
            undisplaced.values = current.values;
            continue;
        }
        const auto &uv = upload.attributes[layer.uv_attribute_index];
        compute_tangents(
            upload,
            UvView{
                .attributes = {
                    uv.values.data(),
                    uv.values.size()},
                .domain = uv.domain,
                .available = true},
            undisplaced.values);
        undisplaced.domain = attribute_domain_corner;
    }
}

}// namespace psycles::luisa_backend::detail
