#include "path_tracer_vector_mapping.h"

#include "surface_vector_mapping.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] ShaderLocatedVectorMappingCallable
make_point_mapping_callable() noexcept {
    ShaderLocatedVectorMappingCallable mapping =
        [](Float3 input,
           Float3 location,
           Float3 rotation,
           Float3 scale) noexcept {
        return map_vector_point_inline(
            input, location, rotation, scale);
    };
    mapping.set_name("surface_mapping_point");
    return mapping;
}

[[nodiscard]] ShaderLocatedVectorMappingCallable
make_texture_mapping_callable() noexcept {
    ShaderLocatedVectorMappingCallable mapping =
        [](Float3 input,
           Float3 location,
           Float3 rotation,
           Float3 scale) noexcept {
        return map_vector_texture_inline(
            input, location, rotation, scale);
    };
    mapping.set_name("surface_mapping_texture");
    return mapping;
}

[[nodiscard]] ShaderLinearVectorMappingCallable
make_vector_mapping_callable() noexcept {
    ShaderLinearVectorMappingCallable mapping =
        [](Float3 input,
           Float3 rotation,
           Float3 scale) noexcept {
        return map_vector_direction_inline(
            input, rotation, scale);
    };
    mapping.set_name("surface_mapping_vector");
    return mapping;
}

[[nodiscard]] ShaderLinearVectorMappingCallable
make_normal_mapping_callable() noexcept {
    ShaderLinearVectorMappingCallable mapping =
        [](Float3 input,
           Float3 rotation,
           Float3 scale) noexcept {
        return map_vector_normal_inline(
            input, rotation, scale);
    };
    mapping.set_name("surface_mapping_normal");
    return mapping;
}

}// namespace

Float3 CallableSurfaceVectorMappingProvider::map_point(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) const noexcept {
    if (!_point) {
        _point.emplace(make_point_mapping_callable());
    }
    return (*_point)(input, location, rotation, scale);
}

Float3 CallableSurfaceVectorMappingProvider::map_texture(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) const noexcept {
    if (!_texture) {
        _texture.emplace(make_texture_mapping_callable());
    }
    return (*_texture)(input, location, rotation, scale);
}

Float3 CallableSurfaceVectorMappingProvider::map_vector(
    Float3 input,
    Float3 rotation,
    Float3 scale) const noexcept {
    if (!_vector) {
        _vector.emplace(make_vector_mapping_callable());
    }
    return (*_vector)(input, rotation, scale);
}

Float3 CallableSurfaceVectorMappingProvider::map_normal(
    Float3 input,
    Float3 rotation,
    Float3 scale) const noexcept {
    if (!_normal) {
        _normal.emplace(make_normal_mapping_callable());
    }
    return (*_normal)(input, rotation, scale);
}

}// namespace psycles::luisa_backend::detail
