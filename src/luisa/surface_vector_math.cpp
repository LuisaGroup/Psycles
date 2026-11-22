#include "surface_vector_math.h"

#include "surface_math.h"

#include <cstdlib>

#include <luisa/dsl/sugar.h>

#include <psycles/luisa/native_vector_math.h>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;

SurfaceVectorMathResult evaluate_surface_vector_math_operation(
    compiler::VectorMathOperation operation,
    Float3 a,
    Float3 b,
    Float3 c,
    Float scale) noexcept {
    const auto safe_divide = [](Float numerator, Float denominator) noexcept {
        const auto valid = denominator != 0.0f;
        return select(0.0f,
                      numerator / select(1.0f, denominator, valid),
                      valid);
    };
    const auto safe_divide_vector = [&](Float3 numerator,
                                        Float3 denominator) noexcept {
        return make_float3(safe_divide(numerator.x, denominator.x),
                           safe_divide(numerator.y, denominator.y),
                           safe_divide(numerator.z, denominator.z));
    };
    const auto safe_normalize_zero = [](Float3 input) noexcept {
        return native_vector_math::safe_normalize_nonzero(input);
    };
    const auto wrap_component = [](Float input,
                                   Float maximum,
                                   Float minimum) noexcept {
        const auto range = maximum - minimum;
        const auto valid = range != 0.0f;
        return select(minimum,
                      input - range *
                                  floor((input - minimum) /
                                        select(1.0f, range, valid)),
                      valid);
    };

    SurfaceVectorMathResult result{
        .value = 0.0f,
        .vector = make_float3(0.0f)};
    switch (operation) {
        case compiler::VectorMathOperation::add:
            result.vector = a + b;
            break;
        case compiler::VectorMathOperation::subtract:
            result.vector = a - b;
            break;
        case compiler::VectorMathOperation::multiply:
            result.vector = a * b;
            break;
        case compiler::VectorMathOperation::divide:
            result.vector = safe_divide_vector(a, b);
            break;
        case compiler::VectorMathOperation::multiply_add:
            result.vector = a * b + c;
            break;
        case compiler::VectorMathOperation::cross_product:
            result.vector = cross(a, b);
            break;
        case compiler::VectorMathOperation::project: {
            const auto length_squared = dot(b, b);
            const auto valid = length_squared != 0.0f;
            result.vector = select(make_float3(0.0f),
                                   safe_divide(dot(a, b), length_squared) * b,
                                   valid);
            break;
        }
        case compiler::VectorMathOperation::reflect: {
            const auto normal = safe_normalize_zero(b);
            result.vector = a - 2.0f * normal * dot(a, normal);
            break;
        }
        case compiler::VectorMathOperation::refract: {
            const auto normal = safe_normalize_zero(b);
            const auto cosine = dot(normal, a);
            const auto k =
                1.0f - scale * scale * (1.0f - cosine * cosine);
            result.vector = select(
                make_float3(0.0f),
                scale * a -
                    (scale * cosine + sqrt(max(k, 0.0f))) * normal,
                k >= 0.0f);
            break;
        }
        case compiler::VectorMathOperation::faceforward:
            result.vector = select(-a, a, dot(c, b) < 0.0f);
            break;
        case compiler::VectorMathOperation::dot_product:
            result.value = dot(a, b);
            break;
        case compiler::VectorMathOperation::distance: {
            const auto delta = a - b;
            result.value = sqrt(dot(delta, delta));
            break;
        }
        case compiler::VectorMathOperation::length:
            result.value = sqrt(dot(a, a));
            break;
        case compiler::VectorMathOperation::scale:
            result.vector = a * scale;
            break;
        case compiler::VectorMathOperation::normalize:
            result.vector = safe_normalize_zero(a);
            break;
        case compiler::VectorMathOperation::absolute:
            result.vector = abs(a);
            break;
        case compiler::VectorMathOperation::power:
            result.vector = make_float3(cycles_safe_power(a.x, b.x),
                                        cycles_safe_power(a.y, b.y),
                                        cycles_safe_power(a.z, b.z));
            break;
        case compiler::VectorMathOperation::sign: {
            const auto sign_component = [](Float input) noexcept {
                return select(select(1.0f, -1.0f, input < 0.0f),
                              0.0f,
                              input == 0.0f);
            };
            result.vector = make_float3(sign_component(a.x),
                                        sign_component(a.y),
                                        sign_component(a.z));
            break;
        }
        case compiler::VectorMathOperation::minimum:
            result.vector = min(a, b);
            break;
        case compiler::VectorMathOperation::maximum:
            result.vector = max(a, b);
            break;
        case compiler::VectorMathOperation::floor:
            result.vector = floor(a);
            break;
        case compiler::VectorMathOperation::ceil:
            result.vector = ceil(a);
            break;
        case compiler::VectorMathOperation::fraction:
            result.vector = a - floor(a);
            break;
        case compiler::VectorMathOperation::modulo:
            result.vector = make_float3(
                select(0.0f, fmod(a.x, b.x), b.x != 0.0f),
                select(0.0f, fmod(a.y, b.y), b.y != 0.0f),
                select(0.0f, fmod(a.z, b.z), b.z != 0.0f));
            break;
        case compiler::VectorMathOperation::wrap:
            result.vector = make_float3(wrap_component(a.x, b.x, c.x),
                                        wrap_component(a.y, b.y, c.y),
                                        wrap_component(a.z, b.z, c.z));
            break;
        case compiler::VectorMathOperation::snap:
            result.vector = floor(safe_divide_vector(a, b)) * b;
            break;
        case compiler::VectorMathOperation::sine:
            result.vector = make_float3(sin(a.x), sin(a.y), sin(a.z));
            break;
        case compiler::VectorMathOperation::cosine:
            result.vector = make_float3(cos(a.x), cos(a.y), cos(a.z));
            break;
        case compiler::VectorMathOperation::tangent:
            result.vector = make_float3(tan(a.x), tan(a.y), tan(a.z));
            break;
        case compiler::VectorMathOperation::round:
            result.vector = floor(a + 0.5f);
            break;
        case compiler::VectorMathOperation::cycles_normalize:
            // Cycles NormalNode uses normalize(), not safe_normalize(). Keep
            // the zero-vector result non-finite, but let the backend select
            // the native implementation rather than freezing sqrt/div.
            result.vector = native_vector_math::normalize_unchecked(a);
            break;
        default:
            std::abort();
    }
    return result;
}

} // namespace psycles::luisa_backend::detail
