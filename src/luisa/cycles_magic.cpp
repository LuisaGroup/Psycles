#include <psycles/luisa/cycles_magic.h>

#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_magic {
namespace {

inline constexpr float two_pi = 6.28318530717958647692f;

using MagicCallable = luisa::compute::Callable<
    luisa::float3(luisa::uint, luisa::float3, float, float)>;

[[nodiscard]] const MagicCallable &magic_callable() noexcept {
    static MagicCallable callable{
        [](UInt requested_depth,
           Float3 point,
           Float scale,
           Float distortion) noexcept {
            const auto depth = min(
                requested_depth, maximum_depth);
            // Exact Cycles svm_magic() ordering. The fmod reduction is
            // semantically required: it prevents large coordinates from
            // losing the smaller dimensions before the trigonometric
            // recurrence.
            const auto px = fmod(point.x * scale, two_pi);
            const auto py = fmod(point.y * scale, two_pi);
            const auto pz = fmod(point.z * scale, two_pi);

            Float x = sin((px + py + pz) * 5.0f);
            Float y = cos((-px + py - pz) * 5.0f);
            Float z = -cos((-px - py + pz) * 5.0f);

            $for (iteration, depth) {
                $switch (iteration) {
                    $case (0u) {
                        x *= distortion;
                        y *= distortion;
                        z *= distortion;
                        y = -cos(x - y + z);
                        y *= distortion;
                    };
                    $case (1u) {
                        x = cos(x - y - z);
                        x *= distortion;
                    };
                    $case (2u) {
                        z = sin(-x - y - z);
                        z *= distortion;
                    };
                    $case (3u) {
                        x = -cos(-x + y - z);
                        x *= distortion;
                    };
                    $case (4u) {
                        y = -sin(-x + y + z);
                        y *= distortion;
                    };
                    $case (5u) {
                        y = -cos(-x + y + z);
                        y *= distortion;
                    };
                    $case (6u) {
                        x = cos(x + y + z);
                        x *= distortion;
                    };
                    $case (7u) {
                        z = sin(x + y - z);
                        z *= distortion;
                    };
                    $case (8u) {
                        x = -cos(-x - y + z);
                        x *= distortion;
                    };
                    $case (9u) {
                        y = -sin(x - y + z);
                        y *= distortion;
                    };
                };
            };

            $if (distortion != 0.0f) {
                const auto denominator = distortion * 2.0f;
                x /= denominator;
                y /= denominator;
                z /= denominator;
            };
            return make_float3(
                0.5f - x,
                0.5f - y,
                0.5f - z);
        }};
    return callable;
}

}// namespace

void prepare() noexcept {
    static_cast<void>(magic_callable());
}

Float3 evaluate(
    UInt depth,
    Float3 vector,
    Float scale,
    Float distortion) noexcept {
    return magic_callable()(depth, vector, scale, distortion);
}

}// namespace psycles::luisa_backend::cycles_magic
