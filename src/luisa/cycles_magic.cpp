#include <psycles/luisa/cycles_magic.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>

#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_magic {
namespace {

inline constexpr float two_pi = 6.28318530717958647692f;

using MagicCallable = luisa::compute::Callable<
    luisa::float3(luisa::float3, float, float)>;

[[nodiscard]] auto &callables() noexcept {
    static std::array<
        std::unique_ptr<MagicCallable>,
        maximum_depth + 1u>
        values;
    return values;
}

[[nodiscard]] auto &callable_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] const MagicCallable &magic_callable(
    std::uint32_t requested_depth) noexcept {
    const auto depth = std::min(requested_depth, maximum_depth);
    std::lock_guard lock{callable_mutex()};
    auto &callable = callables()[depth];
    if (!callable) {
        callable = std::make_unique<MagicCallable>(
            [depth](Float3 point,
                    Float scale,
                    Float distortion) noexcept {
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

                if (depth > 0u) {
                    x *= distortion;
                    y *= distortion;
                    z *= distortion;
                    y = -cos(x - y + z);
                    y *= distortion;
                }
                if (depth > 1u) {
                    x = cos(x - y - z);
                    x *= distortion;
                }
                if (depth > 2u) {
                    z = sin(-x - y - z);
                    z *= distortion;
                }
                if (depth > 3u) {
                    x = -cos(-x + y - z);
                    x *= distortion;
                }
                if (depth > 4u) {
                    y = -sin(-x + y + z);
                    y *= distortion;
                }
                if (depth > 5u) {
                    y = -cos(-x + y + z);
                    y *= distortion;
                }
                if (depth > 6u) {
                    x = cos(x + y + z);
                    x *= distortion;
                }
                if (depth > 7u) {
                    z = sin(x + y - z);
                    z *= distortion;
                }
                if (depth > 8u) {
                    x = -cos(-x - y + z);
                    x *= distortion;
                }
                if (depth > 9u) {
                    y = -sin(x - y + z);
                    y *= distortion;
                }

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
            });
    }
    return *callable;
}

}// namespace

void prepare(std::uint32_t depth) noexcept {
    static_cast<void>(magic_callable(depth));
}

Float3 evaluate(
    std::uint32_t depth,
    Float3 vector,
    Float scale,
    Float distortion) noexcept {
    return magic_callable(depth)(vector, scale, distortion);
}

}// namespace psycles::luisa_backend::cycles_magic
