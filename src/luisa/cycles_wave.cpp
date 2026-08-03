#include <psycles/luisa/cycles_wave.h>

#include <psycles/luisa/cycles_noise.h>

#include <luisa/dsl/func.h>

namespace psycles::luisa_backend::cycles_wave {
namespace {

using DistortionNoiseCallable =
    luisa::compute::Callable<
        float(luisa::float3, float, float)>;

[[nodiscard]] const DistortionNoiseCallable &
distortion_noise_callable() noexcept {
    static DistortionNoiseCallable callable{
        [](Float3 point,
           Float detail,
           Float roughness) noexcept {
            return cycles_noise::fbm(
                point,
                detail,
                roughness,
                2.0f,
                true);
        }};
    return callable;
}

}// namespace

void prepare_distortion_noise() noexcept {
    static_cast<void>(distortion_noise_callable());
}

Float distortion_noise(
    Float3 point,
    Float detail,
    Float roughness) noexcept {
    return distortion_noise_callable()(
        point,
        detail,
        roughness);
}

}// namespace psycles::luisa_backend::cycles_wave
