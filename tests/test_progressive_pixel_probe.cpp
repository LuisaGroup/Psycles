#include <psycles/io/progressive_pixel_probe.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] bool close(double lhs, double rhs) noexcept {
    return std::abs(lhs - rhs) <= 1.0e-12;
}

[[nodiscard]] bool expect(
    bool condition,
    std::string_view message) {
    if (!condition) {
        std::cerr << "failure: " << message << '\n';
    }
    return condition;
}

[[nodiscard]] psycles::io::PixelPassCapture capture(
    std::string_view name,
    std::initializer_list<float> values) {
    return {
        .pass = {
            .kind = psycles::contract::PassKind::combined,
            .name = std::string{name},
            .channels =
                static_cast<std::uint32_t>(values.size())},
        .values = std::vector<float>{values}};
}

}// namespace

int main() {
    using namespace psycles;
    bool ok = true;

    io::PixelOutputSink sink{3u, 2u};
    contract::RenderSettings settings;
    sink.begin(settings);
    const std::array<float, 18u> pixels{
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f};
    sink.write({
        .pass = {
            .kind = contract::PassKind::combined,
            .name = "Combined",
            .channels = 3u},
        .window = {
            .x = 2u,
            .y = 1u,
            .width = 3u,
            .height = 2u},
        .full_extent = {.width = 8u, .height = 6u},
        .pixels = pixels});
    sink.end(false);
    ok &= expect(!sink.cancelled(), "valid tile was cancelled");
    ok &= expect(
        sink.passes().size() == 1u,
        "pass was not captured");
    if (sink.passes().size() == 1u) {
        const auto &values = sink.passes()[0u].values;
        ok &= expect(
            values == std::vector<float>({12.0f, 13.0f, 14.0f}),
            "absolute raster pixel selected the wrong tile element");
    }

    sink.begin(settings);
    sink.write({
        .pass = {
            .kind = contract::PassKind::combined,
            .name = "Combined",
            .channels = 3u},
        .window = {
            .x = 0u,
            .y = 0u,
            .width = 1u,
            .height = 1u},
        .full_extent = {.width = 8u, .height = 6u},
        .pixels = std::span<const float>{pixels}.first(3u)});
    sink.end(false);
    ok &= expect(
        sink.cancelled(),
        "tile not covering the requested pixel was accepted");

    io::ProgressivePixelAccumulator accumulator;
    const std::vector first{
        capture("Combined", {1.0f, 2.0f}),
        capture("Normal", {3.0f})};
    auto first_delta = accumulator.append(7u, 2u, first);
    ok &= expect(first_delta.has_value(), "first chunk was rejected");
    ok &= expect(
        accumulator.rendered_samples() == 2u,
        "first chunk sample count was not committed");
    if (first_delta) {
        ok &= expect(
            close(first_delta->passes[0u].values[0u], 2.0) &&
                close(first_delta->passes[0u].values[1u], 4.0) &&
                close(first_delta->passes[1u].values[0u], 6.0),
            "first scaled-output delta is incorrect");
    }

    const std::vector second{
        capture("Combined", {4.0f, 5.0f}),
        capture("Normal", {6.0f})};
    auto second_delta = accumulator.append(19u, 3u, second);
    ok &= expect(second_delta.has_value(), "second chunk was rejected");
    ok &= expect(
        accumulator.rendered_samples() == 5u,
        "second chunk sample count was not committed");
    if (second_delta) {
        ok &= expect(
            close(second_delta->passes[0u].values[0u], 18.0) &&
                close(second_delta->passes[0u].values[1u], 21.0) &&
                close(second_delta->passes[1u].values[0u], 24.0),
            "progressive scaled-output delta is incorrect");
    }

    auto invalid = second;
    invalid[1u].pass.name = "Changed layout";
    ok &= expect(
        !accumulator.append(22u, 1u, invalid),
        "changed pass layout was accepted");
    ok &= expect(
        accumulator.rendered_samples() == 5u,
        "invalid chunk mutated accumulated sample count");

    const std::vector third{
        capture("Combined", {5.0f, 6.0f}),
        capture("Normal", {7.0f})};
    auto third_delta = accumulator.append(23u, 1u, third);
    ok &= expect(
        third_delta.has_value() &&
            close(third_delta->passes[0u].values[0u], 10.0) &&
            close(third_delta->passes[0u].values[1u], 11.0) &&
            close(third_delta->passes[1u].values[0u], 12.0),
        "invalid chunk changed the following valid delta");

    accumulator.reset();
    ok &= expect(
        accumulator.rendered_samples() == 0u,
        "reset did not clear the sample count");
    ok &= expect(
        accumulator.append(0u, 1u, first).has_value(),
        "reset did not clear the pass layout");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
