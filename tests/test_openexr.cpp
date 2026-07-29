#include <psycles/io/image.h>

#include <OpenImageIO/imageio.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}// namespace

int main() {
    using psycles::contract::PassKind;
    using psycles::io::PassImage;

    const std::vector images{
        PassImage{
            .pass = {
                .kind = PassKind::combined,
                .name = "Combined",
                .channels = 4u},
            .extent = {.width = 2u, .height = 2u},
            .channels = 4u,
            .pixels = {
                1.0f, 2.0f, 3.0f, 0.25f,
                4.0f, 5.0f, 6.0f, 0.50f,
                7.0f, 8.0f, 9.0f, 0.75f,
                10.0f, 11.0f, 12.0f, 1.0f}},
        PassImage{
            .pass = {
                .kind = PassKind::normal,
                .name = "Normal",
                .channels = 3u},
            .extent = {.width = 2u, .height = 2u},
            .channels = 3u,
            .pixels = {
                -1.0f, 0.0f, 1.0f,
                -0.5f, 0.5f, 1.0f,
                0.25f, 0.50f, 0.75f,
                1.0f, 0.0f, -1.0f}}};

    const auto nonce =
        std::chrono::steady_clock::now()
            .time_since_epoch()
            .count();
    const auto path =
        std::filesystem::temp_directory_path() /
        ("psycles-openexr-" + std::to_string(nonce) + ".exr");
    std::string error;
    require(
        psycles::io::write_multilayer_exr(
            images, path, "ViewLayer", &error),
        error);

    auto input = OIIO::ImageInput::open(path.string());
    require(
        static_cast<bool>(input),
        "OpenImageIO could not reopen the generated EXR");
    const auto specification = input->spec();
    require(
        specification.width == 2 &&
            specification.height == 2 &&
            specification.nchannels == 7,
        "generated EXR shape does not match the pass set");
    const std::vector<std::string> expected_names{
        "ViewLayer.Combined.R",
        "ViewLayer.Combined.G",
        "ViewLayer.Combined.B",
        "ViewLayer.Combined.A",
        "ViewLayer.Normal.X",
        "ViewLayer.Normal.Y",
        "ViewLayer.Normal.Z"};
    require(
        specification.channelnames == expected_names,
        "generated EXR channel names are not Cycles-compatible");
    require(
        specification.get_string_attribute(
            "oiio:ColorSpace") == "lin_rec709_scene",
        "generated EXR does not identify Cycles' scene-linear Rec.709 space");
    require(
        specification.get_string_attribute(
            "colorInteropID") == "lin_rec709_scene",
        "OpenEXR color interoperability metadata does not match Cycles");

    std::vector<float> actual(2u * 2u * 7u);
    require(
        input->read_image(
            0,
            0,
            0,
            specification.nchannels,
            OIIO::TypeDesc::FLOAT,
            actual.data()),
        input->geterror());
    require(input->close(), "could not close generated EXR");

    const std::vector<float> expected{
        1.0f, 2.0f, 3.0f, 0.25f, -1.0f, 0.0f, 1.0f,
        4.0f, 5.0f, 6.0f, 0.50f, -0.5f, 0.5f, 1.0f,
        7.0f, 8.0f, 9.0f, 0.75f, 0.25f, 0.50f, 0.75f,
        10.0f, 11.0f, 12.0f, 1.0f, 1.0f, 0.0f, -1.0f};
    require(actual.size() == expected.size(), "EXR readback size mismatch");
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(
            std::abs(actual[index] - expected[index]) <= 1.0e-7f,
            "EXR readback changed a float pass value");
    }

    std::error_code removal_error;
    std::filesystem::remove(path, removal_error);
    require(!removal_error, "could not remove the EXR regression artifact");

    std::cout << "Cycles-compatible multilayer OpenEXR test passed.\n";
    return EXIT_SUCCESS;
}
