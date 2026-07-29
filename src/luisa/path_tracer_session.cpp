#include "path_tracer_internal.h"
#include "sample_dispatch_partition.h"

namespace psycles::luisa_backend::detail {

std::size_t LuisaRenderSession::pixel_count() const noexcept {
    return static_cast<std::size_t>(_window.width) *
           static_cast<std::size_t>(_window.height);
}

void LuisaRenderSession::prepare_sobol_table(
    std::uint32_t total_samples) {
    const auto sequence_size =
        tabulated_sobol::sequence_size_for_samples(
            total_samples);
    if (!_sobol_table ||
        _sobol_sequence_size != sequence_size) {
        const auto generated =
            tabulated_sobol::generate_table(sequence_size);
        luisa::vector<luisa::float4> table;
        table.reserve(generated.size());
        for (const auto sample : generated) {
            table.emplace_back(luisa::make_float4(
                sample.x,
                sample.y,
                sample.z,
                sample.w));
        }
        _sobol_table =
            _scene->device.create_buffer<luisa::float4>(
                table.size());
        _stream
            << _sobol_table.copy_from(luisa::span{table})
            << synchronize();
        _sobol_sequence_size = sequence_size;
    }
    _kernel_parameters.sobol_sequence_size =
        sequence_size;
}

void LuisaRenderSession::write_passes(
    contract::OutputSink &output) {
    const auto count = pixel_count();
    luisa::vector<luisa::float4> combined(count);
    luisa::vector<luisa::float4> normal(count);
    luisa::vector<luisa::float4> albedo(count);
    luisa::vector<luisa::float4> light_passes(
        count * light_pass_buffer_count);
    luisa::vector<luisa::uint> samples(count);
    _stream << _combined.copy_to(luisa::span{combined})
            << _normal.copy_to(luisa::span{normal})
            << _albedo.copy_to(luisa::span{albedo})
            << _light_passes.copy_to(
                   luisa::span{light_passes})
            << _sample_count.copy_to(luisa::span{samples})
            << synchronize();

    output.begin(_settings);
    for (const auto &pass : _settings.passes) {
        if (!supported_pass(pass.kind)) {
            continue;
        }
        const auto channels = pass_channels(pass);
        std::vector<float> pixels(
            count * static_cast<std::size_t>(channels));
        for (std::size_t i = 0u; i < count; ++i) {
            const auto denominator =
                static_cast<float>(std::max(samples[i], 1u));
            const auto exposure =
                _settings.integrator.film_exposure;
            const auto light_pass_base =
                i * light_pass_buffer_count;
            const auto read_light_pass =
                [&](LightPassBuffer kind) noexcept
                -> const luisa::float4 & {
                return light_passes[
                    light_pass_base +
                    light_pass_index(kind)];
            };
            const auto divided_light_pass =
                [&](LightPassBuffer kind,
                    const luisa::float4 &color) noexcept {
                    const auto divided =
                        safe_divide_even_color(
                            read_light_pass(kind), color) *
                        exposure;
                    return luisa::make_float4(
                        divided, 1.0f);
                };
            luisa::float4 value{};
            switch (pass.kind) {
                case PassKind::combined:
                    value =
                        combined[i] *
                        (exposure / denominator);
                    break;
                case PassKind::normal:
                case PassKind::denoising_normal:
                    value = normal[i] / denominator;
                    break;
                case PassKind::albedo:
                case PassKind::denoising_albedo:
                    value = albedo[i] / denominator;
                    break;
                case PassKind::glossy_color:
                    value =
                        read_light_pass(
                            LightPassBuffer::glossy_color) /
                        denominator;
                    break;
                case PassKind::transmission_color:
                    value =
                        read_light_pass(
                            LightPassBuffer::
                                transmission_color) /
                        denominator;
                    break;
                case PassKind::emission:
                    value =
                        read_light_pass(
                            LightPassBuffer::emission) *
                        (exposure / denominator);
                    break;
                case PassKind::environment:
                    value =
                        read_light_pass(
                            LightPassBuffer::environment) *
                        (exposure / denominator);
                    break;
                case PassKind::diffuse_direct:
                    value = divided_light_pass(
                        LightPassBuffer::diffuse_direct,
                        albedo[i]);
                    break;
                case PassKind::diffuse_indirect:
                    value = divided_light_pass(
                        LightPassBuffer::diffuse_indirect,
                        albedo[i]);
                    break;
                case PassKind::glossy_direct:
                    value = divided_light_pass(
                        LightPassBuffer::glossy_direct,
                        read_light_pass(
                            LightPassBuffer::glossy_color));
                    break;
                case PassKind::glossy_indirect:
                    value = divided_light_pass(
                        LightPassBuffer::glossy_indirect,
                        read_light_pass(
                            LightPassBuffer::glossy_color));
                    break;
                case PassKind::transmission_direct:
                    value = divided_light_pass(
                        LightPassBuffer::
                            transmission_direct,
                        read_light_pass(
                            LightPassBuffer::
                                transmission_color));
                    break;
                case PassKind::transmission_indirect:
                    value = divided_light_pass(
                        LightPassBuffer::
                            transmission_indirect,
                        read_light_pass(
                            LightPassBuffer::
                                transmission_color));
                    break;
                case PassKind::sample_count:
                    value = luisa::make_float4(
                        static_cast<float>(samples[i]));
                    break;
                default:
                    break;
            }
            const std::array source{
                value.x, value.y, value.z, value.w};
            for (std::uint32_t channel = 0u;
                 channel < channels;
                 ++channel) {
                pixels[i * channels + channel] =
                    source[std::min<std::uint32_t>(
                        channel, 3u)];
            }
        }
        output.write(PassTile{
            .pass = pass,
            .window = _window,
            .full_extent = _settings.full_extent,
            .pixels = std::span<const float>{pixels}});
    }
    output.end(_cancelled.load());
}

LuisaRenderSession::LuisaRenderSession(
    std::shared_ptr<LuisaSceneData> scene,
    LuisaPathTracerOptions options,
    const RenderSettings &settings)
    : _scene{std::move(scene)},
      _options{options},
      _stream{_scene->device.create_stream()} {
    initialize(settings);
}

void LuisaRenderSession::reset(
    const RenderSettings &settings) {
    _cancelled.store(false);
    initialize(settings);
}

bool LuisaRenderSession::render_samples(
    const SampleRange &samples,
    contract::OutputSink &output) {
    if (_cancelled.load() || samples.count == 0u) {
        return false;
    }
    const auto sample_begin =
        static_cast<std::uint64_t>(samples.first) +
        static_cast<std::uint64_t>(samples.offset);
    const auto sample_end =
        sample_begin +
        static_cast<std::uint64_t>(samples.count);
    if (samples.total == 0u ||
        sample_end >
            static_cast<std::uint64_t>(samples.total) ||
        (_total_aa_samples != 0u &&
         _total_aa_samples != samples.total)) {
        return false;
    }
    _total_aa_samples = samples.total;
    prepare_sobol_table(samples.total);
    auto dispatches = SampleDispatchPartition::make(
        static_cast<std::uint32_t>(sample_begin),
        samples.count,
        _options.max_samples_per_dispatch);
    if (!dispatches) {
        return false;
    }
    const auto dispatch_size =
        static_cast<std::uint32_t>(
            std::max<std::size_t>(pixel_count(), 1u));
    while (const auto batch = dispatches->next()) {
        if (_cancelled.load()) {
            return false;
        }
        _stream
            << _render_shader(
                   _combined,
                   _normal,
                   _albedo,
                   _light_passes,
                   _sample_count,
                   batch->first,
                   batch->count,
                   _sobol_table,
                   _pixel_filter_table,
                   _kernel_parameters)
                   .dispatch(dispatch_size)
            << synchronize();
    }
    if (_cancelled.load()) {
        return false;
    }
    write_passes(output);
    return true;
}

void LuisaRenderSession::cancel() noexcept {
    _cancelled.store(true);
}

}// namespace psycles::luisa_backend::detail
