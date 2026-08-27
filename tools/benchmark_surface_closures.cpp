#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_sampling.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::contract;
using namespace psycles::luisa_backend;

class BenchmarkShaderServices final : public ShaderServices {

  private:
    const BufferFloat4 &_input;

  public:
    explicit BenchmarkShaderServices(const BufferFloat4 &input) noexcept
        : _input{input} {}

    [[nodiscard]] Float4 texture_2d(Expr<std::uint32_t>,
                                    Expr<luisa::float2>,
                                    Expr<luisa::float2>,
                                    Expr<luisa::float2>,
                                    std::uint32_t,
                                    std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<luisa::ulong>, const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return _input.read(dispatch_x()).x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return _input.read(dispatch_x()).xyz();
    }

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t>, Expr<std::uint32_t>) const noexcept override {
        return _input.read(dispatch_x()).xy().bitcast<luisa::ulong>();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 0.73f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>,
                                     std::uint32_t,
                                     Expr<luisa::float3>,
                                     Expr<float>,
                                     Expr<float>,
                                     Expr<float>,
                                     Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

struct ProbeSpec {
    std::string_view name;
    SurfaceClosureKind kind;
    SurfaceClosureLobe lobe;
    SurfaceClosureReachability reachability;
    bool anisotropic;
    bool beckmann;
    bool singular;
};

[[nodiscard]] constexpr SurfaceClosureReachability kind_reachability(
    SurfaceClosureKind kind,
    SurfaceClosureLobe lobe = SurfaceClosureLobe::none,
    bool anisotropic = false) noexcept {
    const auto kind_bit = surface_closure_kind_bit(kind);
    return {
        .kinds = kind_bit,
        .principled_lobes =
            kind == SurfaceClosureKind::principled
                ? surface_closure_lobe_bit(lobe)
                : 0u,
        .anisotropic_microfacet_kinds = anisotropic ? kind_bit : 0u};
}

constexpr std::array probe_specs{
    ProbeSpec{"diffuse", SurfaceClosureKind::diffuse,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::diffuse), false, false,
              false},
    ProbeSpec{"oren_nayar", SurfaceClosureKind::diffuse,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::diffuse), false, false,
              false},
    ProbeSpec{"translucent", SurfaceClosureKind::translucent,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::translucent), false, false,
              false},
    ProbeSpec{"rough_translucent", SurfaceClosureKind::rough_translucent,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::rough_translucent), false,
              false, false},
    ProbeSpec{"transparent", SurfaceClosureKind::transparent,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::transparent), false, false,
              false},
    ProbeSpec{"glossy_ggx_regular", SurfaceClosureKind::glossy,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glossy), false, false,
              false},
    ProbeSpec{"glossy_ggx_singular", SurfaceClosureKind::glossy,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glossy), false, false,
              true},
    ProbeSpec{"glossy_ggx_anisotropic", SurfaceClosureKind::glossy,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glossy,
                                SurfaceClosureLobe::none, true),
              true, false, false},
    ProbeSpec{"glossy_beckmann", SurfaceClosureKind::glossy,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glossy), false, true,
              false},
    ProbeSpec{"glossy_beckmann_singular", SurfaceClosureKind::glossy,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glossy), false, true,
              true},
    ProbeSpec{"glass_ggx", SurfaceClosureKind::glass,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glass), false, false,
              false},
    ProbeSpec{"glass_ggx_singular", SurfaceClosureKind::glass,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::glass), false, false,
              true},
    ProbeSpec{"refraction_ggx", SurfaceClosureKind::refraction,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::refraction), false, false,
              false},
    ProbeSpec{"refraction_ggx_singular", SurfaceClosureKind::refraction,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::refraction), false, false,
              true},
    ProbeSpec{"principled_sheen", SurfaceClosureKind::principled,
              SurfaceClosureLobe::sheen,
              kind_reachability(SurfaceClosureKind::principled,
                                SurfaceClosureLobe::sheen),
              false, false, false},
    ProbeSpec{"principled_metallic", SurfaceClosureKind::principled,
              SurfaceClosureLobe::metallic,
              kind_reachability(SurfaceClosureKind::principled,
                                SurfaceClosureLobe::metallic, true),
              true, false, false},
    ProbeSpec{"principled_dielectric", SurfaceClosureKind::principled,
              SurfaceClosureLobe::dielectric,
              kind_reachability(SurfaceClosureKind::principled,
                                SurfaceClosureLobe::dielectric, true),
              true, false, false},
    ProbeSpec{"thin_glass", SurfaceClosureKind::thin_glass_transmission,
              SurfaceClosureLobe::transmission,
              kind_reachability(SurfaceClosureKind::thin_glass_transmission),
              false, false, false},
    ProbeSpec{"thin_glass_singular",
              SurfaceClosureKind::thin_glass_transmission,
              SurfaceClosureLobe::transmission,
              kind_reachability(SurfaceClosureKind::thin_glass_transmission),
              false, false, true},
    ProbeSpec{"bssrdf", SurfaceClosureKind::bssrdf,
              SurfaceClosureLobe::none,
              kind_reachability(SurfaceClosureKind::bssrdf), false, false,
              false}};

[[nodiscard]] const ProbeSpec &find_probe(std::string_view name) {
    for (const auto &probe : probe_specs) {
        if (probe.name == name) {
            return probe;
        }
    }
    throw std::invalid_argument{"unknown closure probe"};
}

[[nodiscard]] auto make_probe_kernel(const ProbeSpec &spec) {
    return Kernel1D{[spec](BufferFloat4 input,
                           BufferFloat4 output,
                           UInt iterations) noexcept {
        set_block_size(256u);
        const auto seed = input.read(dispatch_x());
        const auto normal = normalize(make_float3(
            seed.x * 0.25f, seed.y * 0.25f, 1.0f));
        const auto incoming = normalize(make_float3(
            0.13f + seed.x * 0.1f,
            -0.21f + seed.y * 0.1f,
            0.968f));
        const SurfaceClosurePoint point{
            normal,
            normal,
            incoming,
            0xffffffffu,
            false,
            false,
            false};
        BenchmarkShaderServices services{input};

        auto closure_record = SurfaceClosureRecord::zero();
        closure_record.kind = static_cast<std::uint32_t>(spec.kind);
        closure_record.lobe = static_cast<std::uint32_t>(spec.lobe);
        // Unit closure and categorical weights make the output exactly the
        // selected-closure (eval, pdf) contract returned by Cycles' bsdf_sample.
        // Real per-lane geometry, roughness, and random values remain dynamic.
        closure_record.weight = make_float3(1.0f);
        closure_record.allocation_weight = 1.0f;
        closure_record.sample_weight = 1.0f;
        closure_record.setup_valid = true;
        closure_record.color = make_float3(0.8f, 0.6f, 0.4f);
        // ThinGlassComponent::setup stores the transmission closure on the
        // opposite side of the sheet. Reproduce that physical-record
        // invariant instead of feeding the handler an impossible orientation.
        closure_record.normal =
            spec.kind == SurfaceClosureKind::thin_glass_transmission
                ? -normal
                : normal;
        if (spec.name == "diffuse") {
            closure_record.roughness = 0.0f;
        } else if (spec.singular) {
            // alpha_x * alpha_y is at most 1.6e-11, strictly below the
            // Cycles/Psycles 2e-10 singular threshold for every lane.
            closure_record.roughness =
                0.001f + 0.001f * abs(seed.z);
        } else {
            // alpha_x * alpha_y is at least 2.56e-3, strictly regular for
            // every lane and far from the delta-classification boundary.
            closure_record.roughness =
                0.2f + 0.6f * abs(seed.z);
        }
        const auto alpha = closure_record.roughness * closure_record.roughness;
        closure_record.microfacet_alpha_x = alpha;
        closure_record.microfacet_alpha_y =
            spec.anisotropic ? alpha * 0.47f : alpha;
        closure_record.microfacet_tangent =
            normalize(make_float3(1.0f, 0.23f, 0.0f));
        closure_record.diffuse_roughness = clamp(abs(seed.w), 0.0f, 1.0f);
        closure_record.metallic =
            spec.lobe == SurfaceClosureLobe::metallic ? 1.0f : 0.0f;
        closure_record.ior = 1.45f;
        closure_record.specular_tint = make_float3(0.9f, 0.8f, 0.7f);
        closure_record.sheen_transform_a = 0.83f;
        closure_record.sheen_transform_b = 0.21f;
        closure_record.evaluation_scale = make_float3(1.0f);
        closure_record.fresnel_f0 = make_float3(0.04f);
        closure_record.fresnel_f90 = make_float3(1.0f);
        closure_record.reflection_tint = make_float3(1.0f);
        closure_record.transmission_tint = make_float3(0.8f, 0.9f, 1.0f);
        closure_record.preserve_ggx_energy = false;
        closure_record.beckmann = spec.beckmann;
        closure_record.bssrdf_method = static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk);
        closure_record.bssrdf_radius = make_float3(1.0f, 0.6f, 0.3f);
        closure_record.bssrdf_albedo = make_float3(0.7f, 0.4f, 0.2f);
        closure_record.bssrdf_ior = 1.4f;
        closure_record.bssrdf_roughness = 0.35f;
        closure_record.bssrdf_anisotropy = 0.1f;
        const auto closure =
            static_cast<SurfaceClosurePhysicalRecord>(closure_record);

        constexpr auto all_lobes = static_cast<std::uint32_t>(
            event_diffuse | event_glossy | event_transmission |
            event_transparent);
        const SurfaceQuery query{
            .lobe_mask = all_lobes,
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto policy = make_surface_closure_evaluation_policy(
            false, Expr<std::uint32_t>{0u});

        Float2 random = clamp(seed.zw(), make_float2(0.001f),
                              make_float2(0.999f));
        Float4 accumulated = make_float4(0.0f);
        UInt iteration = 0u;
        $while(iteration < iterations) {
            const auto sample = surface_closure_conditional_sample(
                services,
                point,
                normal,
                closure,
                incoming,
                normal,
                random,
                random.x,
                query,
                spec.reachability);
            const auto contribution = surface_closure_evaluation_contribution(
                services,
                point,
                normal,
                closure,
                incoming,
                sample.direction,
                query,
                policy,
                false,
                spec.reachability);
            using namespace surface_closure_sample_property;
            const auto transparent_sample =
                (sample.properties & transparent) != 0u;
            const auto singular_sample =
                (sample.properties & singular) != 0u;
            const auto delta_sample = transparent_sample | singular_sample;
            const auto selected_evaluation = select(
                contribution.f,
                select(sample.singular_evaluation,
                       make_float3(1.0e6f),
                       transparent_sample),
                delta_sample);
            const auto selected_pdf = select(
                contribution.weighted_pdf,
                select(sample.singular_pdf, 1.0e6f, transparent_sample),
                delta_sample);
            accumulated += make_float4(
                select(0.0f, selected_evaluation.x, sample.valid),
                sample.direction.x + sample.direction.y,
                select(0.0f, selected_pdf, sample.valid),
                cast<float>(contribution.events | sample.properties));
            random += make_float2(0.61803398875f, 0.41421356237f);
            random = select(random, random - 1.0f, random >= 1.0f);
            iteration += 1u;
        };
        output.write(dispatch_x(), accumulated);
    }};
}

[[nodiscard]] std::uint32_t parse_u32(const char *text,
                                      std::string_view label) {
    char *end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > UINT32_MAX) {
        throw std::invalid_argument{std::string{label} + " must be uint32"};
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: " << argv[0]
                  << " BACKEND PROBE [COUNT [ITERATIONS [REPEATS]]]\nprobes:";
        for (const auto &probe : probe_specs) {
            std::cerr << ' ' << probe.name;
        }
        std::cerr << '\n';
        return EXIT_FAILURE;
    }
    try {
        const auto backend = std::string_view{argv[1]};
        const auto &probe = find_probe(argv[2]);
        const auto count = argc > 3 ? parse_u32(argv[3], "count") : 1u << 22u;
        const auto iterations =
            argc > 4 ? parse_u32(argv[4], "iterations") : 32u;
        const auto repeats = argc > 5 ? parse_u32(argv[5], "repeats") : 5u;
        if (count == 0u || iterations == 0u || repeats == 0u) {
            throw std::invalid_argument{"count, iterations, and repeats must be positive"};
        }

        Context context{argv[0]};
        auto device = context.create_device(backend);
        auto stream = device.create_stream();
        auto input = device.create_buffer<luisa::float4>(count);
        auto output = device.create_buffer<luisa::float4>(count);
        Kernel1D initialize = [](BufferFloat4 values) noexcept {
            set_block_size(256u);
            const auto lane = cast<float>(dispatch_x() & 255u) *
                              (1.0f / 256.0f);
            values.write(dispatch_x(),
                         make_float4(0.13f,
                                     -0.17f,
                                     0.001f + 0.997f * lane,
                                     0.999f - 0.997f * lane));
        };
        auto initialize_shader = device.compile(initialize);
        auto shader = device.compile(make_probe_kernel(probe));
        constexpr auto warmup_count = 3u;
        stream << initialize_shader(input).dispatch(count);
        for (auto warmup = 0u; warmup < warmup_count; ++warmup) {
            stream << shader(input, output, iterations).dispatch(count);
        }
        stream << synchronize();

        for (auto repeat = std::uint32_t{0u}; repeat < repeats; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            stream << shader(input, output, iterations).dispatch(count)
                   << synchronize();
            const auto end = std::chrono::steady_clock::now();
            const auto milliseconds =
                std::chrono::duration<double, std::milli>(end - begin).count();
            const auto ns_per_handler =
                milliseconds * 1.0e6 / static_cast<double>(count) /
                static_cast<double>(iterations);
            std::cout << "repeat=" << repeat
                      << " milliseconds=" << milliseconds
                      << " ns_per_handler=" << ns_per_handler << '\n';
        }
        std::array<luisa::float4, 1u> checksum{};
        stream << output.view(0u, 1u).copy_to(std::span{checksum})
               << synchronize();
        std::cout << "checksum=" << checksum[0u].x << ',' << checksum[0u].y
                  << ',' << checksum[0u].z << ',' << checksum[0u].w
                  << " count=" << count << " iterations=" << iterations
                  << " warmups=" << warmup_count
                  << " contract=selected_closure_sample"
                  << " probe=" << probe.name << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
