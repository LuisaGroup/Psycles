#include <psycles/luisa/path_tracer.h>

#include <array>
#include <cstdint>
#include <cstdlib>

int main() {
    using psycles::luisa_backend::LuisaPathScheduler;
    using psycles::luisa_backend::LuisaPathTracerOptions;
    using psycles::luisa_backend::luisa_path_scheduler_name;
    using psycles::luisa_backend::parse_luisa_path_scheduler;

    constexpr std::array schedulers{
        LuisaPathScheduler::megakernel,
        LuisaPathScheduler::wavefront,
        LuisaPathScheduler::persistent};
    for (const auto scheduler : schedulers) {
        const auto name = luisa_path_scheduler_name(scheduler);
        if (name.empty() ||
            parse_luisa_path_scheduler(name) != scheduler) {
            return EXIT_FAILURE;
        }
    }
    if (parse_luisa_path_scheduler("state-machine") ||
        parse_luisa_path_scheduler("") ||
        !luisa_path_scheduler_name(
             static_cast<LuisaPathScheduler>(255u))
             .empty()) {
        return EXIT_FAILURE;
    }

    // Defaults reproduce the path-tracing experiment in GPU Coroutines:
    // wavefront 2^24; persistent 2^15, block 128, fetch 16, SoA + GME.
    constexpr LuisaPathTracerOptions options;
    static_assert(
        options.scheduler == LuisaPathScheduler::megakernel);
    static_assert(options.wavefront_frame_capacity == (1u << 24u));
    static_assert(options.persistent_worker_count == (1u << 15u));
    static_assert(options.persistent_block_size == 128u);
    static_assert(options.persistent_fetch_size == 16u);
    static_assert(options.persistent_shared_memory_soa);
    static_assert(options.persistent_global_memory_extension);
    return EXIT_SUCCESS;
}
