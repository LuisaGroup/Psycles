#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <luisa/ast/callable_library.h>
#include <luisa/ast/function_builder.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

using namespace luisa::compute;

namespace {

struct CachePayloadSample {
    uint32_t acc;
    uint32_t tag;
    float weight;
    bool flag_a;
    bool flag_b;
};

}// namespace

LUISA_STRUCT(CachePayloadSample, acc, tag, weight, flag_a, flag_b) {};

namespace {

void set_env_var(const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct ScopedTempCacheDir {
    std::filesystem::path dir;

    ScopedTempCacheDir() {
        std::error_code ec;
        const auto pid =
#ifdef _WIN32
            static_cast<unsigned long>(_getpid());
#else
            static_cast<unsigned long>(::getpid());
#endif
        const auto stamp = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        dir = std::filesystem::temp_directory_path(ec) /
              ("psycles_coro_cache_test_" + std::to_string(pid) + "_" + std::to_string(stamp));
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        set_env_var("LUISA_CORO_COMPILE_CACHE_DIR", dir.string().c_str());
    }

    ~ScopedTempCacheDir() {
        std::error_code ec;
        set_env_var("LUISA_CORO_COMPILE_CACHE_DIR", "");
        set_env_var("LUISA_CORO_COMPILE_CACHE", "1");
        std::filesystem::remove_all(dir, ec);
    }

    void clear_entries() const {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }

    [[nodiscard]] std::vector<std::filesystem::path> list_lcc_files() const {
        std::vector<std::filesystem::path> files;
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) { return files; }
        for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".lcc") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }
};

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "FAIL [" << __LINE__ << "]: " << (msg) << "\n";     \
            return false;                                                     \
        }                                                                     \
    } while (false)

[[nodiscard]] luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> build_rich_source_coroutine(
    const Buffer<uint32_t> &in_buf,
    const Buffer<uint32_t> &out_buf,
    const Constant<float> &lut,
    uint32_t salt,
    const char *second_suspend_name,
    bool include_assert,
    bool alt_helper_behavior) {

    // 2-hop transitive custom callables (inner -> outer -> coroutine)
    Callable inner_mix = [alt_helper_behavior](UInt a, UInt b) noexcept {
        UInt r = (a ^ 0x9e3779b9u) + (b * (alt_helper_behavior ? 19u : 17u));
        return r ^ (r >> 13u);
    };

    Callable outer_step = [&inner_mix](UInt a, UInt b, UInt iter) noexcept {
        UInt mixed = inner_mix(a + iter, b ^ (iter * 31u));
        return (mixed << 5u) | (mixed >> 27u);
    };

    return luisa::compute::detail::FunctionBuilder::define_coroutine([&] {
        auto create = [&]() noexcept {
            Var<uint32_t> scale{luisa::compute::detail::FunctionBuilder::current()->argument(Type::of<uint32_t>())};
            UInt tid = dispatch_x();
            UInt seed = in_buf->read(tid) ^ salt;
            if (include_assert) {
                device_assert(scale > 0u, "scale_must_be_positive");
            }

            Var<CachePayloadSample> st;
            st.acc = outer_step(seed, scale, 0u);
            st.tag = (seed + tid) & 7u;
            st.weight = lut.read(st.tag & 3u);
            st.flag_a = (seed & 1u) != 0u;
            st.flag_b = (tid & 2u) != 0u;

            UInt sort_key = (st.acc ^ st.tag) & 63u;
            UInt aux_handle = seed + 3u;
            Bool aux_resident = st.flag_a;
            if (include_assert) {
                $suspend(
                    "stage_one",
                    coro_frame_export("export_key", sort_key),
                    coro_sort_by(sort_key, 64u),
                    coro_stage("psycles.test.aux_stage", 2u)
                        .fallback(CoroSuspendFallback::ignore)
                        .read("tag", st.tag)
                        .write("handle", aux_handle)
                        .read_write("resident", aux_resident)
                        .attribute("domain", "cache-test")
                        .attribute("priority", 7u));
            } else {
                $suspend(
                    "stage_one",
                    coro_frame_export("export_key", sort_key),
                    coro_sort_by(sort_key, 64u));
            }

            st.acc = outer_step(st.acc, aux_handle, 1u);
            $switch(st.tag & 3u) {
                $case(0u) {
                    st.acc += 11u;
                    st.flag_b = !st.flag_b;
                };
                $case(1u, 2u) {
                    st.acc ^= 0x55aau;
                };
                $default {
                    st.acc *= 3u;
                };
            };

            $for(k, 2u) {
                $if(st.flag_a) {
                    st.acc = outer_step(st.acc, cast<uint32_t>(st.weight * 8.0f), k + 1u);
                } $else {
                    st.acc += k + 7u;
                };
                st.flag_a = !st.flag_a;
            };

            $suspend(
                second_suspend_name,
                coro_annotation("psycles.test.watch")
                    .read("acc", st.acc)
                    .attribute("pass", "second"));

            UInt final_val = st.acc ^ ite(st.flag_b, 0x101u, 0x202u);
            final_val += cast<uint32_t>(lut.read((tid + 1u) & 3u) * 4.0f);
            out_buf->write(tid, final_val);
        };
        create();
    });
}

[[nodiscard]] uint32_t cpu_reference_eval(
    uint32_t in_val, uint32_t tid, uint32_t scale, uint32_t salt,
    const std::array<float, 4> &lut) noexcept {
    auto inner_mix = [](uint32_t a, uint32_t b) noexcept -> uint32_t {
        uint32_t r = (a ^ 0x9e3779b9u) + (b * 17u);
        return r ^ (r >> 13u);
    };
    auto outer_step = [&](uint32_t a, uint32_t b, uint32_t iter) noexcept -> uint32_t {
        uint32_t m = inner_mix(a + iter, b ^ (iter * 31u));
        return (m << 5u) | (m >> 27u);
    };
    uint32_t seed = in_val ^ salt;
    uint32_t acc = outer_step(seed, scale, 0u);
    uint32_t tag = (seed + tid) & 7u;
    float weight = lut[tag & 3u];
    bool flag_a = (seed & 1u) != 0u;
    bool flag_b = (tid & 2u) != 0u;
    uint32_t aux_handle = seed + 3u;
    acc = outer_step(acc, aux_handle, 1u);
    switch (tag & 3u) {
        case 0u:
            acc += 11u;
            flag_b = !flag_b;
            break;
        case 1u:
        case 2u:
            acc ^= 0x55aau;
            break;
        default:
            acc *= 3u;
            break;
    }
    for (uint32_t k = 0u; k < 2u; ++k) {
        if (flag_a) {
            acc = outer_step(acc, static_cast<uint32_t>(weight * 8.0f), k + 1u);
        } else {
            acc += k + 7u;
        }
        flag_a = !flag_a;
    }
    uint32_t final_val = acc ^ (flag_b ? 0x101u : 0x202u);
    final_val += static_cast<uint32_t>(lut[(tid + 1u) & 3u] * 4.0f);
    return final_val;
}

[[nodiscard]] bool verify_results_match(
    const luisa::compute::detail::CoroutineCompileResult &cold,
    const luisa::compute::detail::CoroutineCompileResult &warm) {

    CHECK(cold.boundary_verifier_count == 2u, "cold boundary verifier count must be 2");
    CHECK(cold.boundary_verifier_count == warm.boundary_verifier_count, "boundary verifier count mismatch");
    CHECK(cold.nested_pass_boundary_verifier_count == 0u, "cold nested verifier count must be 0");
    CHECK(cold.nested_pass_boundary_verifier_count == warm.nested_pass_boundary_verifier_count,
          "nested verifier count mismatch");

    CHECK(cold.trigger_tokens == warm.trigger_tokens, "trigger_tokens mismatch");
    CHECK(cold.subroutine_source_argument_indices == warm.subroutine_source_argument_indices,
          "subroutine_source_argument_indices mismatch");
    CHECK(cold.subroutines.size() == warm.subroutines.size(), "subroutine count mismatch");
    CHECK(!cold.subroutines.empty(), "empty subroutines");

    for (size_t i = 0u; i < cold.subroutines.size(); ++i) {
        auto &cs = cold.subroutines[i];
        auto &ws = warm.subroutines[i];
        CHECK(cs != nullptr && ws != nullptr, "null subroutine");
        CHECK(cs->hash() == ws->hash(), "subroutine hash mismatch");
        CHECK(cs->xir_cache_hash() == ws->xir_cache_hash(), "xir_cache_hash mismatch");
        CHECK(ws->arguments().size() == ws->bound_arguments().size(),
              "cached subroutine bound_arguments size mismatch");
        CHECK(cs->arguments().size() == ws->arguments().size(),
              "cached subroutine argument count mismatch");
        CHECK(ws->has_xir_cache(), "small cached subroutine should expose XIR cache provider");
    }

    auto const &cfd = cold.frame_desc;
    auto const &wfd = warm.frame_desc;
    CHECK(cfd.field_count() == wfd.field_count(), "frame_desc field_count mismatch");
    CHECK(cfd.total_size() == wfd.total_size(), "frame_desc total_size mismatch");
    CHECK(cfd.frame_field_count() == wfd.frame_field_count(), "frame_field_count mismatch");
    for (size_t i = 0u; i < cfd.frame_field_count(); ++i) {
        CHECK(cfd.frame_field_name(i) == wfd.frame_field_name(i), "frame_field_name mismatch");
        CHECK(cfd.frame_field_type(i) == wfd.frame_field_type(i), "frame_field_type mismatch");
        CHECK(cfd.frame_field_offset(i) == wfd.frame_field_offset(i), "frame_field_offset mismatch");
    }
    CHECK(cfd.field_index("export_key") == wfd.field_index("export_key"),
          "exported frame field alias mismatch");

    auto const &cg = cold.graph;
    auto const &wg = warm.graph;
    CHECK(cg.node_count() == wg.node_count(), "graph node_count mismatch");
    CHECK(cg.edge_count() == wg.edge_count(), "graph edge_count mismatch");
    CHECK(cg.boundary_count() == wg.boundary_count(), "graph boundary_count mismatch");
    CHECK(cg.call_graph().analysis_state_count == wg.call_graph().analysis_state_count,
          "call_graph analysis_state_count mismatch");
    CHECK(cg.call_graph().functions.size() == wg.call_graph().functions.size(),
          "call_graph functions size mismatch");
    CHECK(cg.call_graph().edges.size() == wg.call_graph().edges.size(),
          "call_graph edges size mismatch");
    CHECK(cg.dump() == wg.dump(), "graph.dump() mismatch");
    return true;
}

[[nodiscard]] bool run_all_tests(std::string_view backend, const char *exe_path) {
    ScopedTempCacheDir temp_cache;

    Context ctx{exe_path};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    constexpr uint32_t kCount = 128u;
    const std::array<float, 4> lut_vals{1.5f, 3.25f, 5.75f, 2.0f};
    std::vector<uint32_t> init_in(kCount);
    for (uint32_t i = 0u; i < kCount; ++i) {
        init_in[i] = (i * 2654435761u) ^ 0x13572468u;
    }

    auto in_buf_a = device.create_buffer<uint32_t>(kCount);
    auto out_buf_a = device.create_buffer<uint32_t>(kCount);
    Constant<float> lut_a{luisa::span<const float>{lut_vals.data(), lut_vals.size()}};

    // Hold a second set simultaneously so handles are strictly distinct from *_a!
    auto in_buf_b = device.create_buffer<uint32_t>(kCount);
    auto out_buf_b = device.create_buffer<uint32_t>(kCount);
    Constant<float> lut_b{luisa::span<const float>{lut_vals.data(), lut_vals.size()}};

    // -------------------------------------------------------------------------
    // Check 1: Cold vs Populate vs Warm Hit bit-exact parity (different handles)
    // -------------------------------------------------------------------------
    auto src_a = build_rich_source_coroutine(
        in_buf_a, out_buf_a, lut_a, 0x1234u, "stage_two", true, false);
    auto src_b = build_rich_source_coroutine(
        in_buf_b, out_buf_b, lut_b, 0x1234u, "stage_two", true, false);
    CHECK(src_a->hash() == src_b->hash(),
          "source coroutine hash should depend only on binding kinds, not buffer handles");

    set_env_var("LUISA_CORO_COMPILE_CACHE", "0");
    auto cold = luisa::compute::detail::compile_coroutine_pipeline(src_a);
    CHECK(temp_cache.list_lcc_files().empty(),
          "LUISA_CORO_COMPILE_CACHE=0 must not write any cache files");

    set_env_var("LUISA_CORO_COMPILE_CACHE", "1");
    auto populated = luisa::compute::detail::compile_coroutine_pipeline(src_a);
    CHECK(verify_results_match(cold, populated), "cold vs populated mismatch");
    auto files = temp_cache.list_lcc_files();
    CHECK(files.size() == 1u, "expected exactly 1 .lcc cache file after populate");

    auto warm = luisa::compute::detail::compile_coroutine_pipeline(src_b);
    CHECK(verify_results_match(cold, warm), "cold vs warm cache hit mismatch");
    CHECK(temp_cache.list_lcc_files().size() == 1u,
          "warm cache hit with different buffer handles must hit existing .lcc file");

    // -------------------------------------------------------------------------
    // Check 2: Lazy XIR cache provider works with ast_to_xir_translate + verifier
    // -------------------------------------------------------------------------
    {
        Coroutine<void(uint32_t)> cached_coro{[&](UInt scale) {
            UInt tid = dispatch_x();
            UInt s = in_buf_b->read(tid);
            device_assert(scale > 0u, "lazy_xir_assert");
            UInt tmp = s * scale + cast<uint32_t>(lut_b.read(tid & 3u));
            $suspend("xir_stage_1");
            tmp ^= 0x55u;
            $suspend("xir_stage_2");
            out_buf_b->write(tid, tmp);
        }};
        // Second construction hits disk cache so subroutines own LazyCoroXirCacheProvider!
        Coroutine<void(uint32_t)> hit_coro{[&](UInt scale) {
            UInt tid = dispatch_x();
            UInt s = in_buf_b->read(tid);
            device_assert(scale > 0u, "lazy_xir_assert");
            UInt tmp = s * scale + cast<uint32_t>(lut_b.read(tid & 3u));
            $suspend("xir_stage_1");
            tmp ^= 0x55u;
            $suspend("xir_stage_2");
            out_buf_b->write(tid, tmp);
        }};
        CHECK(hit_coro.subroutine_count() == 3u, "expected 3 subroutines for lazy XIR test");
        Kernel1D consumer = [&](UInt scale) {
            auto frame = hit_coro.instantiate(dispatch_id(), dispatch_size());
            hit_coro.entry()(frame, scale);
            hit_coro.subroutine("xir_stage_1")(frame, scale);
            hit_coro.subroutine("xir_stage_2")(frame, scale);
        };
        xir::AST2XIRTranslationStatistics stats;
        auto ir = xir::ast_to_xir_translate(
            consumer.function()->function(), {.statistics = &stats});
        CHECK(stats.cached_function_imports == hit_coro.subroutine_count(),
              "ast_to_xir_translate should lazily import all cached continuations via XIRCacheProvider");
        CHECK(xir::xir_verify_module(ir.get()).succeeded(),
              "lazily imported XIR module failed xir_verify_module");
    }

    // -------------------------------------------------------------------------
    // Check 3: Corruption & truncation resilience -> graceful fallback + rewrite
    // -------------------------------------------------------------------------
    {
        temp_cache.clear_entries();
        set_env_var("LUISA_CORO_COMPILE_CACHE", "1");
        auto base = luisa::compute::detail::compile_coroutine_pipeline(src_a);
        auto lcc = temp_cache.list_lcc_files();
        CHECK(lcc.size() == 1u, "expected 1 file before corruption test");

        // Corrupt a byte inside the AST/meta body near middle
        {
            std::fstream io{lcc.front(), std::ios::in | std::ios::out | std::ios::binary};
            CHECK(io.is_open(), "failed to open cache file for byte flip");
            io.seekg(0, std::ios::end);
            const auto sz = io.tellg();
            CHECK(sz > 128, "unexpectedly tiny cache file");
            const auto off = sz / 2;
            char ch = 0;
            io.seekg(off, std::ios::beg);
            io.read(&ch, 1);
            ch = static_cast<char>(ch ^ 0x5Au);
            io.seekp(off, std::ios::beg);
            io.write(&ch, 1);
            io.flush();
        }
        auto recovered_flip = luisa::compute::detail::compile_coroutine_pipeline(src_a);
        CHECK(verify_results_match(cold, recovered_flip),
              "corrupted cache file should fall back cleanly and rewrite");

        // Now truncate the file completely and re-test
        std::filesystem::resize_file(lcc.front(), 32u);
        auto recovered_trunc = luisa::compute::detail::compile_coroutine_pipeline(src_a);
        CHECK(verify_results_match(cold, recovered_trunc),
              "truncated cache file should fall back cleanly and rewrite");

        // Re-compiled valid file must now hit cleanly
        auto re_hit = luisa::compute::detail::compile_coroutine_pipeline(src_b);
        CHECK(verify_results_match(cold, re_hit),
              "repaired cache file failed subsequent warm hit");
    }

    // -------------------------------------------------------------------------
    // Check 4: Invalidation on AST / suspend / callee mutation
    // -------------------------------------------------------------------------
    {
        temp_cache.clear_entries();
        auto baseline = luisa::compute::detail::compile_coroutine_pipeline(src_a);
        CHECK(temp_cache.list_lcc_files().size() == 1u, "baseline entry missing");

        auto mut_suspend = build_rich_source_coroutine(
            in_buf_a, out_buf_a, lut_a, 0x1234u, "stage_two_renamed", true, false);
        auto res_mut_suspend = luisa::compute::detail::compile_coroutine_pipeline(mut_suspend);
        CHECK(temp_cache.list_lcc_files().size() == 2u,
              "suspend rename should produce a distinct cache entry");
        CHECK(baseline.graph.dump() != res_mut_suspend.graph.dump(),
              "mutated suspend name must alter CoroGraph");

        auto mut_callee = build_rich_source_coroutine(
            in_buf_a, out_buf_a, lut_a, 0x1234u, "stage_two", true, true);
        auto res_mut_callee = luisa::compute::detail::compile_coroutine_pipeline(mut_callee);
        CHECK(temp_cache.list_lcc_files().size() == 3u,
              "transitive callee body change should produce a distinct cache entry");
        bool any_hash_changed = false;
        if (baseline.subroutines.size() != res_mut_callee.subroutines.size()) {
            any_hash_changed = true;
        } else {
            for (size_t i = 0u; i < baseline.subroutines.size(); ++i) {
                if (baseline.subroutines[i]->hash() != res_mut_callee.subroutines[i]->hash()) {
                    any_hash_changed = true;
                }
            }
        }
        CHECK(any_hash_changed, "mutated transitive callee must change output subroutines");
    }

    // -------------------------------------------------------------------------
    // Check 5: Live GPU execution via WavefrontCoroScheduler with rebound handles
    // -------------------------------------------------------------------------
    {
        temp_cache.clear_entries();
        constexpr uint32_t kScale = 7u;
        constexpr uint32_t kSalt = 0x1234u;

        // Cold GPU run into buffer set A
        set_env_var("LUISA_CORO_COMPILE_CACHE", "0");
        auto gpu_coro_cold = Coroutine<void(uint32_t)>{[&](UInt scale) {
            Callable inner_mix = [](UInt a, UInt b) noexcept {
                UInt r = (a ^ 0x9e3779b9u) + (b * 17u);
                return r ^ (r >> 13u);
            };
            Callable outer_step = [&inner_mix](UInt a, UInt b, UInt iter) noexcept {
                UInt m = inner_mix(a + iter, b ^ (iter * 31u));
                return (m << 5u) | (m >> 27u);
            };
            UInt tid = dispatch_x();
            UInt seed = in_buf_a->read(tid) ^ kSalt;
            Var<CachePayloadSample> st;
            st.acc = outer_step(seed, scale, 0u);
            st.tag = (seed + tid) & 7u;
            st.weight = lut_a.read(st.tag & 3u);
            st.flag_a = (seed & 1u) != 0u;
            st.flag_b = (tid & 2u) != 0u;
            UInt sort_key = (st.acc ^ st.tag) & 63u;
            UInt aux_handle = seed + 3u;
            $suspend(
                "gpu_stage_1",
                coro_frame_export("export_key", sort_key),
                coro_sort_by(sort_key, 64u));
            st.acc = outer_step(st.acc, aux_handle, 1u);
            $switch(st.tag & 3u) {
                $case(0u) {
                    st.acc += 11u;
                    st.flag_b = !st.flag_b;
                };
                $case(1u, 2u) {
                    st.acc ^= 0x55aau;
                };
                $default {
                    st.acc *= 3u;
                };
            };
            $for(k, 2u) {
                $if(st.flag_a) {
                    st.acc = outer_step(st.acc, cast<uint32_t>(st.weight * 8.0f), k + 1u);
                } $else {
                    st.acc += k + 7u;
                };
                st.flag_a = !st.flag_a;
            };
            $suspend(
                "gpu_stage_2",
                coro_annotation("psycles.test.watch")
                    .read("acc", st.acc)
                    .attribute("pass", "second"));
            UInt final_val = st.acc ^ ite(st.flag_b, 0x101u, 0x202u);
            final_val += cast<uint32_t>(lut_a.read((tid + 1u) & 3u) * 4.0f);
            out_buf_a->write(tid, final_val);
        }};

        // Populate cache from set A, then hit cache with distinct buffer set B
        auto make_gpu_coro = [&](const Buffer<uint32_t> &ib,
                                 const Buffer<uint32_t> &ob,
                                 const Constant<float> &lt) {
            return Coroutine<void(uint32_t)>{[&](UInt scale) {
                Callable inner_mix = [](UInt a, UInt b) noexcept {
                    UInt r = (a ^ 0x9e3779b9u) + (b * 17u);
                    return r ^ (r >> 13u);
                };
                Callable outer_step = [&inner_mix](UInt a, UInt b, UInt iter) noexcept {
                    UInt m = inner_mix(a + iter, b ^ (iter * 31u));
                    return (m << 5u) | (m >> 27u);
                };
                UInt tid = dispatch_x();
                UInt seed = ib->read(tid) ^ kSalt;
                Var<CachePayloadSample> st;
                st.acc = outer_step(seed, scale, 0u);
                st.tag = (seed + tid) & 7u;
                st.weight = lt.read(st.tag & 3u);
                st.flag_a = (seed & 1u) != 0u;
                st.flag_b = (tid & 2u) != 0u;
                UInt sort_key = (st.acc ^ st.tag) & 63u;
                UInt aux_handle = seed + 3u;
                $suspend(
                    "gpu_stage_1",
                    coro_frame_export("export_key", sort_key),
                    coro_sort_by(sort_key, 64u));
                st.acc = outer_step(st.acc, aux_handle, 1u);
                $switch(st.tag & 3u) {
                    $case(0u) {
                        st.acc += 11u;
                        st.flag_b = !st.flag_b;
                    };
                    $case(1u, 2u) {
                        st.acc ^= 0x55aau;
                    };
                    $default {
                        st.acc *= 3u;
                    };
                };
                $for(k, 2u) {
                    $if(st.flag_a) {
                        st.acc = outer_step(st.acc, cast<uint32_t>(st.weight * 8.0f), k + 1u);
                    } $else {
                        st.acc += k + 7u;
                    };
                    st.flag_a = !st.flag_a;
                };
                $suspend(
                    "gpu_stage_2",
                    coro_annotation("psycles.test.watch")
                        .read("acc", st.acc)
                        .attribute("pass", "second"));
                UInt final_val = st.acc ^ ite(st.flag_b, 0x101u, 0x202u);
                final_val += cast<uint32_t>(lt.read((tid + 1u) & 3u) * 4.0f);
                ob->write(tid, final_val);
            }};
        };

        set_env_var("LUISA_CORO_COMPILE_CACHE", "1");
        auto gpu_coro_pop = make_gpu_coro(in_buf_a, out_buf_a, lut_a);
        CHECK(temp_cache.list_lcc_files().size() == 1u,
              "expected 1 .lcc file after populate GPU coro");
        auto gpu_coro_warm = make_gpu_coro(in_buf_b, out_buf_b, lut_b);
        CHECK(temp_cache.list_lcc_files().size() == 1u,
              "expected warm hit for GPU coro with buffer set B");

        coro::WavefrontCoroSchedulerConfig wf_cfg;
        wf_cfg.thread_count = 64u;
        wf_cfg.global_memory_soa = true;
        wf_cfg.largest_continuation_first = true;
        wf_cfg.incremental_continuation_counts = true;

        coro::WavefrontCoroScheduler<uint32_t> sched_cold{device, gpu_coro_cold, wf_cfg};
        coro::WavefrontCoroScheduler<uint32_t> sched_warm{device, gpu_coro_warm, wf_cfg};

        std::vector<uint32_t> out_cold(kCount, 0u);
        std::vector<uint32_t> out_warm(kCount, 0u);
        std::vector<uint32_t> zeros(kCount, 0u);

        stream << in_buf_a.copy_from(init_in.data())
               << out_buf_a.copy_from(zeros.data());
        stream << sched_cold(kScale).dispatch(kCount);
        stream << out_buf_a.copy_to(out_cold.data())
               << synchronize();

        stream << in_buf_b.copy_from(init_in.data())
               << out_buf_b.copy_from(zeros.data());
        stream << sched_warm(kScale).dispatch(kCount);
        stream << out_buf_b.copy_to(out_warm.data())
               << synchronize();

        for (uint32_t i = 0u; i < kCount; ++i) {
            const uint32_t ref = cpu_reference_eval(init_in[i], i, kScale, kSalt, lut_vals);
            CHECK(out_cold[i] == ref, "cold GPU output disagrees with CPU reference");
            CHECK(out_warm[i] == ref, "cached GPU output disagrees with CPU reference");
        }
    }

    return true;
}

}// namespace

int main(int argc, char **argv) {
    const std::string_view backend = argc > 1 ? argv[1] : "fallback";
    if (!run_all_tests(backend, argv[0])) {
        return EXIT_FAILURE;
    }
    std::cout << "PASS psycles_luisa_coro_compile_cache_tests (" << backend << ")\n";
    return EXIT_SUCCESS;
}
