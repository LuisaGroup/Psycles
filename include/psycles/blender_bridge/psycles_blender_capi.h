#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#  if defined(PSYCLES_BLENDER_BUILD)
#    define PSYCLES_BLENDER_API __declspec(dllexport)
#  else
#    define PSYCLES_BLENDER_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define PSYCLES_BLENDER_API __attribute__((visibility("default")))
#else
#  define PSYCLES_BLENDER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

constexpr std::uint32_t PSYCLES_CAPI_VERSION = 1u;

constexpr std::uint32_t PSYCLES_SCHEDULER_AUTO = 0xFFFFFFFFu;
constexpr std::uint32_t PSYCLES_SCHEDULER_MEGAKERNEL = 0u;
constexpr std::uint32_t PSYCLES_SCHEDULER_MEGAKERNEL_PER_SAMPLE = 1u;
constexpr std::uint32_t PSYCLES_SCHEDULER_WAVEFRONT = 2u;
constexpr std::uint32_t PSYCLES_SCHEDULER_WAVEFRONT_GRAPH = 3u;
constexpr std::uint32_t PSYCLES_SCHEDULER_WAVEFRONT_STAGED = 4u;
constexpr std::uint32_t PSYCLES_SCHEDULER_PERSISTENT = 5u;

constexpr std::uint32_t PSYCLES_PROJECTION_PERSPECTIVE = 0u;
constexpr std::uint32_t PSYCLES_PROJECTION_ORTHOGRAPHIC = 1u;
constexpr std::uint32_t PSYCLES_PROJECTION_PANORAMA = 2u;

constexpr std::uint32_t PSYCLES_SENSOR_FIT_AUTO = 0u;
constexpr std::uint32_t PSYCLES_SENSOR_FIT_HORIZONTAL = 1u;
constexpr std::uint32_t PSYCLES_SENSOR_FIT_VERTICAL = 2u;

constexpr std::uint8_t PSYCLES_PIXEL_FILTER_BOX = 0u;
constexpr std::uint8_t PSYCLES_PIXEL_FILTER_GAUSSIAN = 1u;
constexpr std::uint8_t PSYCLES_PIXEL_FILTER_BLACKMAN_HARRIS = 2u;
constexpr std::uint8_t PSYCLES_PIXEL_FILTER_FROM_SCENE = 255u;

constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_IDLE = 0u;
constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_COMPILING = 1u;
constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_RENDERING = 2u;
constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_DONE = 3u;
constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_CANCELLED = 4u;
constexpr std::uint32_t PSYCLES_VIEWPORT_STATE_FAILED = 5u;

struct PsyclesEngine;
struct PsyclesScene;
struct PsyclesSession;
struct PsyclesViewportTask;

struct PsyclesViewportStatus {
    std::uint32_t state;
    std::uint32_t rendered_samples;
    std::uint32_t target_samples;
    std::uint32_t width;
    std::uint32_t height;
    std::uint64_t frame_generation;
    std::uint8_t is_finished;
    std::uint8_t has_compiled_scene;
    std::uint8_t has_session;
    std::uint8_t reserved;
};

struct PsyclesEngineConfig {
    const char *cache_dir;
    std::uint32_t scheduler;
    std::uint32_t wavefront_frame_capacity;
    std::uint32_t wavefront_execution_block_size;
    std::uint32_t max_samples_per_dispatch;
    std::uint8_t enable_fast_math;
    std::uint8_t staged_surface_sorting;
    std::uint8_t staged_direct_light_queue;
    std::uint8_t vulkan_require_native_xir;
    std::int32_t hip_visible_device;
};

struct PsyclesSceneInfo {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t samples;
    std::uint32_t seed;
    std::uint32_t geometry_count;
    std::uint32_t instance_count;
    std::uint32_t material_count;
    std::uint32_t light_count;
    std::uint8_t transparent_background;
};

struct PsyclesPassDesc {
    std::uint16_t kind;
    std::uint16_t reserved;
    std::uint32_t channels;
    const char *name;
    const char *light_group;
};

struct PsyclesCameraDesc {
    std::uint32_t projection;
    std::uint32_t sensor_fit;
    float transform[16];
    float field_of_view;
    float horizontal_field_of_view;
    float orthographic_scale;
    float lens_shift_x;
    float lens_shift_y;
    float near_clip;
    float far_clip;
    float aperture_radius;
    float focal_distance;
    std::uint32_t aperture_blades;
    float aperture_rotation;
    float aperture_ratio;
    std::uint8_t has_sensor;
    std::uint8_t reserved[3];
    float sensor_width_mm;
    float sensor_height_mm;
    float sensor_lens_mm;
    float pixel_aspect_x;
    float pixel_aspect_y;
};

struct PsyclesRenderParams {
    std::uint32_t full_width;
    std::uint32_t full_height;
    std::uint32_t window_x;
    std::uint32_t window_y;
    std::uint32_t window_width;
    std::uint32_t window_height;
    std::uint32_t seed;
    std::uint8_t override_seed;
    std::uint8_t transparent_background;
    std::uint8_t override_transparent_background;
    std::uint8_t pixel_filter;
    float filter_width;
    float pass_alpha_threshold;
    std::uint8_t override_pass_alpha_threshold;
    std::uint8_t reserved[3];
};

PSYCLES_BLENDER_API std::uint32_t psycles_capi_version();

PSYCLES_BLENDER_API std::uint32_t psycles_probe_backends(
    const char *runtime_dir,
    char *out_csv,
    std::size_t out_cap);

PSYCLES_BLENDER_API PsyclesEngine *psycles_engine_create(
    const char *runtime_dir,
    const char *backend_name,
    const PsyclesEngineConfig *cfg,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API void psycles_engine_destroy(PsyclesEngine *engine);

PSYCLES_BLENDER_API PsyclesScene *psycles_scene_compile_memory(
    PsyclesEngine *engine,
    const char *scene_json,
    std::size_t json_len,
    const void *geometry_bin,
    std::size_t geom_len,
    const char *asset_base_dir,
    PsyclesSceneInfo *out_info,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API void psycles_scene_destroy(PsyclesScene *scene);

PSYCLES_BLENDER_API PsyclesSession *psycles_session_create(
    PsyclesEngine *engine,
    PsyclesScene *scene,
    const PsyclesRenderParams *params,
    const PsyclesPassDesc *passes,
    std::size_t pass_count,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API int psycles_session_update_camera(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API int psycles_session_reconfigure(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    const PsyclesPassDesc *passes,
    std::size_t pass_count,
    char *err_buf,
    std::size_t err_cap);

// Returns 1 on success, 0 if cancelled, -1 on failure.
PSYCLES_BLENDER_API int psycles_session_render_samples(
    PsyclesSession *session,
    std::uint32_t sample_first,
    std::uint32_t sample_count,
    std::uint32_t total_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API int psycles_session_get_pass_buffer(
    const PsyclesSession *session,
    std::size_t pass_index,
    const float **out_ptr,
    std::uint32_t *out_width,
    std::uint32_t *out_height,
    std::uint32_t *out_channels);

PSYCLES_BLENDER_API int psycles_session_get_last_timings(
    const PsyclesSession *session,
    double *out_integration_s,
    double *out_pass_output_s,
    double *out_render_call_s);

PSYCLES_BLENDER_API void psycles_session_cancel(PsyclesSession *session);

PSYCLES_BLENDER_API void psycles_session_destroy(PsyclesSession *session);

PSYCLES_BLENDER_API PsyclesViewportTask *psycles_viewport_task_start_compile(
    PsyclesEngine *engine,
    const char *scene_json,
    std::size_t json_len,
    const void *geometry_bin,
    std::size_t geom_len,
    const char *asset_base_dir,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API PsyclesViewportTask *psycles_viewport_task_start_scene(
    PsyclesEngine *engine,
    PsyclesScene *scene,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap);

PSYCLES_BLENDER_API PsyclesViewportTask *psycles_viewport_task_start_session(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap);

// Non-blocking: returns 1 if queued onto active worker, 0 if task finished/stopping, -1 on error.
PSYCLES_BLENDER_API int psycles_viewport_task_request_camera(
    PsyclesViewportTask *task,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples);

PSYCLES_BLENDER_API void psycles_viewport_task_cancel(PsyclesViewportTask *task);

// Non-blocking snapshot of worker state/progress/error.
PSYCLES_BLENDER_API int psycles_viewport_task_poll(
    const PsyclesViewportTask *task,
    PsyclesViewportStatus *out_status,
    char *err_buf,
    std::size_t err_cap);

// Non-blocking copy of latest Combined RGBA float buffer.
// Returns 1 if copied, 0 if no frame yet, -1 if out_float_cap is too small.
PSYCLES_BLENDER_API int psycles_viewport_task_copy_frame(
    const PsyclesViewportTask *task,
    float *out_rgba,
    std::size_t out_float_cap,
    std::uint32_t *out_width,
    std::uint32_t *out_height,
    std::uint64_t *out_generation);

// Returns a new owning PsyclesScene/PsyclesSession handle once the task has finished.
PSYCLES_BLENDER_API PsyclesScene *psycles_viewport_task_acquire_scene(
    PsyclesViewportTask *task);

PSYCLES_BLENDER_API PsyclesSession *psycles_viewport_task_acquire_session(
    PsyclesViewportTask *task);

PSYCLES_BLENDER_API void psycles_viewport_task_wait(PsyclesViewportTask *task);

PSYCLES_BLENDER_API void psycles_viewport_task_destroy(PsyclesViewportTask *task);

PSYCLES_BLENDER_API void psycles_shutdown_runtime();

#ifdef __cplusplus
}
#endif
