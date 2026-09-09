// Diagnostic-only observation of original Cycles scene-owned light fields.
// Include after scene/light.h, scene/object.h and scene/scene.h, then call
// after LightManager::device_update_lights has populated the original table.
// This does not construct, normalize or evaluate any light parameter.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

CCL_NAMESPACE_BEGIN
static void psycles_dump_light_parameters(const Scene *scene, const KernelLight *table)
{
  const char *path = std::getenv("PSYCLES_CYCLES_LIGHT_PARAMETERS_DUMP");
  if (!path || !*path) { return; }
  FILE *file = std::fopen(path, "wb");
  if (!file) { std::perror(path); std::abort(); }
  const auto scalar = [&](const char *key, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::fprintf(file, ",\"%s\":%u", key, bits);
  };
  const auto vector = [&](const char *key, float3 value) {
    uint32_t bits[3];
    std::memcpy(&bits[0], &value.x, 4);
    std::memcpy(&bits[1], &value.y, 4);
    std::memcpy(&bits[2], &value.z, 4);
    std::fprintf(file, ",\"%s\":[%u,%u,%u]", key, bits[0], bits[1], bits[2]);
  };
  std::fputs("{\"format\":\"Cycles 5.2 light parameter bits\",\"version\":1,\"lights\":[", file);
  int index = 0;
  bool first = true;
  for (const Object *object : scene->objects) {
    if (!object->get_geometry()->is_light()) { continue; }
    const Light *light = static_cast<const Light *>(object->get_geometry());
    if (!light->get_is_enabled() || light->is_portal_light()) { continue; }
    const KernelLight &k = table[index++];
    // The other union members are not defined for point/background lights.
    if (k.type != LIGHT_SPOT && k.type != LIGHT_AREA) { continue; }
    if (!first) { std::fputc(',', file); }
    first = false;
    std::fprintf(file, "{\"index\":%d,\"type\":%d,\"name_hex\":\"", index - 1, int(k.type));
    for (const unsigned char c : object->name.string()) { std::fprintf(file, "%02x", c); }
    std::fputs("\",\"input\":{\"normalize\":", file);
    std::fputs(light->get_normalize() ? "true" : "false", file);
    vector("axis_x", transform_get_column(&object->get_tfm(), 0));
    vector("axis_y", transform_get_column(&object->get_tfm(), 1));
    vector("axis_z", transform_get_column(&object->get_tfm(), 2));
    vector("position", transform_get_translation(&object->get_tfm()));
    if (k.type == LIGHT_AREA) {
      const AreaLight *area = static_cast<const AreaLight *>(light);
      scalar("spread", area->get_spread());
      scalar("size_u", area->get_sizeu());
      scalar("size_v", area->get_sizev());
      std::fprintf(file, ",\"ellipse\":%s},\"kernel\":{\"type\":%d", area->get_ellipse() ? "true" : "false", int(k.type));
      scalar("tan_half_spread", k.area.tan_half_spread);
      scalar("normalize_spread", k.area.normalize_spread);
      scalar("invarea", k.area.invarea);
      scalar("len_u", k.area.len_u);
      scalar("len_v", k.area.len_v);
      vector("axis_u", k.area.axis_u);
      vector("axis_v", k.area.axis_v);
      vector("dir", k.area.dir);
    } else {
      const SpotLight *spot = static_cast<const SpotLight *>(light);
      scalar("angle", spot->get_angle());
      scalar("smooth", spot->get_smooth());
      scalar("radius", spot->get_radius());
      std::fprintf(file, ",\"is_sphere\":%s},\"kernel\":{\"type\":%d", spot->get_is_sphere() ? "true" : "false", int(k.type));
      scalar("cos_half_spot_angle", k.spot.cos_half_spot_angle);
      scalar("half_cot_half_spot_angle", k.spot.half_cot_half_spot_angle);
      scalar("spot_smooth", k.spot.spot_smooth);
      scalar("cos_half_larger_spread", k.spot.cos_half_larger_spread);
      scalar("ray_segment_dp", k.spot.ray_segment_dp);
      scalar("eval_fac", k.spot.eval_fac);
      scalar("radius", k.spot.radius);
      std::fprintf(file, ",\"is_sphere\":%d", k.spot.is_sphere);
      vector("dir", k.spot.dir);
    }
    std::fputs("}}", file);
  }
  std::fputs("]}\n", file);
  if (std::ferror(file) || std::fclose(file)) { std::abort(); }
}
CCL_NAMESPACE_END
