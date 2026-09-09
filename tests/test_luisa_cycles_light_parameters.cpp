#include "cycles_light_parameters_fixture.h"
#include "path_tracer_analytic_light_scene.h"
#include <psycles/luisa/area_light_sampling.h>
#include <psycles/luisa/volume_analytic_light_sampling.h>
#include <psycles/luisa/volume_light_interval.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace f = psycles::test_support::light_parameters;
namespace sampling = analytic_light_sampling;
namespace contract = psycles::contract;

auto prepare_scene(const std::vector<f::Record> &records) {
  contract::SceneSnapshot snapshot;
  LuisaSceneData scene;
  scene.cycles_svm = std::make_unique<CyclesSvmRuntime>();
  for (unsigned i = 0; i < records.size(); ++i) {
    const auto &r = records[i]; const auto &v = r.v;
    contract::LightDesc light;
    light.type = r.spot ? contract::LightType::spot : contract::LightType::area;
    light.power = 1;
    light.color = {1,1,1};
    for (unsigned column = 0; column < 4; ++column) {
      for (unsigned row = 0; row < 3; ++row) {
        light.transform.elements[column*4+row] = v[column*3+row];
      }
    }
    light.spread = light.spot_angle = v[12];
    light.spot_smooth = v[13]; light.size = v[14]; light.size_y = v[15];
    light.normalize = r.normalize; light.ellipse = r.ellipse; light.is_sphere = r.sphere;
    const contract::LightId id{i+1u};
    snapshot.lights.emplace(id, std::move(light));
    scene.cycles_svm->object_identities.light_indices.emplace(id, i);
  }
  auto upload = AnalyticLightSceneComponent{}.build(snapshot, scene);
  if (!upload.ok() || upload.regular_count != records.size() || upload.portal_count != 0) {
    throw std::runtime_error{"native parameter scene packing failed: " + upload.diagnostic};
  }
  return upload;
}

bool matches(float a, float e, float tolerance = 2.0e-4f) {
  if (std::isnan(e)) { return std::isnan(a); }
  if (std::isinf(e)) { return a == e; }
  return std::isfinite(a) && std::abs(a-e) <= tolerance * std::max(1.0f, std::abs(e));
}

unsigned check_parameters(const auto &upload, const std::vector<f::Record> &records) {
  unsigned failures = 0;
  for (unsigned i = 0; i < records.size(); ++i) {
    const auto &r = records[i]; const auto &light = upload.device_lights[i];
    const auto &s = light.spot;
    const std::array fields = r.spot ?
        std::array{s.cos_half_spot_angle, s.half_cot_half_spot_angle, s.spot_smooth,
                   s.cos_half_larger_spread, s.ray_segment_dp} :
        std::array{light.area.tan_half_spread, light.area.normalize_spread, 0.0f, 0.0f, 0.0f};
    for (unsigned j = 0; j < (r.spot ? 5u : 2u); ++j) {
      const auto expected = r.v[16+j];
      // Keep positive/zero and finite/invalid classifications exact, including
      // the original full-pi subnormal. Arithmetic is not a last-bit gate.
      const auto a = fields[j];
      if (std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(expected)) {
        std::cout << "host arithmetic light=" << i << " field=" << j
                  << " actual_bits=" << std::bit_cast<std::uint32_t>(a)
                  << " native_bits=" << std::bit_cast<std::uint32_t>(expected) << '\n';
      }
      const auto relative_match = matches(a, expected, 0.0f) ||
          (std::isfinite(a) && std::isfinite(expected) &&
           std::abs(a-expected) <= 8e-6f * std::abs(expected));
      if (!relative_match || ((a > 0) != (expected > 0)) || ((a == 0) != (expected == 0))) {
        std::cerr << "parameter light=" << i << " field=" << j << " actual=" << a
                  << " native=" << expected << '\n';
        ++failures;
      }
    }
  }
  return failures;
}

auto read_state(unsigned count, bool flush_denormals) {
  std::ifstream file{flush_denormals ? PSYCLES_LIGHT_PARAMETER_STATE_FTZ : PSYCLES_LIGHT_PARAMETER_STATE};
  std::vector<float> result(count*f::outputs);
  for (unsigned i = 0; i < count; ++i) {
    unsigned ordinal;
    if (!(file >> ordinal) || ordinal != i) { throw std::runtime_error{"invalid native GPU ordinal"}; }
    for (unsigned j = 0; j < f::outputs; ++j) {
      std::uint32_t bits;
      if (!(file >> bits)) { throw std::runtime_error{"truncated native GPU output"}; }
      result[i*f::outputs+j] = std::bit_cast<float>(bits);
    }
  }
  std::string trailing;
  if (file >> trailing) { throw std::runtime_error{"trailing native GPU output"}; }
  return result;
}

void save_sample(const BufferFloat &output, UInt base, const sampling::FiniteLightSample &s) {
  output.write(base, cast<float>(s.valid));
  $if(s.valid) {
    output.write(base+1u, s.evaluation_factor); output.write(base+2u, s.conditional_pdf);
    output.write(base+3u, s.distance);
    for (unsigned i = 0; i < 3; ++i) {
      output.write(base+4u+i, s.direction[i]); output.write(base+7u+i, s.position[i]);
      output.write(base+10u+i, s.normal[i]);
    }
    output.write(base+13u, s.uv.x); output.write(base+14u, s.uv.y);
  };
}

bool flushes_denormals(Device &device, Stream &stream) {
  // Match the oracle's arithmetic environment, not the light function's
  // result. Vulkan's default denormal mode is device-dependent. Normal,
  // signed and zero controls must still agree; only subnormal handling may
  // select the separately captured original-GPU FTZ fixture.
  const std::array bits{0u, 0x80000000u, 0x3f800000u, 0xbf800000u,
                        0x00800000u, 0x007fffffu, 0x00200000u};
  auto buffer = device.create_buffer<unsigned>(bits.size());
  auto out = device.create_buffer<unsigned>(bits.size());
  Kernel1D classify = [](BufferUInt input, BufferUInt result) {
    result.write(dispatch_x(), cast<unsigned>(input.read(dispatch_x()).as<float>() > 0.0f));
  };
  const auto shader = device.compile(classify, ShaderOption{.enable_cache=false,.enable_fast_math=true});
  std::array<unsigned, bits.size()> result{};
  stream << buffer.copy_from(bits.data()) << shader(buffer, out).dispatch(bits.size())
         << out.copy_to(result.data()) << synchronize();
  if (result[0] != 0 || result[1] != 0 || result[2] != 1 || result[3] != 0 ||
      result[4] != 1 || result[5] > 1 || result[5] != result[6]) {
    throw std::runtime_error{"invalid floating-point environment controls"};
  }
  return result[5] == 0;
}
} // namespace

int main(int argc, char **argv) {
  const auto records = f::read(PSYCLES_LIGHT_PARAMETERS);
  auto upload = prepare_scene(records);
  auto failures = check_parameters(upload, records);
  // Test the device consumer on exactly the same scene-owned inputs as the
  // original GPU probe. Host preparation above is a separate, tolerance-based
  // gate: a one-ULP cosf difference must not move a deliberately exact cone
  // boundary in the device-function regression. Full-scene canaries exercise
  // production preparation and consumption together.
  for (unsigned i = 0; i < records.size(); ++i) {
    const auto &r = records[i]; auto &light = upload.device_lights[i];
    if (r.spot) { light.spot = {r.v[16],r.v[17],r.v[18],r.v[19],r.v[20]}; }
    else { light.area = {r.v[16],r.v[17]}; }
  }
  const auto count = unsigned(records.size()) * f::probes;
  std::vector<luisa::float4> inputs(count*5u);
  for (unsigned i = 0; i < records.size(); ++i) {
    for (unsigned j = 0; j < f::probes; ++j) {
      const auto base = (i*f::probes+j)*5u;
      const auto p = f::input(records[i],j);
      const auto vec = [](const auto &v) { return luisa::make_float4(v[0],v[1],v[2],0.0f); };
      inputs[base] = vec(p.offset); inputs[base+1] = {p.random[0],p.random[1],0,0};
      inputs[base+2] = vec(p.local_ray); inputs[base+3] = vec(p.normal);
      inputs[base+4] = vec(p.interval_direction);
    }
  }
  Context context{argv[0]};
  auto device = context.create_device(argc > 1 ? argv[1] : "hip");
  auto stream = device.create_stream();
  const auto flush_denormals = flushes_denormals(device, stream);
  const auto expected = read_state(count, flush_denormals);
  std::cout << "Original GPU oracle mode: " << (flush_denormals ? "FTZ" : "preserve") << '\n';
  auto lights = device.create_buffer<LightGpu>(records.size());
  auto rays = device.create_buffer<luisa::float4>(inputs.size());
  auto output = device.create_buffer<float>(count*f::outputs);
  Kernel1D kernel = [](Var<Buffer<LightGpu>> lights, BufferFloat4 inputs, BufferFloat out) {
    const auto index = dispatch_x(), base = index*f::outputs;
    const auto light = lights.read(index/f::probes);
    const auto P = light.position + inputs.read(index*5u).xyz();
    const auto random = inputs.read(index*5u+1u).xy();
    const auto local_ray = inputs.read(index*5u+2u).xyz();
    const auto N = inputs.read(index*5u+3u).xyz();
    const auto interval_direction = inputs.read(index*5u+4u).xyz();
    const auto sphere = (light.flags & light_flag_sphere) != 0u;
    const auto normalize = (light.flags & light_flag_normalize) != 0u;
    for (unsigned j = 0; j < f::outputs; ++j) { out.write(base+j, 0.0f); }
    VolumeLightIntervalResult interval{{0.0f,10.0f},false};
    $if(light.type == unsigned(contract::LightType::spot)) {
      out.write(base, sampling::spot_light_attenuation(local_ray, light.spot));
      const auto uv = sampling::spot_light_uv(local_ray, light.spot.half_cot_half_spot_angle);
      out.write(base+1u, uv.x); out.write(base+2u, uv.y);
      out.write(base+3u, 1.0f-light.spot.cos_half_larger_spread);
      const auto s = sampling::sample_spot_light(P, N, false, light.position, light.radius,
          sphere, light.axis_x, light.axis_y, light.axis_z, light.axis_scale, light.spot, random, normalize);
      save_sample(out, base+4u, s);
      const VolumeSpotLightSampleInput v{{P,light.position,light.radius,sphere,light.axis_x,
          light.axis_y,light.axis_z,light.axis_scale,random,normalize},light.spot};
      save_sample(out, base+20u, VolumeAnalyticLightSampling{}.spot_from_segment(v));
      interval = VolumeLightInterval{}.spot({P,interval_direction,{0.0f,10.0f},light.position,light.axis_x,
          light.axis_y,light.axis_z,light.axis_scale,light.spot});
    } $else {
      const auto spread = light.area.normalize_spread > 0.0f;
      Float value = 1.0f;
      $if(spread) { value = sampling::area_spread_attenuation(-local_ray,make_float3(0.0f,0.0f,1.0f),light.area); };
      out.write(base, value); out.write(base+3u, cast<float>(spread));
      const auto ellipse = (light.flags & light_flag_ellipse) != 0u;
      const AreaLightSampleInput a{P,light.position,light.axis_x,light.axis_y,light.axis_z,
          light.size_u,light.size_v,light.area,ellipse,random,normalize};
      save_sample(out,base+4u,AreaLightSampling{}.from_position(a));
      save_sample(out,base+20u,AreaLightSampling{}.from_segment(a));
      interval = VolumeLightInterval{}.area({P,interval_direction,{0.0f,10.0f},light.position,light.axis_x,
          light.axis_y,light.axis_z,light.size_u,light.size_v,light.area,ellipse});
    };
    out.write(base+36u,cast<float>(interval.valid));
    $if(interval.valid) {
      out.write(base+37u,interval.interval.minimum); out.write(base+38u,interval.interval.maximum);
    };
  };
  const auto shader = device.compile(kernel, ShaderOption{.enable_cache=false,.enable_fast_math=true});
  std::vector<float> actual(expected.size());
  stream << lights.copy_from(upload.device_lights.data()) << rays.copy_from(inputs.data())
         << shader(lights,rays,output).dispatch(count) << output.copy_to(actual.data()) << synchronize();
  for (unsigned i = 0; i < count; ++i) {
    for (unsigned j = 0; j < f::outputs; ++j) {
      const auto predicate = j == 4 || j == 20 || j == 36 ||
          (j == 3 && !records[i/f::probes].spot);
      const auto a = actual[i*f::outputs+j], e = expected[i*f::outputs+j];
      // Interval endpoints are lengths: compare relative to the authored
      // ten-unit ray segment, not an arbitrary one-unit world coordinate.
      // Validity is still exact. Other outputs retain their native units.
      const auto scale = j == 37 || j == 38 ? 10.0f : 1.0f;
      if (!matches(a/scale,e/scale,predicate ? 0.0f : 2e-4f)) {
        if (failures < 100) { std::cerr << "row=" << i << " light=" << i/f::probes
            << " lane=" << j << " actual=" << a << " native=" << e << '\n'; }
        ++failures;
      }
    }
  }
  std::cout << records.size() << " native scene light tables, " << count
            << " GPU states; failures=" << failures << '\n';
  return failures == 0 ? 0 : 1;
}
