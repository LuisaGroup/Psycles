// Original Cycles 5.2.1 interpreter and typed payloads, on HIP. This tool
// constructs legal instruction streams, not expected results or a renderer.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_svm_volume_oracle.hip -o /tmp/volume-svm-oracle
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
// clang-format on

#include <array>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

SVMInputFloat literal(float x) { return {std::bit_cast<unsigned>(x)}; }
SVMInputFloat3 literal(float x, float y, float z) {
  return {literal(x), literal(y), literal(z)};
}
template<typename T>
void emit(std::vector<unsigned> &words, ShaderNodeType tag, const T &payload) {
  static_assert(sizeof(T) % sizeof(unsigned) == 0);
  words.push_back(tag);
  const auto offset = words.size();
  words.resize(offset + sizeof(T) / sizeof(unsigned));
  std::memcpy(words.data() + offset, &payload, sizeof(T));
}

struct Image { std::string name; std::vector<unsigned> words; };

std::vector<Image> images() {
  std::vector<Image> result;
  const std::array types{CLOSURE_VOLUME_ABSORPTION_ID,
                        CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID,
                        CLOSURE_VOLUME_FOURNIER_FORAND_ID,
                        CLOSURE_VOLUME_RAYLEIGH_ID, CLOSURE_VOLUME_DRAINE_ID,
                        CLOSURE_VOLUME_MIE_ID};
  for (auto model = 0u; model < 20u; ++model) {
    Image image{std::to_string(model), {NODE_SHADER_JUMP, 4u, 4u, 4u}};
    auto &w = image.words;
    emit(w, NODE_VALUE_F, SVMNodeValueF{.value = model == 12u ? 0.0f : 0.375f,
                                       .out_offset = 0u});
    emit(w, NODE_VALUE_F, SVMNodeValueF{.value = 2.5f, .out_offset = 1u});
    emit(w, NODE_VALUE_V, SVMNodeValueV{.out_offset = 2u,
                                       .value = {0.1f, 0.25f, 0.6f}});
    emit(w, NODE_CLOSURE_SET_WEIGHT,
         SVMNodeClosureSetWeight{.rgb = {0.2f, 0.4f, 0.8f}});
    if (model < 13u) {
      const auto type = types[model < 6u ? model : model < 10u ? 5u : 1u];
      const auto param = type == CLOSURE_VOLUME_MIE_ID
                             ? std::array{0.05f, 0.3f, 2.5f, 8.0f}[model < 6u ? 0u : model - 6u]
                         : type == CLOSURE_VOLUME_FOURNIER_FORAND_ID ? 1.33f : -1.4f;
      const SVMNodeClosureVolume node{
          .closure_type = type,
          .density = model == 10u ? literal(-2.0f)
                     : SVMInputFloat{SVM_INPUT_STACK_OFFSET_MASK | 1u},
          .param1 = literal(param), .param_extra = literal(0.23f),
          .mix_weight_offset = model == 11u ? SVM_STACK_INVALID : SVMStackOffset{0u}};
      emit(w, NODE_CLOSURE_VOLUME, node);
      if (model == 11u) { emit(w, NODE_CLOSURE_VOLUME, node); }
    } else if (model < 16u) {
      if (model == 15u) {
        emit(w, NODE_CLOSURE_SET_WEIGHT,
             SVMNodeClosureSetWeight{.rgb = {0.0f, 0.0f, 0.0f}});
      }
      emit(w, NODE_VOLUME_COEFFICIENTS, SVMNodeVolumeCoefficients{
          .closure_type = model == 14u ? CLOSURE_VOLUME_ABSORPTION_ID
                                      : CLOSURE_VOLUME_DRAINE_ID,
          .absorption_coeffs = literal(-0.1f, 0.3f, 0.5f),
          .emission_coeffs = {{SVM_INPUT_STACK_OFFSET_MASK | 2u}, {}, {}},
          .param1 = literal(0.6f), .param_extra = literal(0.7f),
          .mix_weight_offset = 0u});
    } else {
      emit(w, NODE_PRINCIPLED_VOLUME, SVMNodePrincipledVolume{
          .absorption_color = literal(0.04f, 0.25f, 0.81f),
          .emission_color = {{SVM_INPUT_STACK_OFFSET_MASK | 2u}, {}, {}},
          .blackbody_tint = literal(0.75f, 0.5f, 0.25f),
          .density = literal(model == 19u ? 0.0f : 0.75f),
          .anisotropy = literal(-1.2f), .emission = literal(0.8f),
          .blackbody = literal(model == 16u ? 0.0f : 0.65f),
          .temperature = literal(model == 18u ? -200.0f : 1800.0f),
          .attr_density = 1000, .attr_color = 1001, .attr_temperature = 1002,
          .mix_weight_offset = 0u});
    }
    w.push_back(NODE_END);
    result.emplace_back(std::move(image));
  }
  return result;
}

struct Output { unsigned flag, count, left; float values[22]; };

template<ShaderType domain>
__global__ void probe(Output *output, unsigned capacity) {
  const auto i = threadIdx.x;
  if (i >= 16u) { return; }
  ShaderData sd{};
  sd.object = i % 4u == 0u ? OBJECT_NONE : i % 4u - 1u;
  sd.prim = PRIM_NONE;
  sd.type = PRIMITIVE_VOLUME;
  sd.shader = 0;
  sd.flag = SD_IS_VOLUME_SHADER_EVAL | ((i & 8u) ? SD_EXTINCTION | SD_EMISSION : 0);
  sd.closure_transparent_extinction = make_float3(-1.0f, -2.0f, -3.0f);
  sd.closure_emission_background = make_float3(0.125f, 0.25f, 0.5f);
  sd.num_closure = 0;
  sd.num_closure_left = capacity;
  const auto visibility = (i & 4u) ? PATH_RAY_VISIBILITY_SHADOW : PATH_RAY_VISIBILITY_CAMERA;
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_VOLUME, domain>(
      nullptr, IntegratorState{0}, &sd, nullptr, visibility, PATH_RAY_FLAG_NONE);
  auto &out = output[i];
  out.flag = sd.flag;
  out.count = sd.num_closure;
  out.left = sd.num_closure_left;
  const auto put3 = [&](unsigned offset, float3 v) {
    out.values[offset] = v.x; out.values[offset + 1u] = v.y; out.values[offset + 2u] = v.z;
  };
  put3(0u, sd.closure_transparent_extinction);
  put3(3u, sd.closure_emission_background);
  for (auto slot = 0u; slot < 2u; ++slot) {
    const auto base = 6u + slot * 8u;
    for (auto j = 0u; j < 8u; ++j) { out.values[base + j] = 0.0f; }
    if (slot >= sd.num_closure) { continue; }
    const auto *sc = &sd.closure[slot];
    out.values[base] = float(sc->type);
    put3(base + 1u, sc->weight);
    out.values[base + 4u] = sc->sample_weight;
    if (sc->type == CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID) {
      out.values[base + 5u] = reinterpret_cast<const HenyeyGreensteinVolume *>(sc)->g;
    } else if (sc->type == CLOSURE_VOLUME_DRAINE_ID) {
      const auto *phase = reinterpret_cast<const DraineVolume *>(sc);
      out.values[base + 5u] = phase->g; out.values[base + 6u] = phase->alpha;
    } else if (sc->type == CLOSURE_VOLUME_FOURNIER_FORAND_ID) {
      const auto *phase = reinterpret_cast<const FournierForandVolume *>(sc);
      put3(base + 5u, make_float3(phase->c1, phase->c2, phase->c3));
    }
  }
}

__global__ void layout(unsigned *out) {
  out[0] = sizeof(ShaderVolumeClosure);
  out[1] = offsetof(HenyeyGreensteinVolume, g);
  out[2] = offsetof(DraineVolume, g);
  out[3] = offsetof(DraineVolume, alpha);
  out[4] = offsetof(FournierForandVolume, c1);
  out[5] = offsetof(FournierForandVolume, c2);
  out[6] = offsetof(FournierForandVolume, c3);
}
} // namespace

int main() {
  unsigned *words{}, *device_layout{};
  Output *device_output{};
  KernelObject *device_objects{};
  AttributeMap *device_attributes{};
  float *device_floats{};
  packed_float3 *device_colors{};
  check(hipMalloc(&words, 512u * sizeof(unsigned)));
  check(hipMalloc(&device_layout, 7u * sizeof(unsigned)));
  check(hipMalloc(&device_output, 16u * sizeof(Output)));
  std::array<KernelObject, 3u> objects{};
  objects[0].volume_density = 0.0f;
  objects[1].volume_density = 0.5f;
  objects[2].volume_density = 2.0f;
  objects[2].attribute_map_offset = 2u;
  std::array<AttributeMap, 10u> attributes{};
  for (auto j = 0u; j < 3u; ++j) {
    auto &a = attributes[2u + 2u * j];
    a.id = 1000u + j; a.element = ATTR_ELEMENT_MESH;
    a.offset = j == 2u ? 1 : 0;
    a.type = j == 1u ? NODE_ATTR_FLOAT3 : NODE_ATTR_FLOAT;
  }
  const std::array floats{1.5f, 1.1f};
  const std::array<packed_float3, 1u> colors{{{0.5f, 0.8f, 0.3f}}};
  check(hipMalloc(&device_objects, sizeof(objects)));
  check(hipMalloc(&device_attributes, sizeof(attributes)));
  check(hipMalloc(&device_floats, sizeof(floats)));
  check(hipMalloc(&device_colors, sizeof(colors)));
  check(hipMemcpy(device_objects, objects.data(), sizeof(objects), hipMemcpyHostToDevice));
  check(hipMemcpy(device_attributes, attributes.data(), sizeof(attributes), hipMemcpyHostToDevice));
  check(hipMemcpy(device_floats, floats.data(), sizeof(floats), hipMemcpyHostToDevice));
  check(hipMemcpy(device_colors, colors.data(), sizeof(colors), hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.svm_nodes = words; params.objects = device_objects;
  params.attributes_map = device_attributes;
  params.attributes_float = device_floats;
  params.attributes_float3 = device_colors;
  params.data.film.is_rec709 = true;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  layout<<<1u, 1u>>>(device_layout);
  std::array<unsigned, 7u> offsets{};
  check(hipMemcpy(offsets.data(), device_layout, sizeof(offsets), hipMemcpyDeviceToHost));
  std::printf("L");
  for (const auto value : offsets) { std::printf(" %u", value); }
  std::printf("\n");
  const auto fixtures = images();
  for (auto model = 0u; model < fixtures.size(); ++model) {
    const auto &image = fixtures[model];
    std::printf("W %u %zu", model, image.words.size());
    for (const auto word : image.words) { std::printf(" %08x", word); }
    std::printf("\n");
    check(hipMemcpy(words, image.words.data(), image.words.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    for (const auto capacity : {0u, 1u, 64u}) {
      for (auto domain = 0u; domain < 3u; ++domain) {
        if (domain == 0u) { probe<SHADER_TYPE_VOLUME><<<1u, 32u>>>(device_output, capacity); }
        if (domain == 1u) { probe<SHADER_TYPE_SURFACE><<<1u, 32u>>>(device_output, capacity); }
        if (domain == 2u) { probe<SHADER_TYPE_DISPLACEMENT><<<1u, 32u>>>(device_output, capacity); }
        check(hipGetLastError());
        std::array<Output, 16u> output{};
        check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
        for (auto i = 0u; i < output.size(); ++i) {
          const auto &row = output[i];
          std::printf("R %u %u %u %u %u %u %u", model, capacity, domain, i,
                      row.flag, row.count, row.left);
          for (const auto value : row.values) { std::printf(" %.9g", value); }
          std::printf("\n");
        }
      }
    }
  }
  check(hipFree(device_colors)); check(hipFree(device_floats));
  check(hipFree(device_attributes)); check(hipFree(device_objects));
  check(hipFree(device_output)); check(hipFree(device_layout)); check(hipFree(words));
}
