#pragma once

#include <array>
#include <cstdint>

namespace psycles::test_support::ray_portal {

// Boundary inputs only. Expected state is captured by the original Cycles
// GPU integrator; no host selection, shading or transport model lives here.
using RGB = std::array<float, 3>;
enum class Kind : unsigned { portal, diffuse, transparent, bssrdf, holdout };
struct Closure {
  Kind kind{Kind::portal};
  RGB weight{0.25f, 0.5f, 0.75f};
  float sample_weight{0.5f};
};
struct Input {
  std::array<Closure, 3> closures{};
  unsigned count{1};
  RGB point{0.25f, 0.25f, 2.0f};
  RGB portal_position{1.0f, 2.0f, 3.0f};
  RGB portal_direction{0.0f, 0.6f, 0.8f};
  float random_z{0.125f};
  // 0: curve; 1: interior/outside test triangle; 2/3: the two original
  // triangles sharing the exact edge observed in the full-render probe.
  unsigned triangle{0};
  unsigned transform_applied{1};
  unsigned portal_shader{1};
  unsigned transparent_limit{8};
  unsigned transparent_bounce{2};
  unsigned portal_bounce{3};
  // Nonzero: observe original path_state_next directly with this native
  // label. This is a transition control, not a mocked BSDF sampling result.
  unsigned transition_label{0};
  unsigned force_first_closure{0};
};

inline constexpr auto inputs = [] {
  std::array<Input, 26> result{};
  result[1].portal_position = result[1].point;
  result[2].triangle = 1;
  result[2].portal_position = result[2].point;
  result[3] = result[2];
  result[3].point = result[3].portal_position = {4.0f, 4.0f, 2.0f};
  result[4] = result[3];
  result[4].portal_position[0] += 0.00002f;
  result[5] = result[3];
  result[5].portal_position[0] += 0.00004f;
  result[6] = result[2];
  result[6].transform_applied = 0;
  result[7] = result[3];
  result[7].transform_applied = 0;
  result[8].closures[0].weight = {-0.25f, 0.5f, 0.75f};
  result[9].transparent_limit = 0;
  result[10].transparent_limit = 3;
  result[11].portal_shader = 0;
  result[12].count = 3;
  result[12].closures = {{
      {Kind::diffuse, {1.0f, 1.0f, 1.0f}, 0.25f},
      {Kind::portal, {0.25f, 0.5f, 0.75f}, 0.5f},
      {Kind::bssrdf, {1.0f, 1.0f, 1.0f}, 0.75f}}};
  result[12].random_z = 0.25f;
  result[13] = result[12];
  result[13].closures[2] = {Kind::holdout, {9.0f, 8.0f, 7.0f}, 100.0f};
  result[13].random_z = 0.75f;
  result[14].closures[0].sample_weight = 0.0f;
  result[14].force_first_closure = 1;
  result[15] = result[14];
  result[15].closures[0].sample_weight = -1.0f;
  // Native ClosureLabel: TRANSMIT=1, REFLECT=2, DIFFUSE=4, SINGULAR=16,
  // TRANSPARENT=32, TRANSMIT_TRANSPARENT=128, RAY_PORTAL=512.
  result[16].transition_label = 1u | 32u;
  result[17] = result[16];
  result[17].portal_shader = 0;
  result[18].transition_label = 2u | 4u;
  result[19].transition_label = 1u | 16u | 128u;
  result[20].transition_label = 1u | 16u;
  result[21].transition_label = 1u | 512u;
  result[21].portal_shader = 0;
  result[22] = result[16];
  result[22].transparent_limit = 0;
  result[23] = result[18];
  result[23].transparent_limit = 0;
  result[24].triangle = 2;
  result[24].point = result[24].portal_position =
      {0.468780517578125f, 0.468780517578125f, 3.0f};
  result[24].portal_direction = {0.0f, 0.0f, -1.0f};
  result[25] = result[24];
  result[25].triangle = 3;
  return result;
}();

inline constexpr unsigned float_rows = 7;
inline constexpr unsigned integer_rows = 4;

} // namespace psycles::test_support::ray_portal
