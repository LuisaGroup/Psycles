#pragma once

#include <array>
#include <cstdint>

namespace psycles::test_support::holdout_state {

// Immutable boundary inputs, not a host shading/reference implementation.
// The original Cycles GPU functions produce all expected state transitions.
using RGB = std::array<float, 3>;
enum class Kind : unsigned { holdout, diffuse, transparent, bssrdf, volume, portal, none, excluded };
struct Closure {
  Kind kind;
  RGB weight;
  float sample_weight;
};
struct Input {
  std::array<Closure, 4> closures{{
      {Kind::holdout, {0.25f, -0.5f, 1.5f}, 0.125f},
      {Kind::diffuse, {1.0f, 2.0f, 3.0f}, 0.75f},
      {Kind::transparent, {-0.125f, 0.5f, 1.25f}, 0.375f},
      {Kind::holdout, {0.5f, 0.25f, -0.5f}, 0.25f}}};
  unsigned count{2};
  unsigned left{2};
  // Native low ShaderData bits plus SD_MIS_FRONT, statically checked by both
  // observers. SD_CLOSURE_FLAGS deliberately does not include transparent.
  unsigned flags{0x1fffu | (1u << 16)};
  unsigned object_flags{0};
  RGB extinction{0.25f, 0.5f, 0.75f};
  RGB emission{3.0f, -2.0f, 0.125f};
  unsigned node_enabled{0};
  unsigned mix_offset{255};
  float mix{2.0f};
  RGB node_weight{0.75f, -0.25f, 1.5f};
};

inline constexpr std::array<const char *, 36> names{
    "single-node-closure", "signed-holdout-sum", "no-closures", "no-holdout-types",
    "excluded-types", "object-transparent-mixed", "object-opaque",
    "object-volume-only-transparent", "object-transparent-empty",
    "object-transparent-signed-extinction", "object-transparent-only",
    "object-opaque-empty", "node-unlinked", "node-positive-mix", "node-zero-mix",
    "node-negative-zero-mix", "node-negative-mix", "node-zero-weight",
    "node-zero-left", "node-full-pool", "node-full-pool-zero-mix",
    "node-existing-holdout", "node-object-transparent", "node-object-volume-only",
    "node-object-opaque", "node-negative-unlinked-weight", "node-no-free-tail",
    "node-negative-mix-full-pool", "transparent-flag-absent",
    "object-clears-native-subtraction-mask", "object-retains-unrelated-flags",
    "holdout-sum-ignores-flags", "node-zero-mix-retains-flags",
    "object-transparent-zero-weight", "object-transparent-unit-extinction",
    "node-high-stack-offset"};

inline constexpr auto inputs = [] {
  std::array<Input, names.size()> r{};
  r[1].count = 4; r[1].left = 0;
  r[2].count = 0; r[2].left = 4;
  r[3].closures[0].kind = Kind::bssrdf;
  r[4].count = 4; r[4].left = 0;
  r[4].closures[0].kind = Kind::excluded;
  r[4].closures[1].kind = Kind::none;
  r[4].closures[2].kind = Kind::volume;
  r[4].closures[3].kind = Kind::portal;
  r[5] = r[1]; r[5].object_flags = 1;
  r[6] = r[5]; r[6].flags &= ~(1u << 9);
  r[7] = r[5]; r[7].flags |= 1u << 19;
  r[8] = r[5]; r[8].count = 0; r[8].left = 4;
  r[9] = r[5]; r[9].extinction = {-0.5f, 1.25f, 2.0f};
  r[10] = r[5]; r[10].count = 1; r[10].left = 3;
  r[10].closures[0] = r[5].closures[2];
  r[11] = r[6]; r[11].count = 0; r[11].left = 0;
  for (unsigned i = 12; i <= 27; ++i) {
    r[i].node_enabled = 1;
    r[i].count = 0; r[i].left = 4;
    r[i].flags = 1u | (1u << 16);
  }
  r[13].mix_offset = 3; r[13].mix = 0.25f;
  r[14].mix_offset = 3; r[14].mix = 0.0f;
  r[15].mix_offset = 3; r[15].mix = -0.0f;
  r[16].mix_offset = 3; r[16].mix = -2.0f;
  r[17].node_weight = {0.0f, 0.0f, 0.0f};
  r[18].left = 0;
  r[19].count = 4; r[19].left = 0;
  r[20] = r[19]; r[20].mix_offset = 3; r[20].mix = 0.0f;
  r[21].count = 1; r[21].left = 3;
  r[22] = r[5]; r[22].node_enabled = 1; r[22].count = 3; r[22].left = 1;
  r[23] = r[22]; r[23].flags |= 1u << 19;
  r[24] = r[22]; r[24].flags &= ~(1u << 9);
  r[25].node_weight = {-0.5f, -1.5f, -2.0f};
  r[26].count = 1; r[26].left = 0;
  r[27] = r[19]; r[27].mix_offset = 3; r[27].mix = -2.0f;
  r[28] = r[5]; r[28].flags &= ~(1u << 9);
  r[29] = r[5]; r[29].flags = 0x1ffe;
  r[30] = r[5]; r[30].flags |= (1u << 13) | (1u << 14) | (1u << 18);
  r[31] = r[1]; r[31].flags = 0;
  r[32] = r[14]; r[32].flags = 0x1fff;
  r[33] = r[5]; r[33].extinction = {0.0f, 0.0f, 0.0f};
  r[34] = r[5]; r[34].extinction = {1.0f, 1.0f, 1.0f};
  r[35] = r[13]; r[35].mix_offset = 254;
  return r;
}();

inline constexpr unsigned capacity = 4;
inline constexpr unsigned snapshots = 3; // before node, after node, after holdout
// All slots receive sample_weight/N sentinels before count/left is assigned.
// This makes the allocator's untouched fields defined and observable.
inline constexpr RGB normal_sentinel{-0.25f, 0.5f, 0.75f};
inline constexpr unsigned snapshot_float_rows = 11;
inline constexpr unsigned snapshot_integer_rows = 2;
inline constexpr unsigned float_rows = snapshots * snapshot_float_rows;
inline constexpr unsigned integer_rows = snapshots * snapshot_integer_rows;

} // namespace psycles::test_support::holdout_state
