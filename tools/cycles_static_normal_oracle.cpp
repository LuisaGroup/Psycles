// Calls the original Cycles packed_normal and transform functions, in the
// same host sequence as BlenderSync::create_mesh + Mesh::apply_transform.
// This is a geometry-payload oracle, not a CPU reference renderer.
#include "../tests/cycles_static_normal_fixture.h"
// The same namespace definitions supplied by Cycles' own CMakeLists.txt.
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "util/transform.h"
#include "util/types_normal.h"
#include <cstdio>

int main() {
  using namespace ccl;
  using namespace psycles::test_support;
  auto row = 0u;
  for (const auto &t : static_normal_transforms) {
    const Transform ntfm{
        make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
        make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
        make_float4(t.z.x, t.z.y, t.z.z, t.z.w)};
    for (const auto &n : static_normal_inputs) {
      const auto raw = make_float3(n.x, n.y, n.z);
      const ccl::packed_normal first{raw};
      const ccl::packed_normal final{
          normalize(transform_direction(&ntfm, first.decode()))};
      const ccl::packed_normal once{
          normalize(transform_direction(&ntfm, raw))};
      std::printf("%u %u %u %u\n", row++, first.value, final.value, once.value);
    }
  }
}
