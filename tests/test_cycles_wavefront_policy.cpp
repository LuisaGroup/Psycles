#include "cycles_wavefront_policy.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

int main() {
  using psycles::luisa_backend::detail::cycles_shadow_compaction;
  std::ifstream oracle{PSYCLES_WAVEFRONT_POLICY_ORACLE};
  std::string tag, revision, digest;
  if (!(oracle >> tag >> revision >> digest) || tag != "source" ||
      revision != "cb168525138fecc792cc393f94afc39582b0103c" ||
      digest != "2e8d1c32186cfb832792cafb69d16257f72990791edc3762084d73bb7a92a075") {
    std::cerr << "Invalid Cycles host-policy oracle source identity\n";
    return 1;
  }
  std::array<unsigned, 6u> kernels{};
  if (!(oracle >> tag) || tag != "kernels") {
    return 1;
  }
  for (auto &kernel : kernels) {
    if (!(oracle >> kernel)) {
      return 1;
    }
  }
  auto checked = 0u;
  while (oracle >> tag) {
    if (tag == "compact") {
      std::uint32_t extent{}, next{}, compact{}, upload{};
      std::array<std::uint32_t, 4u> counts{};
      if (!(oracle >> extent >> counts[0] >> counts[1] >> counts[2] >> counts[3] >>
            next >> compact >> upload)) {
        return 1;
      }
      const auto live = std::uint64_t{counts[0]} + counts[1] + counts[2] + counts[3];
      if (live > extent || compact > 1u || upload > 1u) {
        return 1;
      }
      const auto actual =
          cycles_shadow_compaction(extent, static_cast<std::uint32_t>(live));
      if (actual.extent != next || actual.compact != (compact != 0u) ||
          actual.upload_extent != (upload != 0u)) {
        std::cerr << "Cycles shadow compaction mismatch at extent=" << extent
                  << " live=" << live << '\n';
        return 1;
      }
      ++checked;
    } else if (tag == "select") {
      // Retain the source's global queue-order oracle for the separate
      // priority integration. This test claims compaction coverage only.
      std::array<unsigned, 5u> row{};
      for (auto &value : row) {
        if (!(oracle >> value)) {
          return 1;
        }
      }
    } else {
      return 1;
    }
  }
  if (!oracle.eof() || checked != 145u) {
    return 1;
  }
  std::cout << checked << " original Cycles shadow-compaction oracle rows passed\n";
  return 0;
}
