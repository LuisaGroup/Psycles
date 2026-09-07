#include "cycles_static_normal_fixture.h"
#include <psycles/compiler/cycles_svm_geometry_scene.h>
#include <fstream>
#include <iostream>

int main() {
  using namespace psycles::test_support;
  using namespace psycles::compiler::cycles_svm;
  std::ifstream oracle{PSYCLES_STATIC_NORMAL_ORACLE};
  auto index = 0u, witnesses = 0u;
  bool passed = true;
  for (const auto &t : static_normal_transforms) {
    for (const auto &n : static_normal_inputs) {
      unsigned row{}, first{}, final{}, once{};
      if (!(oracle >> row >> first >> final >> once) || row != index++) {
        std::cerr << "Malformed Cycles static-normal oracle\n";
        return 1;
      }
      const auto actual_first = pack_geometry_normal(n).value;
      const auto actual_final = pack_transformed_geometry_normal(n, t).value;
      if (actual_first != first || actual_final != final) {
        std::cerr << "Static normal " << row << ": " << actual_first << ','
                  << actual_final << " expected " << first << ',' << final << '\n';
        passed = false;
      }
      witnesses += unsigned(final != once);
    }
  }
  std::string tail;
  if ((oracle >> tail) || !oracle.eof() || witnesses == 0u) { return 1; }
  if (passed) {
    std::cout << index << " Cycles normal pack/transform cases passed; "
              << witnesses << " distinguish the invalid one-pack pipeline\n";
  }
  return passed ? 0 : 1;
}
