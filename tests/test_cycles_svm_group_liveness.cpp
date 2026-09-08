#include "cycles_svm_imported_word_fixture.h"

int main() {
  unsigned failures = 0;
  for (const auto &[stem, count] : {std::pair{"cycles_group_liveness", 16u},
                                   std::pair{"cycles_group_context", 14u}}) {
    try {
      psycles::test_support::check_imported_words(stem, count);
    } catch (const std::exception &error) {
      std::cerr << stem << ": " << error.what() << '\n';
      ++failures;
    }
  }
  if (failures != 0) {
    return 1;
  }
  std::cout << "30 lazy group input/context graphs match original Cycles\n";
}
