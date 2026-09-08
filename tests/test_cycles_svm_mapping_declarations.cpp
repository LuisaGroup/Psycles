#include "cycles_svm_imported_word_fixture.h"

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_mapping_declarations", 18);
    std::cout << "18 native Mapping declaration graphs match original Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
