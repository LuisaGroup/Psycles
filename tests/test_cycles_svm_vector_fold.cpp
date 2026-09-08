#include "cycles_svm_imported_word_fixture.h"

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_vector_fold", 66, true);
    std::cout << "38 constant and 28 dynamic Vector Math graphs match Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
