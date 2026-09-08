#include "cycles_svm_imported_word_fixture.h"

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_closure_inputs", 26);
    std::cout << "26 closure-input graphs match Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
