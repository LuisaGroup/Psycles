#include "cycles_svm_imported_word_fixture.h"

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_bump_alias", 12);
    std::cout << "12 surface/automatic-bump alias graphs match Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
