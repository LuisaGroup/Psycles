#include "cycles_svm_imported_word_fixture.h"

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_hidden_socket", 15);
    std::cout << "15 hidden/default/linked/conversion/bump graphs match Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
