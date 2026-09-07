#include "cycles_svm_shared_closure_test_support.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

int main() {
  using namespace psycles::test_support::shared_closure;
  try {
    const auto oracle = read_words(PSYCLES_SHARED_CLOSURE_WORDS_ORACLE);
    bool matched = true;
    for (auto index = 0u; index < names.size(); ++index) {
      const auto image = compile(index);
      if (image.words != oracle[index]) {
        matched = false;
        std::cerr << names[index]
                  << ": word image differs from original Cycles\n";
        const auto count = std::max(image.words.size(), oracle[index].size());
        for (auto i = 0u; i < count; ++i) {
          if (i >= image.words.size() || i >= oracle[index].size() ||
              image.words[i] != oracle[index][i]) {
            std::cerr << "word " << std::dec << i << " actual ";
            if (i < image.words.size()) {
              std::cerr << std::hex << image.words[i];
            } else {
              std::cerr << "<absent>";
            }
            std::cerr << " expected ";
            if (i < oracle[index].size()) {
              std::cerr << std::hex << oracle[index][i];
            } else {
              std::cerr << "<absent>";
            }
            std::cerr << '\n';
          }
        }
      }
    }
    require(matched, "shared-closure Add/Mix stream regression failed");
    std::cout << "Original Cycles shared-closure word images passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
