#include "alpha_zero.h"

int main() {
  absl::Time now = absl::Now();
  std::cout << absl::StrFormat("Hello, World! The time is: %s\n",
                               absl::FormatTime(now));
  // Torch: create a tensor and print it
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "Random tensor from torch:\n" << tensor << "\n";
  return 0;
}