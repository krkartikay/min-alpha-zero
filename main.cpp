#include "alpha_zero.h"

int main() {
  // Abseil: get the current time and print a formatted string
  absl::Time now = absl::Now();
  std::cout << absl::StrFormat("Hello, World! The time is: %s\n",
                               absl::FormatTime(now));

  // Torch: create a tensor and print it
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "Random tensor from torch:\n" << tensor << "\n";

  // Boost: convert a string to uppercase
  std::string s = "hello boost";
  boost::to_upper(s);
  std::cout << "Boost upper-cased string: " << s << "\n";
  return 0;
}