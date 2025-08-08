#include "alpha_zero.h"

namespace alphazero {

// -----------------------------------------------------------

std::string timestamp() {
  return absl::FormatTime("[%H:%M:%S.%E3S]", absl::Now(),
                          absl::LocalTimeZone());
}

// -----------------------------------------------------------

}  // namespace alphazero