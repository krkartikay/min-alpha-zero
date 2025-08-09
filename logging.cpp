#include "alpha_zero.h"

namespace alphazero {

// -----------------------------------------------------------

std::string timestamp() {
  return absl::FormatTime("[%H:%M:%S.%E3S]", absl::Now(),
                          absl::LocalTimeZone());
}

std::string board_to_string(const chess::Board& board) {
  std::ostringstream oss;
  oss << board;
  return oss.str();
}

// -----------------------------------------------------------

}  // namespace alphazero