#include <filesystem>

#include "alpha_zero.h"

namespace alphazero {

// -----------------------------------------------------------

template <typename T>
void dump_sparse_array(std::array<T, kNumActions> arr, std::ostream& os) {
  os << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] != 0) {
      os << "(" << i << ":" << arr[i] << "),";
    }
  }
  os << "]";
}

void dump_node(const Node& node, std::ostream& os, int depth) {
  std::string indent = std::string(depth * 4, '-');
  os << indent << "========================================================\n";
  os << indent << "Node (" << &node << "): " << node.move_history << "\n";
  os << indent << "Board:\n" << board_to_string(node.board) << "\n";
  os << indent << "Parent node (" << node.parent
     << "): " << (node.parent ? node.parent->move_history : "None") << "\n";
  os << indent << "Is Evaluated? " << (node.is_evaluated ? "Yes" : "No")
     << "\n";
  os << indent << "Is Leaf? " << (node.is_leaf ? "Yes" : "No") << "\n";
  os << indent << "Value: " << node.value << "\n";
  os << indent << "Legal Mask: ";
  dump_sparse_array(node.legal_mask, os);
  os << "\n";
  os << indent << "Policy: ";
  dump_sparse_array(node.policy, os);
  os << "\n";
  os << indent << "Child Visits: ";
  dump_sparse_array(node.child_visits, os);
  os << "\n";
  os << indent << "Child Values: ";
  dump_sparse_array(node.child_values, os);
  os << "\n";
  os << indent << "Number of children: " << node.child_nodes.size() << "\n";
  os << indent << "Children: \n";
  for (const auto& [child_idx, child] : node.child_nodes) {
    os << indent << "  Child (" << child_idx << "):\n";
    dump_node(*child, os, depth + 1);
    os << indent << "-------------------------------------------------------";
  }
  os << indent << "========================================================\n";
}

void dump_game_tree_to_file(const Game& game, int g, int m) {
  // Construct filename
  std::filesystem::create_directories("out");
  const std::string filename = absl::StrFormat("out/game_%d_move_%d.txt", g, m);
  std::ofstream ofs(filename);
  if (!ofs) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return;
  }
  dump_node(*game.root, ofs);
  ofs.close();
}

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