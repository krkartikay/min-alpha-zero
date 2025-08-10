#include <filesystem>

#include "alpha_zero.h"

namespace alphazero {

// -----------------------------------------------------------

template <typename T>
void dump_sparse_array(const Node& node, std::array<T, kNumActions> arr,
                       std::ostream& os) {
  os << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (!node.legal_mask[i]) continue;
    chess::Move move = int_to_move(i, node.board);
    std::string move_str = chess::uci::moveToSan(node.board, move);
    if (arr[i] != 0) {
      os << "(" << move_str << ":" << arr[i] << "),";
    }
  }
  os << "]";
}

void dump_node(const Node& node, std::ostream& os, int chosen_action,
               int depth) {
  std::string indent = std::string(depth * 4, '>');
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
  dump_sparse_array(node, node.legal_mask, os);
  os << "\n";
  os << indent << "Policy: ";
  dump_sparse_array(node, node.policy, os);
  os << "\n";
  os << indent << "Child Visits: ";
  dump_sparse_array(node, node.child_visits, os);
  os << "\n";
  os << indent << "Child Values: ";
  dump_sparse_array(node, node.child_values, os);
  os << "\n";
  if (chosen_action != -1) {
    chess::Move move = int_to_move(chosen_action, node.board);
    os << indent << "Action chosen: " << chosen_action << " ("
       << chess::uci::moveToSan(node.board, move) << ")\n";
  }
  os << indent << "Number of children: " << node.child_nodes.size() << "\n";
  os << indent << "Children index: \n";
  for (const auto& [child_idx, child] : node.child_nodes) {
    os << indent << "  Child (" << child_idx << ") at Node (" << child
       << "):\n";
  }
  os << indent << "Children: \n";
  for (const auto& [child_idx, child] : node.child_nodes) {
    os << indent << "  Child (" << child_idx << "):\n";
    dump_node(*child, os, -1, depth + 1);
    os << indent << "-------------------------------------------------------\n";
  }
  os << indent << "========================================================\n";
}

void dump_game_tree_to_file(const Game& game, int g, int m, int chosen_action) {
  // Construct filename
  std::filesystem::create_directories("out");
  const std::string filename = absl::StrFormat("out/game_%d_move_%d.txt", g, m);
  std::ofstream ofs(filename);
  if (!ofs) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return;
  }
  dump_node(*game.root, ofs, chosen_action, 0);
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