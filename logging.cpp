#include <filesystem>

#include "alpha_zero.h"

namespace alphazero {

// -----------------------------------------------------------

constexpr const char* kSeparatorLine =
    "========================================================\n";
constexpr const char* kChildSeparatorLine =
    "--------------------------------------------------------\n";

template <typename T>
std::string dump_sparse_array(const Node& node,
                              const std::array<T, kNumActions>& arr) {
  std::string out = "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (!node.legal_mask[i]) continue;
    chess::Move move = int_to_move(i, node.board);
    std::string move_str = chess::uci::moveToSan(node.board, move);
    if (arr[i] != 0) {
      absl::StrAppendFormat(&out, "(%s:%v),", move_str, arr[i]);
    }
  }
  out += "]";
  return out;
}

std::string board_to_string(const chess::Board& board) {
  std::ostringstream oss;
  oss << board;
  return oss.str();
}

std::string dump_node(const Node& node, int chosen_action, int depth) {
  std::string indent(depth * 4, '>');
  std::string children_indices, children_details;

  for (const auto& [child_idx, child] : node.child_nodes) {
    absl::StrAppendFormat(&children_indices, "%s  Child (%d) at Node (%p):\n",
                          indent, child_idx,
                          static_cast<const void*>(child.get()));
  }
  for (const auto& [child_idx, child] : node.child_nodes) {
    absl::StrAppendFormat(&children_details, "%s  Child (%d):\n%s%s%s", indent,
                          child_idx, dump_node(*child, -1, depth + 1), indent,
                          kChildSeparatorLine);
  }

  std::string action_str;
  if (chosen_action != -1) {
    chess::Move move = int_to_move(chosen_action, node.board);
    action_str =
        absl::StrFormat("%sAction chosen: %d (%s)\n", indent, chosen_action,
                        chess::uci::moveToSan(node.board, move));
  }

  // Split board string into lines and add indent before each line
  std::string board_str = board_to_string(node.board);
  std::istringstream iss(board_str);
  std::string board_out, line;
  while (std::getline(iss, line)) {
    board_out += indent + line + "\n";
  }

  return absl::StrFormat(
      "%s%s"
      "%sNode (%p): %s\n"
      "%sBoard:\n%s"
      "%sParent node (%p): %s\n"
      "%sIs Evaluated? %s\n"
      "%sIs Leaf? %s\n"
      "%sValue: %v\n"
      "%sLegal Mask: %s\n"
      "%sPolicy: %s\n"
      "%sChild Visits: %s\n"
      "%sChild Values: %s\n"
      "%s%s"
      "%sNumber of children: %d\n"
      "%sChildren index: \n"
      "%s"
      "%sChildren: \n"
      "%s"
      "%s%s",
      indent, kSeparatorLine, indent, static_cast<const void*>(&node),
      node.move_history, indent, board_out, indent,
      static_cast<const void*>(node.parent),
      node.parent ? node.parent->move_history : "None", indent,
      node.is_evaluated ? "Yes" : "No", indent, node.is_leaf ? "Yes" : "No",
      indent, node.value, indent, dump_sparse_array(node, node.legal_mask),
      indent, dump_sparse_array(node, node.policy), indent,
      dump_sparse_array(node, node.child_visits), indent,
      dump_sparse_array(node, node.child_values), indent, action_str, indent,
      static_cast<int>(node.child_nodes.size()), indent, children_indices,
      indent, children_details, indent, kSeparatorLine);
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
  ofs << dump_node(*game.root, chosen_action, 0);
  ofs.close();
}

// -----------------------------------------------------------

std::string timestamp() {
  return absl::FormatTime("[%H:%M:%S.%E3S]", absl::Now(),
                          absl::LocalTimeZone());
}

}  // namespace alphazero