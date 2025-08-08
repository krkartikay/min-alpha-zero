#include "alpha_zero.h"

namespace alphazero {

GameTree g_game_tree;
std::mt19937 g_rng;  // Random number generator for MCTS

// Worker thread main loop
void run_worker() {
  // For now just runs self play on one game_tree
  self_play(g_game_tree);
}

// -----------------------------------------------------------

// Self play logic

// Selects moves on game tree then makes the move until game is finished
void self_play(GameTree& game_tree) {
  while (true) {
    // select move
    select_move(game_tree);
  }
}

void select_move(GameTree& game_tree) {
  // MCTS move selection process

  // Runs N number of simulations to select a move to play
  // All simulations are run on a boost fiber!
  // Store fibers so we can join them later
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < kNumSimulations; ++i) {
    fibers.emplace_back([&game_tree]() { run_simulation(game_tree); });
  }
  // Join all fibers
  for (auto& fiber : fibers) {
    fiber.join();
  }

  // Calculate visit counts at root
  std::array<int, kNumActions> visit_counts = {};
  for (int i = 0; i < kNumActions; ++i) {
    Node* child = game_tree.root->getChildNode(i);
    if (child == nullptr) continue;  // illegal move
    visit_counts[i] = child->visit_count;
  }

  // Sample an action based on visit counts
  std::discrete_distribution<int> dist(visit_counts.begin(),
                                       visit_counts.end());
  int action = dist(g_rng);
  log("Selected action: %d with (%d) visits", action, visit_counts[action]);

  if (!game_tree.root->legal_mask[action]) {
    log("Illegal move selected: %d", action);
    return;  // Illegal move, do not proceed
  }

  // For logging:
  chess::Move move = int_to_move(action, game_tree.root->board);
  std::string move_str = chess::uci::moveToSan(game_tree.root->board, move);

  // Update root node to the selected child
  std::unique_ptr<Node> chosen_child =
      std::move(game_tree.root->child_nodes[action]);
  game_tree.root = std::move(chosen_child);
  game_tree.root->parent = nullptr;  // Reset parent to null
  log("Move played: %s", move_str);
}

// Run MCTS simulation
// Finds a node that is as yet unevaluated and evaluates it,
// And then backpropagates the result.
void run_simulation(GameTree& game_tree) {
  Node* current_node = game_tree.root.get();

  // walk down to unevaluated node
  while (true) {
    std::unique_lock<mutex> lock(current_node->is_processing_mutex);
    if (!current_node->is_evaluated) break;
    if (current_node->is_leaf) break;
    current_node = select_child(*current_node);
  }

  // evaluate node
  // we could reach an unevaluated node
  // or a leaf node, even if it is already evaluated
  if (!current_node->is_evaluated) {
    evaluate(*current_node);
  }
  current_node->visit_count++;

  // backpropagate the result
  while (current_node->parent != nullptr) {
    current_node = current_node->parent;
    current_node->visit_count++;
  }
}

Node* select_child(Node& node) {
  // Gather legal actions
  std::vector<int> legal_moves;
  legal_moves.reserve(kNumActions);
  for (int a = 0; a < kNumActions; ++a)
    if (node.legal_mask[a]) legal_moves.push_back(a);
  if (legal_moves.empty()) return nullptr;  // no children
  std::uniform_int_distribution<int> dist(
      0, static_cast<int>(legal_moves.size()) - 1);
  int idx = dist(g_rng);
  int move = legal_moves[idx];
  return node.getChildNode(move);
}

// -----------------------------------------------------------

GameTree::GameTree() {
  // Initialize the root node
  root = std::make_unique<Node>();
}

Node::Node(const chess::Board& board) : board(board) {
  // also sets legal mask and is_leaf flag
  chess::Movelist movelist;
  chess::movegen::legalmoves(movelist, board);
  if (movelist.empty()) {
    is_leaf = true;
  }
  for (chess::Move move : movelist) {
    int move_idx = move_to_int(move);
    legal_mask[move_idx] = true;
  }
}

Node* Node::getChildNode(int move_idx) {
  if (!legal_mask[move_idx]) {
    // If the move is not legal, return nullptr
    return nullptr;
  }

  // Lazy initialization of nodes
  auto it = child_nodes.find(move_idx);
  if (it != child_nodes.end()) {
    return it->second.get();
  }

  // Create a new node if it doesn't exist
  chess::Board new_board = board;
  chess::Move move = int_to_move(move_idx, board);
  new_board.makeMove(move);
  std::unique_ptr<Node> new_node = std::make_unique<Node>(new_board);
  new_node->parent = this;
  new_node->move_history =
      move_history.empty()
          ? chess::uci::moveToSan(board, move)
          : absl::StrCat(move_history, " ", chess::uci::moveToSan(board, move));
  Node* new_node_ptr = new_node.get();
  child_nodes[move_idx] = std::move(new_node);
  return new_node_ptr;
}

// -----------------------------------------------------------

void evaluate(Node& node) {
  // Ensure no other fibers are processing this node
  std::unique_lock<mutex> lock(node.is_processing_mutex);

  // Create a promise and future pair
  promise<void> promise;
  future<void> future = promise.get_future();

  // Send the node to the channel
  g_evaluation_queue.push(std::make_pair(&node, std::move(promise)));

  // Wait for evaluation to complete
  future.get();
}

// -----------------------------------------------------------

}  // namespace alphazero