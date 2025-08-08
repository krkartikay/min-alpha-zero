#include "alpha_zero.h"

namespace alphazero {

GameTree g_game_tree;

// Worker thread main loop
void run_worker() {
  // For now just runs self play on one game_tree
  self_play(g_game_tree);
}

// -----------------------------------------------------------

// Self play logic

// Selects moves on game tree then makes the move until game is finished
void self_play(GameTree& game_tree) {
  // select move
  select_move(game_tree);
}

void select_move(GameTree& game_tree) {
  // MCTS move selection process

  // Runs N number of simulations to select a move to play
  // All simulations are run on a boost fiber!
  for (int i = 0; i < kNumSimulations; ++i) {
    boost::fibers::fiber([&game_tree]() {
      run_simulation(game_tree);
    }).detach();  // Detach the fiber to run it asynchronously
  }

  // Now we would select the best move based on visit counts at root node
}

void run_simulation(GameTree& game_tree) {
  // Run MCTS simulations to select a move
  // This is where the actual MCTS logic would go
  // For now, we just evaluate the root node
  std::cout << "Running simulation on root node." << std::endl;
  evaluate(*game_tree.root);
}

// -----------------------------------------------------------

GameTree::GameTree() {
  // Initialize the root node
  root = std::make_unique<Node>();
}

Node* Node::getChildNode(int move_idx) {
  // Lazy initialization of nodes
  auto it = child_nodes.find(move_idx);
  if (it != child_nodes.end()) {
    return it->second.get();
  }

  // Create a new node if it doesn't exist
  std::unique_ptr<Node> new_node = std::make_unique<Node>();
  Node* new_node_ptr = new_node.get();
  child_nodes[move_idx] = std::move(new_node);
  return new_node_ptr;
}

// -----------------------------------------------------------

void evaluate(Node& node) {
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