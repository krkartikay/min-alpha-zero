#include "alpha_zero.h"

namespace alphazero {

GameTree g_game_tree;

// Worker thread main loop
void run_worker() {
  // TODO: LEFT OFF HERE
  // Starts a number of fibers, each one sends as many requests
  // to the channel as possible.
  for (int i = 0; i < 100; ++i) {
    boost::fibers::fiber([i]() { work_on(g_game_tree); }).detach();
  }
}

void work_on(const GameTree& game_tree) {
  // Send a request to the channel
  std::cout << "Worker Sending request: " << game_tree.root.get() << std::endl;
  future<void> response = send_request(game_tree.root.get());
  // Wait for the response
  response.get();
  // Print the result
  std::cout << "Worker sent: " << game_tree.root.get() << " and got result."
            << std::endl;
}

future<void> send_request(Node* request) {
  // Create a promise and future pair
  promise<void> promise;
  future<void> future = promise.get_future();

  // Send the request to the channel
  g_evaluation_queue.push(std::make_pair(request, std::move(promise)));
  return future;
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

}  // namespace alphazero