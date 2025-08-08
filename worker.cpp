#include "alpha_zero.h"

namespace alphazero {

std::mt19937 g_rng;  // Random number generator for MCTS

// Worker thread main loop
void run_worker() {
  int game_number = 0;
  while (true) {
    // For now just runs self play on one game
    Game game;
    self_play(game);
    log("Game %d finished.", game_number++);
  }
}

// -----------------------------------------------------------

// Self play logic

// Selects moves on game tree then makes the move until game is finished
void self_play(Game& game) {
  // Keep going while the game is not over
  bool is_game_over = false;
  int moves_played = 0;
  while (!is_game_over && moves_played < 100) {
    // select move
    int action = select_move(game);

    // For logging:
    chess::Move move = int_to_move(action, game.root->board);
    std::string move_str = chess::uci::moveToSan(game.root->board, move);

    // Save game state at current position.
    game.history.emplace_back(
        GameState{chess::board_to_tensor(game.root->board), game.root->policy,
                  game.root->value});

    // Update root node to the selected child
    std::unique_ptr<Node> chosen_child =
        std::move(game.root->child_nodes[action]);
    game.root = std::move(chosen_child);
    game.root->parent = nullptr;  // Reset parent to null

    log("Move played: %s", move_str);
    moves_played++;

    // print game state
    log("Current board state:\n%s", chess::board_to_string(game.root->board));
    is_game_over =
        game.root->board.isGameOver().first != chess::GameResultReason::NONE;
  }

  // At the end of the game, update result and side to move
  // and then write data to training file.
  game.result = game.root->board.isGameOver().first;
  game.side_to_move = game.root->board.sideToMove();
  append_to_training_file(game);
}

int select_move(Game& game) {
  // MCTS move selection process

  // Runs N number of simulations to select a move to play
  // All simulations are run on a boost fiber!
  // Store fibers so we can join them later
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < kNumSimulations; ++i) {
    fibers.emplace_back([&game]() { run_simulation(game); });
  }
  // Join all fibers
  for (auto& fiber : fibers) {
    fiber.join();
  }

  // Calculate visit counts at root
  std::array<int, kNumActions> visit_counts = {};
  for (int i = 0; i < kNumActions; ++i) {
    Node* child = game.root->getChildNode(i);
    if (child == nullptr) continue;  // illegal move
    visit_counts[i] = child->visit_count;
  }

  // Sample an action based on visit counts
  std::discrete_distribution<int> dist(visit_counts.begin(),
                                       visit_counts.end());
  int action = dist(g_rng);
  log("Selected action: %d with (%d) visits", action, visit_counts[action]);

  if (!game.root->legal_mask[action]) {
    log("Illegal move selected: %d", action);
    std::abort();
  }

  return action;
}

// Run MCTS simulation
// Finds a node that is as yet unevaluated and evaluates it,
// And then backpropagates the result.
void run_simulation(Game& game) {
  Node* current_node = game.root.get();

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

Game::Game() {
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

void append_to_training_file(const Game& game) {
  const int N = game.history.size();

  // Pack history as contiguous tensors
  torch::Tensor boards = torch::empty({N, kInputSize}, torch::kFloat32);
  torch::Tensor policies = torch::empty({N, kNumActions}, torch::kFloat32);
  torch::Tensor values = torch::empty({N}, torch::kFloat32);

  float* bptr = boards.data_ptr<float>();
  float* pptr = policies.data_ptr<float>();
  float* vptr = values.data_ptr<float>();

  for (int i = 0; i < N; ++i) {
    const auto& s = game.history[i];
    std::memcpy(bptr + i * kInputSize, s.board_tensor.data(),
                kInputSize * sizeof(float));
    std::memcpy(pptr + i * kNumActions, s.policy.data(),
                kNumActions * sizeof(float));
    vptr[i] = s.value;
  }

  // Scalar metadata (store as 0-dim/int64 tensors for portability)
  torch::Tensor result_t = torch::tensor(int(game.result), torch::kInt64);
  torch::Tensor side_t = torch::tensor(int(game.side_to_move), torch::kInt64);

  // One “record” = map of named tensors (easy to read in Python)
  torch::serialize::OutputArchive ar;
  ar.write("boards", boards);
  ar.write("policies", policies);
  ar.write("values", values);
  ar.write("result_reason", result_t);
  ar.write("side_to_move", side_t);

  // Append this record to the file
  std::ofstream ofs(kTrainingFile, std::ios::binary | std::ios::app);
  TORCH_CHECK(ofs.good(),
              "Failed to open training file for append: ", kTrainingFile);
  ar.save_to(ofs);
  ofs.close();
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