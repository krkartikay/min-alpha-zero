#include "alpha_zero.h"

namespace alphazero {

std::mt19937 g_rng;  // Random number generator for MCTS

constexpr float c_puct = 1.0;

// Worker thread main loop
void run_worker() {
  LOG(INFO) << absl::StrFormat("Starting worker, playing %d games.",
                               g_config.num_games);
  for (int i = 0; i < g_config.num_games; ++i) {
    // For now just runs self play on one game
    Game game;
    self_play(game);
    LOG(INFO) << absl::StrFormat("Game %d finished.", i);
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
    VLOG(1) << absl::StrFormat(
        "Move played: %s",
        chess::uci::moveToSan(game.root->board,
                              int_to_move(action, game.root->board)));

    update_root(game, action);

    // Save game state at current position.
    save_game_state(game);

    moves_played++;
    is_game_over = game.root->is_leaf;

    VLOG(2) << absl::StrFormat("Current board state:\n%s",
                               board_to_string(game.root->board));
  }

  // At the end of the game note final winner and write to training file.
  update_game_history(game);
  append_to_training_file(game);
  LOG(INFO) << absl::StrFormat(
      "Game finished. Moves played: %d, Final value: %d", moves_played,
      game.history[0].final_value);
}

void update_root(Game& game, int action) {
  // Update root node to the selected child
  game.root->getChildNode(action);  // to make sure the child node exists
  std::unique_ptr<Node> chosen_child =
      std::move(game.root->child_nodes[action]);
  game.root = std::move(chosen_child);
  game.root->parent = nullptr;  // Reset parent to null
}

void save_game_state(Game& game) {
  game.history.emplace_back();
  game.history.back().board_tensor = chess::board_to_tensor(game.root->board);
  game.history.back().policy = game.root->policy;
  game.history.back().value = game.root->value;
  game.history.back().child_visit_counts = game.root->child_visits;
}

void update_game_history(Game& game) {
  auto result = game.root->board.isGameOver();
  chess::Color side_to_move = game.root->board.sideToMove();
  bool is_draw = result.second == chess::GameResult::DRAW ||
                 result.second == chess::GameResult::NONE;
  bool is_white_won = side_to_move == chess::Color::BLACK;
  int final_value = is_draw        ? 0    // Draw
                    : is_white_won ? 1    // White won, Black lost
                                   : -1;  // Black lost, White won
  for (size_t i = 0; i < game.history.size(); ++i) {
    game.history[i].final_value = final_value;
    final_value = -final_value;  // Reverse for the other side
  }
}

int select_move(Game& game) {
  // MCTS move selection process

  // Runs N number of simulations to select a move to play
  // All simulations are run on a boost fiber!
  // Store fibers so we can join them later
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < g_config.num_simulations; ++i) {
    fibers.emplace_back([&game]() { run_simulation(game); });
  }
  // Join all fibers
  for (auto& fiber : fibers) {
    fiber.join();
  }

  // Sample an action based on visit counts
  std::discrete_distribution<int> dist(game.root->child_visits.begin(),
                                       game.root->child_visits.end());
  int action = dist(g_rng);

  CHECK(game.root->legal_mask[action]) << "Illegal move selected: " << action;
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
    int selected_action = select_action(*current_node);
    CHECK(selected_action != -1 && current_node->legal_mask[selected_action])
        << "Illegal action selected: " << selected_action
        << " at node: " << current_node->move_history << " with board:\n"
        << board_to_string(current_node->board);
    current_node = current_node->getChildNode(selected_action);
  }

  // evaluate node
  // we could reach an unevaluated node
  // or a leaf node, even if it is already evaluated
  if (current_node->is_leaf) {
    evaluate_leaf_node(*current_node);
  }
  if (!current_node->is_evaluated) {
    evaluate(*current_node);
  }

  // backpropagate the result
  Node* parent_node = current_node->parent;
  while (parent_node != nullptr) {
    parent_node->child_visits[current_node->parent_action] += 1;
    parent_node->child_values[current_node->parent_action] +=
        -1 * current_node->value;  // SIGN FLIP! -> Opponent's perspective
    current_node = parent_node;
    parent_node = parent_node->parent;
  }
}

int select_action(Node& node) {
  CHECK(!node.is_leaf) << "Cannot select action on leaf node: "
                       << node.move_history;
  // current node visit count = sum of child visit counts + 1
  float node_visits =
      std::accumulate(node.child_visits.begin(), node.child_visits.end(), 0) +
      1;

  // for all moves we will calculate a score
  std::array<float, kNumActions> score = {};
  for (int i = 0; i < kNumActions; i++) {
    // Q -> avg evaluation
    float child_value = (node.child_visits[i] == 0)
                            ? 0
                            : node.child_values[i] / node.child_visits[i];
    // P -> prior probability
    float prior_probability = node.policy[i];
    // Exploration term
    float exploration_value =
        std::sqrt(node_visits) / (node.child_visits[i] + 1);
    score[i] = child_value + c_puct * prior_probability * exploration_value;
    score[i] *= node.legal_mask[i];
  }
  std::discrete_distribution<int> dist(score.begin(), score.end());
  int action = dist(g_rng);
  return action;
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
  new_node->parent_action = move_idx;
  new_node->move_history =
      move_history.empty()
          ? chess::uci::moveToSan(board, move)
          : absl::StrCat(move_history, " ", chess::uci::moveToSan(board, move));
  Node* new_node_ptr = new_node.get();
  child_nodes[move_idx] = std::move(new_node);
  return new_node_ptr;
}

// -----------------------------------------------------------
// Write game data to training file

template <class T>
static void write_bin(std::ofstream& out, const T& v) {
  auto bytes = std::as_bytes(std::span{&v, 1});
  out.write((char*)(bytes.data()), bytes.size());
}

template <class T, std::size_t N>
static void write_bin(std::ofstream& out, const std::array<T, N>& a) {
  auto bytes = std::as_bytes(std::span{a});
  out.write((char*)(bytes.data()), bytes.size());
}

void append_to_training_file(const Game& game) {
  std::ofstream out(g_config.training_file, std::ios::binary | std::ios::app);
  if (!out) throw std::runtime_error("open failed: " + g_config.training_file);

  for (const auto& s : game.history) {
    write_bin(out, s.board_tensor);
    write_bin(out, s.policy);
    write_bin(out, s.child_visit_counts);
    write_bin(out, s.value);
    write_bin(out, s.final_value);
  }
}

// -----------------------------------------------------------

void evaluate(Node& node) {
  // Ensure no other fibers are processing this node
  std::unique_lock<mutex> lock(node.is_processing_mutex);

  // Create a promise and future pair
  promise<void> promise;
  future<void> future = promise.get_future();

  // Send the node to the channel
  g_evaluation_queue->push(std::make_pair(&node, std::move(promise)));

  // Wait for evaluation to complete
  future.get();
}

void evaluate_leaf_node(Node& node) {
  // Doesn't need the neural net.
  // If we're checkmated value is -1.
  node.is_evaluated = true;
  auto result = node.board.isGameOver();
  if (result.first == chess::GameResultReason::CHECKMATE) {
    node.value = -1;
  } else {
    node.value = 0;
  }
  // node.policy doesn't need to be set, all 0 is ok
}

// -----------------------------------------------------------

}  // namespace alphazero