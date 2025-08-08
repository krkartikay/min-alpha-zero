#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/torch.h>

#include <boost/fiber/all.hpp>
#include <chess.hpp>
#include <chess_utils.hpp>
#include <iostream>

#ifndef ALPHA_ZERO_H
#define ALPHA_ZERO_H

namespace alphazero {

using boost::fibers::buffered_channel;
using boost::fibers::channel_op_status;
using boost::fibers::future;
using boost::fibers::promise;

struct Node;

using eval_request_t = std::pair<Node*, boost::fibers::promise<void>>;
using eval_channel_t = boost::fibers::buffered_channel<eval_request_t>;

// Config ----------------------------------------------------

const int kChannelSize = 128;     // Must be a power of 2!
const int kNumSimulations = 200;  // Number of MCTS simulations per move

// Constants -------------------------------------------------

constexpr int kNumActions = 64 * 64;

// Global Position Evaluation queue --------------------------

inline eval_channel_t g_evaluation_queue(kChannelSize);

// Game Tree functions ---------------------------------------

struct Node {
  chess::Board board;
  bool is_evaluated = false;
  bool is_leaf = false;
  int visit_count = 0;
  float value = 0.0f;
  std::array<bool, kNumActions> legal_mask = {};
  std::array<float, kNumActions> policy = {};
  // Child node map, move idx -> Node*, Lazily initialized
  std::map<int, std::unique_ptr<Node>> child_nodes;
  Node* getChildNode(int move_idx);
  // Constructor, sets legal_mask and is_leaf
  Node(const chess::Board& board = chess::Board());
  // No copy or move
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&&) = default;
  Node& operator=(Node&&) = default;
};

struct GameTree {
  GameTree();
  std::unique_ptr<Node> root;
};

// Runs N number of MCTS simulations to select a move to play
void self_play(GameTree& game_tree);
void select_move(GameTree& game_tree);
void run_simulation(GameTree& game_tree);
Node* select_child(Node& node);

// Evaluator thread ------------------------------------------

void run_evaluator();

// Worker thread ---------------------------------------------

void run_worker();
void evaluate(Node& node);

// -----------------------------------------------------------

}  // namespace alphazero

#endif  // ALPHA_ZERO_H