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

// Constants -------------------------------------------------

constexpr int kNumActions = 64 * 64;

// Global Position Evaluation queue --------------------------

const int kChannelSize = 128;  // Must be a power of 2!
inline eval_channel_t g_evaluation_queue(kChannelSize);

// Game Tree functions ---------------------------------------

struct Node {
  chess::Board board;
  bool is_evaluated = false;
  bool is_leaf = false;
  int visit_count = 0;
  float value = 0.0f;
  std::array<float, kNumActions> policy = {};
  // Child node map, move idx -> Node*, Lazily initialized
  std::map<int, std::unique_ptr<Node>> child_nodes;
  Node* getChildNode(int move_idx);
  // Constructor
  Node(const chess::Board& board = chess::Board()) : board(board) {}
};

struct GameTree {
  GameTree();
  std::unique_ptr<Node> root;
};

// Evaluator thread ------------------------------------------

void run_evaluator();

// Worker thread ---------------------------------------------

void run_worker();
future<void> send_request(Node* node);
void work_on(const GameTree& game_tree);

// -----------------------------------------------------------

}  // namespace alphazero

#endif  // ALPHA_ZERO_H