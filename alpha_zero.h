#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <boost/fiber/all.hpp>
#include <chess.hpp>
#include <chess_utils.hpp>
#include <chrono>
#include <iostream>

#ifndef ALPHA_ZERO_H
#define ALPHA_ZERO_H

namespace alphazero {

using namespace std::chrono_literals;
using boost::fibers::buffered_channel;
using boost::fibers::channel_op_status;
using boost::fibers::future;
using boost::fibers::mutex;
using boost::fibers::promise;
using std::chrono::steady_clock;

struct Node;

using eval_request_t = std::pair<Node*, boost::fibers::promise<void>>;
using eval_channel_t = boost::fibers::buffered_channel<eval_request_t>;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
using duration_t = std::chrono::duration<int, std::milli>;

// Config ----------------------------------------------------

const int kChannelSize = 128;           // Must be a power of 2!
const int kNumSimulations = 200;        // Number of MCTS simulations per move
const int kBatchSize = 1000;            // Number of nodes to process at once
const duration_t kEvalTimeout = 1ms;    // Timeout for evaluation requests
const std::string kModelPath = "model.pt";  // Path to the model file
const std::string kTrainingFile =
    "training_data.pt";  // File to store training data

// Constants -------------------------------------------------

constexpr int kNumActions = 64 * 64;
constexpr int kInputSize = 7 * 8 * 8;

// Globals ---------------------------------------------------

inline torch::jit::script::Module g_model;
inline eval_channel_t g_evaluation_queue(kChannelSize);

// Game Tree functions ---------------------------------------

struct Node {
  const chess::Board board;
  Node* parent = nullptr;
  bool is_evaluated = false;
  bool is_leaf = false;
  int visit_count = 0;
  float value = 0.0f;
  std::array<bool, kNumActions> legal_mask = {};
  std::array<float, kNumActions> policy = {};
  mutex is_processing_mutex;
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
  // For debugging purposes Only
  std::string move_history;
};

// For storing training data
struct GameState {
  // Board tensor at current state
  std::array<float, kInputSize> board_tensor = {};
  // Policy vector at current state
  std::array<float, kNumActions> policy = {};
  // Value of current move (will be +/- 1 or 0)
  float value = 0.0f;
};

struct Game {
  Game();
  std::unique_ptr<Node> root;
  std::vector<GameState> history;
  // These two will be populated at the end of the game.
  chess::GameResultReason result = chess::GameResultReason::NONE;
  chess::Color side_to_move = chess::Color::WHITE;
};

// Runs N number of MCTS simulations to select a move to play
void self_play(Game& game);
int select_move(Game& game);
void run_simulation(Game& game);
Node* select_child(Node& node);
void append_to_training_file(const Game& game);

// Evaluator thread ------------------------------------------

void run_evaluator();
void init_model();
std::vector<eval_request_t> get_requests_batch();
void process_batch(std::vector<eval_request_t> nodes);

// Worker thread---------------------------------------------

void run_worker();
void evaluate(Node& node);

// Logging and Debugging ---------------------------------

#define log(fmt, ...) \
  std::cout << absl::StrFormat("%s " fmt "\n", timestamp(), ##__VA_ARGS__)

// add time with absl str format in the format [HH:MM:SS.sss]
std::string timestamp();

// -----------------------------------------------------------

}  // namespace alphazero

#endif  // ALPHA_ZERO_H