#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <boost/fiber/all.hpp>
#include <chess.hpp>
#include <chess_utils.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <span>

#ifndef ALPHA_ZERO_H
#define ALPHA_ZERO_H

namespace alphazero {

using namespace std::chrono_literals;
using boost::fibers::buffered_channel;
using boost::fibers::channel_op_status;
using boost::fibers::future;
using boost::fibers::mutex;
using boost::fibers::promise;
using std::milli;
using std::chrono::duration;
using std::chrono::steady_clock;

struct Node;

using eval_request_t = std::pair<Node*, boost::fibers::promise<void>>;
using eval_channel_t = boost::fibers::buffered_channel<eval_request_t>;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
using duration_t = std::chrono::duration<int, std::milli>;

// Constants -------------------------------------------------

constexpr int kNumActions = 64 * 64;
constexpr int kInputSize = 7 * 8 * 8;

// Globals ---------------------------------------------------

inline std::unique_ptr<eval_channel_t> g_evaluation_queue;
inline bool g_stop_evaluator = false;

struct Config {
  int channel_size;
  int num_simulations;
  int batch_size;
  int num_games;
  int num_threads;
  duration_t eval_timeout;
  std::string model_path;
  std::string training_file;
  bool debug = false;
};

inline Config g_config;

// Game Tree functions ---------------------------------------

struct Node {
  const chess::Board board;
  Node* parent = nullptr;
  bool is_evaluated = false;
  bool is_leaf = false;
  int parent_action = -1;
  // value of this node as predicted by the model
  float value = 0.0f;
  // legal_mask[i] = true if action i is legal, false otherwise
  // policy[i] = prior probability of action i (from model)
  std::array<bool, kNumActions> legal_mask = {};
  std::array<float, kNumActions> policy = {};
  // child_visits[i] = number of visits for the i-th action
  // child_values[i] = sum of values for the i-th action
  std::array<int, kNumActions> child_visits = {};
  std::array<float, kNumActions> child_values = {};
  // Mutex to ensure only one fiber processes this node at a time
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
  // Legal mask, we need this during training too
  std::array<bool, kNumActions> legal_mask = {};
  // Policy vector at current state (as predicted by the model)
  std::array<float, kNumActions> policy = {};
  // Child visit counts (in policy order)
  std::array<int, kNumActions> child_visit_counts = {};
  // Value of current move (as predicted by the model)
  float value = 0.0f;
  // Final value (will be set at the end of the game)
  int final_value = 0;
};

struct Game {
  Game();
  std::unique_ptr<Node> root;
  std::vector<GameState> history;
};

// Runs N number of MCTS simulations to select a move to play
void self_play(Game& game, int game_id);
void update_root(Game& game, int action);
void save_game_state(Game& game);
void update_game_history(Game& game);
int select_move(Game& game);
void run_simulation(Game& game);
int select_action(Node& node);
void append_to_training_file(const Game& game);

// Evaluator thread ------------------------------------------

void run_evaluator();
void init_model();
std::vector<eval_request_t> get_requests_batch();
void cpu_stage();
void gpu_stage();
void post_stage();

// Worker thread---------------------------------------------

void run_worker();
void evaluate(Node& node);
void evaluate_leaf_node(Node& node);

// Logging and Debugging ---------------------------------

struct FileSink : absl::LogSink {
  explicit FileSink(const std::string& path);
  ~FileSink() override;
  void Send(const absl::LogEntry& e) override;

 private:
  std::ofstream ofs;
};

// To dump intermediate states to a file for debugging
void dump_game_tree_to_file(const Game& game, int g = 0, int m = 0,
                            int chosen_action = -1);

std::string board_to_string(const chess::Board& board);

// add time with absl str format in the format [HH:MM:SS.sss]
std::string timestamp();

// -----------------------------------------------------------

}  // namespace alphazero

#endif  // ALPHA_ZERO_H