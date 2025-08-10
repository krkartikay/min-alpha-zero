#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include "alpha_zero.h"

// new flag specific to this binary

// existing flags from main.cpp
ABSL_FLAG(int, channel_size, 128, "Channel size (must be power of 2)");
ABSL_FLAG(int, num_simulations, 200, "Number of MCTS simulations per move");
ABSL_FLAG(int, batch_size, 1000, "Number of nodes to process at once");
ABSL_FLAG(int, eval_timeout_ms, 1,
          "Timeout for evaluation requests in milliseconds");
ABSL_FLAG(int, num_games, 100, "Number of games to play in evaluation.");
ABSL_FLAG(std::string, model_path, "model.pt", "Path to the model file");
ABSL_FLAG(std::string, training_file, "training_data.bin",
          "File to store training data");
ABSL_FLAG(bool, debug, false,
          "Enable debug logging and intermediate state dumps");

namespace alphazero {

// Base chess agent interface
class ChessAgent {
 public:
  virtual ~ChessAgent() = default;
  virtual int select_action(Game& game) = 0;
  virtual std::string name() const = 0;
};

// Random agent implementation
class RandomAgent : public ChessAgent {
 private:
  mutable std::mt19937 rng;

 public:
  RandomAgent() : rng(std::random_device{}()) {}
  int select_action(Game& game) override {
    // Get legal moves from the current board
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, game.root->board);
    if (moves.empty()) return -1;

    // Convert to action indices and pick randomly
    std::vector<int> legal_actions;
    for (const auto& move : moves) {
      legal_actions.push_back(move_to_int(move));
    }

    std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
    return legal_actions[dist(rng)];
  }
  std::string name() const override { return "Random"; }
};

// MCTS agent using existing infrastructure
class MCTSAgent : public ChessAgent {
 public:
  MCTSAgent() {}
  int select_action(Game& game) override {
    int action = select_move(game);
    return action;
  }
  std::string name() const override { return "MCTS"; }
};

// Game result structure
struct GameResult {
  int moves_played;
  int agent_wins;
  int draws;
  int other_wins;
};

GameResult play_agent_vs_agent(ChessAgent& agent, ChessAgent& other,
                               int game_num) {
  Game game;  // Start with fresh game
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> coin(0, 1);

  bool agent_is_white = coin(rng) == 0;
  ChessAgent* white = agent_is_white ? &agent : &other;
  ChessAgent* black = agent_is_white ? &other : &agent;

  int moves_played = 0;

  LOG(INFO) << absl::StrFormat("Game start: White=%s, Black=%s", white->name(),
                               black->name());

  while (!game.root->is_leaf && moves_played < 100) {
    ChessAgent* current = (moves_played % 2 == 0) ? white : black;

    // Get action from current agent
    int action = current->select_action(game);
    if (action == -1) break;  // No legal moves

    // Let's store game tree state after the move
    if (g_config.debug) {
      dump_game_tree_to_file(game, game_num, moves_played, action);
    }

    VLOG(1) << absl::StrFormat("Agent %s selected action: %d", current->name(),
                               action);

    // For move logging
    chess::Move move = int_to_move(action, game.root->board);
    std::string move_str = chess::uci::moveToSan(game.root->board, move);
    std::string tab = (moves_played % 2 == 0) ? " " : "\t...";
    VLOG(1) << absl::StrFormat("%3d.%s%s", moves_played / 2 + 1, tab, move_str);
    VLOG(2) << absl::StrFormat("Board:\n%s", board_to_string(game.root->board));

    // Make the move using existing infrastructure
    update_root(game, action);

    moves_played++;
  }

  // Determine game result
  auto result = game.root->board.isGameOver();
  bool is_game_over = (result.first != chess::GameResultReason::NONE);
  chess::GameResult game_result = result.second;

  LOG(INFO) << absl::StrFormat(
      "Game finished. Moves: %d, Result: %s", moves_played,
      !is_game_over ? "Incomplete"
      : game_result == chess::GameResult::DRAW
          ? "Draw"
          : (game.root->board.sideToMove() == chess::Color::BLACK
                 ? "White wins"
                 : "Black wins"));

  if (!is_game_over || game_result == chess::GameResult::DRAW) {
    return {moves_played, 0, 1, 0};
  }

  bool white_won = (game.root->board.sideToMove() == chess::Color::BLACK);
  bool agent_won =
      (white_won && agent_is_white) || (!white_won && !agent_is_white);

  return {moves_played, agent_won ? 1 : 0, 0, agent_won ? 0 : 1};
}

void run_agent_tournament() {
  int total_moves = 0;
  int total_agent_wins = 0;
  int total_draws = 0;
  int total_other_wins = 0;

  int num_games = g_config.num_games;

  LOG(INFO) << absl::StrFormat(
      "Starting tournament: MCTS(%d sims) vs Random (%d games)",
      g_config.num_simulations, num_games);

  std::vector<GameResult> results(num_games);

  // Run games concurrently using boost::fibers
  boost::fibers::mutex results_mutex;
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < num_games; ++i) {
    fibers.emplace_back([i, &results, &results_mutex]() {
      alphazero::RandomAgent random_agent;
      alphazero::MCTSAgent mcts_agent;
      auto result = alphazero::play_agent_vs_agent(mcts_agent, random_agent, i);
      results[i] = result;
      LOG(INFO) << absl::StrFormat("Game %d: %s", i + 1,
                                   result.agent_wins ? "MCTS wins"
                                   : result.draws    ? "Draw"
                                                     : "Random wins");
    });
  }
  for (auto& f : fibers) {
    f.join();
  }

  for (const auto& result : results) {
    total_moves += result.moves_played;
    total_agent_wins += result.agent_wins;
    total_draws += result.draws;
    total_other_wins += result.other_wins;
  }

  LOG(INFO) << "Tournament results:";
  LOG(INFO) << absl::StrFormat("  Games played: %d", num_games);
  LOG(INFO) << absl::StrFormat("  MCTS wins: %d (%.1f%%)", total_agent_wins,
                               100.0f * total_agent_wins / num_games);
  LOG(INFO) << absl::StrFormat("  Draws: %d (%.1f%%)", total_draws,
                               100.0f * total_draws / num_games);
  LOG(INFO) << absl::StrFormat("  Random wins: %d (%.1f%%)", total_other_wins,
                               100.0f * total_other_wins / num_games);
  LOG(INFO) << absl::StrFormat("  Average moves per game: %.1f",
                               float(total_moves) / num_games);
}

void init_globals() {
  g_config.channel_size = absl::GetFlag(FLAGS_channel_size);
  g_config.num_simulations = absl::GetFlag(FLAGS_num_simulations);
  g_config.batch_size = absl::GetFlag(FLAGS_batch_size);
  g_config.eval_timeout =
      duration<int, milli>(absl::GetFlag(FLAGS_eval_timeout_ms));
  g_config.num_games = absl::GetFlag(FLAGS_num_games);
  g_config.model_path = absl::GetFlag(FLAGS_model_path);
  g_config.training_file = absl::GetFlag(FLAGS_training_file);
  g_config.debug = absl::GetFlag(FLAGS_debug);
  // initialize the evaluation queue with the channel size
  g_evaluation_queue = std::make_unique<eval_channel_t>(g_config.channel_size);
}

}  // namespace alphazero

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  alphazero::init_globals();

  LOG(INFO) << "Starting AlphaZero agent tournament...";

  // Start evaluator thread
  std::thread evaluator_thread(&alphazero::run_evaluator);

  // Run the agent tournament
  alphazero::run_agent_tournament();

  // Signal evaluator to stop and join thread
  alphazero::g_stop_evaluator = true;
  evaluator_thread.join();

  return 0;
}