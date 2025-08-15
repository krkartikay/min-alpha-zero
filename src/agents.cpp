#include "alpha_zero.h"

#include <random>
#include <vector>

namespace alphazero {

RandomAgent::RandomAgent() : rng(std::random_device{}()) {}

int RandomAgent::select_action(Game& game) {
  chess::Movelist moves;
  chess::movegen::legalmoves(moves, game.root->board);
  if (moves.empty()) return -1;

  std::vector<int> legal_actions;
  for (const auto& move : moves) {
    legal_actions.push_back(move_to_int(move));
  }

  std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
  return legal_actions[dist(rng)];
}

std::string RandomAgent::name() const { return "Random"; }

MCTSAgent::MCTSAgent() {}

int MCTSAgent::select_action(Game& game) {
  int action = game.selectMove();
  return action;
}

std::string MCTSAgent::name() const { return "MCTS"; }

GameResult play_agent_vs_agent(ChessAgent& agent, ChessAgent& other,
                               int game_num) {
  Game game;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> coin(0, 1);

  bool agent_is_white = coin(rng) == 0;
  ChessAgent* white = agent_is_white ? &agent : &other;
  ChessAgent* black = agent_is_white ? &other : &agent;

  int moves_played = 0;

  while (!game.root->is_leaf && moves_played < 100) {
    ChessAgent* current = (moves_played % 2 == 0) ? white : black;

    int action = current->select_action(game);
    if (action == -1) break;

    if (g_config.debug) {
      dump_game_tree_to_file(game, game_num, moves_played, action);
    }

    VLOG(1) << absl::StrFormat("Agent %s selected action: %d", current->name(),
                               action);

    chess::Move move = int_to_move(action, game.root->board);
    std::string move_str = chess::uci::moveToSan(game.root->board, move);
    std::string tab = (moves_played % 2 == 0) ? " " : "\t...";
    VLOG(1) << absl::StrFormat("%3d.%s%s", moves_played / 2 + 1, tab, move_str);
    VLOG(2) << absl::StrFormat("Board:\n%s", board_to_string(game.root->board));

    game.updateRoot(action);

    moves_played++;
  }

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

  boost::fibers::mutex results_mutex;
  std::vector<boost::fibers::fiber> fibers;
  for (int i = 0; i < num_games; ++i) {
    fibers.emplace_back([i, &results]() {
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

}  // namespace alphazero
