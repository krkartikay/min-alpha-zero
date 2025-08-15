#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/log/log_sink_registry.h>

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
  alphazero::FileSink file_sink("model_eval.log");
  absl::AddLogSink(&file_sink);

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